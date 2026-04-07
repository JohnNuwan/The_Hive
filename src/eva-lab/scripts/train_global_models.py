"""Entraine MuZero sur historique reel puis execute la selection ADN."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import jax

package_root = Path(__file__).resolve().parents[1]
shared_root = package_root.parent / 'shared'
for candidate in (package_root, shared_root):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from eva_lab.arena import Arena
from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.genetic_updater import GeneticUpdater
from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.environment import TradingEnvironment
from eva_lab.muzero.jax_agent import JAXMuZeroAgent
from eva_lab.timescale_store import record_arena_result, record_training_dataset
from eva_lab.training_notifier import send_horizon_summary, send_training_horizon_started
from eva_lab.training_status import (
    append_training_log,
    write_arena_summary,
    load_training_status,
    mark_step_running,
    set_gold_precheck,
    write_precheck_summary,
    write_terminal_summary,
)
from eva_lab.training_utils import (
    build_inventory_report,
    build_muzero_market_data,
    get_horizon_history_bars,
    infer_family_from_symbols,
    load_history_frame,
    resolve_feature_profile,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.train_muzero")



def build_environment(symbol: str, config: MuZeroConfigV3) -> TradingEnvironment | None:
    """Construit l'environnement MuZero a partir de l'historique du symbole."""
    frame = load_history_frame(symbol, config.primary_timeframe)
    if frame is None:
        logger.warning("Historique absent pour %s sur %s.", symbol, config.primary_timeframe)
        return None

    history_bars = get_horizon_history_bars(config.horizon, env_prefix="MUZERO_HISTORY", fallback=4000)
    market_data = build_muzero_market_data(frame.tail(history_bars))
    if market_data.shape[0] < 240:
        logger.warning("Historique insuffisant pour %s sur %s.", symbol, config.primary_timeframe)
        return None

    max_steps = min(config.max_moves, market_data.shape[0] - 101)
    env = TradingEnvironment(data=market_data, symbol=symbol, config=config, max_steps=max_steps)
    setattr(env, "dataset_source", str(frame.attrs.get("dataset_source") or getattr(config, "dataset_source", "csv")))
    return env



def _is_gold_proxy_precheck_enabled(
    *,
    gate_profile: str,
    trial_mode: str | None,
    focus_symbols: list[str],
) -> bool:
    """Retourne vrai si le run courant doit executer un precheck Gold.

    Args:
        gate_profile (str): Profil de gate du run.
        trial_mode (str | None): Mode courant du trial.
        focus_symbols (list[str]): Univers explicite du run.

    Returns:
        bool: ``True`` si le precheck Gold est pertinent, ``False`` sinon.
    """

    normalized_focus = [str(symbol).strip().upper() for symbol in focus_symbols if str(symbol).strip()]
    return (
        str(gate_profile or "").strip().lower() == "gold_demo"
        and str(trial_mode or "").strip().lower() == "proxy_ga"
        and normalized_focus == ["XAUUSD"]
    )


def _evaluate_gold_precheck_verdict(
    *,
    metrics: dict[str, object],
    mechanics: dict[str, object],
) -> dict[str, object]:
    """Etablit un verdict de precheck Gold a partir des metriques intermediaires.

    La politique reste prudente: on coupe uniquement les echantillons
    manifestement faibles. Les cas ambigus continuent jusqu'au proxy complet.

    Args:
        metrics (dict[str, object]): Metriques consolidees du challenger.
        mechanics (dict[str, object]): Metriques de mecanique de position.

    Returns:
        dict[str, object]: Verdict structure ``pass``, ``fail`` ou
            ``inconclusive`` avec raison et mode d'echec stable.
    """

    evaluation_games = int(metrics.get("evaluation_games", 0) or 0)
    total_trades = int(metrics.get("total_trades", 0) or 0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    return_pct = float(metrics.get("return_pct", 0.0) or 0.0)
    net_realized_pct = float(metrics.get("net_realized_pct", 0.0) or 0.0)
    close_quality_score = float(mechanics.get("close_quality_score", 0.0) or 0.0)
    hold_drag_score = float(mechanics.get("hold_drag_score", 0.0) or 0.0)
    directional_bias = str(metrics.get("directional_bias") or "inactive").strip().lower()

    if evaluation_games <= 0 or total_trades <= 0:
        return {
            "status": "fail",
            "reason": "aucun_trade_exploitable",
            "failure_mode": "inactive",
        }

    if directional_bias in {"buy_heavy", "sell_heavy"} and return_pct <= 0.0 and profit_factor < 0.95:
        return {
            "status": "fail",
            "reason": "biais_directionnel_extreme",
            "failure_mode": directional_bias,
        }

    if (
        return_pct <= 0.0
        and net_realized_pct < 0.0
        and profit_factor < 0.90
        and close_quality_score <= 0.05
        and hold_drag_score >= 0.90
    ):
        return {
            "status": "fail",
            "reason": "profil_non_rentable_et_sorties_degradees",
            "failure_mode": "unprofitable",
        }

    if close_quality_score <= 0.02 and hold_drag_score >= 1.00 and total_trades >= 4:
        return {
            "status": "fail",
            "reason": "sorties_trop_degradees",
            "failure_mode": "bad_exit",
        }

    if (
        total_trades >= 6
        and return_pct > 0.0
        and net_realized_pct >= 0.0
        and profit_factor >= 1.0
        and directional_bias == "balanced"
        and close_quality_score >= 0.20
        and hold_drag_score <= 0.80
    ):
        return {
            "status": "pass",
            "reason": "signal_prometteur",
            "failure_mode": None,
        }

    return {
        "status": "inconclusive",
        "reason": "signal_ambigu_a_confirmer",
        "failure_mode": None,
    }


def main() -> dict[str, object]:
    """Orchestre l'entrainement MuZero d'un horizon strategique."""
    config = MuZeroConfigV3()
    horizon = config.horizon
    engine = "muzero"
    step_name = f"muzero_{horizon}"
    initial_family = str(getattr(config, "model_family", "")).strip() or None
    feature_profile_name = (
        str((getattr(config, "feature_profile", {}) or {}).get("profile_name") or "").strip() or None
    )
    mechanics_profile_version = str(getattr(config, "mechanics_profile_version", "")).strip() or None
    dataset_id = str(getattr(config, "dataset_id", "")).strip() or None
    dataset_source = str(getattr(config, "dataset_source", "")).strip() or None
    dataset_coverage = dict(getattr(config, "dataset_coverage", {}) or {})
    ga_status = str(os.getenv("TRAINING_GA_STATUS", "")).strip() or None
    ga_generation = (
        int(os.getenv("TRAINING_GA_GENERATION", "0"))
        if str(os.getenv("TRAINING_GA_GENERATION", "")).strip()
        else None
    )
    ga_trial = str(os.getenv("TRAINING_GA_TRIAL", "")).strip() or None
    ga_campaign_id = str(os.getenv("TRAINING_GA_CAMPAIGN_ID", "")).strip() or None
    ga_scope = str(os.getenv("TRAINING_GA_SCOPE", "")).strip() or None
    ga_parent_champion_id = str(os.getenv("TRAINING_GA_PARENT_CHAMPION_ID", "")).strip() or None
    ga_defer_promotion = str(os.getenv("TRAINING_GA_DEFER_PROMOTION", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    ga_seed_checkpoint_path = str(os.getenv("TRAINING_GA_SEED_CHECKPOINT_PATH", "")).strip() or None
    ga_genome_raw = str(os.getenv("TRAINING_GA_GENOME_JSON", "")).strip()
    ga_genome: dict[str, object] = {}
    if ga_genome_raw:
        try:
            decoded_genome = json.loads(ga_genome_raw)
        except json.JSONDecodeError:
            logger.warning("Genotype GA invalide ignore pour le run %s.", step_name)
        else:
            if isinstance(decoded_genome, dict):
                ga_genome = decoded_genome
    trial_mode = str(os.getenv("TRAINING_TRIAL_MODE", "")).strip() or None
    trial_cost_profile = str(os.getenv("TRAINING_TRIAL_COST_PROFILE", "")).strip() or None
    gate_profile = str(os.getenv("TRAINING_GATE_PROFILE", "")).strip() or "standard"
    mechanics_only_mode = bool(
        ga_scope == "mechanics_only"
        and ga_seed_checkpoint_path
        and ga_status in {"proxy", "proxy_ga", "final", "full"}
    )
    focus_symbols = list(dict.fromkeys(str(symbol).strip() for symbol in config.symbols if str(symbol).strip()))
    replay_cache_key = f"{engine}:{horizon}:{initial_family or 'global'}:{mechanics_profile_version or 'default'}"
    gold_precheck_enabled = _is_gold_proxy_precheck_enabled(
        gate_profile=gate_profile,
        trial_mode=trial_mode,
        focus_symbols=focus_symbols,
    )
    gold_precheck_step = max(1, int(os.getenv("MUZERO_GOLD_PRECHECK_STEP", "3000")))
    gold_precheck_games = max(1, int(os.getenv("MUZERO_GOLD_PRECHECK_GAMES", "6")))
    logger.info("Demarrage MuZero horizon=%s | timeframe=%s", horizon, config.primary_timeframe)
    logger.info("Peripheriques JAX: %s", jax.devices())
    logger.info("Inventaire historique: %s", build_inventory_report())
    logger.info("Univers MuZero: %s", config.symbols)
    append_training_log(
        f"MuZero {horizon} demarre sur {len(config.symbols)} symboles.",
        source="muzero",
    )
    send_training_horizon_started(horizon, len(config.symbols))
    set_gold_precheck(None)
    mark_step_running(
        step_name,
        engine=engine,
        phase="initialisation",
        horizon=horizon,
        family=initial_family,
        symbol_total=len(config.symbols),
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        feature_profile=feature_profile_name,
        mechanics_profile_version=mechanics_profile_version,
        ga_status=ga_status,
        ga_generation=ga_generation,
        ga_trial=ga_trial,
        ga_campaign_id=ga_campaign_id,
        ga_scope=ga_scope,
        ga_parent_champion_id=ga_parent_champion_id,
        trial_mode=trial_mode,
        trial_cost_profile=trial_cost_profile,
        focus_symbols=focus_symbols,
        gate_profile=gate_profile,
        replay_cache_status="warming",
        replay_cache_key=replay_cache_key,
        replay_cache_entries=0,
        replay_cache_source="memoire",
        dataset_coverage=dataset_coverage,
    )
    record_training_dataset(
        dict(getattr(config, "dataset_descriptor", {}) or {}),
        metadata={
            "engine": engine,
            "run_trigger": str(os.getenv("TRAINING_RUN_TRIGGER", "manual") or "manual"),
            "ga_status": ga_status,
            "ga_generation": str(ga_generation) if ga_generation is not None else None,
            "ga_trial": ga_trial,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "gate_profile": gate_profile,
        },
    )

    agent = JAXMuZeroAgent(config)
    weights_dir = Path(config.weights_path)
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    latest_path = weights_dir / f"muzero_{horizon}_latest.pkl"
    if mechanics_only_mode:
        try:
            agent.load(str(ga_seed_checkpoint_path))
            logger.info("Mode GA seede mecanique: checkpoint fixe charge depuis %s", ga_seed_checkpoint_path)
        except Exception as exc:
            raise RuntimeError(
                f"Impossible de charger le checkpoint seed MuZero {ga_seed_checkpoint_path}: {exc}"
            ) from exc
    elif latest_path.exists():
        try:
            agent.load(str(latest_path))
            logger.info("Reprise MuZero depuis %s", latest_path)
        except Exception as exc:
            logger.warning("Checkpoint MuZero ignore: %s", exc)

    games_per_symbol = int(os.getenv("MUZERO_GAMES_PER_SYMBOL", "12"))
    valid_symbols: list[str] = []
    total_games = 0

    family = infer_family_from_symbols(config.symbols, family=getattr(config, "model_family", None))
    feature_profile = resolve_feature_profile(horizon, family)
    dataset_source = str(getattr(config, "dataset_source", "csv") or "csv")
    dataset_descriptor = dict(getattr(config, "dataset_descriptor", {}) or {})
    dataset_id = str(dataset_descriptor.get("dataset_id") or "")
    start_time = datetime.now()
    last_metrics: dict[str, object] | None = None
    gold_precheck_payload: dict[str, object] | None = None
    gold_precheck_executed = False
    killed_after_precheck = False

    if mechanics_only_mode:
        valid_symbols = [str(symbol).strip() for symbol in config.symbols if str(symbol).strip()]
        total_games = 0
        last_metrics = {
            "mode": "seeded_ga_mechanics_only",
            "seed_checkpoint_path": ga_seed_checkpoint_path,
            "feature_profile": str(feature_profile.get("profile_name") or "").strip() or None,
            "mechanics_profile_version": mechanics_profile_version,
        }
        append_training_log(
            (
                f"MuZero {horizon}: mode GA seede mecanique active. "
                f"Checkpoint fixe={ga_seed_checkpoint_path}."
            ),
            source="muzero",
        )
    else:
        logger.info("Phase 1 - collecte historique par self-play guide")
        for symbol_index, symbol in enumerate(config.symbols, start=1):
            env = build_environment(symbol, config)
            if env is None:
                continue
            valid_symbols.append(symbol)
            append_training_log(
                f"MuZero {horizon}: collecte sur {symbol} ({symbol_index}/{len(config.symbols)}).",
                source="muzero",
            )
            for game_index in range(games_per_symbol):
                mark_step_running(
                    step_name,
                    engine=engine,
                    phase="collecte",
                    horizon=horizon,
                    family=initial_family,
                    symbol=symbol,
                    symbol_index=symbol_index,
                    symbol_total=len(config.symbols),
                    part_index=game_index + 1,
                    part_total=games_per_symbol,
                    dataset_id=dataset_id,
                    dataset_source=dataset_source,
                    feature_profile=feature_profile_name,
                    mechanics_profile_version=mechanics_profile_version,
                    ga_status=ga_status,
                    ga_generation=ga_generation,
                    ga_trial=ga_trial,
                    ga_campaign_id=ga_campaign_id,
                    ga_scope=ga_scope,
                    ga_parent_champion_id=ga_parent_champion_id,
                    trial_mode=trial_mode,
                    trial_cost_profile=trial_cost_profile,
                    focus_symbols=focus_symbols,
                    gate_profile=gate_profile,
                    replay_cache_status="warming",
                    replay_cache_key=replay_cache_key,
                    replay_cache_entries=agent.replay_buffer.size,
                    replay_cache_source="memoire",
                    dataset_coverage=dataset_coverage,
                )
                agent.play_game(env, exploration=True)
                summary = env.get_summary()
                total_games += 1
                logger.info(
                    "[%s] %s partie %s/%s | return=%.2f%% | trades=%s | buffer=%s",
                    horizon,
                    symbol,
                    game_index + 1,
                    games_per_symbol,
                    summary.get("return_pct", 0.0),
                    summary.get("total_trades", 0),
                    agent.replay_buffer.size,
                )

        if not valid_symbols:
            raise RuntimeError("Aucun symbole valide pour MuZero.")

        logger.info("Phase 2 - optimisation profonde (%s steps)", config.training_steps)
        append_training_log(
            f"MuZero {horizon}: optimisation profonde sur {config.training_steps} steps.",
            source="muzero",
        )

        for step in range(1, config.training_steps + 1):
            mark_step_running(
                step_name,
                engine=engine,
                phase="optimisation",
                horizon=horizon,
                family=family,
                symbol_total=len(valid_symbols),
                training_step_current=step,
                training_step_total=config.training_steps,
                dataset_id=dataset_id,
                dataset_source=dataset_source,
                feature_profile=(str(feature_profile.get("profile_name") or "").strip() or None),
                mechanics_profile_version=mechanics_profile_version,
                ga_status=ga_status,
                ga_generation=ga_generation,
                ga_trial=ga_trial,
                ga_campaign_id=ga_campaign_id,
                ga_scope=ga_scope,
                ga_parent_champion_id=ga_parent_champion_id,
                trial_mode=trial_mode,
                trial_cost_profile=trial_cost_profile,
                focus_symbols=focus_symbols,
                gate_profile=gate_profile,
                replay_cache_status="memoire",
                replay_cache_key=f"{engine}:{horizon}:{family}:{mechanics_profile_version or 'default'}",
                replay_cache_entries=agent.replay_buffer.size,
                replay_cache_source="memoire",
                dataset_coverage=dataset_coverage,
            )
            metrics = agent.train_step()
            if metrics is None:
                logger.warning("MuZero sans batch suffisant, arret a l'etape %s.", step)
                append_training_log(
                    f"MuZero {horizon}: arret anticipe a l'etape {step} faute de batch suffisant.",
                    level="WARNING",
                    source="muzero",
                )
                break
            last_metrics = metrics

            if (
                gold_precheck_enabled
                and not gold_precheck_executed
                and step >= gold_precheck_step
            ):
                checkpoint_path = weights_dir / f"muzero_{horizon}_gold_precheck_{step}.pkl"
                agent.save(str(checkpoint_path))
                append_training_log(
                    (
                        f"MuZero {horizon}: lancement du precheck Gold a l'etape "
                        f"{step} sur {gold_precheck_games} segments."
                    ),
                    source="muzero",
                )
                running_precheck = {
                    "status": "running",
                    "run_id": str(load_training_status().get("run_id") or "").strip() or None,
                    "trial_id": ga_trial,
                    "engine": engine,
                    "horizon": horizon,
                    "family": family,
                    "feature_profile": str(feature_profile.get("profile_name") or "").strip() or None,
                    "step": step,
                    "eval_symbols": list(focus_symbols),
                    "games": gold_precheck_games,
                    "reason": "precheck_en_cours",
                }
                set_gold_precheck(running_precheck)
                arena = Arena(weights_dir=config.weights_path)
                precheck_report = arena.evaluate_candidate(
                    checkpoint_path.stem,
                    horizon=horizon,
                    eval_symbols=list(focus_symbols),
                    games_per_symbol=gold_precheck_games,
                )
                precheck_metrics = dict(((precheck_report.get("challenger") or {}).get("metrics")) or {})
                precheck_mechanics = dict(precheck_metrics.get("metrics_by_position_mechanics") or {})
                verdict = _evaluate_gold_precheck_verdict(
                    metrics=precheck_metrics,
                    mechanics=precheck_mechanics,
                )
                gold_precheck_payload = {
                    "status": verdict.get("status"),
                    "run_id": str(load_training_status().get("run_id") or "").strip() or None,
                    "trial_id": ga_trial,
                    "engine": engine,
                    "horizon": horizon,
                    "family": family,
                    "feature_profile": str(feature_profile.get("profile_name") or "").strip() or None,
                    "step": step,
                    "eval_symbols": list(precheck_report.get("eval_symbols") or focus_symbols),
                    "games": int(precheck_report.get("games_per_symbol") or gold_precheck_games),
                    "metrics": precheck_metrics,
                    "metrics_by_position_mechanics": precheck_mechanics,
                    "reason": verdict.get("reason"),
                    "failure_mode": verdict.get("failure_mode"),
                }
                precheck_path = write_precheck_summary(gold_precheck_payload)
                gold_precheck_payload["path"] = str(precheck_path)
                set_gold_precheck(gold_precheck_payload)
                gold_precheck_executed = True
                append_training_log(
                    (
                        f"MuZero {horizon}: precheck Gold {gold_precheck_payload.get('status')} "
                        f"({gold_precheck_payload.get('reason')})."
                    ),
                    level="WARNING" if gold_precheck_payload.get("status") == "fail" else "INFO",
                    source="muzero",
                )
                if gold_precheck_payload.get("status") == "fail":
                    logger.warning(
                        "MuZero %s coupe apres precheck Gold a l'etape %s: %s",
                        horizon,
                        step,
                        gold_precheck_payload.get("reason"),
                    )
                    killed_after_precheck = True
                    break

            if step % 50 == 0:
                elapsed = max((datetime.now() - start_time).total_seconds(), 1.0)
                logger.info(
                    "[%s] step %05d/%05d | loss=%.4f | val=%.4f | rew=%.4f | pol=%.4f | %.2f steps/s",
                    horizon,
                    step,
                    config.training_steps,
                    float(metrics["loss_total"]),
                    float(metrics["loss_val"]),
                    float(metrics["loss_rew"]),
                    float(metrics["loss_pol"]),
                    step / elapsed,
                )
                append_training_log(
                    "MuZero "
                    f"{horizon}: step {step}/{config.training_steps} | "
                    f"loss={float(metrics['loss_total']):.4f}",
                    source="muzero",
                )

            if step % config.checkpoint_interval == 0:
                checkpoint_path = weights_dir / f"muzero_{horizon}_ckpt_{step}.pkl"
                agent.save(str(checkpoint_path))
                logger.info("Checkpoint MuZero sauvegarde: %s", checkpoint_path)

        agent.save(str(latest_path))
        logger.info("Checkpoint latest mis a jour: %s", latest_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    challenger_id = f"gen_{horizon}_{timestamp}"
    challenger_path = weights_dir / f"{challenger_id}.pkl"
    if mechanics_only_mode:
        shutil.copy2(Path(ga_seed_checkpoint_path), challenger_path)
    else:
        agent.save(str(challenger_path))
    latest_checkpoint_reference = (
        str(ga_seed_checkpoint_path)
        if mechanics_only_mode and ga_seed_checkpoint_path
        else str(latest_path)
    )
    active_run_id = str(load_training_status().get("run_id") or "").strip() or None

    if killed_after_precheck:
        precheck_metrics = dict((gold_precheck_payload or {}).get("metrics") or {})
        precheck_mechanics = dict((gold_precheck_payload or {}).get("metrics_by_position_mechanics") or {})
        promotion_result = {
            "status": "skipped",
            "reason": "gold_precheck_fail",
            "promotion_gate": {
                "allowed": False,
                "status": "blocked",
                "reason": "gold_precheck_fail",
                "gate_profile": gate_profile,
                "failure_mode": (gold_precheck_payload or {}).get("failure_mode"),
            },
        }
        terminal_summary = {
            "run_id": active_run_id,
            "sequence_id": str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None,
            "sequence_profile": str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None,
            "window_id": str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None,
            "trial_id": str(os.getenv("TRAINING_TRIAL_ID", "")).strip() or ga_trial,
            "engine": engine,
            "horizon": horizon,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "ga_trial": ga_trial,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "terminal_status": "completed",
            "failed_step": None,
            "failure_mode": (gold_precheck_payload or {}).get("failure_mode"),
            "arena_outcome": None,
            "promotion_gate": dict(promotion_result.get("promotion_gate") or {}),
            "metrics": precheck_metrics,
            "metrics_by_symbol": dict(precheck_metrics.get("metrics_by_symbol") or {}),
            "metrics_by_position_mechanics": precheck_mechanics,
            "artifact_state": {
                "precheck_report_present": bool((gold_precheck_payload or {}).get("path")),
                "arena_report_present": False,
                "battle_report_present": False,
                "promotion_present": True,
                "candidate_checkpoint_present": challenger_path.exists(),
            },
            "latest_candidate": challenger_id,
            "latest_verdict": {
                "status": "killed_after_precheck",
                "reason": (gold_precheck_payload or {}).get("reason"),
                "failure_mode": (gold_precheck_payload or {}).get("failure_mode"),
            },
            "gold_precheck": dict(gold_precheck_payload or {}),
            "precheck_status": (gold_precheck_payload or {}).get("status"),
        }
        terminal_summary_path = write_terminal_summary(terminal_summary)
        logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
        append_training_log(
            (
                f"MuZero {horizon}: trial coupe apres precheck Gold "
                f"({(gold_precheck_payload or {}).get('reason')})."
            ),
            level="WARNING",
            source="muzero",
        )
        return {
            "engine": engine,
            "horizon": horizon,
            "timeframe": config.primary_timeframe,
            "symbols": valid_symbols,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "dataset_descriptor": dataset_descriptor,
            "dataset_coverage": dict(getattr(config, "dataset_coverage", {}) or {}),
            "games_per_symbol": games_per_symbol,
            "total_games": total_games,
            "latest_checkpoint": str(latest_path),
            "challenger_path": str(challenger_path),
            "training_metrics": last_metrics,
            "ga_status": ga_status,
            "ga_generation": ga_generation,
            "ga_trial": ga_trial,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "precheck": dict(gold_precheck_payload or {}),
            "promotion": promotion_result,
            "terminal_summary_path": str(terminal_summary_path),
        }

    logger.info("Phase 3 - arena ADN")
    append_training_log(
        f"MuZero {horizon}: lancement de l'arena ADN.",
        source="muzero",
    )
    mark_step_running(
        step_name,
        engine=engine,
        phase="arena",
        horizon=horizon,
        family=family,
        symbol_total=len(valid_symbols),
        part_total=games_per_symbol,
        training_step_current=config.training_steps if last_metrics is not None else None,
        training_step_total=config.training_steps,
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        feature_profile=(str(feature_profile.get("profile_name") or "").strip() or None),
        mechanics_profile_version=mechanics_profile_version,
        ga_status=ga_status,
        ga_generation=ga_generation,
        ga_trial=ga_trial,
        ga_campaign_id=ga_campaign_id,
        ga_scope=ga_scope,
        ga_parent_champion_id=ga_parent_champion_id,
        trial_mode=trial_mode,
        trial_cost_profile=trial_cost_profile,
        replay_cache_status="memoire",
        replay_cache_key=f"{engine}:{horizon}:{family}:{mechanics_profile_version or 'default'}",
        replay_cache_entries=agent.replay_buffer.size,
        replay_cache_source="memoire",
        dataset_coverage=dataset_coverage,
    )
    genetic = GeneticUpdater()
    arena = Arena(weights_dir=config.weights_path)
    promoter = ChampionPromoter(weights_dir=config.weights_path, results_dir=config.results_path)
    live_path, live_meta = promoter.resolve_live_checkpoint(horizon)
    live_champion_id = str(live_meta.get("live_champion_id") or "").strip()
    champion_reference = None
    if live_path is not None:
        champion_reference = str(live_path)
    elif live_champion_id:
        champion_reference = live_champion_id
    else:
        champion_reference = genetic.get_champion(horizon=horizon)

    battle_report = arena.battle(challenger_id, champion_reference, horizon=horizon)
    logger.info("Verdict ADN %s: %s", horizon, battle_report["outcome"])
    logger.info(
        "Validation Arena %s: %s",
        horizon,
        battle_report.get("validation", {}),
    )

    challenger_metrics = battle_report["challenger"]["metrics"]
    registry_metrics = {
        "win_rate": {horizon: challenger_metrics.get("win_rate", 0.0)},
        "return_pct": {horizon: challenger_metrics.get("return_pct", 0.0)},
        "battles_won": {horizon: 1 if battle_report["outcome"] == "VICTORY" else 0},
        "horizon_accuracy": {horizon: challenger_metrics.get("win_rate", 0.0) / 100.0},
    }
    if ga_defer_promotion:
        promotion_gate = promoter.evaluate_promotion_gate(
            battle_report,
            gate_profile="standard",
        )
        live_comparison = promoter._compare_with_live_champion(
            horizon=horizon,
            candidate_gate=promotion_gate,
            engine=engine,
            challenger_id=challenger_id,
        )
        promotion_result = {
            "status": "candidate_only",
            "reason": "deferred_ga_selection",
            "engine": engine,
            "horizon": horizon,
            "source_path": str(challenger_path),
            "champion_paths": [],
            "promotion_gate": promotion_gate,
            "requested_gate_profile": promoter.normalize_gate_profile(gate_profile or "standard"),
            "live_gate_profile": "standard",
            "live_comparison": live_comparison,
            "promotion_state": "candidate_only",
            "deferred_promotion": True,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
        }
    else:
        promotion_result = promoter.promote_muzero_challenger(
            challenger_path=challenger_path,
            horizon=horizon,
            battle_report=battle_report,
            training_metrics=last_metrics,
            latest_checkpoint=(Path(ga_seed_checkpoint_path) if mechanics_only_mode and ga_seed_checkpoint_path else latest_path),
            challenger_id=challenger_id,
            gate_profile=gate_profile,
        )
    logger.info("Promotion live %s: %s", horizon, promotion_result.get("status"))
    append_training_log(
        "MuZero "
        f"{horizon}: arena={battle_report.get('outcome')} | "
        f"promotion={promotion_result.get('status')} | "
        f"gate={promotion_result.get('reason') or promotion_result.get('promotion_gate', {}).get('reason') or 'aucun'}",
        source="muzero",
    )
    genetic.register_new_generation(
        gen_id=challenger_id,
        metrics=registry_metrics,
        is_champion=promotion_result.get("status") == "promoted",
        horizon=horizon,
    )
    champion_paths = promotion_result.get("champion_paths", [])

    report_path = results_dir / f"arena_{horizon}_latest.json"
    report_payload = {
        "engine": engine,
        "horizon": horizon,
        "timeframe": config.primary_timeframe,
        "symbols": valid_symbols,
        "family": family,
        "feature_profile": feature_profile.get("profile_name"),
        "mechanics_profile_version": str(getattr(config, "mechanics_profile_version", "") or "") or None,
        "dataset_id": dataset_id,
        "dataset_source": dataset_source,
        "focus_symbols": focus_symbols,
        "gate_profile": gate_profile,
        "dataset_descriptor": dataset_descriptor,
        "dataset_coverage": dict(getattr(config, "dataset_coverage", {}) or {}),
        "games_per_symbol": games_per_symbol,
        "total_games": total_games,
        "latest_checkpoint": latest_checkpoint_reference,
        "challenger_path": str(challenger_path),
        "live_champion_reference": champion_reference,
        "live_champion_id": live_champion_id or None,
        "champion_paths": champion_paths,
        "training_metrics": last_metrics,
        "ga_status": str(os.getenv("TRAINING_GA_STATUS", "")).strip() or None,
        "ga_generation": (
            int(os.getenv("TRAINING_GA_GENERATION", "0"))
            if str(os.getenv("TRAINING_GA_GENERATION", "")).strip()
            else None
        ),
        "ga_trial": str(os.getenv("TRAINING_GA_TRIAL", "")).strip() or None,
        "ga_campaign_id": ga_campaign_id,
        "ga_scope": ga_scope,
        "ga_parent_champion_id": ga_parent_champion_id,
        "ga_defer_promotion": ga_defer_promotion,
        "ga_genome": ga_genome,
        "trial_mode": trial_mode,
        "trial_cost_profile": trial_cost_profile,
        "gold_precheck": dict(gold_precheck_payload or {}),
        "battle_report": battle_report,
        "promotion": promotion_result,
    }
    unique_report_path = write_arena_summary(report_payload)
    report_payload["battle_report_path"] = str(unique_report_path)
    report_path.write_text(json.dumps(report_payload, indent=2, default=float), encoding="utf-8")
    logger.info("Rapport MuZero ecrit dans %s", report_path)
    promotion_gate = dict(promotion_result.get("promotion_gate") or {})
    challenger_payload = dict((battle_report.get("challenger") or {}))
    challenger_metrics_full = dict(challenger_payload.get("metrics") or {})
    terminal_summary = {
        "run_id": active_run_id,
        "sequence_id": str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None,
        "sequence_profile": str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None,
        "window_id": str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None,
        "trial_id": str(os.getenv("TRAINING_TRIAL_ID", "")).strip() or ga_trial,
        "engine": engine,
        "horizon": horizon,
        "family": family,
        "feature_profile": feature_profile.get("profile_name"),
        "ga_trial": ga_trial,
        "ga_campaign_id": ga_campaign_id,
        "ga_scope": ga_scope,
        "ga_parent_champion_id": ga_parent_champion_id,
        "ga_genome": ga_genome,
        "trial_mode": trial_mode,
        "dataset_id": dataset_id,
        "dataset_source": dataset_source,
        "focus_symbols": focus_symbols,
        "gate_profile": gate_profile,
        "terminal_status": "completed",
        "failed_step": None,
        "failure_mode": (
            str(promotion_gate.get("failure_mode") or "").strip()
            or ("arena_defeat" if str(battle_report.get("outcome") or "").upper() != "VICTORY" else None)
        ),
        "arena_outcome": battle_report.get("outcome"),
        "promotion_gate": promotion_gate,
        "metrics": challenger_metrics_full,
        "metrics_by_symbol": dict(challenger_metrics_full.get("metrics_by_symbol") or {}),
        "metrics_by_position_mechanics": dict(
            challenger_metrics_full.get("metrics_by_position_mechanics") or {}
        ),
        "training_metrics": dict(last_metrics or {}),
        "challenger_path": str(challenger_path),
        "latest_checkpoint": latest_checkpoint_reference,
        "battle_report_path": str(unique_report_path),
        "live_comparison": dict(promotion_result.get("live_comparison") or {}),
        "artifact_state": {
            "arena_report_present": True,
            "battle_report_present": True,
            "promotion_present": bool(promotion_result),
            "candidate_checkpoint_present": challenger_path.exists(),
        },
        "latest_candidate": challenger_id,
        "latest_verdict": {
            "status": promotion_result.get("status"),
            "reason": promotion_result.get("reason") or promotion_gate.get("reason"),
            "failure_mode": promotion_gate.get("failure_mode"),
        },
        "gold_precheck": dict(gold_precheck_payload or {}),
        "precheck_status": (gold_precheck_payload or {}).get("status"),
    }
    terminal_summary_path = write_terminal_summary(terminal_summary)
    logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
    record_arena_result(
        report_payload,
        metadata={
            "engine": engine,
            "run_trigger": str(os.getenv("TRAINING_RUN_TRIGGER", "manual") or "manual"),
            "ga_status": report_payload.get("ga_status"),
            "ga_generation": report_payload.get("ga_generation"),
            "ga_trial": report_payload.get("ga_trial"),
            "trial_mode": report_payload.get("trial_mode"),
            "trial_cost_profile": report_payload.get("trial_cost_profile"),
        },
    )
    send_horizon_summary(horizon, report_payload, promotion_result)
    return report_payload


if __name__ == "__main__":
    summary = main()
    logger.info("MuZero termine: %s", summary)

