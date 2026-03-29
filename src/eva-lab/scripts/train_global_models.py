"""Entraine MuZero sur historique reel puis execute la selection ADN."""

from __future__ import annotations

import json
import logging
import os
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
    load_training_status,
    mark_step_running,
    set_gold_precheck,
    set_training_runtime_state,
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



def _parse_proxy_precheck_steps(raw_steps: str) -> list[int]:
    """Normalise la liste des etapes de precheck proxy.

    Args:
        raw_steps (str): Valeur CSV issue de l'environnement.

    Returns:
        list[int]: Etapes strictement positives, triees et dedoublonnees.
    """

    steps: list[int] = []
    for raw_part in str(raw_steps or "").split(","):
        candidate = str(raw_part).strip()
        if not candidate:
            continue
        try:
            step = int(candidate)
        except ValueError:
            continue
        if step > 0 and step not in steps:
            steps.append(step)
    return sorted(steps)


def _is_gold_proxy_precheck_enabled(
    *,
    gate_profile: str,
    trial_mode: str | None,
    focus_symbols: list[str],
) -> bool:
    """Retourne vrai si le run courant doit executer un precheck proxy.

    Args:
        gate_profile (str): Profil de gate du run.
        trial_mode (str | None): Mode courant du trial.
        focus_symbols (list[str]): Univers explicite du run.

    Returns:
        bool: ``True`` si le precheck proxy est pertinent, ``False`` sinon.
    """

    if str(trial_mode or "").strip().lower() != "proxy_ga":
        return False
    if str(os.getenv("MUZERO_PROXY_PRECHECK_ENABLED", "")).strip():
        return str(os.getenv("MUZERO_PROXY_PRECHECK_ENABLED", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    normalized_focus = [str(symbol).strip().upper() for symbol in focus_symbols if str(symbol).strip()]
    return (
        str(gate_profile or "").strip().lower() == "gold_demo"
        and normalized_focus == ["XAUUSD"]
    )


def _evaluate_gold_precheck_verdict(
    *,
    metrics: dict[str, object],
    mechanics: dict[str, object],
) -> dict[str, object]:
    """Etablit un verdict de precheck proxy a partir des metriques intermediaires.

    Args:
        metrics (dict[str, object]): Metriques consolidees du challenger.
        mechanics (dict[str, object]): Metriques de mecanique de position.

    Returns:
        dict[str, object]: Verdict structure ``pass`` ou ``fail`` avec raison stable.
    """

    evaluation_games = int(metrics.get("evaluation_games", 0) or 0)
    total_trades = int(metrics.get("total_trades", 0) or 0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    return_pct = float(metrics.get("return_pct", 0.0) or 0.0)
    net_realized_pct = float(metrics.get("net_realized_pct", 0.0) or 0.0)
    directional_imbalance = float(metrics.get("directional_imbalance", 1.0) or 1.0)
    close_quality_score = float(mechanics.get("close_quality_score", 0.0) or 0.0)
    hold_drag_score = float(mechanics.get("hold_drag_score", 0.0) or 0.0)
    directional_bias = str(metrics.get("directional_bias") or "inactive").strip().lower()

    if evaluation_games <= 0 or total_trades <= 0:
        return {
            "status": "fail",
            "reason": "insufficient_sample",
            "failure_mode": "inactive",
        }

    if profit_factor <= 1.0:
        return {
            "status": "fail",
            "reason": "profit_factor_too_low",
            "failure_mode": "unprofitable",
        }

    if return_pct <= 0.0:
        return {
            "status": "fail",
            "reason": "return_pct_non_positive",
            "failure_mode": "unprofitable",
        }

    if net_realized_pct <= 0.0:
        return {
            "status": "fail",
            "reason": "net_realized_pct_non_positive",
            "failure_mode": "unprofitable",
        }

    if close_quality_score < 0.40:
        return {
            "status": "fail",
            "reason": "close_quality_too_low",
            "failure_mode": "bad_exit",
        }

    if hold_drag_score > 0.80:
        return {
            "status": "fail",
            "reason": "hold_drag_too_high",
            "failure_mode": "bad_exit",
        }

    if directional_imbalance > 0.70:
        return {
            "status": "fail",
            "reason": "directional_imbalance_too_high",
            "failure_mode": directional_bias or "directional_imbalance",
        }

    if directional_bias in {"buy_heavy", "sell_heavy"} and return_pct <= 0.0:
        return {
            "status": "fail",
            "reason": "directional_bias_negative",
            "failure_mode": directional_bias,
        }

    return {"status": "pass", "reason": "proxy_precheck_pass", "failure_mode": None}


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
    trial_mode = str(os.getenv("TRAINING_TRIAL_MODE", "")).strip() or None
    trial_cost_profile = str(os.getenv("TRAINING_TRIAL_COST_PROFILE", "")).strip() or None
    gate_profile = str(os.getenv("TRAINING_GATE_PROFILE", "")).strip() or "standard"
    focus_symbols = list(dict.fromkeys(str(symbol).strip() for symbol in config.symbols if str(symbol).strip()))
    replay_cache_key = f"{engine}:{horizon}:{initial_family or 'global'}:{mechanics_profile_version or 'default'}"
    gold_precheck_enabled = _is_gold_proxy_precheck_enabled(
        gate_profile=gate_profile,
        trial_mode=trial_mode,
        focus_symbols=focus_symbols,
    )
    proxy_precheck_steps = _parse_proxy_precheck_steps(
        str(
            os.getenv(
                "MUZERO_PROXY_PRECHECK_STEPS",
                os.getenv("MUZERO_GOLD_PRECHECK_STEP", "3000"),
            )
        )
    )
    if not proxy_precheck_steps:
        proxy_precheck_steps = [3000]
    gold_precheck_games = max(
        1,
        int(
            os.getenv(
                "MUZERO_PROXY_PRECHECK_GAMES",
                os.getenv("MUZERO_GOLD_PRECHECK_GAMES", "6"),
            )
        ),
    )
    resume_checkpoint_path = str(os.getenv("MUZERO_RESUME_CHECKPOINT_PATH", "")).strip() or None
    resume_step_override = int(str(os.getenv("MUZERO_RESUME_STEP", "0")).strip() or 0)
    collection_timeout_seconds = max(
        0.0,
        float(str(os.getenv("MUZERO_COLLECTION_GAME_TIMEOUT_SECONDS", "240")).strip() or 0.0),
    )
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
    active_run_id = str(load_training_status().get("run_id") or "").strip() or None
    resume_step = 0
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
    set_training_runtime_state(
        last_successful_step=None,
        last_successful_step_at=None,
        train_step_phase="initialisation",
        phase_durations_ms={},
        resume_checkpoint_path=resume_checkpoint_path,
        resume_step=None,
        stall_detected=False,
        stall_reason=None,
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
    if resume_checkpoint_path:
        resume_path = Path(resume_checkpoint_path)
        if not resume_path.exists():
            raise ValueError(f"Checkpoint de reprise introuvable: {resume_checkpoint_path}")
        resume_metadata = agent.load(str(resume_path))
        resume_step = max(int(resume_metadata.get("step") or 0), resume_step_override)
        logger.info("Reprise MuZero explicite depuis %s a l'etape %s", resume_path, resume_step)
        append_training_log(
            f"MuZero {horizon}: reprise depuis {resume_path.name} a l'etape {resume_step}.",
            source="muzero",
        )
        set_training_runtime_state(
            resume_checkpoint_path=resume_checkpoint_path,
            resume_step=resume_step if resume_step > 0 else None,
        )
    elif latest_path.exists():
        try:
            agent.load(str(latest_path))
            logger.info("Reprise MuZero depuis %s", latest_path)
        except Exception as exc:
            logger.warning("Checkpoint MuZero ignore: %s", exc)

    games_per_symbol = int(os.getenv("MUZERO_GAMES_PER_SYMBOL", "12"))
    if resume_step > 0:
        games_per_symbol = max(
            1,
            int(
                str(
                    os.getenv(
                        "MUZERO_RESUME_GAMES_PER_SYMBOL",
                        str(games_per_symbol),
                    )
                ).strip()
                or games_per_symbol
            ),
        )
    collection_mode = str(
        os.getenv(
            "MUZERO_RESUME_COLLECTION_MODE" if resume_step > 0 else "MUZERO_COLLECTION_MODE",
            "policy_only" if resume_step > 0 else "mcts",
        )
        or ("policy_only" if resume_step > 0 else "mcts")
    ).strip().lower()
    if collection_mode not in {"mcts", "policy_only"}:
        raise ValueError(f"Mode de collecte MuZero invalide: {collection_mode}")
    valid_symbols: list[str] = []
    total_games = 0

    logger.info("Phase 1 - collecte historique par self-play guide")
    if resume_step > 0:
        logger.info(
            "Collecte MuZero de reprise en mode %s avec %s parties par symbole.",
            collection_mode,
            games_per_symbol,
        )
        append_training_log(
            (
                f"MuZero {horizon}: collecte de reprise en mode {collection_mode} "
                f"avec {games_per_symbol} parties par symbole."
            ),
            source="muzero",
        )
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
            agent.play_game(
                env,
                exploration=True,
                collection_mode=collection_mode,
                max_wall_time_seconds=collection_timeout_seconds or None,
            )
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

    family = infer_family_from_symbols(valid_symbols, family=getattr(config, "model_family", None))
    feature_profile = resolve_feature_profile(horizon, family)
    dataset_source = str(getattr(config, "dataset_source", "csv") or "csv")
    dataset_descriptor = dict(getattr(config, "dataset_descriptor", {}) or {})
    dataset_id = str(dataset_descriptor.get("dataset_id") or "")

    logger.info("Phase 2 - optimisation profonde (%s steps)", config.training_steps)
    start_time = datetime.now()
    last_metrics = None
    gold_precheck_payload: dict[str, object] | None = None
    start_optimisation_step = min(resume_step, config.training_steps)
    last_successful_step = start_optimisation_step
    executed_precheck_steps = {
        step for step in proxy_precheck_steps if start_optimisation_step >= step
    } if gold_precheck_enabled else set()
    killed_after_precheck = False
    if executed_precheck_steps:
        append_training_log(
            (
                f"MuZero {horizon}: precheck proxy saute pour les etapes "
                f"{sorted(executed_precheck_steps)} car la reprise commence apres ces seuils."
            ),
            source="muzero",
        )
    append_training_log(
        f"MuZero {horizon}: optimisation profonde sur {config.training_steps} steps.",
        source="muzero",
    )
    mark_step_running(
        step_name,
        engine=engine,
        phase="optimisation",
        horizon=horizon,
        family=family,
        symbol_total=len(valid_symbols),
        training_step_current=start_optimisation_step,
        training_step_total=config.training_steps,
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        feature_profile=(str(feature_profile.get("profile_name") or "").strip() or None),
        mechanics_profile_version=mechanics_profile_version,
        ga_status=ga_status,
        ga_generation=ga_generation,
        ga_trial=ga_trial,
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
    set_training_runtime_state(
        last_successful_step=start_optimisation_step if start_optimisation_step > 0 else None,
        last_successful_step_at=datetime.now().isoformat(),
        train_step_phase="ready",
        phase_durations_ms={},
        resume_checkpoint_path=resume_checkpoint_path,
        resume_step=start_optimisation_step if start_optimisation_step > 0 else None,
        stall_detected=False,
        stall_reason=None,
    )

    def _trace_train_step(phase_name: str) -> None:
        """Publie la sous-phase exacte du `train_step` MuZero."""

        set_training_runtime_state(
            train_step_phase=phase_name,
            stall_detected=False,
            stall_reason=None,
        )

    for step in range(start_optimisation_step + 1, config.training_steps + 1):
        step_result = agent.train_step(trace_hook=_trace_train_step)
        if step_result is None:
            logger.warning("MuZero sans batch suffisant, arret a l'etape %s.", step)
            append_training_log(
                f"MuZero {horizon}: arret anticipe a l'etape {step} faute de batch suffisant.",
                level="WARNING",
                source="muzero",
            )
            set_training_runtime_state(
                train_step_phase="waiting_for_batch",
                stall_detected=False,
                stall_reason=None,
            )
            break
        metrics = dict(step_result.get("metrics") or {})
        phase_durations_ms = dict(step_result.get("phase_durations_ms") or {})
        last_metrics = metrics
        last_successful_step = step
        step_completed_at = datetime.now().isoformat()
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
        set_training_runtime_state(
            last_successful_step=last_successful_step,
            last_successful_step_at=step_completed_at,
            train_step_phase="completed",
            phase_durations_ms=phase_durations_ms,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_step=start_optimisation_step if start_optimisation_step > 0 else None,
            stall_detected=False,
            stall_reason=None,
        )

        next_precheck_step = next(
            (
                candidate_step
                for candidate_step in proxy_precheck_steps
                if candidate_step not in executed_precheck_steps and step >= candidate_step
            ),
            None,
        )
        if gold_precheck_enabled and next_precheck_step is not None:
            checkpoint_path = weights_dir / f"muzero_{horizon}_proxy_precheck_{next_precheck_step}.pkl"
            agent.save(
                str(checkpoint_path),
                step=step,
                run_id=active_run_id,
                trial_id=ga_trial,
                gate_profile=gate_profile,
                focus_symbols=focus_symbols,
            )
            append_training_log(
                (
                    f"MuZero {horizon}: lancement du precheck proxy a l'etape "
                    f"{next_precheck_step} sur {gold_precheck_games} segments."
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
                "step": next_precheck_step,
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
                "step": next_precheck_step,
                "eval_symbols": list(precheck_report.get("eval_symbols") or focus_symbols),
                "games": int(precheck_report.get("games_per_symbol") or gold_precheck_games),
                "metrics": precheck_metrics,
                "metrics_by_position_mechanics": precheck_mechanics,
                "reason": verdict.get("reason"),
                "failure_mode": verdict.get("failure_mode"),
                "early_kill_reason": verdict.get("reason") if verdict.get("status") == "fail" else None,
            }
            precheck_path = write_precheck_summary(gold_precheck_payload)
            gold_precheck_payload["path"] = str(precheck_path)
            set_gold_precheck(gold_precheck_payload)
            executed_precheck_steps.add(next_precheck_step)
            append_training_log(
                (
                    f"MuZero {horizon}: precheck proxy {gold_precheck_payload.get('status')} "
                    f"({gold_precheck_payload.get('reason')})."
                ),
                level="WARNING" if gold_precheck_payload.get("status") == "fail" else "INFO",
                source="muzero",
            )
            if gold_precheck_payload.get("status") == "fail":
                logger.warning(
                    "MuZero %s coupe apres precheck proxy a l'etape %s: %s",
                    horizon,
                    next_precheck_step,
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
            agent.save(
                str(checkpoint_path),
                step=step,
                run_id=active_run_id,
                trial_id=ga_trial,
                gate_profile=gate_profile,
                focus_symbols=focus_symbols,
            )
            logger.info("Checkpoint MuZero sauvegarde: %s", checkpoint_path)

    final_checkpoint_step = last_successful_step if last_successful_step > 0 else start_optimisation_step
    agent.save(
        str(latest_path),
        step=final_checkpoint_step,
        run_id=active_run_id,
        trial_id=ga_trial,
        gate_profile=gate_profile,
        focus_symbols=focus_symbols,
    )
    logger.info("Checkpoint latest mis a jour: %s", latest_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    challenger_id = f"gen_{horizon}_{timestamp}"
    challenger_path = weights_dir / f"{challenger_id}.pkl"
    agent.save(
        str(challenger_path),
        step=final_checkpoint_step,
        run_id=active_run_id,
        trial_id=ga_trial,
        gate_profile=gate_profile,
        focus_symbols=focus_symbols,
    )
    active_run_id = str(load_training_status().get("run_id") or "").strip() or active_run_id

    if killed_after_precheck:
        precheck_metrics = dict((gold_precheck_payload or {}).get("metrics") or {})
        precheck_mechanics = dict((gold_precheck_payload or {}).get("metrics_by_position_mechanics") or {})
        promotion_result = {
            "status": "skipped",
            "reason": "proxy_precheck_fail",
            "promotion_gate": {
                "allowed": False,
                "status": "blocked",
                "reason": "proxy_precheck_fail",
                "gate_profile": gate_profile,
                "failure_mode": (gold_precheck_payload or {}).get("failure_mode"),
                "early_kill_reason": (gold_precheck_payload or {}).get("early_kill_reason"),
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
            "resume_checkpoint_path": resume_checkpoint_path,
            "resume_step": start_optimisation_step if start_optimisation_step > 0 else None,
            "latest_candidate": challenger_id,
            "latest_verdict": {
                "status": "killed_after_precheck",
                "reason": (gold_precheck_payload or {}).get("reason"),
                "failure_mode": (gold_precheck_payload or {}).get("failure_mode"),
            },
            "gold_precheck": dict(gold_precheck_payload or {}),
            "precheck_status": (gold_precheck_payload or {}).get("status"),
            "early_kill_reason": (gold_precheck_payload or {}).get("early_kill_reason"),
        }
        terminal_summary_path = write_terminal_summary(terminal_summary)
        logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
        append_training_log(
            (
                f"MuZero {horizon}: trial coupe apres precheck proxy "
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
            "resume_checkpoint_path": resume_checkpoint_path,
            "resume_step": start_optimisation_step if start_optimisation_step > 0 else None,
            "precheck": dict(gold_precheck_payload or {}),
            "early_kill_reason": (gold_precheck_payload or {}).get("early_kill_reason"),
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
        training_step_current=last_successful_step if last_metrics is not None else None,
        training_step_total=config.training_steps,
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        feature_profile=(str(feature_profile.get("profile_name") or "").strip() or None),
        mechanics_profile_version=mechanics_profile_version,
        ga_status=ga_status,
        ga_generation=ga_generation,
        ga_trial=ga_trial,
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
    promotion_result = promoter.promote_muzero_challenger(
        challenger_path=challenger_path,
        horizon=horizon,
        battle_report=battle_report,
        training_metrics=last_metrics,
        latest_checkpoint=latest_path,
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
        "latest_checkpoint": str(latest_path),
        "challenger_path": str(challenger_path),
        "live_champion_reference": champion_reference,
        "live_champion_id": live_champion_id or None,
        "champion_paths": champion_paths,
        "training_metrics": last_metrics,
        "resume_checkpoint_path": resume_checkpoint_path,
        "resume_step": start_optimisation_step if start_optimisation_step > 0 else None,
        "ga_status": str(os.getenv("TRAINING_GA_STATUS", "")).strip() or None,
        "ga_generation": (
            int(os.getenv("TRAINING_GA_GENERATION", "0"))
            if str(os.getenv("TRAINING_GA_GENERATION", "")).strip()
            else None
        ),
        "ga_trial": str(os.getenv("TRAINING_GA_TRIAL", "")).strip() or None,
        "trial_mode": trial_mode,
        "trial_cost_profile": trial_cost_profile,
        "gold_precheck": dict(gold_precheck_payload or {}),
        "battle_report": battle_report,
        "promotion": promotion_result,
    }
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
        "artifact_state": {
            "arena_report_present": True,
            "battle_report_present": True,
            "promotion_present": bool(promotion_result),
            "candidate_checkpoint_present": challenger_path.exists(),
        },
        "resume_checkpoint_path": resume_checkpoint_path,
        "resume_step": start_optimisation_step if start_optimisation_step > 0 else None,
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

