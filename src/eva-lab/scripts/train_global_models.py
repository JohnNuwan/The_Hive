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
from eva_lab.training_status import append_training_log, mark_step_running
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
    replay_cache_key = f"{engine}:{horizon}:{initial_family or 'global'}:{mechanics_profile_version or 'default'}"
    logger.info("Demarrage MuZero horizon=%s | timeframe=%s", horizon, config.primary_timeframe)
    logger.info("Peripheriques JAX: %s", jax.devices())
    logger.info("Inventaire historique: %s", build_inventory_report())
    logger.info("Univers MuZero: %s", config.symbols)
    append_training_log(
        f"MuZero {horizon} demarre sur {len(config.symbols)} symboles.",
        source="muzero",
    )
    send_training_horizon_started(horizon, len(config.symbols))
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
        },
    )

    agent = JAXMuZeroAgent(config)
    weights_dir = Path(config.weights_path)
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    latest_path = weights_dir / f"muzero_{horizon}_latest.pkl"
    if latest_path.exists():
        try:
            agent.load(str(latest_path))
            logger.info("Reprise MuZero depuis %s", latest_path)
        except Exception as exc:
            logger.warning("Checkpoint MuZero ignore: %s", exc)

    games_per_symbol = int(os.getenv("MUZERO_GAMES_PER_SYMBOL", "12"))
    valid_symbols: list[str] = []
    total_games = 0

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
                trial_mode=trial_mode,
                trial_cost_profile=trial_cost_profile,
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

    family = infer_family_from_symbols(valid_symbols, family=getattr(config, "model_family", None))
    feature_profile = resolve_feature_profile(horizon, family)
    dataset_source = str(getattr(config, "dataset_source", "csv") or "csv")
    dataset_descriptor = dict(getattr(config, "dataset_descriptor", {}) or {})
    dataset_id = str(dataset_descriptor.get("dataset_id") or "")

    logger.info("Phase 2 - optimisation profonde (%s steps)", config.training_steps)
    start_time = datetime.now()
    last_metrics = None
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
            trial_mode=trial_mode,
            trial_cost_profile=trial_cost_profile,
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
    agent.save(str(challenger_path))

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
        "ga_status": str(os.getenv("TRAINING_GA_STATUS", "")).strip() or None,
        "ga_generation": (
            int(os.getenv("TRAINING_GA_GENERATION", "0"))
            if str(os.getenv("TRAINING_GA_GENERATION", "")).strip()
            else None
        ),
        "ga_trial": str(os.getenv("TRAINING_GA_TRIAL", "")).strip() or None,
        "trial_mode": trial_mode,
        "trial_cost_profile": trial_cost_profile,
        "battle_report": battle_report,
        "promotion": promotion_result,
    }
    report_path.write_text(json.dumps(report_payload, indent=2, default=float), encoding="utf-8")
    logger.info("Rapport MuZero ecrit dans %s", report_path)
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

