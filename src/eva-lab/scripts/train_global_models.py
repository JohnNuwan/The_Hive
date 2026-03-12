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
from eva_lab.training_notifier import send_horizon_summary
from eva_lab.training_utils import build_inventory_report, build_muzero_market_data, load_history_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.train_muzero")



def build_environment(symbol: str, config: MuZeroConfigV3) -> TradingEnvironment | None:
    """Construit l'environnement MuZero a partir de l'historique du symbole."""
    frame = load_history_frame(symbol, config.primary_timeframe)
    if frame is None:
        logger.warning("Historique absent pour %s sur %s.", symbol, config.primary_timeframe)
        return None

    market_data = build_muzero_market_data(frame.tail(4000))
    if market_data.shape[0] < 240:
        logger.warning("Historique insuffisant pour %s sur %s.", symbol, config.primary_timeframe)
        return None

    max_steps = min(config.max_moves, market_data.shape[0] - 101)
    return TradingEnvironment(data=market_data, symbol=symbol, config=config, max_steps=max_steps)



def main() -> dict[str, object]:
    """Orchestre l'entrainement MuZero d'un horizon strategique."""
    config = MuZeroConfigV3()
    horizon = config.horizon
    logger.info("Demarrage MuZero horizon=%s | timeframe=%s", horizon, config.primary_timeframe)
    logger.info("Peripheriques JAX: %s", jax.devices())
    logger.info("Inventaire historique: %s", build_inventory_report())
    logger.info("Univers MuZero: %s", config.symbols)

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
    for symbol in config.symbols:
        env = build_environment(symbol, config)
        if env is None:
            continue
        valid_symbols.append(symbol)
        for game_index in range(games_per_symbol):
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
    start_time = datetime.now()
    last_metrics = None

    for step in range(1, config.training_steps + 1):
        metrics = agent.train_step()
        if metrics is None:
            logger.warning("MuZero sans batch suffisant, arret a l'etape %s.", step)
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
    genetic = GeneticUpdater()
    arena = Arena(weights_dir=config.weights_path)
    champion_id = genetic.get_champion(horizon=horizon)
    battle_report = arena.battle(challenger_id, champion_id, horizon=horizon)
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
    promoter = ChampionPromoter(weights_dir=config.weights_path, results_dir=config.results_path)
    promotion_result = promoter.promote_muzero_challenger(
        challenger_path=challenger_path,
        horizon=horizon,
        battle_report=battle_report,
        training_metrics=last_metrics,
        latest_checkpoint=latest_path,
        challenger_id=challenger_id,
    )
    logger.info("Promotion live %s: %s", horizon, promotion_result.get("status"))
    genetic.register_new_generation(
        gen_id=challenger_id,
        metrics=registry_metrics,
        is_champion=promotion_result.get("status") == "promoted",
        horizon=horizon,
    )
    champion_paths = promotion_result.get("champion_paths", [])

    report_path = results_dir / f"arena_{horizon}_latest.json"
    report_payload = {
        "horizon": horizon,
        "timeframe": config.primary_timeframe,
        "symbols": valid_symbols,
        "games_per_symbol": games_per_symbol,
        "total_games": total_games,
        "latest_checkpoint": str(latest_path),
        "challenger_path": str(challenger_path),
        "champion_paths": champion_paths,
        "training_metrics": last_metrics,
        "battle_report": battle_report,
        "promotion": promotion_result,
    }
    report_path.write_text(json.dumps(report_payload, indent=2, default=float), encoding="utf-8")
    logger.info("Rapport MuZero ecrit dans %s", report_path)
    send_horizon_summary(horizon, report_payload, promotion_result)
    return report_payload


if __name__ == "__main__":
    summary = main()
    logger.info("MuZero termine: %s", summary)

