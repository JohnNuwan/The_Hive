"""Rejoue l'Arena contre les vrais champions ADN et resynchronise les rapports."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eva_lab.arena import Arena
from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.genetic_updater import GeneticUpdater

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.reconcile_champions")

HORIZONS = ["scalp", "intraday", "swing"]


def _build_registry_metrics(horizon: str, challenger_metrics: dict[str, Any], outcome: str) -> dict[str, Any]:
    """Construit les metriques ADN minimales a ecrire dans le registre.

    Args:
        horizon (str): Horizon cible.
        challenger_metrics (dict[str, Any]): Metriques Arena du challenger.
        outcome (str): Verdict Arena du duel.

    Returns:
        dict[str, Any]: Metriques compatibles avec le registre genetique.
    """
    win_rate = float(challenger_metrics.get("win_rate", 0.0) or 0.0)
    return_pct = float(challenger_metrics.get("return_pct", 0.0) or 0.0)
    return {
        "win_rate": {horizon: win_rate},
        "return_pct": {horizon: return_pct},
        "battles_won": {horizon: 1 if outcome == "VICTORY" else 0},
        "horizon_accuracy": {horizon: win_rate / 100.0},
    }


def reconcile_horizon(
    horizon: str,
    arena: Arena,
    promoter: ChampionPromoter,
    genetic: GeneticUpdater,
) -> dict[str, Any]:
    """Rejoue un duel Arena et met a jour le rapport d'horizon.

    Args:
        horizon (str): Horizon strategique a resynchroniser.
        arena (Arena): Instance Arena deja initialisee.
        promoter (ChampionPromoter): Promoteur live.
        genetic (GeneticUpdater): Registre ADN.

    Returns:
        dict[str, Any]: Resume du recalcul.

    Raises:
        FileNotFoundError: Si le rapport source est absent.
        ValueError: Si le challenger ne peut pas etre determine.
    """
    report_path = promoter.get_arena_report_path(horizon)
    if not report_path.exists():
        raise FileNotFoundError(f"Rapport Arena absent pour {horizon}: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    battle = report.get("battle_report", {}) or {}
    challenger_id = (battle.get("challenger", {}) or {}).get("id")
    challenger_path = report.get("challenger_path")
    latest_checkpoint = report.get("latest_checkpoint")
    training_metrics = report.get("training_metrics")

    if not challenger_id or not challenger_path:
        raise ValueError(f"Challenger introuvable pour {horizon}.")

    champion_id = genetic.get_champion(horizon)
    logger.info(
        "Reconciliation %s: challenger=%s vs champion=%s",
        horizon,
        challenger_id,
        champion_id,
    )

    battle_report = arena.battle(challenger_id, champion_id, horizon=horizon)
    promotion = promoter.promote_muzero_challenger(
        challenger_path=challenger_path,
        horizon=horizon,
        battle_report=battle_report,
        training_metrics=training_metrics,
        latest_checkpoint=latest_checkpoint,
        challenger_id=challenger_id,
    )

    challenger_metrics = battle_report.get("challenger", {}).get("metrics", {}) or {}
    genetic.register_new_generation(
        gen_id=challenger_id,
        metrics=_build_registry_metrics(horizon, challenger_metrics, str(battle_report.get("outcome", "DEFEAT"))),
        is_champion=promotion.get("status") == "promoted",
        horizon=horizon,
    )

    evaluation_games = int(challenger_metrics.get("evaluation_games", 0) or 0)
    evaluation_symbols = int(challenger_metrics.get("evaluation_symbols", 0) or 0)
    eval_symbols = battle_report.get("eval_symbols", report.get("symbols", []))
    refreshed_report = {
        "horizon": horizon,
        "timeframe": report.get("timeframe"),
        "symbols": eval_symbols,
        "games_per_symbol": (
            evaluation_games // evaluation_symbols if evaluation_games and evaluation_symbols else report.get("games_per_symbol")
        ),
        "total_games": evaluation_games or report.get("total_games"),
        "latest_checkpoint": latest_checkpoint,
        "challenger_path": challenger_path,
        "champion_paths": promotion.get("champion_paths", []),
        "training_metrics": training_metrics,
        "battle_report": battle_report,
        "promotion": promotion,
    }
    report_path.write_text(json.dumps(refreshed_report, indent=2, default=float), encoding="utf-8")
    
    # Garbage collection automatique pour ne pas saturer le disque
    promoter.garbage_collect_checkpoints(horizon)

    return {
        "horizon": horizon,
        "champion_id": champion_id,
        "challenger_id": challenger_id,
        "outcome": battle_report.get("outcome"),
        "promotion_status": promotion.get("status"),
        "promotion_reason": promotion.get("reason"),
        "gate_reason": (promotion.get("promotion_gate") or {}).get("reason"),
    }


def main() -> list[dict[str, Any]]:
    """Recalcule tous les horizons et affiche un resume JSON.

    Returns:
        list[dict[str, Any]]: Resultats par horizon.
    """
    arena = Arena()
    promoter = ChampionPromoter()
    genetic = GeneticUpdater()

    requested_horizons = [
        horizon.strip().lower()
        for horizon in os.getenv("HIVE_RECONCILE_HORIZONS", ",".join(HORIZONS)).split(",")
        if horizon.strip()
    ]
    ordered_horizons = [horizon for horizon in HORIZONS if horizon in requested_horizons]

    results: list[dict[str, Any]] = []
    for horizon in ordered_horizons:
        try:
            results.append(reconcile_horizon(horizon, arena, promoter, genetic))
        except Exception as exc:  # pragma: no cover - diagnostic operateur
            logger.exception("Reconciliation impossible pour %s: %s", horizon, exc)
            results.append(
                {
                    "horizon": horizon,
                    "status": "error",
                    "reason": str(exc),
                }
            )

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
