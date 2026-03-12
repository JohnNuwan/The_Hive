"""Rejoue uniquement la promotion live a partir des rapports Arena existants."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eva_lab.champion_promoter import ChampionPromoter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.reapply_champion_promotion")

HORIZONS = ["scalp", "intraday", "swing"]


def reapply_horizon(horizon: str, promoter: ChampionPromoter) -> dict[str, str | None]:
    """Rejoue la promotion live d'un horizon a partir du rapport existant.

    Args:
        horizon (str): Horizon strategique a traiter.
        promoter (ChampionPromoter): Promoteur live deja initialise.

    Returns:
        dict[str, str | None]: Resume de la promotion ou du blocage.

    Raises:
        FileNotFoundError: Si le rapport Arena n'existe pas.
        ValueError: Si le rapport ne contient pas de duel exploitable.
    """
    report_path = promoter.get_arena_report_path(horizon)
    if not report_path.exists():
        raise FileNotFoundError(f"Rapport Arena absent pour {horizon}: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    battle_report = report.get("battle_report", {}) or {}
    challenger = battle_report.get("challenger", {}) or {}
    challenger_path = report.get("challenger_path")

    if not battle_report or not challenger_path:
        raise ValueError(f"Rapport Arena incomplet pour {horizon}.")

    promotion = promoter.promote_muzero_challenger(
        challenger_path=challenger_path,
        horizon=horizon,
        battle_report=battle_report,
        training_metrics=report.get("training_metrics"),
        latest_checkpoint=report.get("latest_checkpoint"),
        challenger_id=challenger.get("id"),
    )
    report["promotion"] = promotion
    report["champion_paths"] = promotion.get("champion_paths", [])
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    logger.info(
        "Promotion rejouee pour %s: %s (%s).",
        horizon,
        promotion.get("status"),
        promotion.get("reason"),
    )
    return {
        "horizon": horizon,
        "status": promotion.get("status"),
        "reason": promotion.get("reason"),
        "gate_reason": (promotion.get("promotion_gate") or {}).get("reason"),
    }


def main() -> list[dict[str, str | None]]:
    """Traite les horizons demandes et affiche un resume JSON.

    Returns:
        list[dict[str, str | None]]: Resultats de promotion par horizon.
    """
    promoter = ChampionPromoter()
    requested_horizons = [
        horizon.strip().lower()
        for horizon in os.getenv("HIVE_REAPPLY_HORIZONS", ",".join(HORIZONS)).split(",")
        if horizon.strip()
    ]
    ordered_horizons = [horizon for horizon in HORIZONS if horizon in requested_horizons]

    results: list[dict[str, str | None]] = []
    for horizon in ordered_horizons:
        try:
            results.append(reapply_horizon(horizon, promoter))
        except Exception as exc:  # pragma: no cover - diagnostic operateur
            logger.exception("Reapplication impossible pour %s: %s", horizon, exc)
            results.append(
                {
                    "horizon": horizon,
                    "status": "error",
                    "reason": str(exc),
                    "gate_reason": None,
                }
            )

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
