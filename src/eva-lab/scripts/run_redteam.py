#!/usr/bin/env python3
"""
Lance une session Red Team complete : charge les trades live,
analyse les hard negatifs, genere des scenarios adverses,
et produit un rapport de robustesse du champion.

Usage:
    python scripts/run_redteam.py [--output data/redteam] [--window 30]

Options:
    --output DIR    Repertoire de sortie (defaut: data/redteam)
    --window DAYS   Fenetre d'analyse en jours (defaut: 30)
    --max-scenarios N  Nombre max de scenarios (defaut: 20)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva_lab.redteam import RedTeam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Red Team — evaluation de robustesse du champion")
    parser.add_argument("--output", default="data/redteam", help="Repertoire de sortie")
    parser.add_argument("--window", type=int, default=30, help="Fenetre d'analyse en jours")
    parser.add_argument("--max-scenarios", type=int, default=20, help="Nombre max de scenarios")
    args = parser.parse_args()

    logger.info("=== Red Team Session ===")
    logger.info("Fenetre: %d jours, max scenarios: %d", args.window, args.max_scenarios)

    champion_id = os.getenv("REDTEAM_CHAMPION_ID", "muzero_champion_scalp")

    redteam = RedTeam(
        trade_review_dir="data/live_trade_reviews",
        champion_id=champion_id,
    )

    n_loaded = redteam.load_trade_data(window_days=args.window)
    if n_loaded == 0:
        logger.warning("Aucun trade live trouve dans la fenetre %d jours.", args.window)
        sys.exit(1)

    failure_dist = redteam.analyze_failure_distribution()
    logger.info("Repartition des echecs: %s", failure_dist)

    fragility = redteam.compute_symbol_fragility()
    logger.info("Fragilite par symbole: %s", fragility)

    scenarios = redteam.generate_scenarios(max_scenarios=args.max_scenarios)
    logger.info("Scenarios generes: %d", len(scenarios))

    report = redteam.produce_report(scenarios)

    os.makedirs(args.output, exist_ok=True)
    report_path = redteam.save_report(report, output_dir=args.output)
    scenarios_path = redteam.save_scenarios(scenarios, output_dir=args.output)

    logger.info("=== Rapport Red Team ===")
    logger.info("  Score de survie du champion: %.1f/100", report.champion_survival_score)
    logger.info("  Hard negatifs trouves: %d", report.hard_negatives_found)
    logger.info("  Faiblesses detectees: %d symboles", len(report.weaknesses))
    for w in report.weaknesses:
        logger.info("    - %s: fragilite %.3f", w["symbol"], w["fragility_score"])
    logger.info("  Rapport: %s", report_path)
    logger.info("  Scenarios: %s", scenarios_path)


if __name__ == "__main__":
    main()
