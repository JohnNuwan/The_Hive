"""
Lance l'ingestion des historiques MT5 de toute la flotte Banker.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVA_LAB_SRC = PROJECT_ROOT / "src" / "eva-lab"
if str(EVA_LAB_SRC) not in sys.path:
    sys.path.insert(0, str(EVA_LAB_SRC))

from eva_lab.mt5_history_pipeline import IngestionConfig, ingest_mt5_fleet_history


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_mt5_fleet_history")


def parse_args() -> argparse.Namespace:
    """
    Construit les arguments CLI du pipeline d'ingestion.

    Returns:
        argparse.Namespace: Arguments normalises.
    """

    parser = argparse.ArgumentParser(
        description="Ingere les historiques MT5 des bankers actifs vers Shadow/Nemesis.",
    )
    parser.add_argument("--master-url", default="http://127.0.0.1:8100", help="URL du Banker maitre.")
    parser.add_argument("--days", type=int, default=30, help="Fenetre d'historique a importer.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "mt5_history_ingestion"),
        help="Racine des artefacts normalises.",
    )
    parser.add_argument(
        "--shadow-output-dir",
        default=str(PROJECT_ROOT / "data" / "shadow_learning" / "mt5_fleet"),
        help="Dossier Shadow Learning JSONL.",
    )
    parser.add_argument(
        "--state-file",
        default=str(PROJECT_ROOT / "data" / "mt5_history_ingestion" / "state.json"),
        help="Fichier d'etat des positions deja importees.",
    )
    parser.add_argument("--include-disabled", action="store_true", help="Inclut les targets copy desactivees.")
    parser.add_argument("--force", action="store_true", help="Reimporte les positions deja vues.")
    parser.add_argument("--timeout", type=float, default=8.0, help="Timeout HTTP par compte.")
    parser.add_argument("--max-deals", type=int, default=0, help="Limite defensive de deals par compte.")
    return parser.parse_args()


def main() -> None:
    """
    Point d'entree du pipeline CLI.
    """

    args = parse_args()
    config = IngestionConfig(
        master_url=args.master_url,
        days=max(1, int(args.days)),
        output_root=Path(args.output_root),
        shadow_output_dir=Path(args.shadow_output_dir),
        state_file=Path(args.state_file),
        include_disabled=bool(args.include_disabled),
        force=bool(args.force),
        timeout_seconds=max(1.0, float(args.timeout)),
        max_deals_per_account=max(0, int(args.max_deals)),
    )
    report = ingest_mt5_fleet_history(config)
    logger.info(
        "Ingestion terminee: %s positions, %s transitions Shadow, %s slices Nemesis.",
        report["positions_imported"],
        report["shadow_transitions"],
        report["nemesis_records"],
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
