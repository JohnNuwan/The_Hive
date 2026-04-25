"""Envoie un digest Telegram recurrent de l'etat des entrainements."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src" / "eva-lab"))
sys.path.insert(0, str(ROOT_DIR / "src" / "shared"))

from eva_lab.training_notifier import send_training_digest


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI du digest training.

    Returns:
        argparse.Namespace: Arguments normalises.
    """
    parser = argparse.ArgumentParser(
        description="Envoie un digest Telegram de l'etat training EVA Lab.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=0.0,
        help="Intervalle en minutes. 0 envoie un seul digest puis quitte.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="",
        help="Liste d'horizons separes par des virgules. Par defaut: MUZERO_HORIZONS.",
    )
    return parser.parse_args()


def _parse_horizons(raw_value: str) -> list[str] | None:
    """Normalise une liste d'horizons depuis la CLI.

    Args:
        raw_value (str): Valeur brute issue des arguments.

    Returns:
        list[str] | None: Horizons nettoyes ou ``None`` pour le comportement par defaut.
    """
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return None
    horizons = [item.strip().lower() for item in raw_text.split(",") if item.strip()]
    return horizons or None


def main() -> int:
    """Execute le digest training en mode one-shot ou boucle.

    Returns:
        int: Code de retour du processus.
    """
    args = parse_args()
    horizons = _parse_horizons(args.horizons)
    interval_minutes = max(float(args.interval_minutes or 0.0), 0.0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    while True:
        logger.info("Envoi du digest training Telegram.")
        send_training_digest(horizons=horizons)
        if interval_minutes <= 0.0:
            return 0
        logger.info("Prochain digest dans %.2f minutes.", interval_minutes)
        time.sleep(interval_minutes * 60.0)


if __name__ == "__main__":
    raise SystemExit(main())
