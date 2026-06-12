"""Relance un entrainement distant via le lanceur Proxmox moderne."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.deploy import start_training_proxmox as remote_training


def main() -> None:
    """Synchronise le payload utile puis lance le run distant standard."""

    print("Relance de l'entrainement via le lanceur distant moderne...")
    remote_training.start_training()


if __name__ == "__main__":
    main()
