"""Lanceur UI CustomTkinter de l'agent follower distribue."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_path in (ROOT / "src" / "eva-banker", ROOT / "src" / "shared"):
    sys.path.insert(0, str(package_path))

from eva_banker.follower.ui import run_fleet_ui


def main() -> None:
    """Lance directement l'interface follower."""

    parser = argparse.ArgumentParser(description="Lance l'interface follower THE HIVE.")
    parser.add_argument(
        "--config",
        default="data/follower_agent/fleet.config.json",
        help="Chemin du fichier JSON de flotte.",
    )
    args = parser.parse_args()
    run_fleet_ui(Path(args.config))


if __name__ == "__main__":
    main()
