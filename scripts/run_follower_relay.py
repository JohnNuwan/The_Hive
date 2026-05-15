"""Lanceur du relay central des agents followers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_path in (ROOT / "src" / "eva-banker", ROOT / "src" / "shared"):
    sys.path.insert(0, str(package_path))

from eva_banker.follower.relay_server import main


if __name__ == "__main__":
    main()
