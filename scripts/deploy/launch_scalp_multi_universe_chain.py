"""Lance la chaine `MuZero -> Dreamer -> GNN` en arriere-plan sur Windows."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CHAIN_SCRIPT = ROOT_DIR / "scripts" / "deploy" / "run_scalp_multi_universe_chain.py"
LOG_DIR = ROOT_DIR / "data" / "logs"
STDOUT_LOG = LOG_DIR / "scalp_multi_universe_chain.stdout.log"
STDERR_LOG = LOG_DIR / "scalp_multi_universe_chain.stderr.log"


def main() -> int:
    """Demarre le watcher en arriere-plan et affiche son PID.

    Returns:
        int: Code retour shell.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if STDOUT_LOG.exists():
        STDOUT_LOG.unlink()
    if STDERR_LOG.exists():
        STDERR_LOG.unlink()

    env = os.environ.copy()
    env.setdefault("HIVE_SSH_PASSWORD", "Kumara-42/600")
    env.setdefault("HIVE_SUDO_PASSWORD", "Kumara-42/600")

    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    with STDOUT_LOG.open("w", encoding="utf-8") as stdout_handle, STDERR_LOG.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            [sys.executable, str(CHAIN_SCRIPT)],
            cwd=str(ROOT_DIR),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creation_flags,
            close_fds=True,
        )

    print(process.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
