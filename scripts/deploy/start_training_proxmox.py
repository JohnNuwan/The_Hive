"""Synchronise EVA Lab sur Proxmox puis lance l'entrainement nocturne."""

from __future__ import annotations

import os
import shlex
import stat
import sys
import time
from pathlib import Path

import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)
REMOTE_DIR = "/home/aza/The_Hive"
REMOTE_LOG = f"{REMOTE_DIR}/hive_nightly_training.log"
REMOTE_SCRIPT = f"{REMOTE_DIR}/scripts/run_nightly_training_remote.sh"
LOCAL_ROOT = Path(__file__).resolve().parents[2]

SYNC_FILES = [
    Path("src/eva-lab/eva_lab/training_utils.py"),
    Path("src/eva-lab/eva_lab/models/gnn_model.py"),
    Path("src/eva-lab/eva_lab/muzero/config.py"),
    Path("src/eva-lab/eva_lab/muzero/jax_agent.py"),
    Path("src/eva-lab/eva_lab/muzero/jax_mcts.py"),
    Path("src/eva-lab/eva_lab/muzero/jax_networks.py"),
    Path("src/eva-lab/eva_lab/muzero/jax_trainer.py"),
    Path("src/eva-lab/eva_lab/muzero/offline_trainer.py"),
    Path("src/eva-lab/eva_lab/arena.py"),
    Path("src/eva-lab/scripts/train_gnn.py"),
    Path("src/eva-lab/scripts/train_global_models.py"),
    Path("src/eva-lab/scripts/train_nightly_stack.py"),
    Path("src/eva-lab/pyproject.toml"),
    Path("src/eva-lab/Dockerfile.trainer"),
]

PASSTHROUGH_VARS = [
    "REBUILD_TRAINER_IMAGE",
    "RUN_TRAIN_GNN",
    "RUN_TRAIN_MUZERO",
    "RUN_TRAIN_DREAMER",
    "MUZERO_HORIZONS",
    "DREAMER_EPOCHS",
    "TRAIN_GNN_EPOCHS",
    "TRAIN_GNN_BATCH_SIZE",
    "MUZERO_TRAINING_STEPS",
    "MUZERO_GAMES_PER_SYMBOL",
]

REMOTE_LAUNCH_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=\"/home/aza/The_Hive\"
cd \"$PROJECT_DIR\"

echo \"[nightly] Arret temporaire des services GPU lourds\"
docker compose stop vllm comfyui || true

cleanup() {
  echo \"[nightly] Redemarrage des services GPU\"
  cd \"$PROJECT_DIR\"
  docker compose up -d vllm comfyui >/dev/null 2>&1 || true
}
trap cleanup EXIT

export RUN_TRAIN_GNN=\"${RUN_TRAIN_GNN:-1}\"
export RUN_TRAIN_MUZERO=\"${RUN_TRAIN_MUZERO:-1}\"
export RUN_TRAIN_DREAMER=\"${RUN_TRAIN_DREAMER:-1}\"
export REBUILD_TRAINER_IMAGE=\"${REBUILD_TRAINER_IMAGE:-1}\"
export MUZERO_HORIZONS=\"${MUZERO_HORIZONS:-scalp,intraday,swing}\"
export DREAMER_EPOCHS=\"${DREAMER_EPOCHS:-1500}\"
export TRAIN_GNN_EPOCHS=\"${TRAIN_GNN_EPOCHS:-500}\"
export TRAIN_GNN_BATCH_SIZE=\"${TRAIN_GNN_BATCH_SIZE:-64}\"
export MUZERO_TRAINING_STEPS=\"${MUZERO_TRAINING_STEPS:-15000}\"
export MUZERO_GAMES_PER_SYMBOL=\"${MUZERO_GAMES_PER_SYMBOL:-8}\"

if [ \"$REBUILD_TRAINER_IMAGE\" = \"1\" ]; then
  echo \"[nightly] Reconstruction de l'image eva-trainer\"
  if ! docker compose --progress plain build eva-trainer; then
    echo \"[nightly] Build eva-trainer en echec, poursuite avec l'image existante\"
  fi
fi

echo \"[nightly] Installation runtime JAX CUDA validee\"
docker compose run --rm \
  -e PYTHONPATH=/app/eva-lab:/app/shared \
  -e REBUILD_TRAINER_IMAGE=\"$REBUILD_TRAINER_IMAGE\" \
  -e RUN_TRAIN_GNN=\"$RUN_TRAIN_GNN\" \
  -e RUN_TRAIN_MUZERO=\"$RUN_TRAIN_MUZERO\" \
  -e RUN_TRAIN_DREAMER=\"$RUN_TRAIN_DREAMER\" \
  -e MUZERO_HORIZONS=\"$MUZERO_HORIZONS\" \
  -e DREAMER_EPOCHS=\"$DREAMER_EPOCHS\" \
  -e TRAIN_GNN_EPOCHS=\"$TRAIN_GNN_EPOCHS\" \
  -e TRAIN_GNN_BATCH_SIZE=\"$TRAIN_GNN_BATCH_SIZE\" \
  -e MUZERO_TRAINING_STEPS=\"$MUZERO_TRAINING_STEPS\" \
  -e MUZERO_GAMES_PER_SYMBOL=\"$MUZERO_GAMES_PER_SYMBOL\" \
  -v \"$PROJECT_DIR/data:/app/eva-lab/data\" \
  eva-trainer \
  bash -lc \"pip install --no-cache-dir --upgrade jax==0.4.23 jaxlib==0.4.23+cuda11.cudnn86 dm-haiku==0.0.11 optax==0.1.7 chex==0.1.85 flax==0.8.4 orbax-checkpoint==0.5.16 nest_asyncio==1.6.0 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && python -c 'import importlib.metadata as md; import jax, haiku, optax, chex; print(\\\"[nightly] jax\\\", jax.__version__); print(\\\"[nightly] jaxlib\\\", md.version(\\\"jaxlib\\\")); print(\\\"[nightly] haiku\\\", haiku.__version__); print(\\\"[nightly] optax\\\", optax.__version__); print(\\\"[nightly] chex\\\", chex.__version__); print(\\\"[nightly] backend\\\", jax.default_backend()); print(\\\"[nightly] devices\\\", jax.devices())' && python scripts/train_nightly_stack.py\"
"""

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def ensure_remote_parent(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    """Cree le dossier parent distant si necessaire."""
    parts = remote_path.split("/")[:-1]
    current = ""
    for part in parts:
        if not part:
            continue
        current += f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def upload_file(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
    """Envoie un fichier du workspace local vers le serveur."""
    ensure_remote_parent(sftp, remote_path)
    sftp.put(str(local_path), remote_path)
    print(f"SYNC {local_path} -> {remote_path}")


def run_command(client: paramiko.SSHClient, command: str, timeout: int = 120) -> tuple[str, str, int]:
    """Execute une commande SSH et retourne stdout, stderr, code."""
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "ignore")
    err = stderr.read().decode("utf-8", "ignore")
    code = stdout.channel.recv_exit_status()
    return out, err, code


def build_runtime_exports() -> str:
    """Construit les variables d'environnement a propager vers le script distant."""
    exports = []
    for name in PASSTHROUGH_VARS:
        value = os.getenv(name)
        if value:
            exports.append(f"export {name}={shlex.quote(value)}")
    return "; ".join(exports)


def start_training() -> None:
    """Synchronise les scripts EVA Lab et lance l'entrainement distant."""
    print(f"Connexion a Proxmox {HOST}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASS, timeout=15)
        print("Connexion SSH etablie.")

        sftp = client.open_sftp()
        for relative_path in SYNC_FILES:
            local_path = LOCAL_ROOT / relative_path
            remote_path = f"{REMOTE_DIR}/{relative_path.as_posix()}"
            upload_file(sftp, local_path, remote_path)

        ensure_remote_parent(sftp, REMOTE_SCRIPT)
        with sftp.file(REMOTE_SCRIPT, "w") as remote_file:
            remote_file.write(REMOTE_LAUNCH_SCRIPT)
        sftp.chmod(REMOTE_SCRIPT, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IROTH)
        sftp.close()
        print("Script distant mis a jour.")

        runtime_exports = build_runtime_exports()
        runtime_prefix = f"{runtime_exports}; " if runtime_exports else ""
        launch_cmd = (
            f"echo '{SUDO_PASS}' | sudo -S bash -lc 'cd {REMOTE_DIR} && "
            f"{runtime_prefix}nohup {REMOTE_SCRIPT} > {REMOTE_LOG} 2>&1 < /dev/null & echo $!'"
        )
        output, error, code = run_command(client, launch_cmd, timeout=30)
        if code != 0:
            raise RuntimeError(error or output or f"Code {code}")

        pid_lines = [line.strip() for line in output.splitlines() if line.strip()]
        pid = pid_lines[-1] if pid_lines else "inconnu"
        print(f"Entrainement nocturne lance. PID={pid}")

        time.sleep(8)
        tail_out, tail_err, _ = run_command(client, f"tail -n 40 {REMOTE_LOG}", timeout=30)
        print("\n--- Derniers logs ---")
        print(tail_out)
        if tail_err:
            print(tail_err)

    except Exception as exc:
        print(f"Erreur: {exc}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    start_training()


