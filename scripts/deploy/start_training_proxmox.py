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
    Path("src/eva-lab/eva_lab/muzero/dreamer_networks.py"),
    Path("src/eva-lab/eva_lab/muzero/dreamer_trainer.py"),
    Path("src/eva-lab/eva_lab/muzero/imagination.py"),
    Path("src/eva-lab/eva_lab/muzero/jax_agent.py"),
    Path("src/eva-lab/eva_lab/muzero/jax_mcts.py"),
    Path("src/eva-lab/eva_lab/muzero/jax_networks.py"),
    Path("src/eva-lab/eva_lab/muzero/jax_trainer.py"),
    Path("src/eva-lab/eva_lab/muzero/offline_trainer.py"),
    Path("src/eva-lab/eva_lab/muzero/replay_buffer.py"),
    Path("src/eva-lab/eva_lab/muzero/rssm.py"),
    Path("src/eva-lab/eva_lab/arena.py"),
    Path("src/eva-lab/eva_lab/champion_promoter.py"),
    Path("src/eva-lab/eva_lab/shadow_dataset.py"),
    Path("src/eva-lab/eva_lab/training_notifier.py"),
    Path("src/eva-lab/scripts/train_gnn.py"),
    Path("src/eva-lab/scripts/train_global_models.py"),
    Path("src/eva-lab/scripts/train_nightly_stack.py"),
    Path("src/eva-lab/pyproject.toml"),
    Path("src/eva-lab/Dockerfile.trainer"),
]

SYNC_DIRS = [
    Path("data/history"),
    Path("data/shadow_learning/imports"),
]

PASSTHROUGH_VARS = [
    "TRAINING_PROFILE",
    "TRAINING_AUTOMATION_MODE",
    "TRAINING_REFRESH_AFTER_HOURS",
    "TRAINING_MAX_CHAMPION_AGE_HOURS",
    "TRAINING_MIN_SHADOW_RECORDS",
    "NIGHTLY_KEEP_VLLM",
    "NIGHTLY_STOP_COMFYUI",
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
    "MUZERO_BATCH_SIZE",
    "MUZERO_NUM_SIMULATIONS",
    "MUZERO_MAX_MOVES",
    "MUZERO_MAX_SYMBOLS",
    "TRAINING_SYMBOLS",
    "ARENA_MAX_SYMBOLS",
    "ARENA_GAMES_PER_SYMBOL",
    "ARENA_MIN_GAMES",
    "ARENA_MIN_SYMBOLS",
    "ARENA_MIN_SCORE_EDGE",
    "DREAMER_BATCH_SIZE",
    "DREAMER_SEQUENCE_LENGTH",
    "DREAMER_SEQUENCE_STRIDE",
    "DREAMER_MAX_START_STATES",
    "DREAMER_HIDDEN_STATE_SIZE",
    "DREAMER_NETWORK_HIDDEN_DIMS",
    "DREAMER_NUM_UNROLL_STEPS",
    "DREAMER_REPLAY_MAX_GAMES",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "MUZERO_PROMOTION_MIN_TOTAL_TRADES",
    "MUZERO_PROMOTION_MIN_EVAL_GAMES",
    "MUZERO_PROMOTION_MIN_EVAL_SYMBOLS",
    "MUZERO_PROMOTION_MIN_WIN_RATE",
    "MUZERO_PROMOTION_MIN_RETURN_PCT",
    "MUZERO_PROMOTION_MIN_PROFIT_FACTOR",
    "MUZERO_PROMOTION_MIN_EXPECTANCY_PCT",
    "MUZERO_PROMOTION_MAX_DRAWDOWN_PCT",
    "MUZERO_PROMOTION_MIN_POSITIVE_EPISODE_RATE",
    "TELEGRAM_NOTIFY_TRAINING",
]

REMOTE_LAUNCH_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=\"/home/aza/The_Hive\"
cd \"$PROJECT_DIR\"

VLLM_STOPPED=0
COMFYUI_STOPPED=0

export TRAINING_PROFILE=\"${TRAINING_PROFILE:-smart}\"
export TRAINING_AUTOMATION_MODE=\"${TRAINING_AUTOMATION_MODE:-smart}\"

if [ \"$TRAINING_PROFILE\" = \"research\" ]; then
  echo \"[nightly] Profil research actif: priorite a l'entrainement massif\"
  : \"${NIGHTLY_KEEP_VLLM:=0}\"
  : \"${RUN_TRAIN_DREAMER:=0}\"
  : \"${MUZERO_MAX_SYMBOLS:=0}\"
  : \"${ARENA_MAX_SYMBOLS:=0}\"
  : \"${MUZERO_GAMES_PER_SYMBOL:=20}\"
  : \"${ARENA_GAMES_PER_SYMBOL:=8}\"
  : \"${ARENA_MIN_GAMES:=24}\"
  : \"${ARENA_MIN_SYMBOLS:=6}\"
  : \"${MUZERO_TRAINING_STEPS:=32000}\"
fi

export NIGHTLY_KEEP_VLLM=\"${NIGHTLY_KEEP_VLLM:-1}\"
export NIGHTLY_STOP_COMFYUI=\"${NIGHTLY_STOP_COMFYUI:-1}\"

if [ \"$NIGHTLY_KEEP_VLLM\" = \"1\" ]; then
  echo \"[nightly] vLLM conserve en ligne pour le live\"
else
  echo \"[nightly] Arret temporaire de vLLM pour liberer le GPU\"
  docker compose stop vllm || true
  VLLM_STOPPED=1
fi

if [ \"$NIGHTLY_STOP_COMFYUI\" = \"1\" ]; then
  echo \"[nightly] Arret temporaire de ComfyUI\"
  docker compose stop comfyui || true
  COMFYUI_STOPPED=1
fi

cleanup() {
  echo \"[nightly] Redemarrage des services GPU\"
  cd \"$PROJECT_DIR\"
  if [ \"$COMFYUI_STOPPED\" = \"1\" ]; then
    docker compose up -d comfyui >/dev/null 2>&1 || true
  fi
  if [ \"$VLLM_STOPPED\" = \"1\" ]; then
    docker compose up -d vllm >/dev/null 2>&1 || true
    echo \"[nightly] Attente de vLLM apres redemarrage\"
    for attempt in $(seq 1 18); do
      if curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo \"[nightly] vLLM operationnel\"
        break
      fi
      if [ \"$attempt\" -eq 6 ]; then
        echo \"[nightly] vLLM encore indisponible, redemarrage complementaire\"
        docker compose restart vllm >/dev/null 2>&1 || true
      fi
      sleep 10
    done
  fi
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
export MUZERO_TRAINING_STEPS=\"${MUZERO_TRAINING_STEPS:-24000}\"
export MUZERO_GAMES_PER_SYMBOL=\"${MUZERO_GAMES_PER_SYMBOL:-12}\"
export MUZERO_BATCH_SIZE=\"${MUZERO_BATCH_SIZE:-32}\"
export MUZERO_NUM_SIMULATIONS=\"${MUZERO_NUM_SIMULATIONS:-100}\"
export MUZERO_MAX_MOVES=\"${MUZERO_MAX_MOVES:-300}\"
export MUZERO_MAX_SYMBOLS=\"${MUZERO_MAX_SYMBOLS:-12}\"
export TRAINING_SYMBOLS=\"${TRAINING_SYMBOLS:-}\"
export ARENA_MAX_SYMBOLS=\"${ARENA_MAX_SYMBOLS:-12}\"
export ARENA_GAMES_PER_SYMBOL=\"${ARENA_GAMES_PER_SYMBOL:-6}\"
export ARENA_MIN_GAMES=\"${ARENA_MIN_GAMES:-12}\"
export ARENA_MIN_SYMBOLS=\"${ARENA_MIN_SYMBOLS:-3}\"
export ARENA_MIN_SCORE_EDGE=\"${ARENA_MIN_SCORE_EDGE:-0.5}\"
export DREAMER_BATCH_SIZE=\"${DREAMER_BATCH_SIZE:-8}\"
export DREAMER_SEQUENCE_LENGTH=\"${DREAMER_SEQUENCE_LENGTH:-64}\"
export DREAMER_SEQUENCE_STRIDE=\"${DREAMER_SEQUENCE_STRIDE:-32}\"
export DREAMER_MAX_START_STATES=\"${DREAMER_MAX_START_STATES:-256}\"
export DREAMER_HIDDEN_STATE_SIZE=\"${DREAMER_HIDDEN_STATE_SIZE:-128}\"
export DREAMER_NETWORK_HIDDEN_DIMS=\"${DREAMER_NETWORK_HIDDEN_DIMS:-256,256}\"
export DREAMER_NUM_UNROLL_STEPS=\"${DREAMER_NUM_UNROLL_STEPS:-3}\"
export DREAMER_REPLAY_MAX_GAMES=\"${DREAMER_REPLAY_MAX_GAMES:-2500}\"
export XLA_PYTHON_CLIENT_PREALLOCATE=\"${XLA_PYTHON_CLIENT_PREALLOCATE:-false}\"
if [ -z \"${XLA_PYTHON_CLIENT_MEM_FRACTION:-}\" ]; then
  if [ \"$NIGHTLY_KEEP_VLLM\" = \"1\" ]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=\"0.55\"
  else
    export XLA_PYTHON_CLIENT_MEM_FRACTION=\"0.70\"
  fi
else
  export XLA_PYTHON_CLIENT_MEM_FRACTION
fi
export MUZERO_PROMOTION_MIN_TOTAL_TRADES=\"${MUZERO_PROMOTION_MIN_TOTAL_TRADES:-24}\"
export MUZERO_PROMOTION_MIN_EVAL_GAMES=\"${MUZERO_PROMOTION_MIN_EVAL_GAMES:-12}\"
export MUZERO_PROMOTION_MIN_EVAL_SYMBOLS=\"${MUZERO_PROMOTION_MIN_EVAL_SYMBOLS:-3}\"
export MUZERO_PROMOTION_MIN_WIN_RATE=\"${MUZERO_PROMOTION_MIN_WIN_RATE:-50.0}\"
export MUZERO_PROMOTION_MIN_RETURN_PCT=\"${MUZERO_PROMOTION_MIN_RETURN_PCT:-0.0}\"
export MUZERO_PROMOTION_MIN_PROFIT_FACTOR=\"${MUZERO_PROMOTION_MIN_PROFIT_FACTOR:-1.05}\"
export MUZERO_PROMOTION_MIN_EXPECTANCY_PCT=\"${MUZERO_PROMOTION_MIN_EXPECTANCY_PCT:-0.02}\"
export MUZERO_PROMOTION_MAX_DRAWDOWN_PCT=\"${MUZERO_PROMOTION_MAX_DRAWDOWN_PCT:-4.0}\"
export MUZERO_PROMOTION_MIN_POSITIVE_EPISODE_RATE=\"${MUZERO_PROMOTION_MIN_POSITIVE_EPISODE_RATE:-50.0}\"
export TELEGRAM_NOTIFY_TRAINING=\"${TELEGRAM_NOTIFY_TRAINING:-1}\"

if [ \"$REBUILD_TRAINER_IMAGE\" = \"1\" ]; then
  echo \"[nightly] Reconstruction de l'image eva-trainer\"
  if ! docker compose --progress plain build eva-trainer; then
    echo \"[nightly] Build eva-trainer en echec, poursuite avec l'image existante\"
  fi
fi

echo \"[nightly] Installation runtime JAX CUDA validee\"
docker compose run --rm \
  -e PYTHONPATH=/app/eva-lab:/app/shared \
  -e TRAINING_PROFILE=\"$TRAINING_PROFILE\" \
  -e TRAINING_AUTOMATION_MODE=\"$TRAINING_AUTOMATION_MODE\" \
  -e TRAINING_REFRESH_AFTER_HOURS=\"${TRAINING_REFRESH_AFTER_HOURS:-24}\" \
  -e TRAINING_MAX_CHAMPION_AGE_HOURS=\"${TRAINING_MAX_CHAMPION_AGE_HOURS:-72}\" \
  -e TRAINING_MIN_SHADOW_RECORDS=\"${TRAINING_MIN_SHADOW_RECORDS:-25}\" \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=\"$XLA_PYTHON_CLIENT_PREALLOCATE\" \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=\"$XLA_PYTHON_CLIENT_MEM_FRACTION\" \
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
  -e MUZERO_BATCH_SIZE=\"$MUZERO_BATCH_SIZE\" \
  -e MUZERO_NUM_SIMULATIONS=\"$MUZERO_NUM_SIMULATIONS\" \
  -e MUZERO_MAX_MOVES=\"$MUZERO_MAX_MOVES\" \
  -e MUZERO_MAX_SYMBOLS=\"$MUZERO_MAX_SYMBOLS\" \
  -e TRAINING_SYMBOLS=\"$TRAINING_SYMBOLS\" \
  -e ARENA_MAX_SYMBOLS=\"$ARENA_MAX_SYMBOLS\" \
  -e ARENA_GAMES_PER_SYMBOL=\"$ARENA_GAMES_PER_SYMBOL\" \
  -e ARENA_MIN_GAMES=\"$ARENA_MIN_GAMES\" \
  -e ARENA_MIN_SYMBOLS=\"$ARENA_MIN_SYMBOLS\" \
  -e ARENA_MIN_SCORE_EDGE=\"$ARENA_MIN_SCORE_EDGE\" \
  -e DREAMER_BATCH_SIZE=\"$DREAMER_BATCH_SIZE\" \
  -e DREAMER_SEQUENCE_LENGTH=\"$DREAMER_SEQUENCE_LENGTH\" \
  -e DREAMER_SEQUENCE_STRIDE=\"$DREAMER_SEQUENCE_STRIDE\" \
  -e DREAMER_MAX_START_STATES=\"$DREAMER_MAX_START_STATES\" \
  -e DREAMER_HIDDEN_STATE_SIZE=\"$DREAMER_HIDDEN_STATE_SIZE\" \
  -e DREAMER_NETWORK_HIDDEN_DIMS=\"$DREAMER_NETWORK_HIDDEN_DIMS\" \
  -e DREAMER_NUM_UNROLL_STEPS=\"$DREAMER_NUM_UNROLL_STEPS\" \
  -e DREAMER_REPLAY_MAX_GAMES=\"$DREAMER_REPLAY_MAX_GAMES\" \
  -e MUZERO_PROMOTION_MIN_TOTAL_TRADES=\"$MUZERO_PROMOTION_MIN_TOTAL_TRADES\" \
  -e MUZERO_PROMOTION_MIN_EVAL_GAMES=\"$MUZERO_PROMOTION_MIN_EVAL_GAMES\" \
  -e MUZERO_PROMOTION_MIN_EVAL_SYMBOLS=\"$MUZERO_PROMOTION_MIN_EVAL_SYMBOLS\" \
  -e MUZERO_PROMOTION_MIN_WIN_RATE=\"$MUZERO_PROMOTION_MIN_WIN_RATE\" \
  -e MUZERO_PROMOTION_MIN_RETURN_PCT=\"$MUZERO_PROMOTION_MIN_RETURN_PCT\" \
  -e MUZERO_PROMOTION_MIN_PROFIT_FACTOR=\"$MUZERO_PROMOTION_MIN_PROFIT_FACTOR\" \
  -e MUZERO_PROMOTION_MIN_EXPECTANCY_PCT=\"$MUZERO_PROMOTION_MIN_EXPECTANCY_PCT\" \
  -e MUZERO_PROMOTION_MAX_DRAWDOWN_PCT=\"$MUZERO_PROMOTION_MAX_DRAWDOWN_PCT\" \
  -e MUZERO_PROMOTION_MIN_POSITIVE_EPISODE_RATE=\"$MUZERO_PROMOTION_MIN_POSITIVE_EPISODE_RATE\" \
  -e TELEGRAM_NOTIFY_TRAINING=\"$TELEGRAM_NOTIFY_TRAINING\" \
  -v \"$PROJECT_DIR/data:/app/eva-lab/data\" \
  eva-trainer \
  bash -lc \"python -c 'import importlib.metadata as md; import jax, haiku, optax, chex, flax, orbax.checkpoint; print(\\\"[nightly] jax\\\", jax.__version__); print(\\\"[nightly] jaxlib\\\", md.version(\\\"jaxlib\\\")); print(\\\"[nightly] haiku\\\", haiku.__version__); print(\\\"[nightly] optax\\\", optax.__version__); print(\\\"[nightly] chex\\\", chex.__version__); print(\\\"[nightly] flax\\\", flax.__version__); print(\\\"[nightly] backend\\\", jax.default_backend()); print(\\\"[nightly] devices\\\", jax.devices())' || pip install --no-cache-dir --upgrade jax==0.4.23 jaxlib==0.4.23+cuda11.cudnn86 dm-haiku==0.0.11 optax==0.1.7 chex==0.1.85 flax==0.8.4 orbax-checkpoint==0.5.16 nest_asyncio==1.6.0 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && python scripts/train_nightly_stack.py\"
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


def upload_tree(sftp: paramiko.SFTPClient, local_dir: Path, remote_dir: str) -> None:
    """Synchronise recursivement un dossier utile a l'entrainement.

    Args:
        sftp (paramiko.SFTPClient): Canal SFTP actif.
        local_dir (Path): Dossier local source.
        remote_dir (str): Dossier distant cible.
    """
    if not local_dir.exists():
        print(f"SKIP {local_dir} (absent)")
        return

    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(local_dir)
        upload_file(sftp, file_path, f"{remote_dir}/{relative.as_posix()}")


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
        for relative_dir in SYNC_DIRS:
            local_dir = LOCAL_ROOT / relative_dir
            remote_dir = f"{REMOTE_DIR}/{relative_dir.as_posix()}"
            upload_tree(sftp, local_dir, remote_dir)

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


