"""Synchronise EVA Lab sur Proxmox puis lance l'entrainement nocturne."""

from __future__ import annotations

import argparse
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
    Path("src/eva-lab/eva_lab/main.py"),
    Path("src/eva-lab/eva_lab/shadow_dataset.py"),
    Path("src/eva-lab/eva_lab/training_notifier.py"),
    Path("src/eva-lab/eva_lab/training_status.py"),
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
    "MUZERO_PROMOTION_MIN_NET_REALIZED_PCT",
    "TELEGRAM_NOTIFY_TRAINING",
    "TRAINING_RUN_TRIGGER",
    "TRAINING_RUN_LOCK_MAX_AGE_HOURS",
]

REMOTE_ENV_LOADER = """if [ -f .env ]; then
  eval "$(
    python3 - <<'PY'
from __future__ import annotations

import pathlib
import shlex

env_path = pathlib.Path(".env")
for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    print(f"export {key}={shlex.quote(value)}")
PY
  )"
fi
"""

REMOTE_LAUNCH_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=\"/home/aza/The_Hive\"
LOCK_FILE=\"$PROJECT_DIR/data/checkpoints/nightly_training.lock\"
LOCK_DIR=\"$PROJECT_DIR/data/checkpoints/nightly_training.lock.d\"
SUMMARY_FILE=\"$PROJECT_DIR/data/checkpoints/nightly_training_summary.json\"
cd \"$PROJECT_DIR\"

""" + REMOTE_ENV_LOADER + """

VLLM_STOPPED=0
COMFYUI_STOPPED=0
RUN_TRIGGER=\"${TRAINING_RUN_TRIGGER:-manual}\"

mkdir -p \"$(dirname \"$LOCK_FILE\")\"

emit_training_log() {
  local level=\"$1\"
  local source=\"$2\"
  shift 2
  local message=\"$*\"
  PYTHONPATH=\"$PROJECT_DIR/src/eva-lab:$PROJECT_DIR/src/shared\" \
  TRAINING_LOG_LEVEL=\"$level\" \
  TRAINING_LOG_SOURCE=\"$source\" \
  TRAINING_LOG_MESSAGE=\"$message\" \
  python3 - <<'PY'
from __future__ import annotations

import os

from eva_lab.training_status import append_training_log

append_training_log(
    os.environ.get("TRAINING_LOG_MESSAGE", ""),
    level=os.environ.get("TRAINING_LOG_LEVEL", "INFO"),
    source=os.environ.get("TRAINING_LOG_SOURCE", "launcher"),
)
PY
}

emit_launcher_state() {
  local phase=\"$1\"
  local vllm_state=\"${2:-}\"
  local comfyui_state=\"${3:-}\"
  PYTHONPATH=\"$PROJECT_DIR/src/eva-lab:$PROJECT_DIR/src/shared\" \
  TRAINING_LAUNCHER_PHASE=\"$phase\" \
  TRAINING_VLLM_STATE=\"$vllm_state\" \
  TRAINING_COMFYUI_STATE=\"$comfyui_state\" \
  TRAINING_REMOTE_PID=\"$$\" \
  TRAINING_RUN_TRIGGER=\"$RUN_TRIGGER\" \
  python3 - <<'PY'
from __future__ import annotations

import os

from eva_lab.training_status import set_training_launcher_state

set_training_launcher_state(
    phase=os.environ.get("TRAINING_LAUNCHER_PHASE") or None,
    vllm_state=os.environ.get("TRAINING_VLLM_STATE") or None,
    comfyui_state=os.environ.get("TRAINING_COMFYUI_STATE") or None,
    remote_pid=os.environ.get("TRAINING_REMOTE_PID") or None,
    trigger=os.environ.get("TRAINING_RUN_TRIGGER") or None,
)
PY
}

write_skip_marker() {
  local skip_reason=\"$1\"
  SUMMARY_FILE=\"$SUMMARY_FILE\" LOCK_FILE=\"$LOCK_FILE\" RUN_TRIGGER=\"$RUN_TRIGGER\" SKIP_REASON=\"$skip_reason\" python3 - <<'PY'
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

summary_file = Path(os.environ["SUMMARY_FILE"])
lock_file = Path(os.environ["LOCK_FILE"])
skip_reason = os.environ.get("SKIP_REASON", "run_already_active")
run_trigger = os.environ.get("RUN_TRIGGER", "unknown")
project_dir = Path("/home/aza/The_Hive")
for candidate in (project_dir / "src/eva-lab", project_dir / "src/shared"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

payload: dict[str, object]
if summary_file.exists():
    try:
        loaded = json.loads(summary_file.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        payload = {}
else:
    payload = {}

event = {
    "trigger": run_trigger,
    "reason": skip_reason,
    "timestamp": datetime.now().isoformat(),
}
if lock_file.exists():
    try:
        lock_payload = json.loads(lock_file.read_text(encoding="utf-8"))
        event["lock"] = lock_payload if isinstance(lock_payload, dict) else {}
    except Exception:
        event["lock"] = {"path": str(lock_file)}

skip_events = list(payload.get("skip_events") or [])
skip_events.append(event)
payload["skip_events"] = skip_events[-20:]
payload["last_skip_event"] = event
payload.setdefault("strategy", os.environ.get("TRAINING_PROFILE", "smart"))
if payload.get("status") in {None, "skipped"}:
    payload["status"] = "skipped"
    payload["finished_at"] = datetime.now().isoformat()
summary_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

try:
    from eva_lab.training_status import append_training_log, mark_skip_status, set_training_launcher_state

    mark_skip_status(skip_reason, run_trigger, event.get("lock"))
    set_training_launcher_state(last_skip_reason=skip_reason, trigger=run_trigger)
    append_training_log(
        f"Run ignore par le lanceur: {skip_reason}",
        level="WARNING",
        source="launcher",
    )
except Exception:
    pass
PY
}

read_lock_pid() {
  if [ ! -f \"$LOCK_FILE\" ]; then
    return 1
  fi
  python3 - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

lock_file = Path("/home/aza/The_Hive/data/checkpoints/nightly_training.lock")
try:
    payload = json.loads(lock_file.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)

pid = payload.get("pid")
print(pid if isinstance(pid, int) else "")
PY
}

list_active_trainer_containers() {
  docker ps --format '{{.Names}}' | grep '^the_hive-eva-trainer-run-' || true
}

acquire_lock() {
  local active_trainers
  active_trainers=\"$(list_active_trainer_containers)\"
  if [ -n \"$active_trainers\" ]; then
    echo \"[nightly] Conteneur trainer deja actif. Skip propre.\"
    printf '%s\n' \"$active_trainers\"
    write_skip_marker \"trainer_container_active\"
    exit 0
  fi

  if mkdir \"$LOCK_DIR\" 2>/dev/null; then
    :
  else
    local existing_pid
    existing_pid=\"$(read_lock_pid || true)\"
    if [ -n \"$existing_pid\" ] && kill -0 \"$existing_pid\" 2>/dev/null; then
      echo \"[nightly] Run deja actif (pid=$existing_pid). Skip propre.\"
      write_skip_marker \"run_already_active\"
      exit 0
    fi
    echo \"[nightly] Verrou nightly obsolete detecte. Nettoyage.\"
    rm -rf \"$LOCK_DIR\"
    rm -f \"$LOCK_FILE\"
    mkdir \"$LOCK_DIR\"
  fi

  printf '{\"pid\": %s, \"trigger\": \"%s\", \"profile\": \"%s\", \"started_at\": \"%s\"}\\n' \\
    \"$$\" \"$RUN_TRIGGER\" \"${TRAINING_PROFILE:-smart}\" \"$(date -Iseconds)\" > \"$LOCK_FILE\"
}

release_lock() {
  rm -rf \"$LOCK_DIR\"
  rm -f \"$LOCK_FILE\"
}

export TRAINING_PROFILE=\"${TRAINING_PROFILE:-smart}\"
export TRAINING_AUTOMATION_MODE=\"${TRAINING_AUTOMATION_MODE:-smart}\"
export TRAINING_RUN_LOCK_ALREADY_HELD=1

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
acquire_lock
emit_launcher_state \"preflight\" \"online\" \"online\"
emit_training_log INFO launcher \"Verification des services critiques avant entrainement.\"

echo \"[nightly] Verification des services critiques avant entrainement\"
docker ps --format '{{.Names}} {{.Status}}' | grep -E 'the_hive-(lab|vllm|redis|neo4j|mosquitto)' || true

if [ \"$NIGHTLY_KEEP_VLLM\" = \"1\" ]; then
  echo \"[nightly] vLLM conserve en ligne pour le live\"
  emit_launcher_state \"preflight\" \"online\" \"$([ \"$NIGHTLY_STOP_COMFYUI\" = \"1\" ] && echo stop_requested || echo online)\"
  emit_training_log INFO launcher \"vLLM conserve en ligne pour ce run.\"
else
  echo \"[nightly] Arret temporaire de vLLM pour liberer le GPU\"
  docker compose stop vllm || true
  VLLM_STOPPED=1
  emit_launcher_state \"preflight\" \"stopped_for_training\" \"$([ \"$NIGHTLY_STOP_COMFYUI\" = \"1\" ] && echo stop_requested || echo online)\"
  emit_training_log INFO launcher \"vLLM arrete volontairement pour liberer le GPU.\"
fi

if [ \"$NIGHTLY_STOP_COMFYUI\" = \"1\" ]; then
  echo \"[nightly] Arret temporaire de ComfyUI\"
  docker compose stop comfyui || true
  COMFYUI_STOPPED=1
  emit_launcher_state \"preflight\" \"$([ \"$VLLM_STOPPED\" = \"1\" ] && echo stopped_for_training || echo online)\" \"stopped_for_training\"
  emit_training_log INFO launcher \"ComfyUI arrete temporairement pour le run.\"
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
    emit_launcher_state \"cleanup\" \"restarting\" \"$([ \"$COMFYUI_STOPPED\" = \"1\" ] && echo restarting || echo online)\"
    emit_training_log INFO launcher \"Redemarrage de vLLM apres entrainement.\"
    for attempt in $(seq 1 18); do
      if curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo \"[nightly] vLLM operationnel\"
        emit_launcher_state \"cleanup\" \"online\" \"$([ \"$COMFYUI_STOPPED\" = \"1\" ] && echo online || echo online)\"
        emit_training_log INFO launcher \"vLLM de nouveau operationnel apres le run.\"
        break
      fi
      if [ \"$attempt\" -eq 6 ]; then
        echo \"[nightly] vLLM encore indisponible, redemarrage complementaire\"
        docker compose restart vllm >/dev/null 2>&1 || true
      fi
      sleep 10
    done
  fi
  emit_launcher_state \"idle\" \"online\" \"online\"
  release_lock
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
export MUZERO_PROMOTION_MIN_NET_REALIZED_PCT=\"${MUZERO_PROMOTION_MIN_NET_REALIZED_PCT:-0.50}\"
export TELEGRAM_NOTIFY_TRAINING=\"${TELEGRAM_NOTIFY_TRAINING:-1}\"

if [ \"$REBUILD_TRAINER_IMAGE\" = \"1\" ]; then
  echo \"[nightly] Reconstruction de l'image eva-trainer\"
  emit_training_log INFO launcher \"Reconstruction de l'image eva-trainer.\"
  if ! docker compose --progress plain build eva-trainer; then
    echo \"[nightly] Build eva-trainer en echec, poursuite avec l'image existante\"
    emit_training_log WARNING launcher \"Build eva-trainer en echec, poursuite avec l'image existante.\"
  fi
fi

echo \"[nightly] Installation runtime JAX CUDA validee\"
emit_launcher_state \"trainer_running\" \"$([ \"$VLLM_STOPPED\" = \"1\" ] && echo stopped_for_training || echo online)\" \"$([ \"$COMFYUI_STOPPED\" = \"1\" ] && echo stopped_for_training || echo online)\"
emit_training_log INFO launcher \"Demarrage du conteneur trainer pour le run nightly.\"
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
  -e MUZERO_PROMOTION_MIN_NET_REALIZED_PCT=\"$MUZERO_PROMOTION_MIN_NET_REALIZED_PCT\" \
  -e TELEGRAM_NOTIFY_TRAINING=\"$TELEGRAM_NOTIFY_TRAINING\" \
  -e TRAINING_RUN_TRIGGER=\"$RUN_TRIGGER\" \
  -e NIGHTLY_RUN_LOCK_ALREADY_HELD=\"1\" \
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


def build_runtime_exports(overrides: dict[str, str] | None = None) -> str:
    """Construit les variables d'environnement a propager vers le script distant.

    Args:
        overrides (dict[str, str] | None): Surcharges explicites pour un run.

    Returns:
        str: Bloc `export ...;` pret a etre injecte dans la commande distante.
    """
    exports = []
    values: dict[str, str] = {}
    for name in PASSTHROUGH_VARS:
        value = os.getenv(name)
        if value:
            values[name] = value
    if overrides:
        values.update({name: value for name, value in overrides.items() if value})
    for name, value in values.items():
        exports.append(f"export {name}={shlex.quote(value)}")
    return "; ".join(exports)


def _build_manual_massive_overrides() -> dict[str, str]:
    """Construit les surcharges du run massif immediat.

    Returns:
        dict[str, str]: Variables a forcer pour la recherche massive.
    """
    return {
        "TRAINING_PROFILE": "research",
        "TRAINING_AUTOMATION_MODE": "force_research",
        "TRAINING_RUN_TRIGGER": "manual_massive_research",
        "NIGHTLY_KEEP_VLLM": "0",
        "RUN_TRAIN_GNN": "1",
        "RUN_TRAIN_MUZERO": "1",
        "RUN_TRAIN_DREAMER": "1",
        "MUZERO_HORIZONS": "scalp,intraday,swing",
        "MUZERO_MAX_SYMBOLS": "0",
        "ARENA_MAX_SYMBOLS": "0",
    }


def start_training(manual_massive: bool = False) -> None:
    """Synchronise les scripts EVA Lab et lance l'entrainement distant.

    Args:
        manual_massive (bool): Force un run massif immediat de recherche.
    """
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

        runtime_overrides = _build_manual_massive_overrides() if manual_massive else {}
        runtime_exports = build_runtime_exports(runtime_overrides)
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


def parse_args() -> argparse.Namespace:
    """Analyse les options du lanceur Proxmox.

    Returns:
        argparse.Namespace: Arguments normalises.
    """
    parser = argparse.ArgumentParser(description="Lance les entrainements EVA Lab sur Proxmox.")
    parser.add_argument(
        "--manual-massive",
        action="store_true",
        help="Force un run massif immediat de recherche (GNN -> MuZero -> Dreamer).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_training(manual_massive=args.manual_massive)


