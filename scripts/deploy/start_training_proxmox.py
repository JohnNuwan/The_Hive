"""Synchronise EVA Lab sur Proxmox puis lance l'entrainement nocturne."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)
REMOTE_DIR = "/home/aza/The_Hive"
REMOTE_LOG = f"{REMOTE_DIR}/hive_nightly_training.log"
REMOTE_SCRIPT = f"{REMOTE_DIR}/scripts/run_nightly_training_remote.sh"
REMOTE_SEQUENCE_SCRIPT = f"{REMOTE_DIR}/scripts/run_wave1_sequence_remote.sh"
REMOTE_SEQUENCE_LOG = f"{REMOTE_DIR}/hive_wave1_sequence.log"
REMOTE_V4_SEQUENCE_DIR = f"{REMOTE_DIR}/data/checkpoints/v4_ga"
REMOTE_V4_SEQUENCE_RUNNER = f"{REMOTE_DIR}/scripts/deploy/v4_sequence_runner.py"
LOCAL_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SCALP_MULTI_UNIVERSE = [
    "EURUSD",
    "XAUUSD",
    "GBPUSD",
    "USDJPY",
    "US30.cash",
    "GER40.cash",
    "US500.cash",
]

SYNC_FILES = [
    Path("docker-compose.yml"),
    Path("src/eva-lab/eva_lab/training_utils.py"),
    Path("src/eva-lab/eva_lab/timescale_store.py"),
    Path("src/eva-lab/eva_lab/models/gnn_model.py"),
    Path("src/eva-lab/eva_lab/muzero/config.py"),
    Path("src/eva-lab/eva_lab/muzero/environment.py"),
    Path("src/eva-lab/eva_lab/dreamer_gate.py"),
    Path("src/eva-lab/eva_lab/gold_cpu_prep.py"),
    Path("src/eva-lab/eva_lab/live_inference_main.py"),
    Path("src/eva-lab/eva_lab/live_inference_models.py"),
    Path("src/eva-lab/eva_lab/muzero/dreamer_agent.py"),
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
    Path("src/shared/pyproject.toml"),
    Path("src/shared/shared/__init__.py"),
    Path("src/shared/shared/config.py"),
    Path("src/shared/shared/models.py"),
    Path("scripts/deploy/v4_sequence_runner.py"),
    Path("scripts/prepare_gold_cpu_artifacts.py"),
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
    "NIGHTLY_DEFER_VLLM_RESTART",
    "NIGHTLY_STOP_COMFYUI",
    "REBUILD_TRAINER_IMAGE",
    "RUN_TRAIN_GNN",
    "RUN_TRAIN_MUZERO",
    "RUN_TRAIN_DREAMER",
    "TRAINING_ENGINE",
    "TRAINING_TRIAL_MODE",
    "TRAINING_TRIAL_COST_PROFILE",
    "MUZERO_HORIZONS",
    "DREAMER_HORIZON",
    "DREAMER_DEFAULT_HORIZON",
    "DREAMER_EPOCHS",
    "TRAIN_GNN_EPOCHS",
    "TRAIN_GNN_BATCH_SIZE",
    "TRAIN_GNN_CHECKPOINT_EVERY",
    "TRAIN_GNN_SYMBOLS",
    "TRAIN_GNN_FOCUS_SYMBOL",
    "TRAIN_GNN_CONTEXT_SYMBOLS",
    "TRAIN_GNN_DEPLOYMENT_CLASS",
    "MUZERO_TRAINING_STEPS",
    "MUZERO_GAMES_PER_SYMBOL",
    "MUZERO_BATCH_SIZE",
    "MUZERO_NUM_SIMULATIONS",
    "MUZERO_MAX_MOVES",
    "MUZERO_MAX_SYMBOLS",
    "MUZERO_MODEL_FAMILY",
    "MUZERO_DATASET_SOURCE",
    "MUZERO_SYMBOLS",
    "MUZERO_SYMBOLS_SCALP",
    "MUZERO_SYMBOLS_INTRADAY",
    "MUZERO_SYMBOLS_SWING",
    "TRAINING_SYMBOLS",
    "ARENA_MAX_SYMBOLS",
    "ARENA_SYMBOLS",
    "ARENA_SYMBOLS_SCALP",
    "ARENA_SYMBOLS_INTRADAY",
    "ARENA_SYMBOLS_SWING",
    "ARENA_GAMES_PER_SYMBOL",
    "ARENA_MIN_GAMES",
    "ARENA_MIN_SYMBOLS",
    "ARENA_MIN_SCORE_EDGE",
    "MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS",
    "MUZERO_LIVE_TOP_SYMBOLS",
    "MUZERO_LIVE_MIN_SYMBOL_EVAL_GAMES",
    "MUZERO_LIVE_MIN_SYMBOL_TRADES",
    "MUZERO_LIVE_MIN_SYMBOL_RETURN_PCT",
    "MUZERO_LIVE_MIN_SYMBOL_NET_REALIZED_PCT",
    "MUZERO_LIVE_MIN_SYMBOL_PROFIT_FACTOR",
    "MUZERO_LIVE_MAX_SYMBOL_DRAWDOWN_PCT",
    "MUZERO_LIVE_MIN_SYMBOL_LONG_ENTRY_SHARE",
    "MUZERO_LIVE_MIN_SYMBOL_SHORT_ENTRY_SHARE",
    "MUZERO_LIVE_MAX_SYMBOL_DIRECTIONAL_IMBALANCE",
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
    "MUZERO_PROMOTION_MIN_SPLIT_EFFICIENCY",
    "MUZERO_PROMOTION_MIN_PYRAMID_EFFICIENCY",
    "MUZERO_PROMOTION_MIN_SLBE_CAPTURE_RATE",
    "MUZERO_PROMOTION_MAX_HOLD_DRAG_SCORE",
    "MUZERO_PROMOTION_MIN_CLOSE_QUALITY_SCORE",
    "MUZERO_ENTRY_EMA_MODE",
    "MUZERO_ENTRY_REQUIRE_VWAP_ALIGNMENT",
    "MUZERO_ENTRY_REQUIRE_OBV_CONFIRMATION",
    "MUZERO_ENTRY_ALLOW_TREND_FALLBACK",
    "MUZERO_ENTRY_MIN_ADX",
    "MUZERO_ENTRY_TREND_ADX",
    "MUZERO_ACTIVITY_MIN_ENTRIES",
    "MUZERO_ACTIVITY_INACTIVE_EPISODE_PENALTY",
    "MUZERO_ACTIVITY_INSUFFICIENT_ENTRIES_PENALTY",
    "MUZERO_DIRECTIONAL_MIN_ENTRY_SHARE",
    "MUZERO_DIRECTIONAL_MAX_IMBALANCE",
    "MUZERO_DIRECTIONAL_IMBALANCE_PENALTY",
    "MUZERO_HOLD_STALE_PENALTY_AFTER_STEPS",
    "MUZERO_HOLD_STALE_PENALTY",
    "MUZERO_HOLD_TREND_PENALTY",
    "MUZERO_HOLD_RANGE_PENALTY",
    "MUZERO_PYRAMID_MAX_ADDITIONS",
    "MUZERO_PYRAMID_MIN_PROFIT_TO_ADD",
    "MUZERO_PYRAMID_REWARD_BONUS",
    "MUZERO_REWARD_PYRAMID_FAILURE_PENALTY",
    "MUZERO_REWARD_PYRAMID_NEGATIVE_EXIT_PENALTY",
    "MUZERO_SPLIT_MAX_SPLITS",
    "MUZERO_SPLIT_MIN_TRADE_RETURN",
    "MUZERO_SPLIT_MIN_REALIZED_PCT",
    "MUZERO_SPLIT_FAILURE_PENALTY",
    "MUZERO_SLBE_ACTIVATION_RETURN",
    "MUZERO_SLBE_BONUS",
    "MUZERO_SLBE_EXIT_BONUS",
    "MUZERO_CLOSE_WINNER_THRESHOLD",
    "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD",
    "MUZERO_CLOSE_TP_LIKE_THRESHOLD",
    "MUZERO_REWARD_REALIZED_PNL_MULTIPLIER",
    "MUZERO_REWARD_CLOSE_REALIZED_MULTIPLIER",
    "MUZERO_REWARD_SPLIT_REALIZED_MULTIPLIER",
    "MUZERO_REWARD_HOLD_DRAG_MULTIPLIER",
    "TRAINING_TIMESCALE_ENABLED",
    "TRAINING_TIMESCALE_HOST",
    "TRAINING_TIMESCALE_PORT",
    "TRAINING_TIMESCALE_DB",
    "TRAINING_TIMESCALE_USER",
    "TRAINING_TIMESCALE_PASSWORD",
    "TRAINING_TIMESCALE_SSLMODE",
    "TRAINING_TIMESCALE_BARS_TABLE",
    "TRAINING_TIMESCALE_FEATURES_TABLE",
    "TRAINING_TIMESCALE_DATASETS_TABLE",
    "TRAINING_TIMESCALE_ARENA_TABLE",
    "TRAINING_TIMESCALE_GA_TABLE",
    "TRAINING_TIMESCALE_REPLAY_TABLE",
    "TRAINING_TIMESCALE_RUN_WINDOWS_TABLE",
    "TELEGRAM_NOTIFY_TRAINING",
    "TRAINING_RUN_TRIGGER",
    "TRAINING_RUN_LOCK_MAX_AGE_HOURS",
    "TRAINING_GA_STATUS",
    "TRAINING_GA_GENERATION",
    "TRAINING_GA_TRIAL",
    "TRAINING_GA_CAMPAIGN_ID",
    "TRAINING_GA_SCOPE",
    "TRAINING_GA_PARENT_CHAMPION_ID",
    "TRAINING_GA_DEFER_PROMOTION",
    "TRAINING_GA_GENOME_JSON",
    "TRAINING_GA_SEED_CHECKPOINT_PATH",
    "TRAINING_RESUME_CHECKPOINT_PATH",
    "TRAINING_RESUME_STEP",
    "TRAINING_GATE_PROFILE",
    "TRAINING_FOCUS_SYMBOLS",
    "TRAINING_SEQUENCE_ID",
    "TRAINING_SEQUENCE_PROFILE",
    "TRAINING_WINDOW_ID",
    "TRAINING_TRIAL_ID",
    "TRAINING_SUPERVISOR_STATE",
]

WAVE1_FAMILY_SYMBOLS = {
    "fx": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"],
    "indices": ["US30.cash", "US500.cash", "GER40.cash"],
    "metals": ["XAUUSD", "XAGUSD"],
}

WAVE1_PROFILE_ORDER = [
    "scalp_fx",
    "scalp_indices",
    "scalp_metals",
    "intraday_fx",
    "intraday_indices",
    "intraday_metals",
    "swing_fx",
    "swing_indices",
    "swing_metals",
]

WAVE1_SEQUENCE_ORDER = {
    "scalp": ["scalp_fx", "scalp_indices", "scalp_metals"],
}

V3_PROFILE_ORDER = [
    "scalp_metals_v2",
    "scalp_fx_v2",
    "scalp_indices_v2",
]

V3_MODE_ORDER = ["proxy_ga", "full"]

V3_SEQUENCE_ORDER = {
    "scalp": ["scalp_metals_v2", "scalp_fx_v2", "scalp_indices_v2"],
}

V4_PROFILE_ORDER = list(V3_PROFILE_ORDER)

V4_ENGINE_ORDER = ["muzero", "dreamer"]

V4_MODE_ORDER = ["proxy_ga", "full"]

V4_SEQUENCE_ORDER = {
    "scalp": ["scalp_metals_v2", "scalp_fx_v2", "scalp_indices_v2"],
    "gold_monday_xauusd": ["scalp_metals_v2"],
}

GOLD_MONDAY_SEQUENCE_NAME = "gold_monday_xauusd"
GOLD_MONDAY_PROFILE = "scalp_metals_v2"
GOLD_MONDAY_FOCUS_SYMBOL = "XAUUSD"
GOLD_MONDAY_CONTEXT_SYMBOLS = [
    "XAGUSD",
    "DXY.cash",
    "US500.cash",
    "EURUSD",
]

V4_WINDOW_ORDER = [
    ("muzero", "proxy_ga"),
    ("dreamer", "proxy_ga"),
    ("muzero", "full"),
    ("dreamer", "full"),
]

HISTORY_DIR = Path("data/history")

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
  local trainer_container=\"${4:-}\"
  PYTHONPATH=\"$PROJECT_DIR/src/eva-lab:$PROJECT_DIR/src/shared\" \
  TRAINING_LAUNCHER_PHASE=\"$phase\" \
  TRAINING_VLLM_STATE=\"$vllm_state\" \
  TRAINING_COMFYUI_STATE=\"$comfyui_state\" \
  TRAINING_TRAINER_CONTAINER=\"$trainer_container\" \
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
    trainer_container=os.environ.get("TRAINING_TRAINER_CONTAINER") or None,
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
export NIGHTLY_DEFER_VLLM_RESTART=\"${NIGHTLY_DEFER_VLLM_RESTART:-0}\"
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
    if [ \"$NIGHTLY_DEFER_VLLM_RESTART\" = \"1\" ]; then
      echo \"[nightly] Redemarrage de vLLM differe jusqu'a la fin de la sequence\"
      emit_launcher_state \"cleanup\" \"deferred\" \"$([ \"$COMFYUI_STOPPED\" = \"1\" ] && echo restarting || echo online)\"
      emit_training_log INFO launcher \"Redemarrage de vLLM differe a la fin de la sequence Gold.\"
      return
    fi
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
export TRAINING_ENGINE=\"${TRAINING_ENGINE:-}\"
export TRAINING_TRIAL_MODE=\"${TRAINING_TRIAL_MODE:-}\"
export TRAINING_TRIAL_COST_PROFILE=\"${TRAINING_TRIAL_COST_PROFILE:-}\"
export REBUILD_TRAINER_IMAGE=\"${REBUILD_TRAINER_IMAGE:-1}\"
export MUZERO_HORIZONS=\"${MUZERO_HORIZONS:-scalp,intraday,swing}\"
export DREAMER_HORIZON=\"${DREAMER_HORIZON:-}\"
export DREAMER_DEFAULT_HORIZON=\"${DREAMER_DEFAULT_HORIZON:-}\"
export DREAMER_EPOCHS=\"${DREAMER_EPOCHS:-1500}\"
export TRAIN_GNN_EPOCHS=\"${TRAIN_GNN_EPOCHS:-500}\"
export TRAIN_GNN_BATCH_SIZE=\"${TRAIN_GNN_BATCH_SIZE:-64}\"
export TRAIN_GNN_CHECKPOINT_EVERY=\"${TRAIN_GNN_CHECKPOINT_EVERY:-25}\"
export TRAIN_GNN_SYMBOLS=\"${TRAIN_GNN_SYMBOLS:-}\"
export TRAIN_GNN_FOCUS_SYMBOL=\"${TRAIN_GNN_FOCUS_SYMBOL:-}\"
export TRAIN_GNN_CONTEXT_SYMBOLS=\"${TRAIN_GNN_CONTEXT_SYMBOLS:-}\"
export TRAIN_GNN_DEPLOYMENT_CLASS=\"${TRAIN_GNN_DEPLOYMENT_CLASS:-consultative}\"
export MUZERO_TRAINING_STEPS=\"${MUZERO_TRAINING_STEPS:-24000}\"
export MUZERO_GAMES_PER_SYMBOL=\"${MUZERO_GAMES_PER_SYMBOL:-12}\"
export MUZERO_BATCH_SIZE=\"${MUZERO_BATCH_SIZE:-32}\"
export MUZERO_NUM_SIMULATIONS=\"${MUZERO_NUM_SIMULATIONS:-100}\"
export MUZERO_MAX_MOVES=\"${MUZERO_MAX_MOVES:-300}\"
export MUZERO_MAX_SYMBOLS=\"${MUZERO_MAX_SYMBOLS:-12}\"
export MUZERO_MODEL_FAMILY=\"${MUZERO_MODEL_FAMILY:-}\"
export MUZERO_DATASET_SOURCE=\"${MUZERO_DATASET_SOURCE:-auto}\"
export MUZERO_SYMBOLS=\"${MUZERO_SYMBOLS:-}\"
export MUZERO_SYMBOLS_SCALP=\"${MUZERO_SYMBOLS_SCALP:-}\"
export MUZERO_SYMBOLS_INTRADAY=\"${MUZERO_SYMBOLS_INTRADAY:-}\"
export MUZERO_SYMBOLS_SWING=\"${MUZERO_SYMBOLS_SWING:-}\"
export TRAINING_SYMBOLS=\"${TRAINING_SYMBOLS:-}\"
export ARENA_MAX_SYMBOLS=\"${ARENA_MAX_SYMBOLS:-12}\"
export ARENA_SYMBOLS=\"${ARENA_SYMBOLS:-}\"
export ARENA_SYMBOLS_SCALP=\"${ARENA_SYMBOLS_SCALP:-}\"
export ARENA_SYMBOLS_INTRADAY=\"${ARENA_SYMBOLS_INTRADAY:-}\"
export ARENA_SYMBOLS_SWING=\"${ARENA_SYMBOLS_SWING:-}\"
export ARENA_GAMES_PER_SYMBOL=\"${ARENA_GAMES_PER_SYMBOL:-6}\"
export ARENA_MIN_GAMES=\"${ARENA_MIN_GAMES:-12}\"
export ARENA_MIN_SYMBOLS=\"${ARENA_MIN_SYMBOLS:-3}\"
export ARENA_MIN_SCORE_EDGE=\"${ARENA_MIN_SCORE_EDGE:-0.5}\"
export MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS=\"${MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS:-12}\"
export MUZERO_LIVE_TOP_SYMBOLS=\"${MUZERO_LIVE_TOP_SYMBOLS:-5}\"
export MUZERO_LIVE_MIN_SYMBOL_EVAL_GAMES=\"${MUZERO_LIVE_MIN_SYMBOL_EVAL_GAMES:-2}\"
export MUZERO_LIVE_MIN_SYMBOL_TRADES=\"${MUZERO_LIVE_MIN_SYMBOL_TRADES:-4}\"
export MUZERO_LIVE_MIN_SYMBOL_RETURN_PCT=\"${MUZERO_LIVE_MIN_SYMBOL_RETURN_PCT:-0.0}\"
export MUZERO_LIVE_MIN_SYMBOL_NET_REALIZED_PCT=\"${MUZERO_LIVE_MIN_SYMBOL_NET_REALIZED_PCT:-0.0}\"
export MUZERO_LIVE_MIN_SYMBOL_PROFIT_FACTOR=\"${MUZERO_LIVE_MIN_SYMBOL_PROFIT_FACTOR:-1.0}\"
export MUZERO_LIVE_MAX_SYMBOL_DRAWDOWN_PCT=\"${MUZERO_LIVE_MAX_SYMBOL_DRAWDOWN_PCT:-5.0}\"
export MUZERO_LIVE_MIN_SYMBOL_LONG_ENTRY_SHARE=\"${MUZERO_LIVE_MIN_SYMBOL_LONG_ENTRY_SHARE:-0.10}\"
export MUZERO_LIVE_MIN_SYMBOL_SHORT_ENTRY_SHARE=\"${MUZERO_LIVE_MIN_SYMBOL_SHORT_ENTRY_SHARE:-0.10}\"
export MUZERO_LIVE_MAX_SYMBOL_DIRECTIONAL_IMBALANCE=\"${MUZERO_LIVE_MAX_SYMBOL_DIRECTIONAL_IMBALANCE:-0.85}\"
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
export MUZERO_PROMOTION_MIN_SPLIT_EFFICIENCY=\"${MUZERO_PROMOTION_MIN_SPLIT_EFFICIENCY:-0.45}\"
export MUZERO_PROMOTION_MIN_PYRAMID_EFFICIENCY=\"${MUZERO_PROMOTION_MIN_PYRAMID_EFFICIENCY:-0.45}\"
export MUZERO_PROMOTION_MIN_SLBE_CAPTURE_RATE=\"${MUZERO_PROMOTION_MIN_SLBE_CAPTURE_RATE:-0.40}\"
export MUZERO_PROMOTION_MAX_HOLD_DRAG_SCORE=\"${MUZERO_PROMOTION_MAX_HOLD_DRAG_SCORE:-0.40}\"
export MUZERO_PROMOTION_MIN_CLOSE_QUALITY_SCORE=\"${MUZERO_PROMOTION_MIN_CLOSE_QUALITY_SCORE:-0.45}\"
export TRAINING_TIMESCALE_ENABLED=\"${TRAINING_TIMESCALE_ENABLED:-1}\"
export TRAINING_TIMESCALE_HOST=\"${TRAINING_TIMESCALE_HOST:-timescaledb}\"
export TRAINING_TIMESCALE_PORT=\"${TRAINING_TIMESCALE_PORT:-5432}\"
export TRAINING_TIMESCALE_DB=\"${TRAINING_TIMESCALE_DB:-thehive}\"
export TRAINING_TIMESCALE_USER=\"${TRAINING_TIMESCALE_USER:-eva}\"
export TRAINING_TIMESCALE_PASSWORD=\"${TRAINING_TIMESCALE_PASSWORD:-${TIMESCALE_PASSWORD:-devpassword}}\"
export TRAINING_TIMESCALE_SSLMODE=\"${TRAINING_TIMESCALE_SSLMODE:-prefer}\"
export TRAINING_TIMESCALE_BARS_TABLE=\"${TRAINING_TIMESCALE_BARS_TABLE:-market.market_bars}\"
export TRAINING_TIMESCALE_FEATURES_TABLE=\"${TRAINING_TIMESCALE_FEATURES_TABLE:-market.market_features}\"
export TRAINING_TIMESCALE_DATASETS_TABLE=\"${TRAINING_TIMESCALE_DATASETS_TABLE:-training.training_datasets}\"
export TRAINING_TIMESCALE_ARENA_TABLE=\"${TRAINING_TIMESCALE_ARENA_TABLE:-training.arena_results}\"
export TRAINING_TIMESCALE_RUN_WINDOWS_TABLE=\"${TRAINING_TIMESCALE_RUN_WINDOWS_TABLE:-training.run_windows}\"
export TELEGRAM_NOTIFY_TRAINING=\"${TELEGRAM_NOTIFY_TRAINING:-1}\"
export TRAINING_GA_STATUS=\"${TRAINING_GA_STATUS:-}\"
export TRAINING_GA_GENERATION=\"${TRAINING_GA_GENERATION:-}\"
export TRAINING_GA_TRIAL=\"${TRAINING_GA_TRIAL:-}\"
export TRAINING_GATE_PROFILE=\"${TRAINING_GATE_PROFILE:-}\"
export TRAINING_FOCUS_SYMBOLS=\"${TRAINING_FOCUS_SYMBOLS:-}\"
export TRAINING_SEQUENCE_ID=\"${TRAINING_SEQUENCE_ID:-}\"
export TRAINING_SEQUENCE_PROFILE=\"${TRAINING_SEQUENCE_PROFILE:-}\"
export TRAINING_WINDOW_ID=\"${TRAINING_WINDOW_ID:-}\"
export TRAINING_TRIAL_ID=\"${TRAINING_TRIAL_ID:-}\"
export TRAINING_SUPERVISOR_STATE=\"${TRAINING_SUPERVISOR_STATE:-}\"

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
  -e TRAINING_ENGINE=\"$TRAINING_ENGINE\" \
  -e TRAINING_TRIAL_MODE=\"$TRAINING_TRIAL_MODE\" \
  -e TRAINING_TRIAL_COST_PROFILE=\"$TRAINING_TRIAL_COST_PROFILE\" \
  -e MUZERO_HORIZONS=\"$MUZERO_HORIZONS\" \
  -e DREAMER_HORIZON=\"$DREAMER_HORIZON\" \
  -e DREAMER_DEFAULT_HORIZON=\"$DREAMER_DEFAULT_HORIZON\" \
  -e DREAMER_EPOCHS=\"$DREAMER_EPOCHS\" \
  -e TRAIN_GNN_EPOCHS=\"$TRAIN_GNN_EPOCHS\" \
  -e TRAIN_GNN_BATCH_SIZE=\"$TRAIN_GNN_BATCH_SIZE\" \
  -e TRAIN_GNN_CHECKPOINT_EVERY=\"$TRAIN_GNN_CHECKPOINT_EVERY\" \
  -e TRAIN_GNN_SYMBOLS=\"$TRAIN_GNN_SYMBOLS\" \
  -e TRAIN_GNN_FOCUS_SYMBOL=\"$TRAIN_GNN_FOCUS_SYMBOL\" \
  -e TRAIN_GNN_CONTEXT_SYMBOLS=\"$TRAIN_GNN_CONTEXT_SYMBOLS\" \
  -e TRAIN_GNN_DEPLOYMENT_CLASS=\"$TRAIN_GNN_DEPLOYMENT_CLASS\" \
  -e MUZERO_TRAINING_STEPS=\"$MUZERO_TRAINING_STEPS\" \
  -e MUZERO_GAMES_PER_SYMBOL=\"$MUZERO_GAMES_PER_SYMBOL\" \
  -e MUZERO_BATCH_SIZE=\"$MUZERO_BATCH_SIZE\" \
  -e MUZERO_NUM_SIMULATIONS=\"$MUZERO_NUM_SIMULATIONS\" \
  -e MUZERO_MAX_MOVES=\"$MUZERO_MAX_MOVES\" \
  -e MUZERO_MAX_SYMBOLS=\"$MUZERO_MAX_SYMBOLS\" \
  -e MUZERO_MODEL_FAMILY=\"$MUZERO_MODEL_FAMILY\" \
  -e MUZERO_DATASET_SOURCE=\"$MUZERO_DATASET_SOURCE\" \
  -e MUZERO_SYMBOLS=\"$MUZERO_SYMBOLS\" \
  -e MUZERO_SYMBOLS_SCALP=\"$MUZERO_SYMBOLS_SCALP\" \
  -e MUZERO_SYMBOLS_INTRADAY=\"$MUZERO_SYMBOLS_INTRADAY\" \
  -e MUZERO_SYMBOLS_SWING=\"$MUZERO_SYMBOLS_SWING\" \
  -e TRAINING_SYMBOLS=\"$TRAINING_SYMBOLS\" \
  -e ARENA_MAX_SYMBOLS=\"$ARENA_MAX_SYMBOLS\" \
  -e ARENA_SYMBOLS=\"$ARENA_SYMBOLS\" \
  -e ARENA_SYMBOLS_SCALP=\"$ARENA_SYMBOLS_SCALP\" \
  -e ARENA_SYMBOLS_INTRADAY=\"$ARENA_SYMBOLS_INTRADAY\" \
  -e ARENA_SYMBOLS_SWING=\"$ARENA_SYMBOLS_SWING\" \
  -e ARENA_GAMES_PER_SYMBOL=\"$ARENA_GAMES_PER_SYMBOL\" \
  -e ARENA_MIN_GAMES=\"$ARENA_MIN_GAMES\" \
  -e ARENA_MIN_SYMBOLS=\"$ARENA_MIN_SYMBOLS\" \
  -e ARENA_MIN_SCORE_EDGE=\"$ARENA_MIN_SCORE_EDGE\" \
  -e MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS=\"$MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS\" \
  -e MUZERO_LIVE_TOP_SYMBOLS=\"$MUZERO_LIVE_TOP_SYMBOLS\" \
  -e MUZERO_LIVE_MIN_SYMBOL_EVAL_GAMES=\"$MUZERO_LIVE_MIN_SYMBOL_EVAL_GAMES\" \
  -e MUZERO_LIVE_MIN_SYMBOL_TRADES=\"$MUZERO_LIVE_MIN_SYMBOL_TRADES\" \
  -e MUZERO_LIVE_MIN_SYMBOL_RETURN_PCT=\"$MUZERO_LIVE_MIN_SYMBOL_RETURN_PCT\" \
  -e MUZERO_LIVE_MIN_SYMBOL_NET_REALIZED_PCT=\"$MUZERO_LIVE_MIN_SYMBOL_NET_REALIZED_PCT\" \
  -e MUZERO_LIVE_MIN_SYMBOL_PROFIT_FACTOR=\"$MUZERO_LIVE_MIN_SYMBOL_PROFIT_FACTOR\" \
  -e MUZERO_LIVE_MAX_SYMBOL_DRAWDOWN_PCT=\"$MUZERO_LIVE_MAX_SYMBOL_DRAWDOWN_PCT\" \
  -e MUZERO_LIVE_MIN_SYMBOL_LONG_ENTRY_SHARE=\"$MUZERO_LIVE_MIN_SYMBOL_LONG_ENTRY_SHARE\" \
  -e MUZERO_LIVE_MIN_SYMBOL_SHORT_ENTRY_SHARE=\"$MUZERO_LIVE_MIN_SYMBOL_SHORT_ENTRY_SHARE\" \
  -e MUZERO_LIVE_MAX_SYMBOL_DIRECTIONAL_IMBALANCE=\"$MUZERO_LIVE_MAX_SYMBOL_DIRECTIONAL_IMBALANCE\" \
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
  -e MUZERO_PROMOTION_MIN_SPLIT_EFFICIENCY=\"$MUZERO_PROMOTION_MIN_SPLIT_EFFICIENCY\" \
  -e MUZERO_PROMOTION_MIN_PYRAMID_EFFICIENCY=\"$MUZERO_PROMOTION_MIN_PYRAMID_EFFICIENCY\" \
  -e MUZERO_PROMOTION_MIN_SLBE_CAPTURE_RATE=\"$MUZERO_PROMOTION_MIN_SLBE_CAPTURE_RATE\" \
  -e MUZERO_PROMOTION_MAX_HOLD_DRAG_SCORE=\"$MUZERO_PROMOTION_MAX_HOLD_DRAG_SCORE\" \
  -e MUZERO_PROMOTION_MIN_CLOSE_QUALITY_SCORE=\"$MUZERO_PROMOTION_MIN_CLOSE_QUALITY_SCORE\" \
  -e TRAINING_TIMESCALE_ENABLED=\"$TRAINING_TIMESCALE_ENABLED\" \
  -e TRAINING_TIMESCALE_HOST=\"$TRAINING_TIMESCALE_HOST\" \
  -e TRAINING_TIMESCALE_PORT=\"$TRAINING_TIMESCALE_PORT\" \
  -e TRAINING_TIMESCALE_DB=\"$TRAINING_TIMESCALE_DB\" \
  -e TRAINING_TIMESCALE_USER=\"$TRAINING_TIMESCALE_USER\" \
  -e TRAINING_TIMESCALE_PASSWORD=\"$TRAINING_TIMESCALE_PASSWORD\" \
  -e TRAINING_TIMESCALE_SSLMODE=\"$TRAINING_TIMESCALE_SSLMODE\" \
  -e TRAINING_TIMESCALE_BARS_TABLE=\"$TRAINING_TIMESCALE_BARS_TABLE\" \
  -e TRAINING_TIMESCALE_FEATURES_TABLE=\"$TRAINING_TIMESCALE_FEATURES_TABLE\" \
  -e TRAINING_TIMESCALE_DATASETS_TABLE=\"$TRAINING_TIMESCALE_DATASETS_TABLE\" \
  -e TRAINING_TIMESCALE_ARENA_TABLE=\"$TRAINING_TIMESCALE_ARENA_TABLE\" \
  -e TRAINING_TIMESCALE_RUN_WINDOWS_TABLE=\"$TRAINING_TIMESCALE_RUN_WINDOWS_TABLE\" \
  -e TELEGRAM_NOTIFY_TRAINING=\"$TELEGRAM_NOTIFY_TRAINING\" \
  -e TRAINING_GA_STATUS=\"$TRAINING_GA_STATUS\" \
  -e TRAINING_GA_GENERATION=\"$TRAINING_GA_GENERATION\" \
  -e TRAINING_GA_TRIAL=\"$TRAINING_GA_TRIAL\" \
  -e TRAINING_GATE_PROFILE=\"$TRAINING_GATE_PROFILE\" \
  -e TRAINING_FOCUS_SYMBOLS=\"$TRAINING_FOCUS_SYMBOLS\" \
  -e TRAINING_SEQUENCE_ID=\"$TRAINING_SEQUENCE_ID\" \
  -e TRAINING_SEQUENCE_PROFILE=\"$TRAINING_SEQUENCE_PROFILE\" \
  -e TRAINING_WINDOW_ID=\"$TRAINING_WINDOW_ID\" \
  -e TRAINING_TRIAL_ID=\"$TRAINING_TRIAL_ID\" \
  -e TRAINING_SUPERVISOR_STATE=\"$TRAINING_SUPERVISOR_STATE\" \
  -e TRAINING_RUN_TRIGGER=\"$RUN_TRIGGER\" \
  -e NIGHTLY_RUN_LOCK_ALREADY_HELD=\"1\" \
  -v \"$PROJECT_DIR/data:/app/eva-lab/data\" \
  eva-trainer \
  bash -lc \"python -c 'import importlib.metadata as md; import jax, haiku, optax, chex, flax, orbax.checkpoint; print(\\\"[nightly] jax\\\", jax.__version__); print(\\\"[nightly] jaxlib\\\", md.version(\\\"jaxlib\\\")); print(\\\"[nightly] haiku\\\", haiku.__version__); print(\\\"[nightly] optax\\\", optax.__version__); print(\\\"[nightly] chex\\\", chex.__version__); print(\\\"[nightly] flax\\\", flax.__version__); print(\\\"[nightly] backend\\\", jax.default_backend()); print(\\\"[nightly] devices\\\", jax.devices())' || pip install --no-cache-dir --upgrade jax==0.4.23 jaxlib==0.4.23+cuda11.cudnn86 dm-haiku==0.0.11 optax==0.1.7 chex==0.1.85 flax==0.8.4 orbax-checkpoint==0.5.16 nest_asyncio==1.6.0 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && python scripts/train_nightly_stack.py\" &
TRAINER_RUN_PID=$!
TRAINER_CONTAINER_NAME=\"\"
for attempt in $(seq 1 20); do
  TRAINER_CONTAINER_NAME=\"$(list_active_trainer_containers | tail -n 1)\"
  if [ -n \"$TRAINER_CONTAINER_NAME\" ]; then
    emit_launcher_state \"trainer_running\" \
      \"$([ \"$VLLM_STOPPED\" = \"1\" ] && echo stopped_for_training || echo online)\" \
      \"$([ \"$COMFYUI_STOPPED\" = \"1\" ] && echo stopped_for_training || echo online)\" \
      \"$TRAINER_CONTAINER_NAME\"
    break
  fi
  sleep 2
done
wait \"$TRAINER_RUN_PID\"
"""

def _require_remote_credentials() -> tuple[str, str]:
    """Valide la presence des secrets SSH utilises pour Proxmox.

    Returns:
        tuple[str, str]: Mot de passe SSH et mot de passe sudo.

    Raises:
        RuntimeError: Si les secrets requis ne sont pas charges.
    """
    if not PASS:
        raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")
    if not SUDO_PASS:
        raise RuntimeError("Variable d'environnement HIVE_SUDO_PASSWORD manquante.")
    return PASS, SUDO_PASS


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


def _resolve_history_symbols_for_profile(profile_hint: str | None) -> list[str] | None:
    """Resout un sous-ensemble de symboles historiques a synchroniser.

    Args:
        profile_hint (str | None): Profil de run optionnel.

    Returns:
        list[str] | None: Symboles cibles si un filtrage fin est possible,
        sinon ``None`` pour demander une synchronisation complete.
    """
    normalized = str(profile_hint or "").strip().lower()
    if not normalized:
        return None

    family: str | None = None
    if normalized in WAVE1_PROFILE_ORDER:
        _horizon, family = normalized.split("_", 1)
    elif normalized in V3_PROFILE_ORDER:
        _horizon, family = _split_v3_profile(normalized)

    if not family:
        return None
    return list(WAVE1_FAMILY_SYMBOLS.get(family) or []) or None


def _upload_history_subset(
    sftp: paramiko.SFTPClient,
    *,
    symbols: list[str],
    remote_dir: str,
) -> None:
    """Synchronise uniquement l'historique utile a une famille cible.

    Args:
        sftp (paramiko.SFTPClient): Canal SFTP actif.
        symbols (list[str]): Symboles a conserver.
        remote_dir (str): Dossier distant cible.
    """
    history_root = LOCAL_ROOT / HISTORY_DIR
    if not history_root.exists():
        print(f"SKIP {history_root} (absent)")
        return

    normalized_symbols = tuple(f"{symbol.lower()}_" for symbol in symbols if symbol)
    if not normalized_symbols:
        upload_tree(sftp, history_root, remote_dir)
        return

    matched = 0
    for file_path in sorted(history_root.glob("*.csv")):
        stem = file_path.stem.lower()
        if not stem.startswith(normalized_symbols):
            continue
        relative = file_path.relative_to(history_root)
        upload_file(sftp, file_path, f"{remote_dir}/{relative.as_posix()}")
        matched += 1

    if matched == 0:
        print(
            "Aucun historique cible trouve pour %s. Repli sur une synchronisation complete."
            % ",".join(symbols)
        )
        upload_tree(sftp, history_root, remote_dir)
    else:
        print(f"Historique cible synchronise pour {','.join(symbols)} ({matched} fichiers).")


def _sync_remote_training_payload(client: paramiko.SSHClient, profile_hint: str | None = None) -> None:
    """Synchronise le payload d'entrainement utile sur le serveur.

    Args:
        client (paramiko.SSHClient): Session SSH active.
        profile_hint (str | None): Profil cible optionnel pour filtrer
            l'historique utile.
    """
    sftp = client.open_sftp()
    try:
        for relative_path in SYNC_FILES:
            local_path = LOCAL_ROOT / relative_path
            remote_path = f"{REMOTE_DIR}/{relative_path.as_posix()}"
            upload_file(sftp, local_path, remote_path)
        history_symbols = _resolve_history_symbols_for_profile(profile_hint)
        for relative_dir in SYNC_DIRS:
            local_dir = LOCAL_ROOT / relative_dir
            remote_dir = f"{REMOTE_DIR}/{relative_dir.as_posix()}"
            if relative_dir == HISTORY_DIR and history_symbols is not None:
                _upload_history_subset(
                    sftp,
                    symbols=history_symbols,
                    remote_dir=remote_dir,
                )
                continue
            upload_tree(sftp, local_dir, remote_dir)

        ensure_remote_parent(sftp, REMOTE_SCRIPT)
        with sftp.file(REMOTE_SCRIPT, "w") as remote_file:
            remote_file.write(REMOTE_LAUNCH_SCRIPT)
        sftp.chmod(REMOTE_SCRIPT, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        sftp.close()


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


def _build_v4_supervisor_overrides() -> dict[str, str]:
    """Construit les variables minimales requises par le superviseur V4.

    Returns:
        dict[str, str]: Variables a exporter avant le lancement du superviseur.
    """

    return {
        "TRAINING_TIMESCALE_ENABLED": os.getenv("TRAINING_TIMESCALE_ENABLED", "1"),
        "TRAINING_TIMESCALE_HOST": os.getenv("TRAINING_TIMESCALE_HOST", "timescaledb"),
        "TRAINING_TIMESCALE_PORT": os.getenv("TRAINING_TIMESCALE_PORT", "5432"),
        "TRAINING_TIMESCALE_DB": os.getenv("TRAINING_TIMESCALE_DB", "thehive"),
        "TRAINING_TIMESCALE_USER": os.getenv("TRAINING_TIMESCALE_USER", "eva"),
        "TRAINING_TIMESCALE_PASSWORD": os.getenv(
            "TRAINING_TIMESCALE_PASSWORD",
            os.getenv("TIMESCALE_PASSWORD", "devpassword"),
        ),
        "TRAINING_TIMESCALE_SSLMODE": os.getenv("TRAINING_TIMESCALE_SSLMODE", "prefer"),
        "TRAINING_TIMESCALE_BARS_TABLE": os.getenv(
            "TRAINING_TIMESCALE_BARS_TABLE",
            "market.market_bars",
        ),
        "TRAINING_TIMESCALE_FEATURES_TABLE": os.getenv(
            "TRAINING_TIMESCALE_FEATURES_TABLE",
            "market.market_features",
        ),
        "TRAINING_TIMESCALE_DATASETS_TABLE": os.getenv(
            "TRAINING_TIMESCALE_DATASETS_TABLE",
            "training.training_datasets",
        ),
        "TRAINING_TIMESCALE_ARENA_TABLE": os.getenv(
            "TRAINING_TIMESCALE_ARENA_TABLE",
            "training.arena_results",
        ),
        "TRAINING_TIMESCALE_GA_TABLE": os.getenv(
            "TRAINING_TIMESCALE_GA_TABLE",
            "training.ga_trials",
        ),
        "TRAINING_TIMESCALE_REPLAY_TABLE": os.getenv(
            "TRAINING_TIMESCALE_REPLAY_TABLE",
            "training.replay_metadata",
        ),
        "TRAINING_TIMESCALE_RUN_WINDOWS_TABLE": os.getenv(
            "TRAINING_TIMESCALE_RUN_WINDOWS_TABLE",
            "training.run_windows",
        ),
    }


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


def _build_scalp_reduced_overrides(symbols: list[str] | None = None) -> dict[str, str]:
    """Construit un profil de relance court pour `scalp` uniquement.

    Args:
        symbols (list[str] | None): Univers reduit a imposer. Si absent,
            un panier par defaut multi-actifs est utilise.

    Returns:
        dict[str, str]: Variables a forcer pour une relance `scalp` ciblee.
    """
    reduced_symbols = symbols or [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "XAUUSD",
        "BTCUSD",
        "ETHUSD",
        "US30.cash",
        "US500.cash",
        "GER40.cash",
    ]
    symbol_csv = ",".join(reduced_symbols)
    symbol_count = str(len(reduced_symbols))
    return {
        "TRAINING_PROFILE": "refresh",
        "TRAINING_AUTOMATION_MODE": "force_refresh",
        "TRAINING_RUN_TRIGGER": "manual_scalp_reduced",
        "NIGHTLY_KEEP_VLLM": "0",
        "RUN_TRAIN_GNN": "0",
        "RUN_TRAIN_MUZERO": "1",
        "RUN_TRAIN_DREAMER": "0",
        "MUZERO_HORIZONS": "scalp",
        "MUZERO_SYMBOLS_SCALP": symbol_csv,
        "ARENA_SYMBOLS_SCALP": symbol_csv,
        "MUZERO_MAX_SYMBOLS": symbol_count,
        "ARENA_MAX_SYMBOLS": symbol_count,
        "MUZERO_GAMES_PER_SYMBOL": "10",
        "ARENA_GAMES_PER_SYMBOL": "4",
        "ARENA_MIN_GAMES": "18",
        "ARENA_MIN_SYMBOLS": "5",
        "MUZERO_TRAINING_STEPS": "12000",
        "MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS": "5",
        "MUZERO_LIVE_TOP_SYMBOLS": "5",
    }


def _normalize_scalp_multi_universe_symbols(symbols: list[str] | None = None) -> list[str]:
    """Normalise l'univers canonique `scalp` multi-univers a 7 symboles.

    Args:
        symbols (list[str] | None): Liste optionnelle a normaliser.

    Returns:
        list[str]: Univers final dedoublonne et ordonne.
    """
    alias_map = {
        "US30.CASH": "US30.cash",
        "GER40.CASH": "GER40.cash",
        "US500.CASH": "US500.cash",
    }
    requested = symbols or CANONICAL_SCALP_MULTI_UNIVERSE
    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in requested:
        label = alias_map.get(str(symbol or "").strip().upper(), str(symbol or "").strip())
        if not label or label in seen:
            continue
        normalized.append(label)
        seen.add(label)
    return normalized


def _build_muzero_scalp_multi_universe_full_overrides(
    symbols: list[str] | None = None,
) -> dict[str, str]:
    """Construit un `full` MuZero `scalp` force sur 7 symboles.

    Args:
        symbols (list[str] | None): Univers explicite a imposer.

    Returns:
        dict[str, str]: Variables d'environnement pretes pour Proxmox.
    """
    full_symbols = _normalize_scalp_multi_universe_symbols(symbols)
    symbol_csv = ",".join(full_symbols)
    symbol_count = str(len(full_symbols))
    return {
        "TRAINING_PROFILE": "refresh",
        "TRAINING_AUTOMATION_MODE": "force_refresh",
        "TRAINING_RUN_TRIGGER": "manual_muzero_scalp_multi_universe_full",
        "TRAINING_ENGINE": "muzero",
        "TRAINING_TRIAL_MODE": "full",
        "TRAINING_TRIAL_COST_PROFILE": "full",
        "TRAINING_GA_STATUS": "full",
        "TRAINING_GATE_PROFILE": "standard",
        "NIGHTLY_KEEP_VLLM": "0",
        "RUN_TRAIN_GNN": "0",
        "RUN_TRAIN_MUZERO": "1",
        "RUN_TRAIN_DREAMER": "0",
        "MUZERO_HORIZONS": "scalp",
        "MUZERO_SYMBOLS_SCALP": symbol_csv,
        "ARENA_SYMBOLS_SCALP": symbol_csv,
        "MUZERO_MAX_SYMBOLS": symbol_count,
        "ARENA_MAX_SYMBOLS": symbol_count,
        "MUZERO_TRAINING_STEPS": "18000",
        "MUZERO_GAMES_PER_SYMBOL": "12",
        "ARENA_GAMES_PER_SYMBOL": "6",
        "ARENA_MIN_GAMES": "12",
        "ARENA_MIN_SYMBOLS": "7",
        "MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS": symbol_count,
        "MUZERO_LIVE_TOP_SYMBOLS": symbol_count,
    }


def _build_dreamer_scalp_multi_universe_full_overrides(
    symbols: list[str] | None = None,
) -> dict[str, str]:
    """Construit un `full` Dreamer `scalp` force sur 7 symboles.

    Args:
        symbols (list[str] | None): Univers explicite a imposer.

    Returns:
        dict[str, str]: Variables d'environnement pretes pour Proxmox.
    """
    full_symbols = _normalize_scalp_multi_universe_symbols(symbols)
    symbol_csv = ",".join(full_symbols)
    symbol_count = str(len(full_symbols))
    return {
        "TRAINING_PROFILE": "refresh",
        "TRAINING_AUTOMATION_MODE": "force_refresh",
        "TRAINING_RUN_TRIGGER": "manual_dreamer_scalp_multi_universe_full",
        "TRAINING_ENGINE": "dreamer",
        "TRAINING_TRIAL_MODE": "full",
        "TRAINING_TRIAL_COST_PROFILE": "full",
        "TRAINING_GA_STATUS": "full",
        "TRAINING_GATE_PROFILE": "standard",
        "NIGHTLY_KEEP_VLLM": "0",
        "RUN_TRAIN_GNN": "0",
        "RUN_TRAIN_MUZERO": "0",
        "RUN_TRAIN_DREAMER": "1",
        "MUZERO_HORIZONS": "scalp",
        "DREAMER_HORIZON": "scalp",
        "MUZERO_SYMBOLS_SCALP": symbol_csv,
        "ARENA_SYMBOLS_SCALP": symbol_csv,
        "MUZERO_MAX_SYMBOLS": symbol_count,
        "ARENA_MAX_SYMBOLS": symbol_count,
        "DREAMER_EPOCHS": "220",
        "DREAMER_BATCH_SIZE": "6",
        "DREAMER_REPLAY_MAX_GAMES": "1800",
        "DREAMER_SEQUENCE_LENGTH": "96",
        "DREAMER_SEQUENCE_STRIDE": "4",
        "DREAMER_NUM_UNROLL_STEPS": "18",
        "ARENA_GAMES_PER_SYMBOL": "6",
        "ARENA_MIN_GAMES": "12",
        "ARENA_MIN_SYMBOLS": "7",
    }


def _build_intraday_reduced_overrides(symbols: list[str] | None = None) -> dict[str, str]:
    """Construit un profil de relance court pour `intraday` uniquement.

    Args:
        symbols (list[str] | None): Univers reduit a imposer. Si absent,
            un panier coeur multi-actifs est utilise.

    Returns:
        dict[str, str]: Variables a forcer pour une relance `intraday` ciblee.
    """
    reduced_symbols = symbols or [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "XAUUSD",
        "US30.cash",
        "US500.cash",
        "GER40.cash",
    ]
    symbol_csv = ",".join(reduced_symbols)
    symbol_count = str(len(reduced_symbols))
    return {
        "TRAINING_PROFILE": "refresh",
        "TRAINING_AUTOMATION_MODE": "force_refresh",
        "TRAINING_RUN_TRIGGER": "manual_intraday_reduced",
        "NIGHTLY_KEEP_VLLM": "0",
        "RUN_TRAIN_GNN": "0",
        "RUN_TRAIN_MUZERO": "1",
        "RUN_TRAIN_DREAMER": "0",
        "MUZERO_HORIZONS": "intraday",
        "MUZERO_SYMBOLS_INTRADAY": symbol_csv,
        "ARENA_SYMBOLS_INTRADAY": symbol_csv,
        "MUZERO_MAX_SYMBOLS": symbol_count,
        "ARENA_MAX_SYMBOLS": symbol_count,
        "MUZERO_GAMES_PER_SYMBOL": "8",
        "ARENA_GAMES_PER_SYMBOL": "4",
        "ARENA_MIN_GAMES": "14",
        "ARENA_MIN_SYMBOLS": "4",
        "MUZERO_TRAINING_STEPS": "8000",
        "MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS": "4",
        "MUZERO_LIVE_TOP_SYMBOLS": "4",
    }


def _build_swing_reduced_overrides(symbols: list[str] | None = None) -> dict[str, str]:
    """Construit un profil de relance court pour `swing` uniquement.

    Args:
        symbols (list[str] | None): Univers reduit a imposer. Si absent,
            un panier coeur multi-actifs est utilise.

    Returns:
        dict[str, str]: Variables a forcer pour une relance `swing` ciblee.
    """
    reduced_symbols = symbols or [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "XAUUSD",
        "US30.cash",
        "US500.cash",
        "GER40.cash",
    ]
    symbol_csv = ",".join(reduced_symbols)
    symbol_count = str(len(reduced_symbols))
    return {
        "TRAINING_PROFILE": "refresh",
        "TRAINING_AUTOMATION_MODE": "force_refresh",
        "TRAINING_RUN_TRIGGER": "manual_swing_reduced",
        "NIGHTLY_KEEP_VLLM": "0",
        "RUN_TRAIN_GNN": "0",
        "RUN_TRAIN_MUZERO": "1",
        "RUN_TRAIN_DREAMER": "0",
        "MUZERO_HORIZONS": "swing",
        "MUZERO_SYMBOLS_SWING": symbol_csv,
        "ARENA_SYMBOLS_SWING": symbol_csv,
        "MUZERO_MAX_SYMBOLS": symbol_count,
        "ARENA_MAX_SYMBOLS": symbol_count,
        "MUZERO_GAMES_PER_SYMBOL": "6",
        "ARENA_GAMES_PER_SYMBOL": "3",
        "ARENA_MIN_GAMES": "10",
        "ARENA_MIN_SYMBOLS": "4",
        "MUZERO_TRAINING_STEPS": "6000",
        "MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS": "4",
        "MUZERO_LIVE_TOP_SYMBOLS": "4",
    }


def _build_all_reduced_overrides(symbols: list[str] | None = None) -> dict[str, str]:
    """Construit un profil reduit multi-horizons pour le socle trading.

    Args:
        symbols (list[str] | None): Univers reduit a imposer sur les trois
            horizons. Si absent, un panier coeur multi-actifs est utilise.

    Returns:
        dict[str, str]: Variables a forcer pour une relance `scalp+intraday+swing`.
    """
    reduced_symbols = symbols or [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "XAUUSD",
        "BTCUSD",
        "ETHUSD",
        "US30.cash",
        "US500.cash",
        "GER40.cash",
    ]
    symbol_csv = ",".join(reduced_symbols)
    symbol_count = str(len(reduced_symbols))
    return {
        "TRAINING_PROFILE": "refresh",
        "TRAINING_AUTOMATION_MODE": "force_refresh",
        "TRAINING_RUN_TRIGGER": "manual_core_reduced",
        "NIGHTLY_KEEP_VLLM": "0",
        "RUN_TRAIN_GNN": "0",
        "RUN_TRAIN_MUZERO": "1",
        "RUN_TRAIN_DREAMER": "0",
        "MUZERO_HORIZONS": "scalp,intraday,swing",
        "MUZERO_SYMBOLS_SCALP": symbol_csv,
        "MUZERO_SYMBOLS_INTRADAY": symbol_csv,
        "MUZERO_SYMBOLS_SWING": symbol_csv,
        "ARENA_SYMBOLS_SCALP": symbol_csv,
        "ARENA_SYMBOLS_INTRADAY": symbol_csv,
        "ARENA_SYMBOLS_SWING": symbol_csv,
        "MUZERO_MAX_SYMBOLS": symbol_count,
        "ARENA_MAX_SYMBOLS": symbol_count,
        "MUZERO_GAMES_PER_SYMBOL": "8",
        "ARENA_GAMES_PER_SYMBOL": "4",
        "ARENA_MIN_GAMES": "14",
        "ARENA_MIN_SYMBOLS": "4",
        "MUZERO_TRAINING_STEPS": "8000",
        "MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS": "5",
        "MUZERO_LIVE_TOP_SYMBOLS": "5",
    }


def _normalize_wave1_profile(profile: str) -> str:
    """Valide un profil `horizon x famille` de la vague 1.

    Args:
        profile (str): Profil brut a normaliser.

    Returns:
        str: Profil valide au format `horizon_famille`.

    Raises:
        ValueError: Si le profil demande est inconnu.
    """
    normalized = str(profile or "").strip().lower()
    if normalized not in WAVE1_PROFILE_ORDER:
        raise ValueError(f"Profil de vague 1 non supporte: {profile}")
    return normalized


def _build_wave1_profile_overrides(profile: str) -> dict[str, str]:
    """Construit les surcharges d'un profil `horizon x famille`.

    Args:
        profile (str): Profil cible au format `horizon_famille`.

    Returns:
        dict[str, str]: Variables d'environnement prêtes pour le lanceur.
    """
    normalized_profile = _normalize_wave1_profile(profile)
    horizon, family = normalized_profile.split("_", 1)
    symbols = list(WAVE1_FAMILY_SYMBOLS[family])
    symbol_csv = ",".join(symbols)
    symbol_count = len(symbols)
    min_symbol_threshold = min(3, symbol_count)

    profile_defaults = {
        "scalp": {
            "steps": "16000",
            "games_per_symbol": "14",
            "arena_games_per_symbol": "8",
            "arena_min_games": "24",
            "live_top_symbols": str(min(5, symbol_count)),
        },
        "intraday": {
            "steps": "12000",
            "games_per_symbol": "10",
            "arena_games_per_symbol": "5",
            "arena_min_games": "15",
            "live_top_symbols": str(min(4, symbol_count)),
        },
        "swing": {
            "steps": "9000",
            "games_per_symbol": "8",
            "arena_games_per_symbol": "4",
            "arena_min_games": "12",
            "live_top_symbols": str(min(3, symbol_count)),
        },
    }
    defaults = profile_defaults[horizon]
    horizon_env = horizon.upper()
    overrides = {
        "TRAINING_PROFILE": "refresh",
        "TRAINING_AUTOMATION_MODE": "force_refresh",
        "TRAINING_RUN_TRIGGER": f"manual_{normalized_profile}",
        "NIGHTLY_KEEP_VLLM": "0",
        "RUN_TRAIN_GNN": "0",
        "RUN_TRAIN_MUZERO": "1",
        "RUN_TRAIN_DREAMER": "0",
        "MUZERO_HORIZONS": horizon,
        "MUZERO_MODEL_FAMILY": family,
        f"MUZERO_SYMBOLS_{horizon_env}": symbol_csv,
        f"ARENA_SYMBOLS_{horizon_env}": symbol_csv,
        "MUZERO_MAX_SYMBOLS": str(symbol_count),
        "ARENA_MAX_SYMBOLS": str(symbol_count),
        "MUZERO_GAMES_PER_SYMBOL": defaults["games_per_symbol"],
        "ARENA_GAMES_PER_SYMBOL": defaults["arena_games_per_symbol"],
        "ARENA_MIN_GAMES": defaults["arena_min_games"],
        "ARENA_MIN_SYMBOLS": str(min_symbol_threshold),
        "MUZERO_PROMOTION_MIN_EVAL_SYMBOLS": str(min_symbol_threshold),
        "MUZERO_TRAINING_STEPS": defaults["steps"],
        "MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS": str(min(5, symbol_count)),
        "MUZERO_LIVE_TOP_SYMBOLS": defaults["live_top_symbols"],
        "MUZERO_DATASET_SOURCE": "auto",
    }
    if horizon == "scalp":
        overrides.update(_build_scalp_mechanics_overrides(family))
    return overrides


def _build_scalp_mechanics_overrides(family: str) -> dict[str, str]:
    """Construit des surcharges mécaniques plus agressives pour `scalp`.

    Args:
        family (str): Famille d'actifs ciblee.

    Returns:
        dict[str, str]: Variables d'environnement de mecanique adaptees.
    """
    normalized_family = str(family or "").strip().lower()
    family_defaults: dict[str, dict[str, str]] = {
        "fx": {
            "MUZERO_ENTRY_EMA_MODE": "moderate",
            "MUZERO_ENTRY_REQUIRE_VWAP_ALIGNMENT": "1",
            "MUZERO_ENTRY_REQUIRE_OBV_CONFIRMATION": "0",
            "MUZERO_ENTRY_MIN_ADX": "14",
            "MUZERO_ENTRY_TREND_ADX": "19",
            "MUZERO_HOLD_STALE_PENALTY_AFTER_STEPS": "56",
            "MUZERO_HOLD_STALE_PENALTY": "0.95",
            "MUZERO_HOLD_TREND_PENALTY": "0.58",
            "MUZERO_HOLD_RANGE_PENALTY": "0.07",
            "MUZERO_PYRAMID_MAX_ADDITIONS": "2",
            "MUZERO_PYRAMID_MIN_PROFIT_TO_ADD": "0.0007",
            "MUZERO_PYRAMID_REWARD_BONUS": "0.16",
            "MUZERO_SPLIT_MAX_SPLITS": "3",
            "MUZERO_SPLIT_MIN_TRADE_RETURN": "0.0045",
            "MUZERO_SLBE_ACTIVATION_RETURN": "0.0025",
            "MUZERO_SLBE_BONUS": "7.0",
            "MUZERO_CLOSE_WINNER_THRESHOLD": "0.0060",
            "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD": "0.0110",
            "MUZERO_CLOSE_TP_LIKE_THRESHOLD": "0.0050",
        },
        "indices": {
            "MUZERO_ENTRY_EMA_MODE": "relaxed",
            "MUZERO_ENTRY_REQUIRE_VWAP_ALIGNMENT": "1",
            "MUZERO_ENTRY_REQUIRE_OBV_CONFIRMATION": "1",
            "MUZERO_ENTRY_MIN_ADX": "13",
            "MUZERO_ENTRY_TREND_ADX": "18",
            "MUZERO_HOLD_STALE_PENALTY_AFTER_STEPS": "52",
            "MUZERO_HOLD_STALE_PENALTY": "0.90",
            "MUZERO_HOLD_TREND_PENALTY": "0.50",
            "MUZERO_HOLD_RANGE_PENALTY": "0.06",
            "MUZERO_PYRAMID_MAX_ADDITIONS": "3",
            "MUZERO_PYRAMID_MIN_PROFIT_TO_ADD": "0.0006",
            "MUZERO_PYRAMID_REWARD_BONUS": "0.18",
            "MUZERO_SPLIT_MAX_SPLITS": "4",
            "MUZERO_SPLIT_MIN_TRADE_RETURN": "0.0042",
            "MUZERO_SLBE_ACTIVATION_RETURN": "0.0022",
            "MUZERO_SLBE_BONUS": "7.0",
            "MUZERO_CLOSE_WINNER_THRESHOLD": "0.0055",
            "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD": "0.0100",
            "MUZERO_CLOSE_TP_LIKE_THRESHOLD": "0.0045",
        },
        "metals": {
            "MUZERO_ENTRY_EMA_MODE": "relaxed",
            "MUZERO_ENTRY_REQUIRE_VWAP_ALIGNMENT": "1",
            "MUZERO_ENTRY_REQUIRE_OBV_CONFIRMATION": "1",
            "MUZERO_ENTRY_MIN_ADX": "12",
            "MUZERO_ENTRY_TREND_ADX": "17",
            "MUZERO_HOLD_STALE_PENALTY_AFTER_STEPS": "50",
            "MUZERO_HOLD_STALE_PENALTY": "0.90",
            "MUZERO_HOLD_TREND_PENALTY": "0.52",
            "MUZERO_HOLD_RANGE_PENALTY": "0.06",
            "MUZERO_PYRAMID_MAX_ADDITIONS": "3",
            "MUZERO_PYRAMID_MIN_PROFIT_TO_ADD": "0.0008",
            "MUZERO_PYRAMID_REWARD_BONUS": "0.18",
            "MUZERO_SPLIT_MAX_SPLITS": "4",
            "MUZERO_SPLIT_MIN_TRADE_RETURN": "0.0040",
            "MUZERO_SLBE_ACTIVATION_RETURN": "0.0020",
            "MUZERO_SLBE_BONUS": "7.0",
            "MUZERO_CLOSE_WINNER_THRESHOLD": "0.0050",
            "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD": "0.0090",
            "MUZERO_CLOSE_TP_LIKE_THRESHOLD": "0.0042",
        },
    }
    return dict(family_defaults.get(normalized_family, family_defaults["fx"]))


def _normalize_v3_profile(profile: str) -> str:
    """Valide un profil V3 explicite.

    Args:
        profile (str): Profil V3 demande.

    Returns:
        str: Profil V3 normalise.

    Raises:
        ValueError: Si le profil est inconnu.
    """
    normalized = str(profile or "").strip().lower()
    if normalized not in V3_PROFILE_ORDER:
        raise ValueError(f"Profil V3 non supporte: {profile}")
    return normalized


def _normalize_v3_mode(mode: str) -> str:
    """Valide un mode de lancement V3.

    Args:
        mode (str): Mode de lancement attendu.

    Returns:
        str: Mode V3 normalise.

    Raises:
        ValueError: Si le mode est inconnu.
    """
    normalized = str(mode or "").strip().lower()
    if normalized not in V3_MODE_ORDER:
        raise ValueError(f"Mode V3 non supporte: {mode}")
    return normalized


def _split_v3_profile(profile: str) -> tuple[str, str]:
    """Decode un profil V3 en horizon et famille.

    Args:
        profile (str): Profil V3 normalise.

    Returns:
        tuple[str, str]: Couple ``(horizon, famille)``.
    """
    normalized = _normalize_v3_profile(profile)
    horizon, family, _version = normalized.split("_", 2)
    return horizon, family


def _build_scalp_v3_base_overrides(family: str) -> dict[str, str]:
    """Construit les regles V3 communes d'un `scalp` par famille.

    Args:
        family (str): Famille ciblee.

    Returns:
        dict[str, str]: Surcouches mecaniques et directionnelles V3.
    """
    normalized_family = str(family or "").strip().lower()
    base = _build_scalp_mechanics_overrides(normalized_family)
    base.update(
        {
            "MUZERO_ENTRY_ALLOW_TREND_FALLBACK": "1",
            "MUZERO_ACTIVITY_MIN_ENTRIES": "3" if normalized_family == "fx" else "2",
            "MUZERO_ACTIVITY_INACTIVE_EPISODE_PENALTY": "18.0",
            "MUZERO_ACTIVITY_INSUFFICIENT_ENTRIES_PENALTY": "8.0",
            "MUZERO_DIRECTIONAL_MIN_ENTRY_SHARE": "0.20",
            "MUZERO_DIRECTIONAL_MAX_IMBALANCE": "0.60",
            "MUZERO_DIRECTIONAL_IMBALANCE_PENALTY": "10.0",
            "MUZERO_SPLIT_MIN_REALIZED_PCT": "0.04",
            "MUZERO_SPLIT_FAILURE_PENALTY": "0.35",
            "MUZERO_SLBE_EXIT_BONUS": "4.0",
            "MUZERO_REWARD_REALIZED_PNL_MULTIPLIER": "1.25",
            "MUZERO_REWARD_CLOSE_REALIZED_MULTIPLIER": "1.55",
            "MUZERO_REWARD_SPLIT_REALIZED_MULTIPLIER": "1.35",
            "MUZERO_REWARD_HOLD_DRAG_MULTIPLIER": "0.45",
            "MUZERO_REWARD_PYRAMID_FAILURE_PENALTY": "0.18",
            "MUZERO_REWARD_PYRAMID_NEGATIVE_EXIT_PENALTY": "0.25",
        }
    )
    if normalized_family == "metals":
        base.update(
            {
                "MUZERO_ENTRY_MIN_ADX": "10",
                "MUZERO_ENTRY_TREND_ADX": "15",
                "MUZERO_SPLIT_MIN_TRADE_RETURN": "0.0032",
                "MUZERO_SPLIT_MIN_REALIZED_PCT": "0.03",
                "MUZERO_SLBE_ACTIVATION_RETURN": "0.0016",
                "MUZERO_CLOSE_WINNER_THRESHOLD": "0.0040",
                "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD": "0.0075",
                "MUZERO_CLOSE_TP_LIKE_THRESHOLD": "0.0035",
                "MUZERO_ACTIVITY_MIN_ENTRIES": "2",
                "MUZERO_REWARD_CLOSE_REALIZED_MULTIPLIER": "1.7",
                "MUZERO_REWARD_SPLIT_REALIZED_MULTIPLIER": "1.5",
            }
        )
    elif normalized_family == "fx":
        base.update(
            {
                "MUZERO_ENTRY_MIN_ADX": "12",
                "MUZERO_ENTRY_TREND_ADX": "17",
                "MUZERO_HOLD_STALE_PENALTY_AFTER_STEPS": "44",
                "MUZERO_HOLD_TREND_PENALTY": "0.72",
                "MUZERO_SPLIT_MIN_TRADE_RETURN": "0.0038",
                "MUZERO_CLOSE_WINNER_THRESHOLD": "0.0048",
                "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD": "0.0082",
                "MUZERO_CLOSE_TP_LIKE_THRESHOLD": "0.0041",
            }
        )
    elif normalized_family == "indices":
        base.update(
            {
                "MUZERO_ENTRY_REQUIRE_VWAP_ALIGNMENT": "0",
                "MUZERO_ENTRY_REQUIRE_OBV_CONFIRMATION": "0",
                "MUZERO_ENTRY_MIN_ADX": "9",
                "MUZERO_ENTRY_TREND_ADX": "13",
                "MUZERO_ACTIVITY_MIN_ENTRIES": "1",
                "MUZERO_ACTIVITY_INACTIVE_EPISODE_PENALTY": "24.0",
                "MUZERO_CLOSE_WINNER_THRESHOLD": "0.0045",
                "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD": "0.0080",
                "MUZERO_CLOSE_TP_LIKE_THRESHOLD": "0.0038",
            }
        )
    return base


def _get_v3_trial_catalog(profile: str) -> list[dict[str, Any]]:
    """Retourne les profils d'essai du proxy GA pour une famille.

    Args:
        profile (str): Profil V3 cible.

    Returns:
        list[dict[str, Any]]: Catalogue d'essais courts.
    """
    normalized_profile = _normalize_v3_profile(profile)
    _horizon, family = _split_v3_profile(normalized_profile)
    common_trials = [
        {
            "trial_id": "momentum_close",
            "overrides": {
                "MUZERO_CLOSE_WINNER_THRESHOLD": "0.0038",
                "MUZERO_CLOSE_TP_LIKE_THRESHOLD": "0.0032",
                "MUZERO_REWARD_CLOSE_REALIZED_MULTIPLIER": "1.75",
                "MUZERO_HOLD_TREND_PENALTY": "0.82",
            },
        },
        {
            "trial_id": "split_capture",
            "overrides": {
                "MUZERO_SPLIT_MIN_TRADE_RETURN": "0.0029",
                "MUZERO_SPLIT_MIN_REALIZED_PCT": "0.025",
                "MUZERO_REWARD_SPLIT_REALIZED_MULTIPLIER": "1.65",
                "MUZERO_SPLIT_FAILURE_PENALTY": "0.45",
            },
        },
        {
            "trial_id": "balanced_activity",
            "overrides": {
                "MUZERO_ACTIVITY_MIN_ENTRIES": "2",
                "MUZERO_ACTIVITY_INSUFFICIENT_ENTRIES_PENALTY": "10.0",
                "MUZERO_DIRECTIONAL_IMBALANCE_PENALTY": "14.0",
                "MUZERO_REWARD_HOLD_DRAG_MULTIPLIER": "0.62",
            },
        },
    ]
    family_trials: dict[str, list[dict[str, Any]]] = {
        "metals": common_trials
        + [
            {
                "trial_id": "metals_vwap_obv",
                "overrides": {
                    "MUZERO_ENTRY_REQUIRE_VWAP_ALIGNMENT": "1",
                    "MUZERO_ENTRY_REQUIRE_OBV_CONFIRMATION": "1",
                    "MUZERO_ENTRY_MIN_ADX": "9",
                    "MUZERO_SLBE_ACTIVATION_RETURN": "0.0014",
                },
            }
        ],
        "fx": common_trials
        + [
            {
                "trial_id": "fx_exit_focus",
                "overrides": {
                    "MUZERO_ENTRY_REQUIRE_VWAP_ALIGNMENT": "1",
                    "MUZERO_ENTRY_REQUIRE_OBV_CONFIRMATION": "0",
                    "MUZERO_ACTIVITY_MIN_ENTRIES": "3",
                    "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD": "0.0072",
                },
            }
        ],
        "indices": common_trials
        + [
            {
                "trial_id": "indices_unfreeze",
                "overrides": {
                    "MUZERO_ENTRY_REQUIRE_VWAP_ALIGNMENT": "0",
                    "MUZERO_ENTRY_REQUIRE_OBV_CONFIRMATION": "0",
                    "MUZERO_ENTRY_ALLOW_TREND_FALLBACK": "1",
                    "MUZERO_ENTRY_MIN_ADX": "8",
                    "MUZERO_ACTIVITY_MIN_ENTRIES": "1",
                },
            }
        ],
    }
    return list(family_trials.get(family, common_trials))


def _build_v3_profile_overrides(
    profile: str,
    *,
    mode: str,
    trial: dict[str, Any] | None = None,
    finalist_rank: int | None = None,
) -> dict[str, str]:
    """Construit les variables d'environnement d'un profil V3.

    Args:
        profile (str): Profil V3 cible.
        mode (str): Mode `proxy_ga` ou `full`.
        trial (dict[str, Any] | None): Essai GA cible.
        finalist_rank (int | None): Rang du finaliste pour le mode `full`.

    Returns:
        dict[str, str]: Bloc d'environnement pret a etre exporte.
    """
    normalized_profile = _normalize_v3_profile(profile)
    normalized_mode = _normalize_v3_mode(mode)
    horizon, family = _split_v3_profile(normalized_profile)
    base_profile = f"{horizon}_{family}"
    overrides = _build_wave1_profile_overrides(base_profile)
    overrides.update(_build_scalp_v3_base_overrides(family))
    trial_id = str((trial or {}).get("trial_id") or "").strip() or "baseline"
    mode_label = "proxy" if normalized_mode == "proxy_ga" else "full"
    if normalized_mode == "proxy_ga":
        overrides.update(
            {
                "TRAINING_RUN_TRIGGER": f"manual_{normalized_profile}_{trial_id}",
                "MUZERO_TRAINING_STEPS": "6000",
                "MUZERO_GAMES_PER_SYMBOL": "8",
                "ARENA_GAMES_PER_SYMBOL": "3",
                "ARENA_MIN_GAMES": "10",
                "MUZERO_PROMOTION_MIN_EVAL_GAMES": "8",
                "TRAINING_GA_STATUS": "proxy_ga",
            }
        )
    else:
        rank_label = str(finalist_rank or 0)
        overrides.update(
            {
                "TRAINING_RUN_TRIGGER": f"manual_{normalized_profile}_finalist_{rank_label}_{trial_id}",
                "MUZERO_TRAINING_STEPS": "16000",
                "MUZERO_GAMES_PER_SYMBOL": "14",
                "ARENA_GAMES_PER_SYMBOL": "8",
                "ARENA_MIN_GAMES": "24",
                "TRAINING_GA_STATUS": "full",
            }
        )
    overrides["TRAINING_GA_GENERATION"] = "1"
    overrides["TRAINING_GA_TRIAL"] = trial_id
    if trial:
        overrides.update({key: str(value) for key, value in dict(trial.get("overrides") or {}).items()})
    overrides["MUZERO_MODEL_FAMILY"] = family
    overrides["MUZERO_HORIZONS"] = horizon
    overrides["MUZERO_DATASET_SOURCE"] = "auto"
    overrides["TRAINING_PROFILE"] = "refresh"
    overrides["TRAINING_AUTOMATION_MODE"] = "force_refresh"
    overrides["TRAINING_MECHANICS_PROFILE_VERSION"] = normalized_profile
    return overrides


def _normalize_v4_profile(profile: str) -> str:
    """Valide un profil V4 en reutilisant la nomenclature V3."""

    return _normalize_v3_profile(profile)


def _normalize_v4_mode(mode: str) -> str:
    """Valide un mode V4."""

    return _normalize_v3_mode(mode)


def _normalize_v4_engine(engine: str) -> str:
    """Valide le moteur de training V4.

    Args:
        engine (str): Moteur cible.

    Returns:
        str: Nom de moteur normalise.

    Raises:
        ValueError: Si le moteur n'est pas supporte.
    """
    normalized = str(engine or "").strip().lower()
    if normalized not in V4_ENGINE_ORDER:
        raise ValueError(f"Moteur V4 non supporte: {engine}")
    return normalized


def _get_dreamer_v4_trial_catalog(profile: str) -> list[dict[str, Any]]:
    """Retourne le catalogue d'essais Dreamer V4 pour une famille."""

    normalized_profile = _normalize_v4_profile(profile)
    _horizon, family = _split_v3_profile(normalized_profile)
    common_trials = [
        {
            "trial_id": "world_model_long_seq",
            "overrides": {
                "DREAMER_SEQUENCE_LENGTH": "96",
                "DREAMER_SEQUENCE_STRIDE": "4",
                "DREAMER_NUM_UNROLL_STEPS": "18",
                "DREAMER_EPOCHS": "8",
            },
        },
        {
            "trial_id": "balanced_stride",
            "overrides": {
                "DREAMER_SEQUENCE_LENGTH": "80",
                "DREAMER_SEQUENCE_STRIDE": "5",
                "DREAMER_NUM_UNROLL_STEPS": "14",
                "DREAMER_BATCH_SIZE": "64",
            },
        },
        {
            "trial_id": "fast_unroll",
            "overrides": {
                "DREAMER_SEQUENCE_LENGTH": "64",
                "DREAMER_SEQUENCE_STRIDE": "4",
                "DREAMER_NUM_UNROLL_STEPS": "12",
                "DREAMER_BATCH_SIZE": "96",
                "DREAMER_EPOCHS": "6",
            },
        },
    ]
    family_trials: dict[str, list[dict[str, Any]]] = {
        "metals": common_trials
        + [
            {
                "trial_id": "metals_memory_focus",
                "overrides": {
                    "DREAMER_SEQUENCE_LENGTH": "112",
                    "DREAMER_SEQUENCE_STRIDE": "3",
                    "DREAMER_HIDDEN_STATE_SIZE": "384",
                    "DREAMER_NUM_UNROLL_STEPS": "20",
                },
            }
        ],
        "fx": common_trials
        + [
            {
                "trial_id": "fx_exit_focus",
                "overrides": {
                    "DREAMER_SEQUENCE_LENGTH": "88",
                    "DREAMER_SEQUENCE_STRIDE": "4",
                    "DREAMER_NUM_UNROLL_STEPS": "16",
                    "DREAMER_EPOCHS": "9",
                },
            }
        ],
        "indices": common_trials
        + [
            {
                "trial_id": "indices_activity_unlock",
                "overrides": {
                    "DREAMER_SEQUENCE_LENGTH": "72",
                    "DREAMER_SEQUENCE_STRIDE": "2",
                    "DREAMER_NUM_UNROLL_STEPS": "14",
                    "DREAMER_BATCH_SIZE": "80",
                },
            }
        ],
    }
    return list(family_trials.get(family, common_trials))


def _get_v4_trial_catalog(profile: str, engine: str) -> list[dict[str, Any]]:
    """Retourne le catalogue d'essais V4 d'un moteur."""

    normalized_engine = _normalize_v4_engine(engine)
    normalized_profile = _normalize_v4_profile(profile)
    if normalized_engine == "muzero":
        return _get_v3_trial_catalog(normalized_profile)
    return _get_dreamer_v4_trial_catalog(normalized_profile)


def _build_muzero_v4_profile_overrides(
    profile: str,
    *,
    mode: str,
    trial: dict[str, Any] | None = None,
    finalist_rank: int | None = None,
) -> dict[str, str]:
    """Construit l'environnement MuZero V4 explicite."""

    normalized_profile = _normalize_v4_profile(profile)
    normalized_mode = _normalize_v4_mode(mode)
    trial_id = str((trial or {}).get("trial_id") or "").strip() or "baseline"
    overrides = _build_v3_profile_overrides(
        normalized_profile,
        mode=normalized_mode,
        trial=trial,
        finalist_rank=finalist_rank,
    )
    if normalized_mode == "proxy_ga":
        overrides["TRAINING_RUN_TRIGGER"] = f"manual_{normalized_profile}_muzero_{trial_id}"
    else:
        rank_label = str(finalist_rank or 0)
        overrides["TRAINING_RUN_TRIGGER"] = (
            f"manual_{normalized_profile}_muzero_finalist_{rank_label}_{trial_id}"
        )
    overrides["TRAINING_ENGINE"] = "muzero"
    overrides["TRAINING_TRIAL_MODE"] = normalized_mode
    overrides["TRAINING_TRIAL_COST_PROFILE"] = "proxy" if normalized_mode == "proxy_ga" else "full"
    overrides["RUN_TRAIN_MUZERO"] = "1"
    overrides["RUN_TRAIN_DREAMER"] = "0"
    return overrides


def _build_dreamer_v4_profile_overrides(
    profile: str,
    *,
    mode: str,
    trial: dict[str, Any] | None = None,
    finalist_rank: int | None = None,
) -> dict[str, str]:
    """Construit l'environnement Dreamer V4 explicite."""

    normalized_profile = _normalize_v4_profile(profile)
    normalized_mode = _normalize_v4_mode(mode)
    horizon, family = _split_v3_profile(normalized_profile)
    base_profile = f"{horizon}_{family}"
    overrides = _build_wave1_profile_overrides(base_profile)
    overrides.update(_build_scalp_v3_base_overrides(family))
    trial_id = str((trial or {}).get("trial_id") or "").strip() or "baseline"
    mode_label = "proxy" if normalized_mode == "proxy_ga" else "full"
    overrides.update(
        {
            "TRAINING_PROFILE": "refresh",
            "TRAINING_AUTOMATION_MODE": "force_refresh",
            "TRAINING_ENGINE": "dreamer",
            "TRAINING_TRIAL_MODE": normalized_mode,
            "TRAINING_TRIAL_COST_PROFILE": mode_label,
            "TRAINING_GA_STATUS": normalized_mode,
            "TRAINING_GA_GENERATION": "1",
            "TRAINING_GA_TRIAL": trial_id,
            "TRAINING_MECHANICS_PROFILE_VERSION": normalized_profile,
            "MUZERO_MODEL_FAMILY": family,
            "MUZERO_HORIZONS": horizon,
            "DREAMER_HORIZON": horizon,
            "MUZERO_DATASET_SOURCE": "auto",
            "RUN_TRAIN_GNN": "0",
            "RUN_TRAIN_MUZERO": "0",
            "RUN_TRAIN_DREAMER": "1",
        }
    )
    if normalized_mode == "proxy_ga":
        overrides.update(
            {
                "TRAINING_RUN_TRIGGER": f"manual_{normalized_profile}_dreamer_{trial_id}",
                "DREAMER_EPOCHS": "8",
                "DREAMER_SEQUENCE_LENGTH": "72",
                "DREAMER_SEQUENCE_STRIDE": "4",
                "DREAMER_NUM_UNROLL_STEPS": "12",
                "DREAMER_BATCH_SIZE": "64",
            }
        )
    else:
        rank_label = str(finalist_rank or 0)
        overrides.update(
            {
                "TRAINING_RUN_TRIGGER": (
                    f"manual_{normalized_profile}_dreamer_finalist_{rank_label}_{trial_id}"
                ),
                "DREAMER_EPOCHS": "18",
                "DREAMER_SEQUENCE_LENGTH": "96",
                "DREAMER_SEQUENCE_STRIDE": "4",
                "DREAMER_NUM_UNROLL_STEPS": "18",
                "DREAMER_BATCH_SIZE": "64",
            }
        )
    if trial:
        overrides.update({key: str(value) for key, value in dict(trial.get("overrides") or {}).items()})
    return overrides


def _build_v4_profile_overrides(
    profile: str,
    *,
    engine: str,
    mode: str,
    trial: dict[str, Any] | None = None,
    finalist_rank: int | None = None,
) -> dict[str, str]:
    """Construit les surcharges d'environnement V4 pour un moteur."""

    normalized_engine = _normalize_v4_engine(engine)
    if normalized_engine == "muzero":
        return _build_muzero_v4_profile_overrides(
            profile,
            mode=mode,
            trial=trial,
            finalist_rank=finalist_rank,
        )
    return _build_dreamer_v4_profile_overrides(
        profile,
        mode=mode,
        trial=trial,
        finalist_rank=finalist_rank,
    )


def _build_gold_monday_common_overrides(
    profile: str,
    *,
    engine: str,
) -> dict[str, str]:
    """Construit les surcharges communes du profil Monday Gold.

    Args:
        profile (str): Profil V4 cible.
        engine (str): Moteur cible.

    Returns:
        dict[str, str]: Variables communes au mode Gold-only.
    """

    normalized_profile = _normalize_v4_profile(profile)
    horizon, family = _split_v3_profile(normalized_profile)
    return {
        "TRAINING_PROFILE": "refresh",
        "TRAINING_AUTOMATION_MODE": "force_refresh",
        "TRAINING_ENGINE": _normalize_v4_engine(engine),
        "TRAINING_GATE_PROFILE": "gold_demo",
        "MUZERO_PROMOTION_GATE_PROFILE": "gold_demo",
        "TRAINING_FOCUS_SYMBOLS": GOLD_MONDAY_FOCUS_SYMBOL,
        "MUZERO_HORIZONS": horizon,
        "DREAMER_HORIZON": horizon,
        "MUZERO_MODEL_FAMILY": family,
        "MUZERO_DATASET_SOURCE": "auto",
        "MUZERO_SYMBOLS": GOLD_MONDAY_FOCUS_SYMBOL,
        "MUZERO_SYMBOLS_SCALP": GOLD_MONDAY_FOCUS_SYMBOL,
        "ARENA_SYMBOLS": GOLD_MONDAY_FOCUS_SYMBOL,
        "ARENA_SYMBOLS_SCALP": GOLD_MONDAY_FOCUS_SYMBOL,
        "MUZERO_MAX_SYMBOLS": "1",
        "ARENA_MAX_SYMBOLS": "1",
        "ARENA_MIN_SYMBOLS": "1",
        "MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS": "1",
        "MUZERO_LIVE_TOP_SYMBOLS": "1",
        "NIGHTLY_KEEP_VLLM": "0",
        "NIGHTLY_DEFER_VLLM_RESTART": "1",
        "RUN_TRAIN_GNN": "0",
    }


def _build_gold_monday_muzero_overrides(
    profile: str,
    *,
    mode: str,
    trial: dict[str, Any] | None = None,
    finalist_rank: int | None = None,
) -> dict[str, str]:
    """Construit les surcharges MuZero du mode Monday Gold.

    Args:
        profile (str): Profil cible.
        mode (str): Mode `proxy_ga` ou `full`.
        trial (dict[str, Any] | None): Definition du trial.
        finalist_rank (int | None): Rang du finaliste en mode `full`.

    Returns:
        dict[str, str]: Environnement complet d'un trial MuZero Gold.
    """

    normalized_mode = _normalize_v4_mode(mode)
    base = _build_muzero_v4_profile_overrides(
        profile,
        mode="proxy_ga" if normalized_mode == "proxy_ga" else "full",
        trial=trial,
        finalist_rank=finalist_rank,
    )
    trial_id = str((trial or {}).get("trial_id") or "").strip() or "baseline"
    base.update(_build_gold_monday_common_overrides(profile, engine="muzero"))
    base.update(
        {
            "TRAINING_RUN_TRIGGER": (
                f"manual_{GOLD_MONDAY_SEQUENCE_NAME}_muzero_{trial_id}"
                if normalized_mode == "proxy_ga"
                else (
                    f"manual_{GOLD_MONDAY_SEQUENCE_NAME}_muzero_finalist_"
                    f"{str(finalist_rank or 0)}_{trial_id}"
                )
            ),
            "TRAINING_TRIAL_MODE": normalized_mode,
            "TRAINING_TRIAL_COST_PROFILE": "proxy" if normalized_mode == "proxy_ga" else "full",
            "TRAINING_GA_STATUS": normalized_mode,
            "TRAINING_GA_TRIAL": trial_id,
            "RUN_TRAIN_MUZERO": "1",
            "RUN_TRAIN_DREAMER": "0",
        }
    )
    if normalized_mode == "proxy_ga":
        base.update(
            {
                "MUZERO_TRAINING_STEPS": "10000",
                "MUZERO_GAMES_PER_SYMBOL": "12",
                "MUZERO_NUM_SIMULATIONS": "128",
                "MUZERO_MAX_MOVES": "260",
                "MUZERO_GOLD_PRECHECK_STEP": "3000",
                "MUZERO_GOLD_PRECHECK_GAMES": "6",
                "ARENA_GAMES_PER_SYMBOL": "6",
                "ARENA_MIN_GAMES": "12",
            }
        )
    else:
        base.update(
            {
                "MUZERO_TRAINING_STEPS": "32000",
                "MUZERO_GAMES_PER_SYMBOL": "32",
                "MUZERO_NUM_SIMULATIONS": "192",
                "MUZERO_MAX_MOVES": "360",
                "ARENA_GAMES_PER_SYMBOL": "16",
                "ARENA_MIN_GAMES": "32",
            }
        )
    return base


def _build_gold_monday_dreamer_overrides(
    profile: str,
    *,
    mode: str,
    trial: dict[str, Any] | None = None,
    finalist_rank: int | None = None,
) -> dict[str, str]:
    """Construit les surcharges Dreamer du mode Monday Gold.

    Args:
        profile (str): Profil cible.
        mode (str): Mode `smoke`, `proxy_ga` ou `full`.
        trial (dict[str, Any] | None): Definition du trial.
        finalist_rank (int | None): Rang du finaliste en mode `full`.

    Returns:
        dict[str, str]: Environnement complet d'un trial Dreamer Gold.
    """

    normalized_profile = _normalize_v4_profile(profile)
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"smoke", "proxy_ga", "full"}:
        raise ValueError(f"Mode Dreamer Gold non supporte: {mode}")
    horizon, family = _split_v3_profile(normalized_profile)
    base_profile = f"{horizon}_{family}"
    overrides = _build_wave1_profile_overrides(base_profile)
    overrides.update(_build_scalp_v3_base_overrides(family))
    overrides.update(_build_gold_monday_common_overrides(profile, engine="dreamer"))
    trial_id = str((trial or {}).get("trial_id") or "").strip() or "baseline"
    overrides.update(
        {
            "TRAINING_ENGINE": "dreamer",
            "TRAINING_GA_STATUS": normalized_mode,
            "TRAINING_GA_GENERATION": "1",
            "TRAINING_GA_TRIAL": trial_id,
            "TRAINING_TRIAL_MODE": normalized_mode,
            "TRAINING_TRIAL_COST_PROFILE": (
                "smoke"
                if normalized_mode == "smoke"
                else ("proxy" if normalized_mode == "proxy_ga" else "full")
            ),
            "TRAINING_MECHANICS_PROFILE_VERSION": normalized_profile,
            "RUN_TRAIN_GNN": "0",
            "RUN_TRAIN_MUZERO": "0",
            "RUN_TRAIN_DREAMER": "1",
            "DREAMER_HORIZON": horizon,
        }
    )
    if normalized_mode == "smoke":
        overrides["TRAINING_RUN_TRIGGER"] = f"manual_{GOLD_MONDAY_SEQUENCE_NAME}_dreamer_smoke_{trial_id}"
    elif normalized_mode == "proxy_ga":
        overrides["TRAINING_RUN_TRIGGER"] = f"manual_{GOLD_MONDAY_SEQUENCE_NAME}_dreamer_{trial_id}"
    else:
        overrides["TRAINING_RUN_TRIGGER"] = (
            f"manual_{GOLD_MONDAY_SEQUENCE_NAME}_dreamer_finalist_"
            f"{str(finalist_rank or 0)}_{trial_id}"
        )
    if trial:
        trial_overrides = {key: str(value) for key, value in dict(trial.get("overrides") or {}).items()}
        if normalized_mode == "full":
            overrides.update(
                {
                    "DREAMER_EPOCHS": "220",
                    "DREAMER_BATCH_SIZE": "6",
                    "DREAMER_REPLAY_MAX_GAMES": "1500",
                }
            )
            for key in (
                "DREAMER_SEQUENCE_LENGTH",
                "DREAMER_SEQUENCE_STRIDE",
                "DREAMER_NUM_UNROLL_STEPS",
                "DREAMER_HIDDEN_STATE_SIZE",
                "DREAMER_NETWORK_HIDDEN_DIMS",
                "DREAMER_MAX_START_STATES",
            ):
                if key in trial_overrides:
                    overrides[key] = trial_overrides[key]
        else:
            overrides.update(trial_overrides)
    return overrides


def _get_gold_monday_dreamer_smoke_trials() -> list[dict[str, Any]]:
    """Retourne le catalogue de smoke tests Dreamer Gold."""

    return [
        {
            "trial_id": "gold_smoke_primary",
            "overrides": {
                "DREAMER_EPOCHS": "10",
                "DREAMER_SEQUENCE_LENGTH": "24",
                "DREAMER_SEQUENCE_STRIDE": "6",
                "DREAMER_NUM_UNROLL_STEPS": "6",
                "DREAMER_BATCH_SIZE": "2",
                "DREAMER_REPLAY_MAX_GAMES": "600",
            },
        },
        {
            "trial_id": "gold_smoke_rescue",
            "overrides": {
                "DREAMER_EPOCHS": "10",
                "DREAMER_SEQUENCE_LENGTH": "16",
                "DREAMER_SEQUENCE_STRIDE": "4",
                "DREAMER_NUM_UNROLL_STEPS": "4",
                "DREAMER_BATCH_SIZE": "2",
                "DREAMER_REPLAY_MAX_GAMES": "600",
            },
        },
    ]


def _get_gold_monday_dreamer_proxy_trials() -> list[dict[str, Any]]:
    """Retourne le catalogue proxy GA Dreamer pour Monday Gold."""

    return [
        {
            "trial_id": "gold_balanced_short_seq",
            "overrides": {
                "DREAMER_SEQUENCE_LENGTH": "32",
                "DREAMER_SEQUENCE_STRIDE": "8",
                "DREAMER_NUM_UNROLL_STEPS": "8",
                "DREAMER_BATCH_SIZE": "4",
                "DREAMER_EPOCHS": "60",
                "DREAMER_REPLAY_MAX_GAMES": "1200",
            },
        },
        {
            "trial_id": "gold_fast_close",
            "overrides": {
                "DREAMER_SEQUENCE_LENGTH": "24",
                "DREAMER_SEQUENCE_STRIDE": "6",
                "DREAMER_NUM_UNROLL_STEPS": "8",
                "DREAMER_BATCH_SIZE": "4",
                "DREAMER_EPOCHS": "80",
                "DREAMER_REPLAY_MAX_GAMES": "1200",
            },
        },
        {
            "trial_id": "gold_memory_mid",
            "overrides": {
                "DREAMER_SEQUENCE_LENGTH": "40",
                "DREAMER_SEQUENCE_STRIDE": "8",
                "DREAMER_NUM_UNROLL_STEPS": "10",
                "DREAMER_HIDDEN_STATE_SIZE": "192",
                "DREAMER_BATCH_SIZE": "4",
                "DREAMER_EPOCHS": "80",
                "DREAMER_REPLAY_MAX_GAMES": "1200",
            },
        },
    ]


def _build_gold_monday_sequence_config(
    *,
    sequence_id: str,
    stdout_log_path: str,
    stderr_log_path: str,
) -> dict[str, Any]:
    """Construit la configuration du superviseur pour Monday Gold.

    Args:
        sequence_id (str): Identifiant unique de sequence.
        stdout_log_path (str): Journal stdout du superviseur.
        stderr_log_path (str): Journal stderr du superviseur.

    Returns:
        dict[str, Any]: Configuration serialisable de la sequence Gold.
    """

    muzero_proxy_trials = []
    muzero_full_catalog: dict[str, dict[str, str]] = {}
    for trial in _get_v3_trial_catalog(GOLD_MONDAY_PROFILE):
        trial_id = str(trial.get("trial_id") or "").strip()
        muzero_proxy_trials.append(
            {
                **dict(trial),
                "runtime_overrides": _build_gold_monday_muzero_overrides(
                    GOLD_MONDAY_PROFILE,
                    mode="proxy_ga",
                    trial=trial,
                ),
            }
        )
        if trial_id:
            muzero_full_catalog[trial_id] = _build_gold_monday_muzero_overrides(
                GOLD_MONDAY_PROFILE,
                mode="full",
                trial=trial,
            )

    dreamer_smoke_trials = []
    for trial in _get_gold_monday_dreamer_smoke_trials():
        dreamer_smoke_trials.append(
            {
                **dict(trial),
                "runtime_overrides": _build_gold_monday_dreamer_overrides(
                    GOLD_MONDAY_PROFILE,
                    mode="smoke",
                    trial=trial,
                ),
            }
        )

    dreamer_proxy_trials = []
    dreamer_full_catalog: dict[str, dict[str, str]] = {}
    for trial in _get_gold_monday_dreamer_proxy_trials():
        trial_id = str(trial.get("trial_id") or "").strip()
        dreamer_proxy_trials.append(
            {
                **dict(trial),
                "runtime_overrides": _build_gold_monday_dreamer_overrides(
                    GOLD_MONDAY_PROFILE,
                    mode="proxy_ga",
                    trial=trial,
                ),
            }
        )
        if trial_id:
            dreamer_full_catalog[trial_id] = _build_gold_monday_dreamer_overrides(
                GOLD_MONDAY_PROFILE,
                mode="full",
                trial=trial,
            )

    return {
        "sequence_id": sequence_id,
        "sequence_name": GOLD_MONDAY_SEQUENCE_NAME,
        "profiles": [GOLD_MONDAY_PROFILE],
        "focus_symbol": GOLD_MONDAY_FOCUS_SYMBOL,
        "gate_profile": "gold_demo",
        "retry_limit": 1,
        "stdout_log_path": stdout_log_path,
        "stderr_log_path": stderr_log_path,
        "catalogs": {
            "muzero": {GOLD_MONDAY_PROFILE: muzero_proxy_trials},
            "dreamer": {GOLD_MONDAY_PROFILE: dreamer_proxy_trials},
        },
        "full_catalogs": {
            "muzero": {GOLD_MONDAY_PROFILE: muzero_full_catalog},
            "dreamer": {GOLD_MONDAY_PROFILE: dreamer_full_catalog},
        },
        "smoke_catalogs": {
            "dreamer": {GOLD_MONDAY_PROFILE: dreamer_smoke_trials},
        },
        "steps": [
            {
                "kind": "window",
                "profile": GOLD_MONDAY_PROFILE,
                "engine": "muzero",
                "mode": "proxy_ga",
            },
            {
                "kind": "window",
                "profile": GOLD_MONDAY_PROFILE,
                "engine": "muzero",
                "mode": "full",
            },
            {
                "kind": "window",
                "profile": GOLD_MONDAY_PROFILE,
                "engine": "dreamer",
                "mode": "smoke",
            },
            {
                "kind": "window",
                "profile": GOLD_MONDAY_PROFILE,
                "engine": "dreamer",
                "mode": "proxy_ga",
            },
            {
                "kind": "window",
                "profile": GOLD_MONDAY_PROFILE,
                "engine": "dreamer",
                "mode": "full",
            },
            {
                "kind": "gnn_refresh",
                "profile": GOLD_MONDAY_PROFILE,
                "focus_symbol": GOLD_MONDAY_FOCUS_SYMBOL,
                "gate_profile": "gold_demo",
                "refresh_payload": {
                    "symbols": [
                        GOLD_MONDAY_FOCUS_SYMBOL,
                        *GOLD_MONDAY_CONTEXT_SYMBOLS,
                    ],
                    "focus_symbol": GOLD_MONDAY_FOCUS_SYMBOL,
                    "context_symbols": GOLD_MONDAY_CONTEXT_SYMBOLS,
                    "deployment_class": "consultative_gold",
                    "epochs": 300,
                    "batch_size": 32,
                    "checkpoint_every": 25,
                    "max_symbols": 5,
                },
            },
            {
                "kind": "service_action",
                "action": "restart_vllm",
                "profile": GOLD_MONDAY_PROFILE,
            },
        ],
    }


def _get_v3_results_dir() -> Path:
    """Retourne le dossier local de resultats V3."""
    target_dir = LOCAL_ROOT / "data" / "checkpoints" / "v3_ga"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _get_v4_results_dir() -> Path:
    """Retourne le dossier local de resultats V4."""

    target_dir = LOCAL_ROOT / "data" / "checkpoints" / "v4_ga"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _archive_local_v4_snapshots(sequence_id: str) -> Path:
    """Archive les instantanes V4 locaux avant une nouvelle sequence.

    Args:
        sequence_id (str): Identifiant de sequence qui servira de dossier
            d'archive.

    Returns:
        Path: Dossier d'archive cree localement.
    """

    results_dir = _get_v4_results_dir()
    archive_dir = results_dir / "archive" / sequence_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in results_dir.iterdir():
        if path.name == "archive" or not path.is_file():
            continue
        shutil.move(str(path), str(archive_dir / path.name))
    return archive_dir


def _write_remote_text_file(
    sftp: paramiko.SFTPClient,
    remote_path: str,
    content: str,
    *,
    executable: bool = False,
) -> None:
    """Ecrit un fichier texte distant via SFTP.

    Args:
        sftp (paramiko.SFTPClient): Canal SFTP actif.
        remote_path (str): Chemin absolu cible sur le serveur.
        content (str): Contenu texte a ecrire.
        executable (bool): Rend le fichier executable si demande.
    """

    ensure_remote_parent(sftp, remote_path)
    with sftp.file(remote_path, "w") as remote_file:
        remote_file.write(content)
    if executable:
        sftp.chmod(
            remote_path,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IROTH,
        )


def _build_v4_sequence_config(
    sequence_name: str,
    *,
    sequence_id: str,
    profiles: list[str],
    stdout_log_path: str,
    stderr_log_path: str,
) -> dict[str, Any]:
    """Construit la configuration JSON du superviseur distant V4.

    Args:
        sequence_name (str): Nom logique de la sequence.
        sequence_id (str): Identifiant unique de sequence.
        profiles (list[str]): Profils V4 a jouer dans l'ordre.
        stdout_log_path (str): Chemin du journal stdout du superviseur.
        stderr_log_path (str): Chemin du journal stderr du superviseur.

    Returns:
        dict[str, Any]: Configuration serialisable pour le superviseur distant.
    """

    catalogs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    full_catalogs: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for engine in V4_ENGINE_ORDER:
        engine_catalog: dict[str, list[dict[str, Any]]] = {}
        engine_full_catalog: dict[str, dict[str, dict[str, str]]] = {}
        for profile in profiles:
            trials: list[dict[str, Any]] = []
            full_trials: dict[str, dict[str, str]] = {}
            for trial in _get_v4_trial_catalog(profile, engine):
                trial_id = str(trial.get("trial_id") or "").strip()
                trials.append(
                    {
                        **dict(trial),
                        "runtime_overrides": _build_v4_profile_overrides(
                            profile,
                            engine=engine,
                            mode="proxy_ga",
                            trial=trial,
                        ),
                    }
                )
                if trial_id:
                    full_trials[trial_id] = _build_v4_profile_overrides(
                        profile,
                        engine=engine,
                        mode="full",
                        trial=trial,
                    )
            engine_catalog[profile] = trials
            engine_full_catalog[profile] = full_trials
        catalogs[engine] = engine_catalog
        full_catalogs[engine] = engine_full_catalog
    return {
        "sequence_id": sequence_id,
        "sequence_name": sequence_name,
        "profiles": profiles,
        "window_order": [{"engine": engine, "mode": mode} for engine, mode in V4_WINDOW_ORDER],
        "retry_limit": 1,
        "stdout_log_path": stdout_log_path,
        "stderr_log_path": stderr_log_path,
        "catalogs": catalogs,
        "full_catalogs": full_catalogs,
    }


def _prepare_remote_v4_sequence_workspace(client: paramiko.SSHClient, sequence_id: str) -> str:
    """Archive les artefacts V4 distants avant une nouvelle sequence.

    Args:
        client (paramiko.SSHClient): Session SSH active.
        sequence_id (str): Identifiant de sequence utilise pour l'archive.

    Returns:
        str: Dossier d'archive cree cote serveur.

    Raises:
        RuntimeError: Si le nettoyage distant echoue.
    """

    remote_archive_dir = f"{REMOTE_V4_SEQUENCE_DIR}/archive/{sequence_id}"
    remote_body = f"""
set -euo pipefail
SEQUENCE_DIR={shlex.quote(REMOTE_V4_SEQUENCE_DIR)}
ARCHIVE_DIR={shlex.quote(remote_archive_dir)}
mkdir -p "$ARCHIVE_DIR"
if [ -f "$SEQUENCE_DIR/sequence_supervisor.pid" ]; then
  SUPERVISOR_PID="$(cat "$SEQUENCE_DIR/sequence_supervisor.pid" 2>/dev/null || true)"
  if [ -n "$SUPERVISOR_PID" ] && kill -0 "$SUPERVISOR_PID" 2>/dev/null; then
    kill "$SUPERVISOR_PID" || true
    sleep 2
  fi
fi
find "$SEQUENCE_DIR" -maxdepth 1 -type f \\( -name '*.json' -o -name '*.log' -o -name '*.pid' \\) -print0 | while IFS= read -r -d '' FILE; do
  mv "$FILE" "$ARCHIVE_DIR"/
done
"""
    output, error, code = run_command(client, f"bash -lc {shlex.quote(remote_body)}", timeout=60)
    if code != 0:
        raise RuntimeError(error or output or "Archivage distant V4 impossible.")
    return remote_archive_dir


def _fetch_remote_champion_status() -> dict:
    """Lit le statut HTTP des champions distants.

    Returns:
        dict: Charge JSON de `GET /champions/status`.

    Raises:
        RuntimeError: Si la lecture echoue.
    """
    url = f"http://{HOST}:8600/champions/status"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Lecture impossible de {url}: {exc}") from exc


def _fetch_remote_sequence_status() -> dict[str, Any]:
    """Lit le statut HTTP du superviseur distant de sequence.

    Returns:
        dict[str, Any]: Charge JSON de `GET /sequence/status`.

    Raises:
        RuntimeError: Si la lecture echoue.
    """

    url = f"http://{HOST}:8600/sequence/status"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Lecture impossible de {url}: {exc}") from exc


def _extract_horizon_metrics(status_payload: dict[str, Any], horizon: str) -> dict[str, Any]:
    """Extrait les metriques du challenger depuis le statut champion.

    Args:
        status_payload (dict[str, Any]): Reponse `/champions/status`.
        horizon (str): Horizon a lire.

    Returns:
        dict[str, Any]: Metriques de promotion consolidees.
    """
    horizon_status = dict((status_payload.get("horizons") or {}).get(horizon) or {})
    promotion_gate = dict(horizon_status.get("promotion_gate") or {})
    metrics = dict(promotion_gate.get("metrics") or {})
    if not metrics:
        metrics = dict(((horizon_status.get("arena_report") or {}).get("challenger") or {}).get("metrics") or {})
    if not metrics:
        metrics = dict(((horizon_status.get("arena_report") or {}).get("battle_report") or {}).get("challenger", {}).get("metrics") or {})
    return {
        "status": horizon_status,
        "metrics": metrics,
        "mechanics": dict(
            horizon_status.get("metrics_by_position_mechanics")
            or metrics.get("metrics_by_position_mechanics")
            or {}
        ),
    }


def _score_v3_trial_result(profile: str, champion_status: dict[str, Any]) -> dict[str, Any]:
    """Calcule le fitness d'un essai V3 a partir du statut champion.

    Args:
        profile (str): Profil V3 cible.
        champion_status (dict[str, Any]): Reponse complete `/champions/status`.

    Returns:
        dict[str, Any]: Score, mode d'echec et metriques clefs.
    """
    horizon, family = _split_v3_profile(profile)
    extracted = _extract_horizon_metrics(champion_status, horizon)
    horizon_status = dict(extracted.get("status") or {})
    metrics = dict(extracted.get("metrics") or {})
    mechanics = dict(extracted.get("mechanics") or {})

    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    return_pct = float(metrics.get("return_pct", 0.0) or 0.0)
    net_realized_pct = float(metrics.get("net_realized_pct", 0.0) or 0.0)
    positive_episode_rate = float(metrics.get("positive_episode_rate", 0.0) or 0.0)
    long_entry_share = float(metrics.get("long_entry_share", 0.0) or 0.0)
    short_entry_share = float(metrics.get("short_entry_share", 0.0) or 0.0)
    directional_imbalance = float(metrics.get("directional_imbalance", 1.0) or 1.0)
    close_quality_score = float(mechanics.get("close_quality_score", 0.0) or 0.0)
    split_efficiency = float(mechanics.get("split_efficiency", 0.0) or 0.0)
    pyramid_efficiency = float(mechanics.get("pyramid_efficiency", 0.0) or 0.0)
    slbe_capture_rate = float(mechanics.get("slbe_capture_rate", 0.0) or 0.0)
    hold_drag_score = float(mechanics.get("hold_drag_score", 0.0) or 0.0)
    total_trades = int(metrics.get("total_trades", 0) or 0)
    failure_mode = str(horizon_status.get("failure_mode") or "unknown")
    score = (
        return_pct * 180.0
        + net_realized_pct * 140.0
        + max(0.0, profit_factor - 1.0) * 55.0
        + positive_episode_rate * 0.35
        + min(long_entry_share, short_entry_share) * 45.0
        + close_quality_score * 30.0
        + split_efficiency * 18.0
        + pyramid_efficiency * 16.0
        + slbe_capture_rate * 14.0
        - hold_drag_score * 12.0
        - directional_imbalance * 24.0
    )
    if total_trades <= 0:
        score -= 80.0
    if failure_mode in {"inactive", "sell_heavy", "buy_heavy", "bad_exit"}:
        score -= 25.0

    return {
        "profile": profile,
        "family": family,
        "score": round(score, 4),
        "failure_mode": failure_mode,
        "metrics": metrics,
        "mechanics": mechanics,
    }


def _extract_engine_horizon_metrics(
    status_payload: dict[str, Any],
    *,
    engine: str,
    horizon: str,
) -> dict[str, Any]:
    """Extrait les metriques d'un moteur et horizon depuis `/champions/status`."""

    normalized_engine = _normalize_v4_engine(engine)
    engine_matrix = dict(status_payload.get("engines") or {})
    horizon_status = dict(((engine_matrix.get(normalized_engine) or {}).get(horizon)) or {})
    if not horizon_status and normalized_engine == "muzero":
        horizon_status = dict((status_payload.get("horizons") or {}).get(horizon) or {})
    promotion_gate = dict(horizon_status.get("promotion_gate") or {})
    metrics = dict(horizon_status.get("candidate_metrics") or {})
    if not metrics:
        metrics = dict(((horizon_status.get("arena_report") or {}).get("challenger") or {}).get("metrics") or {})
    if not metrics:
        metrics = dict(
            ((horizon_status.get("arena_report") or {}).get("battle_report") or {})
            .get("challenger", {})
            .get("metrics")
            or {}
        )
    if not metrics:
        metrics = dict(promotion_gate.get("metrics") or {})
    return {
        "status": horizon_status,
        "metrics": metrics,
        "mechanics": dict(
            horizon_status.get("candidate_metrics", {}).get("metrics_by_position_mechanics")
            or horizon_status.get("metrics_by_position_mechanics")
            or metrics.get("metrics_by_position_mechanics")
            or {}
        ),
    }


def _score_v4_trial_result(profile: str, champion_status: dict[str, Any], *, engine: str) -> dict[str, Any]:
    """Calcule le fitness V4 d'un essai moteur."""

    normalized_profile = _normalize_v4_profile(profile)
    normalized_engine = _normalize_v4_engine(engine)
    horizon, family = _split_v3_profile(normalized_profile)
    extracted = _extract_engine_horizon_metrics(champion_status, engine=normalized_engine, horizon=horizon)
    horizon_status = dict(extracted.get("status") or {})
    metrics = dict(extracted.get("metrics") or {})
    mechanics = dict(extracted.get("mechanics") or {})

    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    return_pct = float(metrics.get("return_pct", 0.0) or 0.0)
    net_realized_pct = float(metrics.get("net_realized_pct", 0.0) or 0.0)
    positive_episode_rate = float(metrics.get("positive_episode_rate", 0.0) or 0.0)
    long_entry_share = float(metrics.get("long_entry_share", 0.0) or 0.0)
    short_entry_share = float(metrics.get("short_entry_share", 0.0) or 0.0)
    directional_imbalance = float(metrics.get("directional_imbalance", 1.0) or 1.0)
    close_quality_score = float(mechanics.get("close_quality_score", 0.0) or 0.0)
    split_efficiency = float(mechanics.get("split_efficiency", 0.0) or 0.0)
    pyramid_efficiency = float(mechanics.get("pyramid_efficiency", 0.0) or 0.0)
    slbe_capture_rate = float(mechanics.get("slbe_capture_rate", 0.0) or 0.0)
    hold_drag_score = float(mechanics.get("hold_drag_score", 0.0) or 0.0)
    total_trades = int(metrics.get("total_trades", 0) or 0)
    failure_mode = str(horizon_status.get("failure_mode") or "unknown")
    score = (
        return_pct * 180.0
        + net_realized_pct * 140.0
        + max(0.0, profit_factor - 1.0) * 55.0
        + positive_episode_rate * 0.35
        + min(long_entry_share, short_entry_share) * 45.0
        + close_quality_score * 30.0
        + split_efficiency * 18.0
        + pyramid_efficiency * 16.0
        + slbe_capture_rate * 14.0
        - hold_drag_score * 12.0
        - directional_imbalance * 24.0
    )
    if total_trades <= 0:
        score -= 80.0
    if failure_mode in {"inactive", "sell_heavy", "buy_heavy", "bad_exit"}:
        score -= 25.0

    return {
        "engine": normalized_engine,
        "profile": normalized_profile,
        "family": family,
        "score": round(score, 4),
        "failure_mode": failure_mode,
        "metrics": metrics,
        "mechanics": mechanics,
    }


def _load_v3_result_entries(results_path: Path, key: str = "results") -> list[dict[str, Any]]:
    """Charge les resultats V3 deja produits si le fichier existe.

    Args:
        results_path (Path): Fichier JSON local des resultats.
        key (str): Cle contenant la liste des entrees.

    Returns:
        list[dict[str, Any]]: Resultats deja persistés.
    """
    if not results_path.exists():
        return []
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = payload.get(key)
    if not isinstance(entries, list):
        return []
    return [dict(item) for item in entries if isinstance(item, dict)]


def _write_v3_results_snapshot(
    results_path: Path,
    *,
    profile: str,
    mode: str,
    key: str,
    entries: list[dict[str, Any]],
) -> None:
    """Ecrit un instantane local des resultats V3.

    Args:
        results_path (Path): Fichier JSON cible.
        profile (str): Profil V3 evalue.
        mode (str): Mode V3 associe.
        key (str): Cle de la liste d'entrees a ecrire.
        entries (list[dict[str, Any]]): Resultats a persister.
    """
    results_path.write_text(
        json.dumps(
            {
                "profile": profile,
                "mode": mode,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                key: entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _read_active_remote_run() -> dict[str, Any]:
    """Retourne une vue compacte du run distant actuellement visible.

    Returns:
        dict[str, Any]: Resume du run courant expose par `/training/status`.
    """
    status = _fetch_remote_training_status()
    run = dict(status.get("run") or {})
    return {
        "run_id": str(run.get("run_id") or "") or None,
        "active": bool(run.get("active")),
        "status": str(run.get("status") or "") or None,
        "trigger": str(run.get("trigger") or "") or None,
        "engine": str(run.get("engine") or "") or None,
        "family": str(run.get("family") or "") or None,
        "ga_status": str(run.get("ga_status") or "") or None,
        "ga_generation": run.get("ga_generation"),
        "ga_trial": str(run.get("ga_trial") or "") or None,
        "trial_mode": str(run.get("trial_mode") or "") or None,
        "trial_cost_profile": str(run.get("trial_cost_profile") or "") or None,
        "sequence_id": str(run.get("sequence_id") or "") or None,
        "window_id": str(run.get("window_id") or "") or None,
        "trial_id": str(run.get("trial_id") or "") or None,
        "terminal_summary_path": str(run.get("terminal_summary_path") or "") or None,
        "supervisor_state": str(run.get("supervisor_state") or "") or None,
    }


def _get_v3_expected_trigger(profile: str, mode: str, trial: dict[str, Any], finalist_rank: int | None = None) -> str:
    """Calcule le trigger attendu d'un essai V3 a partir des surcharges.

    Args:
        profile (str): Profil V3 cible.
        mode (str): Mode `proxy_ga` ou `full`.
        trial (dict[str, Any]): Definition de l'essai.
        finalist_rank (int | None): Rang du finaliste en mode `full`.

    Returns:
        str: Trigger distant attendu.
    """
    overrides = _build_v3_profile_overrides(
        profile,
        mode=mode,
        trial=trial,
        finalist_rank=finalist_rank,
    )
    return str(overrides.get("TRAINING_RUN_TRIGGER") or "")


def _run_matches_v3_trial(run_snapshot: dict[str, Any], expected_trigger: str) -> bool:
    """Indique si le run courant correspond a un essai V3 attendu.

    Args:
        run_snapshot (dict[str, Any]): Vue compacte du run distant.
        expected_trigger (str): Trigger attendu pour l'essai.

    Returns:
        bool: True si le run correspond a l'essai cible.
    """
    return str(run_snapshot.get("trigger") or "") == expected_trigger and bool(run_snapshot.get("run_id"))


def _get_v4_expected_trigger(
    profile: str,
    *,
    engine: str,
    mode: str,
    trial: dict[str, Any],
    finalist_rank: int | None = None,
) -> str:
    """Calcule le trigger attendu d'un essai V4."""

    overrides = _build_v4_profile_overrides(
        profile,
        engine=engine,
        mode=mode,
        trial=trial,
        finalist_rank=finalist_rank,
    )
    return str(overrides.get("TRAINING_RUN_TRIGGER") or "")


def _run_matches_v4_trial(
    run_snapshot: dict[str, Any],
    *,
    expected_trigger: str,
    engine: str,
) -> bool:
    """Indique si le run courant correspond a un essai V4 moteur."""

    if not bool(run_snapshot.get("run_id")):
        return False
    if str(run_snapshot.get("trigger") or "") != expected_trigger:
        return False
    run_engine = str(run_snapshot.get("engine") or "").strip().lower()
    normalized_engine = _normalize_v4_engine(engine)
    return not run_engine or run_engine == normalized_engine


def _launch_remote_training_process(
    client: paramiko.SSHClient,
    runtime_overrides: dict[str, str],
) -> str:
    """Lance un run distant avec surcharges d'environnement.

    Args:
        client (paramiko.SSHClient): Session SSH distante.
        runtime_overrides (dict[str, str]): Variables a injecter.

    Returns:
        str: PID shell du processus lance.

    Raises:
        RuntimeError: Si le lancement distant echoue.
    """
    _, sudo_password = _require_remote_credentials()
    runtime_exports = build_runtime_exports(runtime_overrides)
    runtime_prefix = f"{runtime_exports}; " if runtime_exports else ""
    launch_cmd = (
        f"echo '{sudo_password}' | sudo -S bash -lc 'cd {REMOTE_DIR} && "
        f"{runtime_prefix}nohup {REMOTE_SCRIPT} > {REMOTE_LOG} 2>&1 < /dev/null & echo $!'"
    )
    output, error, code = run_command(client, launch_cmd, timeout=30)
    if code != 0:
        raise RuntimeError(error or output or f"Code {code}")
    pid_lines = [line.strip() for line in output.splitlines() if line.strip()]
    return pid_lines[-1] if pid_lines else "inconnu"


def launch_v3_profile_remote(
    profile: str,
    *,
    mode: str,
    stop_existing: bool = False,
    stop_reason: str = "manual_v3_champion_rework",
) -> None:
    """Lance un profil V3 en mode `proxy_ga` ou `full`.

    Args:
        profile (str): Profil V3 cible.
        mode (str): Mode `proxy_ga` ou `full`.
        stop_existing (bool): Coupe le run actif avant le premier essai.
        stop_reason (str): Motif explicite de coupure.
    """
    normalized_profile = _normalize_v3_profile(profile)
    normalized_mode = _normalize_v3_mode(mode)
    horizon, _family = _split_v3_profile(normalized_profile)
    results_dir = _get_v3_results_dir()
    proxy_results_path = results_dir / f"{normalized_profile}_proxy_results.json"
    finalists_path = results_dir / f"{normalized_profile}_finalists.json"

    print(f"Connexion a Proxmox {HOST}...")
    ssh_password, _sudo_password = _require_remote_credentials()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=ssh_password, timeout=15)
        print("Connexion SSH etablie.")

        active_run = _read_active_remote_run()
        same_v3_family_active = bool(active_run.get("active")) and str(active_run.get("trigger") or "").startswith(
            f"manual_{normalized_profile}_"
        )
        if stop_existing and not same_v3_family_active:
            stop_remote_training(client, reason=stop_reason)
            active_run = _read_active_remote_run()
        elif stop_existing and same_v3_family_active:
            print("Un run V3 de ce profil est deja actif. Reprise sans coupure.")

        _sync_remote_training_payload(client, profile_hint=normalized_profile)
        print("Payload V3 synchronise.")

        if normalized_mode == "proxy_ga":
            previous_run_id = str((active_run.get("run_id") or "")) or None
            proxy_results = _load_v3_result_entries(proxy_results_path, key="results")
            completed_trials = {str(item.get("trial_id") or "").strip() for item in proxy_results if item.get("trial_id")}
            for generation, trial in enumerate(_get_v3_trial_catalog(normalized_profile), start=1):
                trial_id = str(trial.get("trial_id") or "").strip()
                if trial_id in completed_trials:
                    print(f"Essai {trial_id} deja score, saut.")
                    continue
                runtime_overrides = _build_v3_profile_overrides(
                    normalized_profile,
                    mode=normalized_mode,
                    trial=trial,
                )
                runtime_overrides["TRAINING_GA_GENERATION"] = str(generation)
                expected_trigger = str(runtime_overrides.get("TRAINING_RUN_TRIGGER") or "")
                active_run = _read_active_remote_run()
                run_id = None
                if _run_matches_v3_trial(active_run, expected_trigger):
                    run_id = str(active_run.get("run_id") or "")
                    if active_run.get("active"):
                        print(f"Essai {trial_id} deja actif. Reprise sur le run {run_id}.")
                        _wait_for_remote_run_completion(run_id)
                    else:
                        print(f"Essai {trial_id} deja termine. Scoring du run {run_id}.")
                else:
                    pid = _launch_remote_training_process(client, runtime_overrides)
                    print(f"Essai {trial_id} lance. PID={pid}")
                    run_id = _wait_for_remote_run_start(previous_run_id, expected_trigger)
                    _wait_for_remote_run_completion(run_id)
                champion_status = _fetch_remote_champion_status()
                scored = _score_v3_trial_result(normalized_profile, champion_status)
                scored["trial_id"] = trial_id
                scored["generation"] = generation
                scored["run_id"] = run_id
                proxy_results.append(scored)
                _write_v3_results_snapshot(
                    proxy_results_path,
                    profile=normalized_profile,
                    mode=normalized_mode,
                    key="results",
                    entries=proxy_results,
                )
                completed_trials.add(trial_id)
                previous_run_id = run_id
                print(
                    f"[proxy_ga] {trial_id} | score={scored['score']} | "
                    f"failure_mode={scored['failure_mode']}"
                )

            proxy_results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            finalists = proxy_results[:2]
            _write_v3_results_snapshot(
                proxy_results_path,
                profile=normalized_profile,
                mode=normalized_mode,
                key="results",
                entries=proxy_results,
            )
            finalists_path.write_text(
                json.dumps(
                    {
                        "profile": normalized_profile,
                        "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "finalists": finalists,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"Finalistes V3 retenus pour {normalized_profile}:")
            for finalist in finalists:
                print(f" - {finalist['trial_id']} | score={finalist['score']} | mode={finalist['failure_mode']}")
            return

        if not finalists_path.exists():
            raise RuntimeError(
                f"Aucun finaliste proxy GA pour {normalized_profile}. Lancez d'abord --v3-mode proxy_ga."
            )

        finalists_payload = json.loads(finalists_path.read_text(encoding="utf-8"))
        finalists = list(finalists_payload.get("finalists") or [])
        if not finalists:
            raise RuntimeError(f"Aucun finaliste disponible pour {normalized_profile}.")

        previous_run_id = str((_read_active_remote_run().get("run_id") or "")) or None
        full_results = _load_v3_result_entries(results_dir / f"{normalized_profile}_full_results.json", key="results")
        completed_trials = {str(item.get("trial_id") or "").strip() for item in full_results if item.get("trial_id")}
        for finalist_rank, finalist in enumerate(finalists, start=1):
            trial = {
                "trial_id": str(finalist.get("trial_id") or f"finalist_{finalist_rank}"),
                "overrides": {},
            }
            for catalog_trial in _get_v3_trial_catalog(normalized_profile):
                if str(catalog_trial.get("trial_id")) == trial["trial_id"]:
                    trial = dict(catalog_trial)
                    break
            trial_id = str(trial.get("trial_id") or "").strip()
            if trial_id in completed_trials:
                print(f"Finaliste {trial_id} deja score, saut.")
                continue
            runtime_overrides = _build_v3_profile_overrides(
                normalized_profile,
                mode=normalized_mode,
                trial=trial,
                finalist_rank=finalist_rank,
            )
            runtime_overrides["TRAINING_GA_GENERATION"] = str(finalist_rank)
            expected_trigger = str(runtime_overrides.get("TRAINING_RUN_TRIGGER") or "")
            active_run = _read_active_remote_run()
            run_id = None
            if _run_matches_v3_trial(active_run, expected_trigger):
                run_id = str(active_run.get("run_id") or "")
                if active_run.get("active"):
                    print(f"Finaliste {trial_id} deja actif. Reprise sur le run {run_id}.")
                    _wait_for_remote_run_completion(run_id)
                else:
                    print(f"Finaliste {trial_id} deja termine. Scoring du run {run_id}.")
            else:
                pid = _launch_remote_training_process(client, runtime_overrides)
                print(f"Finaliste {trial_id} lance. PID={pid}")
                run_id = _wait_for_remote_run_start(previous_run_id, expected_trigger)
                _wait_for_remote_run_completion(run_id)
            champion_status = _fetch_remote_champion_status()
            scored = _score_v3_trial_result(normalized_profile, champion_status)
            scored["trial_id"] = trial_id
            scored["generation"] = finalist_rank
            scored["run_id"] = run_id
            full_results.append(scored)
            _write_v3_results_snapshot(
                results_dir / f"{normalized_profile}_full_results.json",
                profile=normalized_profile,
                mode=normalized_mode,
                key="results",
                entries=full_results,
            )
            completed_trials.add(trial_id)
            previous_run_id = run_id
            print(
                f"[full] {trial_id} | score={scored['score']} | failure_mode={scored['failure_mode']}"
            )

        _write_v3_results_snapshot(
            results_dir / f"{normalized_profile}_full_results.json",
            profile=normalized_profile,
            mode=normalized_mode,
            key="results",
            entries=full_results,
        )
        best = sorted(full_results, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if best:
            print(f"Meilleur finaliste {normalized_profile}: {best[0]['trial_id']} | score={best[0]['score']}")
    finally:
        client.close()


def launch_v3_sequence_remote(
    sequence_name: str,
    *,
    stop_existing: bool = False,
    stop_reason: str = "manual_v3_champion_rework",
) -> None:
    """Execute une sequence V3 stricte profil par profil et mode par mode.

    Args:
        sequence_name (str): Nom logique de la sequence V3.
        stop_existing (bool): Coupe le run actif avant le premier profil.
        stop_reason (str): Motif explicite d'arret initial.

    Raises:
        ValueError: Si la sequence demandee est inconnue.
    """
    normalized = str(sequence_name or "").strip().lower()
    profiles = V3_SEQUENCE_ORDER.get(normalized)
    if not profiles:
        raise ValueError(f"Sequence V3 non supportee: {sequence_name}")

    first_step = True
    print(f"Sequence V3 '{normalized}' preparee: {', '.join(profiles)}")
    for profile in profiles:
        for mode in V3_MODE_ORDER:
            print(f"[v3-sequence] Lancement de {profile} en mode {mode}.")
            launch_v3_profile_remote(
                profile,
                mode=mode,
                stop_existing=stop_existing if first_step else False,
                stop_reason=stop_reason,
            )
            first_step = False
            print(f"[v3-sequence] {profile} / {mode} termine.")


def launch_v4_profile_remote(
    profile: str,
    *,
    engine: str,
    mode: str,
    stop_existing: bool = False,
    stop_reason: str = "manual_v4_unified_factory",
) -> None:
    """Lance un profil V4 pour un moteur et un mode donnes."""

    normalized_profile = _normalize_v4_profile(profile)
    normalized_engine = _normalize_v4_engine(engine)
    normalized_mode = _normalize_v4_mode(mode)
    results_dir = _get_v4_results_dir()
    proxy_results_path = results_dir / f"v4_{normalized_engine}_{normalized_profile}_proxy_results.json"
    finalists_path = results_dir / f"v4_{normalized_engine}_{normalized_profile}_finalists.json"
    full_results_path = results_dir / f"v4_{normalized_engine}_{normalized_profile}_full_results.json"

    print(f"Connexion a Proxmox {HOST}...")
    ssh_password, _sudo_password = _require_remote_credentials()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=ssh_password, timeout=15)
        print("Connexion SSH etablie.")

        active_run = _read_active_remote_run()
        same_v4_profile_active = bool(active_run.get("active")) and str(active_run.get("trigger") or "").startswith(
            f"manual_{normalized_profile}_{normalized_engine}_"
        )
        if stop_existing and not same_v4_profile_active:
            stop_remote_training(client, reason=stop_reason)
            active_run = _read_active_remote_run()
        elif stop_existing and same_v4_profile_active:
            print("Un run V4 de ce profil est deja actif. Reprise sans coupure.")

        _sync_remote_training_payload(client, profile_hint=normalized_profile)
        print(f"Payload V4 synchronise pour {normalized_profile} [{normalized_engine}].")

        if normalized_mode == "proxy_ga":
            previous_run_id = str((active_run.get("run_id") or "")) or None
            proxy_results = _load_v3_result_entries(proxy_results_path, key="results")
            completed_trials = {str(item.get("trial_id") or "").strip() for item in proxy_results if item.get("trial_id")}
            for generation, trial in enumerate(_get_v4_trial_catalog(normalized_profile, normalized_engine), start=1):
                trial_id = str(trial.get("trial_id") or "").strip()
                if trial_id in completed_trials:
                    print(f"[v4-{normalized_engine}] Essai {trial_id} deja score, saut.")
                    continue
                runtime_overrides = _build_v4_profile_overrides(
                    normalized_profile,
                    engine=normalized_engine,
                    mode=normalized_mode,
                    trial=trial,
                )
                runtime_overrides["TRAINING_GA_GENERATION"] = str(generation)
                expected_trigger = str(runtime_overrides.get("TRAINING_RUN_TRIGGER") or "")
                active_run = _read_active_remote_run()
                run_id = None
                if _run_matches_v4_trial(
                    active_run,
                    expected_trigger=expected_trigger,
                    engine=normalized_engine,
                ):
                    run_id = str(active_run.get("run_id") or "")
                    if active_run.get("active"):
                        print(
                            f"[v4-{normalized_engine}] Essai {trial_id} deja actif. "
                            f"Reprise sur le run {run_id}."
                        )
                        _wait_for_remote_run_completion(run_id)
                    else:
                        print(f"[v4-{normalized_engine}] Essai {trial_id} deja termine. Scoring du run {run_id}.")
                else:
                    pid = _launch_remote_training_process(client, runtime_overrides)
                    print(f"[v4-{normalized_engine}] Essai {trial_id} lance. PID={pid}")
                    run_id = _wait_for_remote_run_start(previous_run_id, expected_trigger)
                    _wait_for_remote_run_completion(run_id)

                champion_status = _fetch_remote_champion_status()
                scored = _score_v4_trial_result(normalized_profile, champion_status, engine=normalized_engine)
                scored["trial_id"] = trial_id
                scored["generation"] = generation
                scored["run_id"] = run_id
                proxy_results.append(scored)
                _write_v3_results_snapshot(
                    proxy_results_path,
                    profile=f"{normalized_engine}:{normalized_profile}",
                    mode=normalized_mode,
                    key="results",
                    entries=proxy_results,
                )
                completed_trials.add(trial_id)
                previous_run_id = run_id
                print(
                    f"[v4-{normalized_engine}][proxy_ga] {trial_id} | score={scored['score']} | "
                    f"failure_mode={scored['failure_mode']}"
                )

            proxy_results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            finalists = proxy_results[:2]
            _write_v3_results_snapshot(
                proxy_results_path,
                profile=f"{normalized_engine}:{normalized_profile}",
                mode=normalized_mode,
                key="results",
                entries=proxy_results,
            )
            finalists_path.write_text(
                json.dumps(
                    {
                        "profile": normalized_profile,
                        "engine": normalized_engine,
                        "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "finalists": finalists,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"Finalistes V4 retenus pour {normalized_profile} [{normalized_engine}]:")
            for finalist in finalists:
                print(
                    f" - {finalist['trial_id']} | score={finalist['score']} | "
                    f"mode={finalist['failure_mode']}"
                )
            return

        if not finalists_path.exists():
            raise RuntimeError(
                f"Aucun finaliste proxy GA pour {normalized_profile} [{normalized_engine}]. "
                "Lancez d'abord le mode proxy_ga."
            )

        finalists_payload = json.loads(finalists_path.read_text(encoding="utf-8"))
        finalists = list(finalists_payload.get("finalists") or [])
        if not finalists:
            raise RuntimeError(f"Aucun finaliste disponible pour {normalized_profile} [{normalized_engine}].")

        previous_run_id = str((_read_active_remote_run().get("run_id") or "")) or None
        full_results = _load_v3_result_entries(full_results_path, key="results")
        completed_trials = {str(item.get("trial_id") or "").strip() for item in full_results if item.get("trial_id")}
        for finalist_rank, finalist in enumerate(finalists, start=1):
            trial = {
                "trial_id": str(finalist.get("trial_id") or f"finalist_{finalist_rank}"),
                "overrides": {},
            }
            for catalog_trial in _get_v4_trial_catalog(normalized_profile, normalized_engine):
                if str(catalog_trial.get("trial_id")) == trial["trial_id"]:
                    trial = dict(catalog_trial)
                    break
            trial_id = str(trial.get("trial_id") or "").strip()
            if trial_id in completed_trials:
                print(f"[v4-{normalized_engine}] Finaliste {trial_id} deja score, saut.")
                continue
            runtime_overrides = _build_v4_profile_overrides(
                normalized_profile,
                engine=normalized_engine,
                mode=normalized_mode,
                trial=trial,
                finalist_rank=finalist_rank,
            )
            runtime_overrides["TRAINING_GA_GENERATION"] = str(finalist_rank)
            expected_trigger = str(runtime_overrides.get("TRAINING_RUN_TRIGGER") or "")
            active_run = _read_active_remote_run()
            run_id = None
            if _run_matches_v4_trial(
                active_run,
                expected_trigger=expected_trigger,
                engine=normalized_engine,
            ):
                run_id = str(active_run.get("run_id") or "")
                if active_run.get("active"):
                    print(
                        f"[v4-{normalized_engine}] Finaliste {trial_id} deja actif. "
                        f"Reprise sur le run {run_id}."
                    )
                    _wait_for_remote_run_completion(run_id)
                else:
                    print(
                        f"[v4-{normalized_engine}] Finaliste {trial_id} deja termine. "
                        f"Scoring du run {run_id}."
                    )
            else:
                pid = _launch_remote_training_process(client, runtime_overrides)
                print(f"[v4-{normalized_engine}] Finaliste {trial_id} lance. PID={pid}")
                run_id = _wait_for_remote_run_start(previous_run_id, expected_trigger)
                _wait_for_remote_run_completion(run_id)

            champion_status = _fetch_remote_champion_status()
            scored = _score_v4_trial_result(normalized_profile, champion_status, engine=normalized_engine)
            scored["trial_id"] = trial_id
            scored["generation"] = finalist_rank
            scored["run_id"] = run_id
            full_results.append(scored)
            _write_v3_results_snapshot(
                full_results_path,
                profile=f"{normalized_engine}:{normalized_profile}",
                mode=normalized_mode,
                key="results",
                entries=full_results,
            )
            completed_trials.add(trial_id)
            previous_run_id = run_id
            print(
                f"[v4-{normalized_engine}][full] {trial_id} | score={scored['score']} | "
                f"failure_mode={scored['failure_mode']}"
            )

        _write_v3_results_snapshot(
            full_results_path,
            profile=f"{normalized_engine}:{normalized_profile}",
            mode=normalized_mode,
            key="results",
            entries=full_results,
        )
        best = sorted(full_results, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if best:
            print(
                f"Meilleur finaliste V4 {normalized_profile} [{normalized_engine}]: "
                f"{best[0]['trial_id']} | score={best[0]['score']}"
            )
    finally:
        client.close()


def launch_v4_sequence_remote(
    sequence_name: str,
    *,
    stop_existing: bool = False,
    stop_reason: str = "manual_v4_unified_factory",
) -> None:
    """Lance une sequence V4 durable via un superviseur distant.

    Args:
        sequence_name (str): Nom logique de sequence, par exemple `scalp`.
        stop_existing (bool): Coupe le run distant actif avant relance.
        stop_reason (str): Motif explicite de coupure initiale.

    Raises:
        ValueError: Si la sequence demandee est inconnue.
        RuntimeError: Si le superviseur distant ne peut pas etre lance.
    """

    normalized = str(sequence_name or "").strip().lower()
    profiles = list(V4_SEQUENCE_ORDER.get(normalized) or [])
    if not profiles:
        raise ValueError(f"Sequence V4 non supportee: {sequence_name}")

    sequence_id = f"v4_{normalized}_{time.strftime('%Y%m%d_%H%M%S')}"
    stdout_log_path = f"{REMOTE_V4_SEQUENCE_DIR}/sequence_{sequence_id}.out.log"
    stderr_log_path = f"{REMOTE_V4_SEQUENCE_DIR}/sequence_{sequence_id}.err.log"
    if normalized == GOLD_MONDAY_SEQUENCE_NAME:
        config_payload = _build_gold_monday_sequence_config(
            sequence_id=sequence_id,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
        )
    else:
        config_payload = _build_v4_sequence_config(
            normalized,
            sequence_id=sequence_id,
            profiles=profiles,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
        )
    local_archive_dir = _archive_local_v4_snapshots(sequence_id)

    print(f"Connexion a Proxmox {HOST}...")
    ssh_password, _sudo_password = _require_remote_credentials()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=ssh_password, timeout=15)
        print("Connexion SSH etablie.")

        active_run = _read_active_remote_run()
        if active_run.get("active") and not stop_existing:
            raise RuntimeError(
                "Un run distant est deja actif. Utilisez --stop-existing pour relancer proprement la sequence V4."
            )
        if active_run.get("active") and stop_existing:
            stop_remote_training(client, reason=stop_reason)

        _sync_remote_training_payload(client, profile_hint=None)
        print("Payload V4 synchronise pour toute la sequence.")

        remote_archive_dir = _prepare_remote_v4_sequence_workspace(client, sequence_id)
        remote_config_path = f"{REMOTE_V4_SEQUENCE_DIR}/sequence_config_{sequence_id}.json"
        sftp = client.open_sftp()
        try:
            _write_remote_text_file(
                sftp,
                remote_config_path,
                json.dumps(config_payload, indent=2, ensure_ascii=False),
            )
        finally:
            sftp.close()

        supervisor_exports = build_runtime_exports(_build_v4_supervisor_overrides())
        supervisor_prefix = f"{supervisor_exports}; " if supervisor_exports else ""
        launch_cmd = (
            "bash -lc "
            + shlex.quote(
                "set -euo pipefail\n"
                f"cd {REMOTE_DIR}\n"
                f"mkdir -p {REMOTE_V4_SEQUENCE_DIR}\n"
                f"{supervisor_prefix}"
                "nohup env "
                f"PYTHONPATH={REMOTE_DIR}/src/eva-lab:{REMOTE_DIR}/src/shared "
                f"python3 {REMOTE_V4_SEQUENCE_RUNNER} --config {remote_config_path} "
                f"> {stdout_log_path} 2> {stderr_log_path} < /dev/null &\n"
                "echo $!"
            )
        )
        output, error, code = run_command(client, launch_cmd, timeout=30)
        if code != 0:
            raise RuntimeError(error or output or "Lancement du superviseur distant impossible.")

        pid_lines = [line.strip() for line in output.splitlines() if line.strip()]
        remote_pid = pid_lines[-1] if pid_lines else "inconnu"
        print(f"Sequence V4 '{normalized}' demarree.")
        print(f" - sequence_id: {sequence_id}")
        print(f" - pid distant: {remote_pid}")
        print(f" - archive locale: {local_archive_dir}")
        print(f" - archive distante: {remote_archive_dir}")
        print(f" - config distante: {remote_config_path}")
        print(f" - stdout: {stdout_log_path}")
        print(f" - stderr: {stderr_log_path}")

        time.sleep(5)
        try:
            sequence_status = _fetch_remote_sequence_status()
            print(
                "[v4-sequence] Etat superviseur: "
                f"{sequence_status.get('state')} | trial={sequence_status.get('current_trial')} "
                f"| next={sequence_status.get('next_step')}"
            )
        except RuntimeError as exc:
            print(f"[v4-sequence] Lecture initiale du superviseur indisponible: {exc}")
    finally:
        client.close()


def _fetch_remote_training_status() -> dict:
    """Lit le statut HTTP de l'entrainement distant.

    Returns:
        dict: Charge JSON de `GET /training/status`.

    Raises:
        RuntimeError: Si le statut distant est illisible.
    """
    url = f"http://{HOST}:8600/training/status"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Lecture impossible de {url}: {exc}") from exc


def _wait_for_remote_run_start(
    previous_run_id: str | None,
    expected_trigger: str,
    timeout_seconds: int = 300,
) -> str:
    """Attend le demarrage visible d'un nouveau run distant.

    Args:
        previous_run_id (str | None): Dernier run observe avant lancement.
        expected_trigger (str): Trigger attendu pour ce run.
        timeout_seconds (int): Delai maximal d'attente.

    Returns:
        str: Identifiant du run demarre.

    Raises:
        RuntimeError: Si aucun nouveau run n'apparait dans le delai.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status = _fetch_remote_training_status()
        run = dict(status.get("run") or {})
        run_id = run.get("run_id")
        trigger = str(run.get("trigger") or "")
        if run_id and run_id != previous_run_id and trigger == expected_trigger:
            print(f"Run {run_id} demarre pour {expected_trigger}.")
            return str(run_id)
        time.sleep(5)
    raise RuntimeError(f"Le run {expected_trigger} n'a pas demarre dans le delai imparti.")


def _wait_for_remote_run_completion(run_id: str, poll_interval_seconds: int = 60) -> dict:
    """Attend la fin d'un run distant deja demarre.

    Args:
        run_id (str): Identifiant du run a surveiller.
        poll_interval_seconds (int): Frequence de poll.

    Returns:
        dict: Etat terminal structure du run, y compris le statut, la
            derniere etape connue et les metadonnees de sequence visibles.
    """
    last_step_label = ""
    while True:
        status = _fetch_remote_training_status()
        run = dict(status.get("run") or {})
        current_run_id = str(run.get("run_id") or "")
        current_step = str(run.get("step_label") or run.get("effective_step_label") or "")
        if current_step and current_step != last_step_label:
            print(f"[{run_id}] Etape: {current_step}")
            last_step_label = current_step
        if current_run_id == run_id and not bool(run.get("active")):
            final_status = str(run.get("status") or "unknown")
            print(f"Run {run_id} termine avec statut {final_status}.")
            return {
                "status": final_status,
                "run_id": run_id,
                "failed_step": dict(run.get("failed_step") or {}),
                "reason": str(run.get("reason") or ""),
                "step_label": current_step or last_step_label,
                "terminal_summary_path": str(run.get("terminal_summary_path") or ""),
                "sequence_id": str(run.get("sequence_id") or ""),
                "window_id": str(run.get("window_id") or ""),
                "trial_id": str(run.get("trial_id") or ""),
                "payload": status,
            }
        time.sleep(max(10, poll_interval_seconds))


def _build_remote_sequence_script(sequence_name: str) -> str:
    """Construit un script shell distant pour une sequence ordonnee.

    Args:
        sequence_name (str): Nom logique de la sequence.

    Returns:
        str: Contenu du script bash distant.

    Raises:
        ValueError: Si la sequence demandee est inconnue.
    """
    normalized = str(sequence_name or "").strip().lower()
    profiles = WAVE1_SEQUENCE_ORDER.get(normalized)
    if not profiles:
        raise ValueError(f"Sequence de vague 1 non supportee: {sequence_name}")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'PROJECT_DIR="{REMOTE_DIR}"',
        'cd "$PROJECT_DIR"',
        "",
    ]
    for profile in profiles:
        export_block = build_runtime_exports(_build_wave1_profile_overrides(profile))
        lines.append(f'echo "[sequence] Lancement de {profile}"')
        if export_block:
            lines.append(export_block)
        lines.append(f"bash {shlex.quote(REMOTE_SCRIPT)}")
        lines.append(f'echo "[sequence] Profil {profile} termine."')
        lines.append("")
    return "\n".join(lines) + "\n"


def launch_training_sequence_remote(
    sequence_name: str,
    *,
    stop_existing: bool = False,
    stop_reason: str = "manual_factory_cutover",
) -> None:
    """Lance une sequence ordonnee de profils V2 en arriere-plan sur le serveur.

    Args:
        sequence_name (str): Nom logique de sequence, par exemple `scalp`.
        stop_existing (bool): Coupe le run courant avant le premier lancement.
        stop_reason (str): Motif explicite si un run actif est coupe.

    Raises:
        ValueError: Si la sequence demandee est inconnue.
    """
    normalized = str(sequence_name or "").strip().lower()
    if normalized not in WAVE1_SEQUENCE_ORDER:
        raise ValueError(f"Sequence de vague 1 non supportee: {sequence_name}")

    print(f"Connexion a Proxmox {HOST}...")
    ssh_password, sudo_password = _require_remote_credentials()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=ssh_password, timeout=15)
        print("Connexion SSH etablie.")

        if stop_existing:
            stop_remote_training(client, reason=stop_reason)

        _sync_remote_training_payload(client)

        sequence_script = _build_remote_sequence_script(normalized)
        sftp = client.open_sftp()
        try:
            ensure_remote_parent(sftp, REMOTE_SEQUENCE_SCRIPT)
            with sftp.file(REMOTE_SEQUENCE_SCRIPT, "w") as remote_file:
                remote_file.write(sequence_script)
            sftp.chmod(
                REMOTE_SEQUENCE_SCRIPT,
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IROTH,
            )
        finally:
            sftp.close()

        launch_cmd = (
            f"echo '{sudo_password}' | sudo -S bash -lc 'cd {REMOTE_DIR} && "
            f"nohup {REMOTE_SEQUENCE_SCRIPT} > {REMOTE_SEQUENCE_LOG} 2>&1 < /dev/null & echo $!'"
        )
        output, error, code = run_command(client, launch_cmd, timeout=30)
        if code != 0:
            raise RuntimeError(error or output or f"Code {code}")

        pid_lines = [line.strip() for line in output.splitlines() if line.strip()]
        pid = pid_lines[-1] if pid_lines else "inconnu"
        print(f"Sequence {normalized} lancee sur le serveur. PID={pid}")
        print(f"Log sequence: {REMOTE_SEQUENCE_LOG}")
    finally:
        client.close()


def stop_remote_training(client: paramiko.SSHClient, reason: str = "manual_factory_cutover") -> None:
    """Stoppe proprement le run distant actif si present.

    Args:
        client (paramiko.SSHClient): Session SSH distante deja ouverte.
        reason (str): Motif de l'arret a enregistrer.

    Raises:
        RuntimeError: Si l'arret distant echoue.
    """
    _, sudo_password = _require_remote_credentials()
    remote_body = f"""
cd {REMOTE_DIR}
LOCK_FILE="{REMOTE_DIR}/data/checkpoints/nightly_training.lock"
LOCK_PID=""
if [ -f "$LOCK_FILE" ]; then
  LOCK_PID="$(python3 - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

lock_file = Path("{REMOTE_DIR}/data/checkpoints/nightly_training.lock")
try:
    payload = json.loads(lock_file.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)

pid = payload.get("pid")
print(pid if isinstance(pid, int) else "")
PY
)"
fi

if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
  kill -TERM "$LOCK_PID" || true
  sleep 5
fi

TRAINERS="$(docker ps --format '{{{{.Names}}}}' | grep '^the_hive-eva-trainer-run-' || true)"
if [ -n "$TRAINERS" ]; then
  printf '%s\\n' "$TRAINERS" | xargs -r docker stop
fi

rm -rf "{REMOTE_DIR}/data/checkpoints/nightly_training.lock.d"
rm -f "{REMOTE_DIR}/data/checkpoints/nightly_training.lock"

PYTHONPATH="{REMOTE_DIR}/src/eva-lab:{REMOTE_DIR}/src/shared" python3 - <<'PY'
from __future__ import annotations

from eva_lab.training_status import (
    append_training_log,
    finalize_training_status,
    set_training_launcher_state,
)

append_training_log(
    "Run interrompu manuellement pour preparer une relance de l'usine trading.",
    level="WARNING",
    source="launcher",
)
set_training_launcher_state(
    phase="idle",
    last_stop_reason="{reason}",
)
finalize_training_status("aborted", reason="{reason}")
PY
"""
    command = f"echo {shlex.quote(sudo_password)} | sudo -S bash -lc {shlex.quote(remote_body)}"
    output, error, code = run_command(client, command, timeout=90)
    if code != 0:
        raise RuntimeError(error or output or f"Code {code}")
    print("Run distant stoppe proprement.")


def start_training(
    manual_massive: bool = False,
    *,
    scalp_reduced: bool = False,
    muzero_scalp_full_7: bool = False,
    dreamer_scalp_full_7: bool = False,
    intraday_reduced: bool = False,
    swing_reduced: bool = False,
    all_reduced: bool = False,
    wave1_profile: str | None = None,
    v3_sequence: str | None = None,
    v3_profile: str | None = None,
    v3_mode: str | None = None,
    stop_existing: bool = False,
    stop_reason: str = "manual_factory_cutover",
    symbols: list[str] | None = None,
) -> None:
    """Synchronise les scripts EVA Lab et lance l'entrainement distant.

    Args:
        manual_massive (bool): Force un run massif immediat de recherche.
        scalp_reduced (bool): Force une relance `scalp` reduite.
        muzero_scalp_full_7 (bool): Lance un `full` MuZero `scalp` 7-symboles.
        dreamer_scalp_full_7 (bool): Lance un `full` Dreamer `scalp` 7-symboles.
        intraday_reduced (bool): Force une relance `intraday` reduite.
        swing_reduced (bool): Force une relance `swing` reduite.
        all_reduced (bool): Force une relance reduite `scalp+intraday+swing`.
        wave1_profile (str | None): Profil `horizon x famille` explicite.
        v3_sequence (str | None): Sequence V3 explicite a orchestrer localement.
        v3_profile (str | None): Profil V3 explicite `scalp_*_v2`.
        v3_mode (str | None): Mode V3 `proxy_ga` ou `full`.
        stop_existing (bool): Stoppe le run actif avant lancement si necessaire.
        stop_reason (str): Motif explicite a enregistrer si le run courant est coupe.
        symbols (list[str] | None): Univers reduit optionnel pour `scalp`.
    """
    print(f"Connexion a Proxmox {HOST}...")
    ssh_password, sudo_password = _require_remote_credentials()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=ssh_password, timeout=15)
        print("Connexion SSH etablie.")

        if stop_existing:
            stop_remote_training(client, reason=stop_reason)

        sync_profile_hint = (
            str(v3_profile or "").strip()
            or str(wave1_profile or "").strip()
            or None
        )
        _sync_remote_training_payload(client, profile_hint=sync_profile_hint)
        print("Script distant mis a jour.")

        runtime_overrides: dict[str, str] = {}
        if manual_massive:
            runtime_overrides = _build_manual_massive_overrides()
        elif scalp_reduced:
            runtime_overrides = _build_scalp_reduced_overrides(symbols)
        elif muzero_scalp_full_7:
            runtime_overrides = _build_muzero_scalp_multi_universe_full_overrides(symbols)
        elif dreamer_scalp_full_7:
            runtime_overrides = _build_dreamer_scalp_multi_universe_full_overrides(symbols)
        elif intraday_reduced:
            runtime_overrides = _build_intraday_reduced_overrides(symbols)
        elif swing_reduced:
            runtime_overrides = _build_swing_reduced_overrides(symbols)
        elif all_reduced:
            runtime_overrides = _build_all_reduced_overrides(symbols)
        elif wave1_profile:
            runtime_overrides = _build_wave1_profile_overrides(wave1_profile)
        elif v3_profile:
            runtime_overrides = _build_v3_profile_overrides(
                v3_profile,
                mode=(v3_mode or "full"),
                trial={"trial_id": "baseline", "overrides": {}},
                finalist_rank=1 if str(v3_mode or "full").strip().lower() == "full" else None,
            )
        runtime_exports = build_runtime_exports(runtime_overrides)
        runtime_prefix = f"{runtime_exports}; " if runtime_exports else ""
        launch_cmd = (
            f"echo '{sudo_password}' | sudo -S bash -lc 'cd {REMOTE_DIR} && "
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
    parser.add_argument(
        "--scalp-reduced",
        action="store_true",
        help="Relance uniquement `scalp` sur un univers reduit multi-actifs.",
    )
    parser.add_argument(
        "--muzero-scalp-full-7",
        action="store_true",
        help="Lance un `full` MuZero `scalp` force sur l'univers canonique a 7 symboles.",
    )
    parser.add_argument(
        "--dreamer-scalp-full-7",
        action="store_true",
        help="Lance un `full` Dreamer `scalp` force sur l'univers canonique a 7 symboles.",
    )
    parser.add_argument(
        "--intraday-reduced",
        action="store_true",
        help="Relance uniquement `intraday` sur un univers reduit coeur.",
    )
    parser.add_argument(
        "--swing-reduced",
        action="store_true",
        help="Relance uniquement `swing` sur un univers reduit coeur.",
    )
    parser.add_argument(
        "--all-reduced",
        action="store_true",
        help="Relance `scalp`, `intraday` et `swing` sur un univers reduit commun.",
    )
    parser.add_argument(
        "--wave1-profile",
        default="",
        help=(
            "Profil explicite de la vague 1 au format `horizon_famille` "
            f"parmi: {', '.join(WAVE1_PROFILE_ORDER)}."
        ),
    )
    parser.add_argument(
        "--wave1-sequence",
        default="",
        help=(
            "Sequence ordonnee de la vague 1 a lancer en serie. "
            f"Valeurs supportees: {', '.join(WAVE1_SEQUENCE_ORDER)}."
        ),
    )
    parser.add_argument(
        "--v3-profile",
        default="",
        help=(
            "Profil V3 explicite a lancer. "
            f"Valeurs supportees: {', '.join(V3_PROFILE_ORDER)}."
        ),
    )
    parser.add_argument(
        "--v3-sequence",
        default="",
        help=(
            "Sequence V3 stricte a lancer profil par profil. "
            f"Valeurs supportees: {', '.join(V3_SEQUENCE_ORDER)}."
        ),
    )
    parser.add_argument(
        "--v3-mode",
        default="proxy_ga",
        help=(
            "Mode de lancement V3. "
            f"Valeurs supportees: {', '.join(V3_MODE_ORDER)}."
        ),
    )
    parser.add_argument(
        "--v4-profile",
        default="",
        help=(
            "Profil V4 explicite a lancer. "
            f"Valeurs supportees: {', '.join(V4_PROFILE_ORDER)}."
        ),
    )
    parser.add_argument(
        "--v4-engine",
        default="muzero",
        help=(
            "Moteur V4 cible. "
            f"Valeurs supportees: {', '.join(V4_ENGINE_ORDER)}."
        ),
    )
    parser.add_argument(
        "--v4-mode",
        default="proxy_ga",
        help=(
            "Mode de lancement V4. "
            f"Valeurs supportees: {', '.join(V4_MODE_ORDER)}."
        ),
    )
    parser.add_argument(
        "--v4-sequence",
        default="",
        help=(
            "Sequence V4 unifiee a lancer moteur par moteur. "
            f"Valeurs supportees: {', '.join(V4_SEQUENCE_ORDER)}."
        ),
    )
    parser.add_argument(
        "--stop-existing",
        action="store_true",
        help="Stoppe proprement le run distant actif avant de lancer le nouveau.",
    )
    parser.add_argument(
        "--stop-reason",
        default="manual_factory_cutover",
        help="Motif explicite a enregistrer lors de l'arret du run actif.",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Liste CSV de symboles a imposer pour les profils `reduced` uniquement.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_profiles = sum(
        1
        for flag in (
            args.manual_massive,
            args.scalp_reduced,
            args.muzero_scalp_full_7,
            args.dreamer_scalp_full_7,
            args.intraday_reduced,
            args.swing_reduced,
            args.all_reduced,
            bool(str(args.wave1_profile or "").strip()),
            bool(str(args.wave1_sequence or "").strip()),
            bool(str(args.v3_sequence or "").strip()),
            bool(str(args.v3_profile or "").strip()),
            bool(str(args.v4_sequence or "").strip()),
            bool(str(args.v4_profile or "").strip()),
        )
        if flag
    )
    if selected_profiles > 1:
        raise SystemExit(
            "Choisissez un seul profil parmi --manual-massive, --scalp-reduced, --muzero-scalp-full-7, --dreamer-scalp-full-7, --intraday-reduced, --swing-reduced, --all-reduced, --wave1-profile, --wave1-sequence, --v3-sequence, --v3-profile, --v4-sequence, --v4-profile."
        )
    requested_symbols = [
        item.strip()
        for item in str(args.symbols or "").split(",")
        if item.strip()
    ]
    wave1_sequence = str(args.wave1_sequence or "").strip()
    v3_sequence = str(args.v3_sequence or "").strip()
    v3_profile = str(args.v3_profile or "").strip()
    v4_sequence = str(args.v4_sequence or "").strip()
    v4_profile = str(args.v4_profile or "").strip()
    if wave1_sequence:
        launch_training_sequence_remote(
            wave1_sequence,
            stop_existing=args.stop_existing,
            stop_reason=str(args.stop_reason or "manual_factory_cutover"),
        )
    elif v3_sequence:
        launch_v3_sequence_remote(
            v3_sequence,
            stop_existing=args.stop_existing,
            stop_reason=str(args.stop_reason or "manual_v3_champion_rework"),
        )
    elif v3_profile:
        launch_v3_profile_remote(
            v3_profile,
            mode=str(args.v3_mode or "proxy_ga"),
            stop_existing=args.stop_existing,
            stop_reason=str(args.stop_reason or "manual_v3_champion_rework"),
        )
    elif v4_sequence:
        launch_v4_sequence_remote(
            v4_sequence,
            stop_existing=args.stop_existing,
            stop_reason=str(args.stop_reason or "manual_v4_unified_factory"),
        )
    elif v4_profile:
        launch_v4_profile_remote(
            v4_profile,
            engine=str(args.v4_engine or "muzero"),
            mode=str(args.v4_mode or "proxy_ga"),
            stop_existing=args.stop_existing,
            stop_reason=str(args.stop_reason or "manual_v4_unified_factory"),
        )
    else:
        start_training(
            manual_massive=args.manual_massive,
            scalp_reduced=args.scalp_reduced,
            muzero_scalp_full_7=args.muzero_scalp_full_7,
            dreamer_scalp_full_7=args.dreamer_scalp_full_7,
            intraday_reduced=args.intraday_reduced,
            swing_reduced=args.swing_reduced,
            all_reduced=args.all_reduced,
            wave1_profile=(str(args.wave1_profile or "").strip() or None),
            v3_sequence=(v3_sequence or None),
            v3_profile=(v3_profile or None),
            v3_mode=(str(args.v3_mode or "proxy_ga").strip() or None),
            stop_existing=args.stop_existing,
            stop_reason=str(args.stop_reason or "manual_factory_cutover"),
            symbols=requested_symbols or None,
        )


