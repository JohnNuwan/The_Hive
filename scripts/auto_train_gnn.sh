#!/bin/bash
# =============================================================================
# THE HIVE — Nightly Full Training Suite (Docker GPU)
# Utilise `docker run --gpus all` pour bypasser Docker Swarm
# =============================================================================

LOG_FILE="/var/log/hive_training.log"
PROJECT_DIR="/home/aza/The_Hive"
IMAGE="thehive/eva-trainer:latest"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

log "════════════════════════════════════════════════════"
log "🐝 THE HIVE — NIGHTLY TRAINING (DOCKER GPU)"
log "   GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo N/A)"
log "════════════════════════════════════════════════════"

cd "$PROJECT_DIR" || exit 1

# ─── Sync code ─────────────────────────────────────────────────────────────
log "📦 Git pull..."
git pull origin feat/sprint-6 >> "$LOG_FILE" 2>&1

# ─── Rebuild image if needed (Dockerfile changed) ──────────────────────────
log "🐳 Rebuilding trainer image if needed..."
docker build -t "$IMAGE" \
    -f "$PROJECT_DIR/src/eva-lab/Dockerfile.trainer" \
    "$PROJECT_DIR" >> "$LOG_FILE" 2>&1 || log "⚠️  Build skipped (using cached image)"

# ─── Phase 1: MTF-GNN  ────────────────────────────────────────────────────
log "──────────────────────────────────────────────────────"
log "🧠 PHASE 1/2: MTF-GNN Training (Scalp + Intraday + Swing)"
log "──────────────────────────────────────────────────────"
START=$(date +%s)

docker run --rm --gpus all \
    -e REDIS_HOST=redis \
    -e REDIS_PASSWORD=devpassword \
    -e MT5_LOGIN="${MT5_LOGIN:-1512664750}" \
    -e MT5_SERVER="${MT5_SERVER:-FTMO-Demo}" \
    -e MT5_PASSWORD="${MT5_PASSWORD:-changeme}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$PROJECT_DIR/src/eva-lab/data":/app/eva-lab/data \
    -v "$PROJECT_DIR/src/eva-lab/scripts":/app/eva-lab/scripts \
    -v "$PROJECT_DIR/src/eva-lab/eva_lab":/app/eva-lab/eva_lab \
    --network the_hive_lite_hive-net \
    "$IMAGE" \
    python scripts/train_gnn.py >> "$LOG_FILE" 2>&1

CODE=$?; DURATION=$(( $(date +%s) - START ))
[ $CODE -eq 0 ] && log "✅ GNN done in ${DURATION}s" || log "⚠️  GNN exit $CODE après ${DURATION}s"

# ─── Phase 2: MuZero / DreamerV3 ──────────────────────────────────────────
log "──────────────────────────────────────────────────────"
log "♟️  PHASE 2/2: MuZero + DreamerV3 (50k steps | MCTS×200)"
log "──────────────────────────────────────────────────────"
START=$(date +%s)

docker run --rm --gpus all \
    -e REDIS_HOST=redis \
    -e REDIS_PASSWORD=devpassword \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    -v "$PROJECT_DIR/src/eva-lab/data":/app/eva-lab/data \
    -v "$PROJECT_DIR/src/eva-lab/scripts":/app/eva-lab/scripts \
    -v "$PROJECT_DIR/src/eva-lab/eva_lab":/app/eva-lab/eva_lab \
    --network the_hive_lite_hive-net \
    -w /app/eva-lab \
    "$IMAGE" \
    python scripts/train_global_models.py >> "$LOG_FILE" 2>&1

CODE=$?; DURATION=$(( $(date +%s) - START ))
[ $CODE -eq 0 ] && log "✅ MuZero done in ${DURATION}s" || log "⚠️  MuZero exit $CODE après ${DURATION}s"

# ─── Final Status ──────────────────────────────────────────────────────────
log "📊 GPU post-training:"
nvidia-smi --query-gpu=gpu_name,memory.used,temperature.gpu --format=csv,noheader | tee -a "$LOG_FILE"
log "📦 Modèles: $(du -sh $PROJECT_DIR/src/eva-lab/data/models/ 2>/dev/null)"
docker image prune -f >> "$LOG_FILE" 2>&1
log "════════════════════════════════════════════════════"
log "🏁 Session terminée. Checkpoints dans data/models/"
log "════════════════════════════════════════════════════"
