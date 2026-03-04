#!/bin/bash
# =============================================================================
# THE HIVE — Nightly Full Training Suite (Docker GPU)
# RTX 3090 FE — Pause vLLM before training, resume after
# =============================================================================

LOG_FILE="/var/log/hive_training.log"
PROJECT_DIR="/home/aza/The_Hive"
IMAGE="thehive/eva-trainer:latest"
PASS="Kumara-42/600"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

cd "$PROJECT_DIR" || exit 1

log "════════════════════════════════════════════════════"
log "🐝 THE HIVE — NIGHTLY TRAINING SESSION"
log "   GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo N/A)"
log "════════════════════════════════════════════════════"

# ─── Sync code ─────────────────────────────────────────────────────────────
log "📦 Git pull..."
git pull origin feat/sprint-6 >> "$LOG_FILE" 2>&1

# ─── Pause vLLM to free the GPU ─────────────────────────────────────────────
log "⏸️  Pausing vLLM container (libérer GPU pour entraînement)..."
echo "$PASS" | sudo -S docker stop the_hive-vllm-1 2>/dev/null \
  && log "   ✅ vLLM stoppé." \
  || log "   ℹ️  vLLM déjà arrêté."

sleep 5  # Wait for GPU memory to flush

# ─── Ensure trainer image is up to date ────────────────────────────────────
log "🐳 Vérification de l'image ea-trainer..."
echo "$PASS" | sudo -S docker build -q -t "$IMAGE" \
    -f "$PROJECT_DIR/src/eva-lab/Dockerfile.trainer" \
    "$PROJECT_DIR" >> "$LOG_FILE" 2>&1 \
  && log "   ✅ Image OK." \
  || log "   ⚠️  Build skipped (cached)."

# ─── Phase 1: MTF-GNN Training ─────────────────────────────────────────────
log "──────────────────────────────────────────────────────"
log "🧠 PHASE 1/2: MTF-GNN (500 epochs | Scalp + Intraday + Swing)"
log "──────────────────────────────────────────────────────"
START=$(date +%s)

echo "$PASS" | sudo -S docker run --rm --gpus all \
    -e REDIS_HOST=host.docker.internal \
    -e REDIS_PASSWORD=devpassword \
    -e MT5_LOGIN="${MT5_LOGIN:-1512664750}" \
    -e MT5_SERVER="${MT5_SERVER:-FTMO-Demo}" \
    -e MT5_PASSWORD="${MT5_PASSWORD:-changeme}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$PROJECT_DIR/src/eva-lab/data":/app/eva-lab/data \
    -v "$PROJECT_DIR/src/eva-lab/scripts":/app/eva-lab/scripts \
    -v "$PROJECT_DIR/src/eva-lab/eva_lab":/app/eva-lab/eva_lab \
    --add-host host.docker.internal:host-gateway \
    "$IMAGE" \
    python scripts/train_gnn.py >> "$LOG_FILE" 2>&1

CODE=$?; DURATION=$(( $(date +%s) - START ))
[ $CODE -eq 0 ] && log "✅ GNN terminé en ${DURATION}s" || log "⚠️  GNN exit $CODE (${DURATION}s)"

# ─── Phase 2: MuZero / DreamerV3 ──────────────────────────────────────────
log "──────────────────────────────────────────────────────"
log "♟️  PHASE 2/2: MuZero + DreamerV3 (50k steps | MCTS×200)"
log "──────────────────────────────────────────────────────"
START=$(date +%s)

echo "$PASS" | sudo -S docker run --rm --gpus all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    -v "$PROJECT_DIR/src/eva-lab/data":/app/eva-lab/data \
    -v "$PROJECT_DIR/src/eva-lab/scripts":/app/eva-lab/scripts \
    -v "$PROJECT_DIR/src/eva-lab/eva_lab":/app/eva-lab/eva_lab \
    -w /app/eva-lab \
    "$IMAGE" \
    python scripts/train_global_models.py >> "$LOG_FILE" 2>&1

CODE=$?; DURATION=$(( $(date +%s) - START ))
[ $CODE -eq 0 ] && log "✅ MuZero terminé en ${DURATION}s" || log "⚠️  MuZero exit $CODE (${DURATION}s)"

# ─── Cleanup dangling images ────────────────────────────────────────────────
echo "$PASS" | sudo -S docker image prune -f >> "$LOG_FILE" 2>&1

# ─── Resume vLLM ───────────────────────────────────────────────────────────
log "▶️  Redémarrage de vLLM..."
echo "$PASS" | sudo -S docker start the_hive-vllm-1 >> "$LOG_FILE" 2>&1 \
  && log "   ✅ vLLM redémarré." \
  || log "   ⚠️  Impossible de redémarrer vLLM."

# ─── Final Status ──────────────────────────────────────────────────────────
log "📊 GPU post-training:"
nvidia-smi --query-gpu=gpu_name,memory.used,temperature.gpu --format=csv,noheader | tee -a "$LOG_FILE"
log "📦 Modèles: $(du -sh $PROJECT_DIR/src/eva-lab/data/models/ 2>/dev/null || echo 'N/A')"
log "════════════════════════════════════════════════════"
log "🏁 Session terminée. Checkpoints dans data/models/"
log "════════════════════════════════════════════════════"
