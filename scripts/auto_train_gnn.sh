#!/bin/bash
# =============================================================================
# THE HIVE — Nightly Full Training Suite (Docker Edition)
# RTX 3090 FE via docker compose run eva-trainer
# =============================================================================

LOG_FILE="/var/log/hive_training.log"
PROJECT_DIR="/home/aza/The_Hive"
COMPOSE="docker compose"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

cd "$PROJECT_DIR" || exit 1

# ─── Header ───────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════════════"
log "🐝 THE HIVE — NIGHTLY TRAINING (DOCKER GPU SESSION)"
GPU_INFO=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
log "   GPU: RTX 3090 FE | ${GPU_INFO}MB VRAM free"
log "════════════════════════════════════════════════════"

# ─── Sync latest code ──────────────────────────────────────────────────────────
log "📦 Pulling latest code..."
git pull origin feat/sprint-6 >> "$LOG_FILE" 2>&1

# ─── Rebuild trainer image if Dockerfile changed ───────────────────────────────
log "🐳 Checking trainer image..."
$COMPOSE build eva-trainer 2>&1 | tail -5 | tee -a "$LOG_FILE"

# ─── Phase 1: MTF-GNN (PyTorch CUDA inside Docker) ────────────────────────────
log "──────────────────────────────────────────────────────"
log "🧠 PHASE 1/2: MTF-GNN Training (Scalp + Intraday + Swing)"
log "   500 epochs | batch=128 | M5(2000) + H1(2000) + D1(1000)"
log "──────────────────────────────────────────────────────"

START=$(date +%s)
$COMPOSE run --rm eva-trainer \
    python scripts/train_gnn.py \
    >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
DURATION=$(( $(date +%s) - START ))
[ $EXIT_CODE -eq 0 ] && log "✅ GNN done in ${DURATION}s" \
                      || log "⚠️  GNN exit code $EXIT_CODE après ${DURATION}s"

# ─── Phase 2: MuZero / DreamerV3 (JAX inside Docker) ─────────────────────────
log "──────────────────────────────────────────────────────"
log "♟️  PHASE 2/2: MuZero + DreamerV3 (50k steps | MCTS×200)"
log "──────────────────────────────────────────────────────"

START=$(date +%s)
$COMPOSE run --rm \
    -w /app \
    eva-trainer \
    python src/eva-lab/scripts/train_global_models.py \
    >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
DURATION=$(( $(date +%s) - START ))
[ $EXIT_CODE -eq 0 ] && log "✅ MuZero done in ${DURATION}s" \
                      || log "⚠️  MuZero exit code $EXIT_CODE après ${DURATION}s"

# ─── Cleanup dangling images ────────────────────────────────────────────────
docker image prune -f >> "$LOG_FILE" 2>&1

# ─── Final Status ──────────────────────────────────────────────────────────────
log "────────────────────────────────────────────"
log "📊 GPU Status Post-Training:"
nvidia-smi --query-gpu=gpu_name,memory.used,temperature.gpu --format=csv,noheader | tee -a "$LOG_FILE"
log "📦 Model Weights:"
du -sh "$PROJECT_DIR/src/eva-lab/data/models/" 2>/dev/null | tee -a "$LOG_FILE"
log "════════════════════════════════════════════════════"
log "🏁 Session complète. Checkpoints dans data/models/"
log "════════════════════════════════════════════════════"
