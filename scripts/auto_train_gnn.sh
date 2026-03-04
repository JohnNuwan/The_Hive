#!/bin/bash
# =============================================================================
# THE HIVE — Nightly Full Training Suite (23H00 Cron)
# RTX 3090 FE Optimized — Runs all AI models sequentially
# Ordre: GNN (PyTorch/CUDA) → MuZero/DreamerV3 (JAX)
# =============================================================================

LOG_FILE="/var/log/hive_training.log"
PROJECT_DIR="/home/aza/The_Hive"
PYTHON="python3"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# ─── Header ───────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════════════"
log "🐝 THE HIVE — NIGHTLY TRAINING SESSION"
log "   GPU: RTX 3090 FE | $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)MB VRAM free"
log "════════════════════════════════════════════════════"

cd "$PROJECT_DIR" || exit 1

# ─── Sync latest code ─────────────────────────────────────────────────────────
log "📦 Syncing latest code from GitHub..."
git pull origin feat/sprint-6 >> "$LOG_FILE" 2>&1

# ─── Phase 1: MTF GNN Training (PyTorch CUDA) ─────────────────────────────────
log "──────────────────────────────────────────────────"
log "🧠 PHASE 1/3: MTF-GNN Training (Scalp + Intraday + Swing)"
log "   Config: 500 epochs | batch=128 | M5(2000) + H1(2000) + D1(1000)"
log "──────────────────────────────────────────────────"

START=$(date +%s)
cd "$PROJECT_DIR/src/eva-lab" || exit 1
$PYTHON scripts/train_gnn.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
DURATION=$(( $(date +%s) - START ))
if [ $EXIT_CODE -eq 0 ]; then
    log "✅ GNN Training done in ${DURATION}s"
else
    log "⚠️  GNN Training exited with code $EXIT_CODE (après ${DURATION}s)"
fi

# ─── Phase 2: MuZero / DreamerV3 Training (JAX) ───────────────────────────────
log "──────────────────────────────────────────────────"
log "♟️  PHASE 2/3: MuZero Global Training (50k steps | MCTS×200)"
log "──────────────────────────────────────────────────"

START=$(date +%s)
cd "$PROJECT_DIR" || exit 1
$PYTHON src/eva-lab/scripts/train_global_models.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
DURATION=$(( $(date +%s) - START ))
if [ $EXIT_CODE -eq 0 ]; then
    log "✅ MuZero Training done in ${DURATION}s"
else
    log "⚠️  MuZero Training exited with code $EXIT_CODE (après ${DURATION}s)"
fi

# ─── Phase 3: Cleanup & Status Report ─────────────────────────────────────────
log "──────────────────────────────────────────────────"
log "📊 PHASE 3/3: GPU & Disk Status"
nvidia-smi --query-gpu=gpu_name,memory.used,memory.free,temperature.gpu --format=csv,noheader | tee -a "$LOG_FILE"
df -h "$PROJECT_DIR/src/eva-lab/data" | tee -a "$LOG_FILE"
log "──────────────────────────────────────────────────"

# ─── Summary ──────────────────────────────────────────────────────────────────
log "🏁 Session d'entraînement terminée."
log "   → Modèles dans: $PROJECT_DIR/src/eva-lab/data/models/"
log "   → Log complet:  $LOG_FILE"
log "════════════════════════════════════════════════════"
