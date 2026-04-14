#!/bin/bash
# =============================================================================
# THE HIVE - Pont legacy vers le lanceur nocturne gouverne
# =============================================================================

set -euo pipefail

PROJECT_DIR="/home/aza/The_Hive"
MODERN_SCRIPT="$PROJECT_DIR/scripts/run_nightly_training_remote.sh"
LOG_FILE="$PROJECT_DIR/hive_training.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

if [ ! -x "$MODERN_SCRIPT" ]; then
  log "ERREUR: lanceur nightly moderne introuvable: $MODERN_SCRIPT"
  exit 1
fi

log "Pont legacy detecte: redirection de auto_train_gnn.sh vers run_nightly_training_remote.sh."
exec "$MODERN_SCRIPT"
