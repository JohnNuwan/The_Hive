#!/bin/bash
# =============================================================================
# THE HIVE — Auto-Training MTF-GNN (23H00 Scheduler)
# Lancé chaque nuit par cron sur le serveur Proxmox
# =============================================================================

LOG_FILE="/var/log/hive_gnn_training.log"
PROJECT_DIR="/root/The_Hive"
PYTHON="python3"

echo "========================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🐝 Démarrage de l'entraînement MTF-GNN..." >> "$LOG_FILE"

# Pull latest code
cd "$PROJECT_DIR" || exit 1
git pull >> "$LOG_FILE" 2>&1

# Activate virtual environment if it exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Run GNN training script
cd "$PROJECT_DIR/src/eva-lab" || exit 1
$PYTHON scripts/train_gnn.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Entraînement terminé avec succès." >> "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Erreur d'entraînement (code $EXIT_CODE)." >> "$LOG_FILE"
fi

echo "========================================" >> "$LOG_FILE"
