#!/bin/bash
# 🐝 THE HIVE - BOOTSTRAP SCRIPT (PROD READY)

set -e

echo "🚀 Initialisation de l'unité de production THE HIVE..."

# 1. Vérification Hardware (Loi 0)
echo "🔍 Analyse des ressources physiques..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU Nvidia détecté"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu --format=csv
else
    echo "⚠️ Aucun GPU détecté. Mode dégradé (CPU-only) activé pour E.V.A."
fi

# 2. Unité de Survie (Python & UV)
echo "📦 Déploiement de l'environnement virtuel..."
if ! command -v uv &> /dev/null; then
    pip install uv
fi
uv venv
source .venv/bin/activate

# 3. Installation des Agents & Shared
echo "🧬 Intégration des modules neuronaux..."
uv pip install -e src/shared
uv pip install -e src/eva-core
uv pip install -e src/eva-banker
uv pip install -e src/eva-sentinel
uv pip install -e src/eva-lab

# 4. Stack AI Spécialisée (RTX 3090 & TPU Ready)
echo "🧠 Calibrage de la stack IA (JAX / Torch-Geometric / Rich)..."
uv pip install torch torch-geometric jax jaxlib rich pandas redis

# 5. Vérification Kernel & Redis
echo "📡 Test des liaisons de communication (Redis)..."
if command -v redis-cli &> /dev/null && redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis est en ligne"
else
    echo "❌ Erreur: Redis est requis pour la coordination du Swarm."
    exit 1
fi

echo "✨ E.V.A. est configurée pour le PC de production. Mission prête."
