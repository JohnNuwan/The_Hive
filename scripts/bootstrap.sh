#!/bin/bash
# 🐝 THE HIVE - Bootstrap Script
# Installe les dépendances nécessaires pour E.V.A. sur une machine vierge.

set -e

echo "🚀 Démarrage de l'installation de THE HIVE..."

# 1. Vérification Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Erreur: Python3 n'est pas installé. Veuillez l'installer."
    exit 1
fi

# 2. Création de l'environnement virtuel
if [ ! -d ".venv" ]; then
    echo "📦 Création de l'unité de survie (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 3. Installation des modules en mode éditable
echo "🧬 Intégration des agents..."
pip install -e src/shared
pip install -e src/eva-core
pip install -e src/eva-banker
pip install -e src/eva-sentinel
pip install -e src/eva-shadow
pip install -e src/eva-builder

# 4. Vérification Docker
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️ Attention: Docker Compose n'est pas détecté. L'infrastructure ne pourra pas démarrer."
else
    echo "🐳 Infrastructure prête. Lancement suggéré : docker-compose -f Documentation/Config/docker_compose.yaml up -d"
fi

echo "✅ Installation terminée. E.V.A. est prête pour la Phase Genesis."
