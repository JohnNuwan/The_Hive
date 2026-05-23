# GUIDE D'ONBOARDING TECHNIQUE — THE HIVE

Bienvenue sur **THE HIVE**, une plateforme souveraine de trading algorithmique et d'intelligence artificielle. Ce document est votre guide de démarrage rapide pour comprendre notre stack technique, configurer votre environnement local et commencer à contribuer efficacement.

---

## 1. Structure de l'Écosystème & Packages

Le codebase est organisé sous forme de mono-répertoire divisé en packages Python modulaires situés dans `src/` :

* **`src/shared`** : Le socle commun. Contient les configurations globales, le routeur de notifications (Telegram/Discord), les calculs d'indicateurs techniques (`IndicatorFactory`) et le **MemoryBridge** (Mem0 + HippoRAG 2).
* **`src/eva-banker`** : Le cerveau de trading live. Conçu pour s'exécuter localement sur Windows au plus près des terminaux MetaTrader 5 (MT5). Il est hautement optimisé pour la faible latence de passage d'ordres et la gestion des risques FTMO/FTUK.
* **`src/eva-lab`** : Notre laboratoire quantitatif de Deep Learning. S'exécute principalement à distance sur les deux GPU RTX 3090 du serveur Proxmox. Il contient les moteurs de JAX (MuZero, Market-JEPA, DreamerV3) et les orchestrateurs nocturnes.
* **`src/eva-nexus`** : Notre interface d'administration Web construite en Nuxt.js pour monitorer l'état de la flotte en temps réel.
* **`src/eva-sentinel`** : Notre garde-fou autonome assurant le monitoring système, la détection d'anomalies de trading et l'auto-healing.

---

## 2. Configuration de l'Environnement de Développement

### A. Pré-requis Locaux (Développement Windows)
1. **Python 3.10+** et un gestionnaire d'environnement virtuel (Venv ou Poetry).
2. **MetaTrader 5 (MT5)** installé localement avec des comptes de démonstration ou de challenge FTMO/FTUK.
3. Fichier `.env` configuré à la racine du projet (copié depuis `.env.example`).

### B. Base de Données & Services (Docker Compose)
Pour faire tourner la stack locale de tests, utilisez le fichier `docker-compose.yml` :
```bash
docker-compose up -d --build
```
Cela démarrera les services suivants :
* **TimescaleDB** (`port 5432`) : Stockage analytique des bars OHLC et des résultats d'Arena.
* **Qdrant** (`port 6333`) : Base de données vectorielle pour les souvenirs Mem0.
* **Redis** (`port 6379`) : Cache et messagerie à court terme.
* **Neo4j** (`port 7474/7687`) : Graphe de connaissances pour la mémoire associative **HippoRAG 2**.

---

## 3. Routine Opérationnelle : Travailler avec le Serveur Distant

Le gros de l'entraînement quantitatif s'effectue sur le serveur Proxmox (`192.168.1.6`). Ne lancez jamais d'entraînement MuZero lourd sur votre machine locale !

### A. Déploiement du Code Local
Pour pousser vos modifications de code locales (`src/eva-lab`) vers le serveur d'entraînement à distance, nous utilisons un script de synchronisation automatisé :
```bash
python scratch/deploy_all.py
```
Ce script copie vos fichiers locaux dans le conteneur JAX Trainer de Proxmox (`the_hive-eva-trainer-run-cdbd0d29e6c3`) et exécute automatiquement un contrôle de syntaxe (`Agent import successful!`).

### B. Audit de l'Entraînement Actif
Pour vérifier l'état de la séquence nocturne active et des conteneurs GPU sur le serveur, utilisez nos utilitaires d'audit :
```bash
# Vérifier l'usage VRAM et CPU du conteneur remote
python scratch/check_active.py

# Lire les métriques d'apprentissage de MuZero en direct
python scratch/check_active_metrics.py
```

---

## 4. Conventions de Code (Strictes)

THE HIVE a des règles strictes régies par [AGENTS.md](file:///c:/Users/nandi/Desktop/The%20Hive/The_Hive/AGENTS.md) :

1. **La Langue (Français)** :
   * **Tous les docstrings**, commentaires, logs console (`logger.info`), exceptions et erreurs doivent être rédigés en **Français**.
   * Les noms de classes, variables et fonctions restent en **Anglais** (ex: `RiskValidator`, `validate_order`).
2. **Le Style de Docstrings (Google Style)** :
   * Chaque fonction ou classe publique doit documenter explicitement ses arguments (`Args`), son retour (`Returns`) et les exceptions possibles (`Raises`).
3. **Pas de CD Commandes** :
   * Dans vos scripts ou commandes Windows, évitez l'utilisation directe de la commande `cd` pour éviter les dysfonctionnements de chemin absolu. Définissez toujours vos chemins à partir de la racine du répertoire.
