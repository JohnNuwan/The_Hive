# Suivi de l'évolution du projet THE HIVE

Ce document récapitule l'état actuel du système EVA et les étapes futures.

## État des Lieux (Sprint 3.2 Terminé)

### Infrastructure & Core
- **Compatibilité Python** : Tous les services alignés sur Python 3.10 (compatibilité CUDA/PyTorch).
- **Core Orchestrator** : Endpoint `/chat` enrichi avec les traces de raisonnement (`thoughts`).
- **Memory Service** : Intégration Qdrant pour la mémoire sémantique et extraction de graphes GNN.
- **Thought Visualization** : Extraction automatique des balises `<thought>` dans les réponses LLM.
- **Infrastructure Hardening** : Fix des healthchecks Nexus et migration Loki vers TSDB (v13).

### Frontend (Nexus)
- **Nexus Graph** : Visualisation Force-Directed des relations sémantiques.
- **Memory Explorer** : Interface de recherche et de navigation dans les fragments Qdrant.
- **Persistence** : Sessions de chat persistantes via LocalStorage.
- **UI/UX** : Thème bioluminescent avec panneaux de raisonnement expert.

## Informations Utiles pour le Futur
- **Ports** : 8080 (Backend Core), 3030 (Frontend Nexus).
- **Modèles LLM** : Utilise Ollama (`llama3:8b`) et vLLM.
- **Embedding** : `nomic-embed-text` via Ollama.
- **Volume Tablet** : Constitution Genesis accessible sur `/mnt/tablet`.

## Ce qu'il reste à faire

### Sprint 3.3 : Polyglot Hardening (Performance)
- [ ] **Sentinel (Rust)** : Durcir la communication P2P (libp2p).
- [ ] **Nervous (Go)** : Migration gRPC pour un routage < 1ms.
- [ ] **Quant-Lab (Julia)** : Intégration pour simulations de portefeuille lourdes.
- [ ] **Lab (JAX)** : Implémentation du cœur DreamerV3.

### Sprint 4 : Vision & Simulation
- [ ] **World Model** : Workspace 8x8 pour la simulation d'actions.
- [ ] **Gymnasium Arena** : Environnement d'entraînement pour DreamerV3.

### Sprint 5 : Autonomie
- [ ] **Orchestration Phoenix** : Auto-réparation et auto-évolution du code expert.
