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

### Sprint 3.3 : Polyglot Hardening (Performance) - 🟡 EN COURS
- [x] **Sentinel (Rust)** : Hardened P2P communication (libp2p & Redis fix).
- [ ] **Nervous (Go)** : Migration gRPC pour un routage < 1ms.
- [ ] **Quant-Lab (Julia)** : Intégration pour simulations de portefeuille lourdes.

### Sprint 3.0-Reboot (Sovereign Stack V3.0) - ✅ TERMINÉ
- [x] **Architecture** : vLLM + Gemma 3 + EAGLE-3 (Moteur Hybride).
- [x] **Mémoire** : HippoRAG 2 (Neo4j).
- [x] **Mémoire Suite** : Mem0 (Bridge intégré).
- [x] **OpenClaw** : Agentivité (Kernel + Skills + Teams).
- [x] **War Rooms** : Débat contradictoire DEFCON + Scénarios (Dojo, Council, Quiet Room).

### Sprint 4 : L'Auto-Évolution (RLM) - ✅ TERMINÉ
- [x] **Evaluator** : Scan logs (regex), probes Docker, détection patterns récurrents.
- [x] **Patcher** : Génération patches via LLM, backup/rollback, diff, auto-commit Git.
- [x] **Evolver** : Boucle complète scan → diagnose → patch → validate → apply → learn.
- [x] **Intégration Phoenix Protocol** : Hook Self-Healing → resurrection_events pour RLM.

### Sprint 5 : World Model (Activation Conditionnelle) - ✅ TERMINÉ
- [x] **Feature Flag** : `ENABLE_DREAMER_TRAINING` dans Settings + .env.
- [x] **Shadow Learning** : Collecte passive (buffer circulaire 10k, flush JSONL, trade/signal/probe).
- [x] **DreamerGate** : Gating conditionnel (inference-only RTX 2060 / training RTX 3090).
- [x] **API Lab** : 6 endpoints (/shadow/record, /shadow/flush, /shadow/stats, /dreamer/status, /dreamer/predict, /dreamer/train).
