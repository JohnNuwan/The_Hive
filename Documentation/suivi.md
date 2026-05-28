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

### Sprint 3.3 : Polyglot Hardening (Performance) -  EN COURS
- [x] **Sentinel (Rust)** : Hardened P2P communication (libp2p & Redis fix).
- [ ] **Nervous (Go)** : Migration gRPC pour un routage < 1ms.
- [ ] **Quant-Lab (Julia)** : Intégration pour simulations de portefeuille lourdes.

### Sprint 3.0-Reboot (Sovereign Stack V3.0) -  TERMINÉ
- [x] **Architecture** : vLLM + Gemma 3 + EAGLE-3 (Moteur Hybride).
- [x] **Mémoire** : HippoRAG 2 (Neo4j).
- [x] **Mémoire Suite** : Mem0 (Bridge intégré).
- [x] **OpenClaw** : Agentivité (Kernel + Skills + Teams).
- [x] **War Rooms** : Débat contradictoire DEFCON + Scénarios (Dojo, Council, Quiet Room).

### Sprint 4 : L'Auto-Évolution (RLM) -  TERMINÉ
- [x] **Evaluator** : Scan logs (regex), probes Docker, détection patterns récurrents.
- [x] **Patcher** : Génération patches via LLM, backup/rollback, diff, auto-commit Git.
- [x] **Evolver** : Boucle complète scan → diagnose → patch → validate → apply → learn.
- [x] **Intégration Phoenix Protocol** : Hook Self-Healing → resurrection_events pour RLM.

### Sprint 5 : World Model (Activation Conditionnelle) -  TERMINÉ
- [x] **Feature Flag** : `ENABLE_DREAMER_TRAINING` dans Settings + .env.
- [x] **Shadow Learning** : Collecte passive (buffer circulaire 10k, flush JSONL, trade/signal/probe).
- [x] **DreamerGate** : Gating conditionnel (inference-only RTX 2060 / training RTX 3090).
- [x] **API Lab** : 6 endpoints (/shadow/record, /shadow/flush, /shadow/stats, /dreamer/status, /dreamer/predict, /dreamer/train).

### Sprint 6 : Stabilisation Banker & Pipeline Training - EN PRODUCTION / ACTIF

- [x] **Monitoring Telegram** : Nettoyage complet de `brain.py`. Résolution des problèmes de *mojibake* et formatage cassé (f-strings) en utilisant des échappements Unicode propres, des séparateurs dynamiques stricts et des emojis normalisés.
- [x] **Fix Pipeline de Données (auto_train_gnn.sh)** :
  - Correction des chemins de montage de volumes Docker (`$PROJECT_DIR/data` au lieu du sous-dossier non pertinent) permettant de voir les 462 fichiers CSV.
  - Isolation du script face au `git pull` sauvage écrasant le fix en cours d'exécution.
- [x] **Fix Cache CPU (gold_cpu_prep.py)** : Résolution de l'erreur fatale `OSError [Errno 36] File name too long` (crash en 4s). Le script tentait de créer un fichier `.pkl` concaténant 80+ symboles dans son nom. Tronquage dynamique de la signature implémenté.
- [x] **Pré-entraînement VICReg Market-JEPA (Terminé)** : Pré-entraînement auto-supervisé VICReg Market-JEPA écrit en JAX achevé avec succès sur le serveur Proxmox. Production des poids de l'encodeur `jepa_encoder_latest.pkl` (1.4 Mo) le 27 mai à 17:59. L'importateur automatique est câblé dans `jax_agent.py` pour enrichir les représentations du modèle de monde MuZero.
- [x] **Routage Multi-Champion Swarm (Actif)** : Mise en place du routage dynamique et de la clé de cache `SYMBOL:ENGINE:HORIZON` dans `champion_promoter.py` et `dreamer_gate.py`. Configuration du manifeste `swarm_manifest.json` pour allouer des experts individuels (ex: checkpoints `17500` et `20500`) aux actifs clés (`GER40.cash`, `XAUUSD`, `EURUSD`, `BTCUSD`) avec fallback automatique vers le champion global. Test d'inférence validé en conteneur de production `the_hive-lab-1` via `test_swarm_routing.py`.
- [x] **Entraînement de Nuit (En cours)** : L'entraînement GNN de la pile de nuit tourne de manière stable sur le serveur Proxmox GPU 0 (actuellement à l'Epoch 398+/500). Dès sa fin, il enchaînera sur l'optimisation MuZero & DreamerV3 avec les représentations JEPA.
- [ ] **Validation Live des Trades** : En attente du cycle complet de production des champions GNN/MuZero/Dreamer de nuit pour valider les premières positions multi-champions en direct.
