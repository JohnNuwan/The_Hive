# 🍯 THE HIVE — ARCHITECTURE ET WORKFLOWS DE TRADING AUTONOMES

THE HIVE est une plateforme souveraine de trading algorithmique et d'intelligence artificielle de pointe. Elle orchestre à la fois l'exécution en direct, la flotte de copy-trading locale sur Windows, et un laboratoire d'entraînement distant (EVA Lab) sur serveur Proxmox (`192.168.1.6`).

---

## 🏗️ 1. ARCHITECTURE MULTI-SERVICES & PORTS

La plateforme repose sur un maillage de micro-services hautement spécialisés, configurés pour tourner en harmonie sans conflits :

| Service | Port | Description |
| :--- | :--- | :--- |
| **Core API** | `8080` | Backend FastAPI central gérant l'état général et les bases cognitives. |
| **Nexus Frontend** | `3030` | Interface utilisateur construite en Nuxt.js pour la visualisation de la flotte. |
| **vLLM Inference** | `8000` | Serveur d'inférence LLM ultra-rapide (exclusif sur **GPU 0** host). |
| **Hermes Coordinator**| `9500` | Directeur de recherche quantitative, stress-testing et post-mortem (port `9500`). |
| **Sentinel Notifier** | `8200` | Service de monitoring, détection d'anomalies et auto-healing Sentinel. |
| **Banker Master** | `8100` | Instance décisionnelle de trading live (FTMO 10K) connectée à MT5. |
| **Banker Followers** | `8110+`| Instances de copy-trading esclaves (FTMO 50K, FTUK 100K). |
| **Shadow Learning** | `8900` | Tampon de stockage des données de self-play et comportements réels. |

---

## ⚡ 2. ENTRAÎNEMENT DE NUIT STABLE (JAX / MUZERO & DREAMER)

Chaque nuit, l'orchestrateur `train_nightly_stack.py` (**PID 1**) exécute de manière séquentielle le cycle d'apprentissage profond :

1.  **Entraînement GNN** (`train_gnn.py`) : Analyse des graphes de corrélation de marché.
2.  **Entraînement MuZero Scalp** (`train_global_models.py` - **PID 1568**) :
    -   Exécuté en mode **GPU 1** physique dédié (device local `0` dans le conteneur) via les liaisons `TRAINING_CHILD_CUDA_VISIBLE_DEVICES=0` et `TRAINING_CHILD_JAX_PLATFORMS=cuda`.
    -   Consommation mémoire GPU bridée à **85 %** pour éviter tout débordement de mémoire (out-of-memory).
    -   **Budgets de simulations adaptatifs** : Simulations réduites pour les indices (192 simulations MCTS) et les métaux pour optimiser la vitesse de self-play.
3.  **Entraînement DreamerV3 Offline** (`offline_trainer.py`) : Modélisation du monde basée sur le buffer consolidé de Shadow Learning.

---

## ⚔️ 3. LIGUE DARWINIENNE ARENA (RED-TEAMING)

L'évaluation des candidats se fait dans une arène de combat redoutable (`arena.py`) pour empêcher tout surapprentissage (overfitting) :

-   **Red-Teaming Directionnel** : Le Challenger est confronté à son miroir directionnel exact (`inverse_action`). S'il se fait battre par sa version inversée (`inverse_ok`), il est éliminé.
-   **Ligue de combat AlphaStar** : Matchmaking complet contre :
    -   Le Champion en titre actuel.
    -   Un pool de **3 champions historiques** archivés (`historical_ok`).
    -   Les baselines passives directionnelles (`Always Long` et `Always Short`).
-   **Scénario de Stress Hermes** : Requête cognitive à l'agent Hermes sur le port `9500` pour synthétiser un régime de marché adverse sur-mesure (haute volatilité, range) et évaluer la résistance du Challenger.

---

## 🧬 4. PONT DE RÉTROACTION ALPHAEVOLVE (LIVE BRIDGING)

À la fin de la séquence nocturne, si le Challenger sort victorieux de l'arène :
-   Le script `apply_alphaevolve_best.py` prend le relais.
-   Il extrait le meilleur génome évolué et l'injecte **automatiquement** dans les configurations de production active du **Banker**.
-   Cela garantit une boucle d'amélioration continue et fermée sans intervention humaine.

---

## 🧬 5. ARCHITECTURE SWARM MULTI-CHAMPION & ENTRAÎNEMENT JEPA

Le système intègre désormais un routage multi-champion dynamique et une représentation auto-supervisée du marché :

### Routage Dynamique par Symbole (Swarm Routing)
- **Manifeste de Routage** : Les instances en direct consultent `swarm_manifest.json` pour déterminer précisément quel expert (`muzero_scalp_ckpt_XXXXX.pkl`) charger en fonction du symbole (ex: `GER40.cash`, `XAUUSD`, `EURUSD`) et de l'horizon.
- **Clé de cache unique** : `DreamerGate` et `ChampionPromoter` indexent le cache d'inférence en combinant `SYMBOL:ENGINE:HORIZON` pour éviter les conflits et maximiser la réutilisation de mémoire RAM/GPU.
- **Stratégie de Fallback** : Si aucun checkpoint expert n'est assigné à un symbole, le système retombe élégamment sur le modèle champion global de l'horizon (`muzero_champion_scalp.pkl`).

### Représentations Auto-Supervisées (VICReg Market-JEPA)
- **Market-JEPA** : Pré-entraînement auto-supervisé VICReg (Variance-Invariance-Covariance Regularization) écrit en JAX, complété avec succès sur le serveur Proxmox.
- **Fichier de Poids** : Les poids de l'encodeur sont stockés dans `jepa_encoder_latest.pkl`.
- **Intégration Directe** : Le module `jax_agent.py` importe automatiquement ces poids à chaque instanciation du réseau MuZero si le fichier est détecté, injectant ainsi les caractéristiques temporelles avancées du marché dans les couches de représentation.

---

## 🔔 6. DISPATCHER DE NOTIFICATIONS DISCORD MULTI-SALONS

Pour remplacer le flux unique et désordonné de Telegram, les notifications sont maintenant dispatchées intelligemment vers un serveur Discord via **Webhooks** d'après le fichier `Liste_salon.csv` :

### Table de Routage Automatique (Keyword-Based)

-   **Salon `🚨 disclamer`** (Webhook) : Reçoit les alertes critiques, urgences système, déclenchements du kill switch, drawdowns excessifs.
    -   *Mots-clés déclencheurs* : `🚨`, `EMERGENCY`, `CRITICAL`, `DANGER`, `KILL SWITCH`, `FATAL`.
-   **Salon `📋 certification`** (Webhook) : Reçoit les audits quotidiens FTMO/FTUK d'Hermes et les vérifications de drawdown de fin de journée.
    -   *Mots-clés déclencheurs* : `FTMO`, `FTUK`, `COMPLIANCE`, `LOSS AUDITOR`, `AUDIT`, `CERTIFICATION`.
-   **Salon `📊 analyse-technique`** (Webhook) : Reçoit les diagnostics complets d'Hermes, revues techniques détaillées et analyses d'indicateurs (RSI, ADX, VWAP, EMA-200).
    -   *Mots-clés déclencheurs* : `ANALYSE TECHNIQUE`, `RSI`, `MACD`, `EMA-200`, `ADX`, `HERMES REPORT`.
-   **Salon `⚡ scalping`** (Webhook) : Reçoit les signaux de trading et ouvertures de positions à horizon Scalping (M5).
    -   *Mots-clés déclencheurs* : `SCALP`, `SCALPING`, `MUZERO_SCALP`.
-   **Salon `🕒 intraday`** (Webhook) : Reçoit les positions de trading Intraday (H1).
    -   *Mots-clés déclencheurs* : `INTRADAY`, `H1`.
-   **Salon `📈 swing`** (Webhook) : Reçoit les positions de trading Swing (D1).
    -   *Mots-clés déclencheurs* : `SWING`, `D1`.
-   **Salon `💬 général`** (Webhook) : Salon général pour les démarrages du Hive, les résumés d'entraînement nocturnes et les messages informatifs de routine.
    -   *Fallback par défaut*.

---

## 🛡️ RÈGLES DE MAINTENANCE CLÉS

1.  **Isolation GPU** : Conserver `GPU 0` exclusivement pour vLLM, et `GPU 1` exclusivement pour JAX MuZero et Dreamer.
2.  **No CD commands** : Ne jamais utiliser de commande shell `cd` directement dans l'environnement de commande Windows.
3.  **Drawdown Compliance** : L'audit journalier `hermes_loss_auditor.py` s'exécute à 23h50 UTC pour valider le respect des limites FTMO/FTUK et couper les instances Banker en cas de déviation majeure.
