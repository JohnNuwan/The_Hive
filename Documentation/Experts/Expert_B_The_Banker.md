# Expert B: THE BANKER (L'Analyste Financier)

## 1. Identité & Rôle
*   **Nom** : The Banker
*   **Rôle** : Trading, Gestion du Risque, Exécution.
*   **Type** : Permanent (Processus Python indépendant).

## 2. Architecture Technique
*   **Modèle IA** : `DeepSeek-Coder-V2` ou `FinGPT` (Appelé uniquement pour l'analyse de sentiment news/chart, pas pour le clic bouton).
*   **Moteur d'Exécution** : Script Python pur (`banker_engine.py`).
*   **Interface Broker** : MetaTrader 5 (MT5) via `MetaTrader5` Python package.

## 3. Fonctions Critiques
### A. Risk Officer (Le Cerveau Froid)
*   Avant tout trade, calcul du **Risk Per Trade**.
*   Formule : `LotSize = (AccountEquity * Risk%) / (StopLossDistance * TickValue)`.
*   *Hard Rule* : Si `Risk% > 1.0%`, REJET immédiat.

### B. The Trade Copier (Hydra)
*   Réplique les ordres du compte "Master" vers les comptes Prop Firm "Slaves".
*   Gère le slippage (Tolérance max 5 points).

### C. News Filter
*   Interroge *The Shadow* pour savoir si une "Red Folder News" est imminente (-30 min / +30 min).
*   Si OUI -> Interdiction d'ouvrir un nouveau trade.

## 4. Flux de Données
*   **Tick Data** : MT5 -> Python -> TimescaleDB (Toutes les ticks).
*   **Bar Data** : Agrégation en M1/M5/H1/H4 pour les algos.

## 5. Roadmap Dév
*   **J1** : Hello World MT5 (Connexion, Get Balance).
*   **J2** : Implémentation du `RiskCalculator` (Unit Tested ++).
*   **J3** : Algo simple "Random Entry" en Demo pour tester la chaine d'exécution complète.
