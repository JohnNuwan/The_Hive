# Tech Spec: Financial & Trading Testing (The Validator)

## 💹 1. Concept
On ne déploie JAMAIS un algorithme de trading sans qu'il ait prouvé sa robustesse statistique. C'est le rôle de la triple validation : **Backtest -> Forward Test -> Monte Carlo**.

## 🏗️ 2. Protocole de Validation

### A. Backtesting Statistique (Historique)
*   **Moteur** : Julia (pour la vitesse) ou Backtrader (Python).
*   **Données** : 5 ans d'historique (M1/M5) incluant des périodes de haute volatilité (COVID, guerres, crises).
*   **KPIs Exigés** :
    *   Profit Factor > 1.5.
    *   Max Drawdown < 2.0%.
    *   Recovery Factor > 3.0.

### B. Forward Testing (Paper Trading)
*   L'algorithme tourne sur les prix RÉELS du marché mais avec un compte démo.
*   Durée : Minimum 2 semaines consécutives.
*   Validation : Si les résultats démo divergent de >10% des résultats backtest sur la même période -> **REJET** (Overfitting suspecté).

### C. Inférence de Monte Carlo (Analyse de Robustesse)
*   Lancer 10,000 simulations en mélangeant aléatoirement l'ordre des trades passés.
*   *Objectif* : Vérifier la probabilité de "ruine" (Drawdown > 8%). Elle doit être < 0.1%.

## 🛡️ 3. Stress Tests (Black Swan Simulation)
*   On injecte des sauts de prix artificiels (Slippage de 50 points, Gap de 100 points) pour vérifier que les Stop-Loss du Kernel Rust se déclenchent correctement même en cas de panique marché.

## 🗓️ Roadmap
*   **Phase 1** : Script de backtest simple sur 1 an de Gold.
*   **Phase 2** : Automatisation du rapport Monte Carlo hebdomadaire.
