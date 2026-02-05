# Tech Architect: JULIA (The Quant)

## 📌 Rôle
Julia est utilisé ponctuellement pour le **Calcul Scientifique**, la **Simulations Monte Carlo**, et le **Risk Modelling Complexe**. C'est le "Cerveau Mathématique".

## 🏗️ Composants Julia

### 1. The Risk Simulator (`eva-risk`)
*   **Type** : Service API (Genie.jl) ou Script Batch.
*   **Responsabilité** :
    *   Calculer la VaR (Value at Risk) du portefeuille global.
    *   Lancer 100,000 simulations Monte Carlo pour prédire la probabilité de dépasser les 4% de drawdown Daily avec la stratégie actuelle.
*   **Pourquoi Julia ?** : Vitesse proche du C pour les boucles mathématiques, syntaxe proche de Python/Matlab.

### 2. Arbitrage Matrix
*   Si on fait de l'arbitrage (Crypto), Julia gère les calculs de déséquilibre matriciel sur 50 paires en < 5ms.

## 🛡️ Règles de Dév Julia
*   **Type Stability** : Écrire du code "Type Stable" pour que le JIT Compiler optimise à fond.
*   **Multiple Dispatch** : Utiliser la force de Julia pour modéliser les instruments financiers.

## 🔄 Interaction
*   Julia n'est pas un service "Always On" critique comme Rust. C'est un Oracle.
*   Python (Banker) demande à Julia : "Quelle est la probabilité de crash aujourd'hui ?" -> Julia calcule 10s -> Répond "5%".
