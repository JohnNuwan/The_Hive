# Tech Spec: World Models & RL (Le Rêve)

##  1. Concept (Model-Based RL)
Contrairement aux modèles "Model-Free" (PPO/DQN) qui apprennent par essai-erreur direct, E.V.A. apprend un **Modèle du Monde** (World Model). Elle apprend à *prédire* ce qui va se passer, puis planifie dans son imagination ("Dreaming").

##  Architecture
*   **Algorithme Cible** : **DreamerV3** (efficace) ou **MuZero** (puissant mais lourd).
*   **Framework** : `Ray RLLib` ou implémentation Custom PyTorch.
*   **Environnement** : `Gymnasium` (TradingEnv custom).

##  Le Cycle "Dreaming" (La Nuit)
Quand les marchés sont fermés (ou la nuit), E.V.A. "rêve" :
1.  **Exploration Latente** : Elle simule des millions de trajectoires de marché dans son espace latent (pas besoin de données réelles, elle imagine des scénarios plausibles basés sur l'historique).
2.  **Policy Improvement** : Elle entraîne son Agent (The Banker) sur ces rêves.
3.  **Reality Check** : Au matin, on teste l'agent sur des données réelles. S'il performe mieux, on met à jour la prod.

##  Contraintes Hardware Genesis
*   MuZero est trop lourd pour 1x 3090 si LLM tourne aussi.
*   **Stratégie Genesis** : Utiliser un RL simple (PPO) pour commencer. Le World Model ne sera activé qu'avec le 2ème GPU ou en Cloud Spot Instance pour les entraînements lourds.

##  Roadmap
*   **Phase 0-1** : PPO (Proximal Policy Optimization) `stable-baselines3`.
*   **Phase 2** : Implmentation DreamerV3 simplifie (SLM - Small Language Model as World Model-).
