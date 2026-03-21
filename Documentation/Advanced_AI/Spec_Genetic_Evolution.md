# Tech Spec: Genetic Evolution Engine (Darwinism Digital)

##  1. Concept
L'apprentissage n'est pas seulement neuronal (Backprop). E.V.A. utilise l'**Évolution** pour optimiser ce que les gradients ne peuvent pas atteindre : les hyper-paramètres et les stratégies de trading discrètes.

##  Architecture
*   **Librairie** : `DEAP` (Distributed Evolutionary Algorithms in Python) ou `PyGad`.
*   **Cible** :
    *   Optimisation des stratégies Trading (StopLoss, TakeProfit, Indicateurs).
    *   Architecture Search (NAS) pour petits réseaux de neurones.

##  Le Cycle de Vie (The Epoch)
1.  **Population Initiale** : 100 stratégies aléatoires (ex: MA Cross 50/200, MA Cross 10/50, etc.).
2.  **Evaluation (Fitness)** : Backtest rapide sur les données de la semaine dernière (TimescaleDB).
    *   *Fitness Function* : `SharpeRatio * 0.7 + (1 / MaxDrawdown) * 0.3`.
3.  **Selection** : On garde les top 20%.
4.  **Crossover** : On mélange les paramètres des gagnants.
    *   Parent A (SL: 10, TP: 20) + Parent B (SL: 50, TP: 100) -> Enfant (SL: 10, TP: 100).
5.  **Mutation** : On modifie aléatoirement un gène (ex: SL: 10 -> 11).
6.  **Next Gen** : On répète.

##  Sécurité
*   Les stratégies générées sont **sandboxées**. Elles ne tradent pas en réel tant qu'elles n'ont pas survécu à 50 générations ET passé un "Forward Test" en Paper Trading pendant 48h.

##  Roadmap
*   **Phase 0** : Simple script `optimize_ma.py` avec DEAP.
*   **Phase 2** : Évolution continue sur serveur dédié (Worker).
