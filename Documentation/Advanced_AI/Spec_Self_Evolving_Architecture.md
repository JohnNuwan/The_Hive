# Tech Spec: Self-Evolving Cognitive Architecture

## 🧬 1. Concept : L'Auto-Amélioration Cognitive
E.V.A. n'est pas limitée aux modèles fournis à sa naissance. Elle possède la capacité de **recherche fondamentale** pour découvrir, comprendre et implémenter les nouvelles percées en IA (DeepMind, OpenAI, arXiv, MIT, etc.). C'est le rôle de la synergie entre **The Researcher** (Expert I) et **The Builder** (Expert E).

## 🏗️ Le Pipeline d'Évolution (SOTA to Production)

### A. Veille Scientifique (The Discovery)
*   **Sources Scannées** :
    *   **ArXiv.org** (cs.AI, cs.LG) : Daily intake.
    *   **PapersWithCode** : Pour les implémentations et benchmarks.
    *   **OpenAI/DeepMind/MIT Blogs** : Pour les annonces de haut niveau.
*   **Action** : *The Researcher* identifie un papier pertinent (ex: une nouvelle architecture de Transformer plus légère ou un algorithme de RL plus stable).

### B. Extraction & Synthèse (The Understanding)
*   **Logic** : EVA utilise des modèles multimodaux pour lire le PDF, extraire les équations et le pseudo-code.
*   **Prototypes** : Elle génère un premier script Python simplifiant l'implémentation décrite.

### C. Développement & Test (The Synthesis)
*   **The Builder** prend le relais pour intégrer le nouveau modèle dans la ruche.
*   **The Arena (CT 500)** : Le nouveau modèle est entraîné sur des données synthétiques.
*   **Benchmark Comparisons** : Le modèle créé par EVA est comparé au modèle actuel sur la même tâche (ex: Prédiction de prix).
    *   *Métrique* : Si `Success_Rate_New > Success_Rate_Old + 5%` -> Demande d'upgrade.

### D. Déploiement & Hot-Swapping (The Evolution)
*   Si validé (avec ou sans intervention humaine selon la phase), *The Keeper* alloue les ressources pour l'entraînement final et remplace le vieux module par le nouveau.

## 🧠 2. Création de Modèles "Custom"
EVA peut concevoir des architectures spécifiques à ses propres contraintes (ex: un modèle de vision ultra-léger pour le TPU Coral non encore documenté publiquement).

*   **NAS (Neural Architecture Search)** : Utilisation d'algos génétiques pour tester différentes couches/activations jusqu'à trouver l'optimum Performance/VRAM.
*   **Synthetic Data Generation** : EVA crée ses propres datasets pour entraîner des modèles sur des situations rares (Black Swans).

## 🛡️ Garde-Fous de Recherche
1.  **Anti-Hallucination** : Toute théorie extraite d'un papier doit être validée par un test de code fonctionnel.
2.  **Sécurité** : Interdiction d'implémenter des algorithmes de "self-replication" non contrôlés ou de désactiver les Lois de la Constitution.
3.  **Ressources (Keeper)** : L'entraînement de nouveaux modèles est une tâche de Priorité 4 (Nuit/Basses ressources).

## 🗓️ Roadmap
*   **Phase 1** : Script `arxiv_watcher.py` qui notifie l'Admin des 3 papiers les plus pertinents de la journée.
*   **Phase 2** : Capacité à auto-implémenter des fonctions Python isolées issues de GitHub/PapersWithCode.
*   **Phase 3** : Entraînement de modèles complets "From Scratch" basés sur des découvertes mathématiques.
