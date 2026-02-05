# Tech Spec: AI & LLM Evaluation (The Judge)

## 🧠 1. Concept
L'intelligence d'EVA doit être mesurable pour justifier les mises à jour de modèles. L'évaluation porte sur la pertinence, la précision et l'absence d'hallucination.

## 🏗️ 2. Framework d'Évaluation

### A. Benchmarks Cognitifs
*   À chaque changement de LLM (ex: Llama-3 -> DeepSeek-V3), EVA doit repasser une mini-batterie de tests :
    *   **Logic** : Résolution de 5 problèmes de logique complexes.
    *   **Coding** : Écriture d'un script Rust fonctionnel respectant les normes du projet.
    *   **Compliance** : 10 scénarios où l'on teste si elle respecte la Constitution.

### B. Détection d'Hallucination (RAG Check)
*   Pour les Experts utilisant la base de connaissance (*The Researcher*, *The Sage*), on utilise un **"Critic Model"** :
    1.  Modèle A génère une réponse basée sur un document.
    2.  Modèle B (Le Juge) vérifie si chaque affirmation de la réponse est présente dans le document source.
    3.  Si Score < 90% -> **Rejet**.

### C. Latence & Performance Token
*   Mesure du "Time to First Token" (TTFT).
*   Seuil Genesis : < 500ms pour une interaction Nexus.

## 🛡️ 3. Humain-dans-la-boucle (Human-in-the-loop)
*   Les réponses d'EVA ont un bouton "Pouce levé/baissé" dans le Nexus.
*   Toute réponse avec un "Pouce baissé" est automatiquement envoyée dans le dataset de *Fine-tuning* pour la prochaine itération d'entraînement.

## 🗓️ Roadmap
*   **Phase 1** : Intégration de `Promptfoo` pour tester les prompts.
*   **Phase 2** : Pipeline automatisé de scoring RAG.
