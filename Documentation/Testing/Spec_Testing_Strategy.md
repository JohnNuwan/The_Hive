# Tech Spec: Global Testing Strategy (E.V.A Quality Assurance)

##  1. Philosophie de Test
La sûreté d'une IA distribuée comme E.V.A. repose sur une validation multicouche. Nous appliquons le principe du **"Test Early, Test Often"** avec une pyramide de tests adaptée à l'IA et au Rust.

##  2. Niveaux de Tests

### A. Tests Unitaires (Unit Tests)
*   **Rust (Kernel)** : Utilisation de `cargo test`. Focus sur les fonctions de calcul de hash, de thermal monitoring et de validation de lois.
*   **Python (Agents)** : Utilisation de `pytest`. Focus sur le parsing de messages, le routage LangGraph et les fonctions utilitaires.
*   **Requirement** : 100% de coverage sur les modules financiers (*The Banker*).

### B. Tests d'Intégration (Integration Tests)
*   Validation de la boucle : `User Input -> Core -> Expert -> Service -> Response`.
*   Simulation des I/O : Mock des APIs brokers (MT5) et des serveurs LLM (vLLM) pour tester la logique de décision sans consommer de tokens ou d'argent.

### C. Tests de Régression IA (Golden Sets)
*   On maintient un fichier `golden_queries.json` avec 50 questions types et la réponse attendue.
*   À chaque modification du prompt système ou du modèle, on vérifie que les réponses d'EVA ne dégradent pas en qualité ou en sécurité.

##  3. CI/CD Pipeline
À chaque `git push` :
1.  **Linter** : `ruff` (Python), `clippy` (Rust).
2.  **Safety Scan** : `bandit` (Python), `cargo audit` (Rust).
3.  **Units Execution** : Exécution de tous les tests unitaires.
4.  **Sandbox Test** : Le code est déployé dans un container éphémère identique à la prod pour un test d'intégration automatisé.

##  Roadmap
*   **J1** : Setup Boilerplate Pytest & Cargo Test.
*   **J2** : Création du premier set de Golden Queries.
