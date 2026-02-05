# Expert E: THE BUILDER (L'Architecte Système)

## 1. Identité & Rôle
*   **Nom** : The Builder
*   **Rôle** : DevOps, Maintenance, Auto-Coding.
*   **Type** : Permanent (En tâche de fond).

## 2. Architecture Technique
*   **Modèle LLM** : `DeepSeek-Coder-33B` (Si RAM permet) ou `CodeLlama-13B`.
*   **Accès** :
    *   System : SSH, Docker Socket.
    *   Filesystem : Accès R/W sur `/app/src`.

## 3. Fonctions Clés
### A. The Librarian (Documentation)
*   Dès qu'un fichier `.py` est modifié, The Builder génère/met à jour son fichier `.md` associé.
*   Vérifie que les Docstrings sont à jour.

### B. The Handyman (Maintenance)
*   Surveille les logs d'erreurs (Sentry/Log file).
*   Si une StackTrace est détectée :
    1.  Analyse l'erreur.
    2.  Localise le fichier coupable.
    3.  Propose un `git patch`.

### C. The Refinery (Optimisation)
*   Analyse statique (Ruff) hebdomadaire.
*   Refactorise le code spaghetti.

## 4. Sécurité (The Sandbox)
*   Tout code généré par The Builder doit être testé dans **CT 500 (Arena)** avant d'être mergé sur Main.
*   Pas d'accès direct au Kernel Rust.

## 5. Roadmap Dév
*   **J1** : Script `doc_gen.py` (Simple parsing AST -> Markdown).
*   **J2** : Integration git (Auto-commit des docs).
