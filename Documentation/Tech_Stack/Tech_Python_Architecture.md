# Tech Architect: PYTHON (The Brain)

##  Rôle
Python est le langage principal pour l'**Orchestration**, l'**IA**, et la **Logique Métier**. C'est le "Système Nerveux" d'EVA.

##  Architecture des Services
Nous utilisons une architecture modulaire basée sur `FastAPI` et `LangGraph`.

### 1. Structure du Code
```
/src-python
  /eva_core          # API Gateway + LangGraph Router
  /eva_agents        # Les Experts (Banker, Muse, etc.)
     /banker
     /shadow
  /eva_shared        # Modèles Pydantic, Utils
  /tests             # Pytest
```

### 2. Frameworks Clés
*   **FastAPI** : Pour tous les endpoints HTTP internes.
    *   Pattern : Pas de logique métier dans les routes. Les routes appellent des `ServiceControllers`.
    *   Async : Utilisation stricte de `async/await` pour ne pas bloquer le thread principal lors des I/O (Database, API Broker).
*   **LangGraph** :
    *   State : Un objet `EvaState` (TypedDict) passe de nœud en nœud.
    *   Cycles : Le graphe supporte les boucles (Réflexion/Correction d'erreur).
*   **Pydantic V2** : Validation stricte des données (Typing).

### 3. Gestion des Dépendances
*   Outil : **Poetry** ou **UV** (Plus rapide).
*   Isolation : Chaque expert majeur (qui a des deps lourdes conflictuelles) tourne dans son propre **VirtualEnv** ou Container.

### 4. Directives de Code (Style Guide)
*   **Type Hinting** : OBLIGATOIRE partout (`mypy --strict`).
*   **Docstrings** : Google Style.
*   **Erreurs** : Pas de `try/except: pass`. Log every error.

##  Interaction
*   Python ne parle **JAMAIS** directement au Hardware critique (Reset Switch) ou à la Mémoire Protégée (Vault) sans passer par l'API du Kernel Rust.
*   Python **PEUT** appeler des librairies C++/Rust via `PyO3` si besoin de performance extrême.
