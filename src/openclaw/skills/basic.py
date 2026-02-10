"""
OpenClaw Basic Skills
Part of Sovereign Stack V3.0

Outils fondamentaux pour l'agent.
"""

import os
from .registry import skill

@skill("fs_read", "Lit le contenu d'un fichier (read-only safe)")
def fs_read(path: str) -> str:
    if not os.path.exists(path):
        return f"Error: File {path} not found."
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@skill("fs_list", "Liste le contenu d'un répertoire")
def fs_list(path: str = ".") -> str:
    try:
        items = os.listdir(path)
        return "\n".join(items)
    except Exception as e:
        return f"Error listing dir: {e}"

@skill("web_search", "Recherche sur le web (Placeholder Exa/Tavily)")
def web_search(query: str) -> str:
    # TODO: Intégrer Exa.ai ou Tavily API
    return f"[MOCK] Résultats de recherche pour '{query}':\n1. Résultat pertinent A\n2. Résultat pertinent B"

@skill("get_public_apis", "Liste des catégories d'APIs publiques disponibles")
def get_public_apis_list() -> str:
    # Placeholder pour l'intégration future du repo public-apis
    return "Animals, Anime, Anti-Malware, Art & Design, Authentication, Blockchain, Books..."
