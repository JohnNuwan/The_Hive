"""
OpenClaw Basic Skills
Part of Sovereign Stack V3.0

Outils fondamentaux pour l'interaction avec le système de fichiers
et le web. Ces skills sont les briques de base utilisées par les
agents OpenClaw pour lire, lister et rechercher.

Skills disponibles :
    - fs_read        : Lecture sécurisée d'un fichier (read-only).
    - fs_list        : Listing du contenu d'un répertoire.
    - web_search     : Recherche web (Exa/Tavily avec fallback mock).
    - get_public_apis: Catalogue des catégories d'APIs publiques.
"""

import os
import logging
import httpx
from .registry import skill

logger = logging.getLogger(__name__)


@skill("fs_read", "Lit le contenu d'un fichier (read-only safe)")
def fs_read(path: str) -> str:
    """Lit et retourne le contenu textuel d'un fichier.

    Sécurité : lecture seule, pas de modification possible.
    Les fichiers binaires ou les erreurs d'encodage sont gérés gracieusement.

    Args:
        path: Chemin absolu ou relatif vers le fichier à lire.

    Returns:
        Le contenu du fichier, ou un message d'erreur descriptif.
    """
    if not os.path.exists(path):
        return f"Error: File '{path}' not found."
    if not os.path.isfile(path):
        return f"Error: '{path}' is not a file (directory?)."
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Limiter la taille pour éviter un context overflow
        if len(content) > 10_000:
            logger.warning(f"fs_read: File '{path}' truncated (>{10_000} chars).")
            return content[:10_000] + "\n\n[... TRUNCATED ...]"
        return content
    except UnicodeDecodeError:
        return f"Error: File '{path}' is not a text file (binary?)."
    except Exception as e:
        return f"Error reading file: {e}"


@skill("fs_list", "Liste le contenu d'un répertoire")
def fs_list(path: str = ".") -> str:
    """Liste les fichiers et sous-dossiers d'un répertoire.

    Affiche le type (FILE/DIR) et la taille pour chaque entrée.

    Args:
        path: Chemin du répertoire à lister (défaut: répertoire courant).

    Returns:
        Liste formatée des entrées, ou message d'erreur.
    """
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a directory."
    try:
        entries = []
        for item in sorted(os.listdir(path)):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                entries.append(f"  [DIR]  {item}/")
            else:
                size = os.path.getsize(full_path)
                entries.append(f"  [FILE] {item} ({size} bytes)")
        return f"Contents of '{path}' ({len(entries)} items):\n" + "\n".join(entries)
    except PermissionError:
        return f"Error: Permission denied for '{path}'."
    except Exception as e:
        return f"Error listing dir: {e}"


def _tavily_search(query: str, api_key: str) -> str:
    """Effectue une recherche via l'API Tavily."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                "https://api.tavily.com/search",
                json={"query": query},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])

            output = [f"Résultats Tavily pour '{query}':"]
            for i, res in enumerate(results[:5], 1):
                title = res.get("title", "No title")
                url = res.get("url", "#")
                content = res.get("content", "")[:200] + "..."
                output.append(f"{i}. {title}\n   URL: {url}\n   Snippet: {content}")

            return "\n\n".join(output)
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Error performing Tavily search: {str(e)}"


def _exa_search(query: str, api_key: str) -> str:
    """Effectue une recherche via l'API Exa.ai."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                "https://api.exa.ai/search",
                json={
                    "query": query,
                    "numResults": 5,
                    "contents": {"text": True}
                },
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])

            output = [f"Résultats Exa pour '{query}':"]
            for i, res in enumerate(results, 1):
                title = res.get("title", "No title")
                url = res.get("url", "#")
                text = res.get("text", "")[:200].replace("\n", " ") + "..."
                output.append(f"{i}. {title}\n   URL: {url}\n   Snippet: {text}")

            return "\n\n".join(output)
    except Exception as e:
        logger.error(f"Exa search error: {e}")
        return f"Error performing Exa search: {str(e)}"


@skill("web_search", "Recherche sur le web (Exa/Tavily avec fallback)")
def web_search(query: str) -> str:
    """Effectue une recherche web et retourne les résultats.

    Utilise Tavily API ou Exa.ai si les clés API sont présentes.
    Sinon, retourne un résultat simulé (mock).

    Args:
        query: La requête de recherche.

    Returns:
        Résultats de recherche.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY")
    exa_key = os.environ.get("EXA_API_KEY")

    if tavily_key:
        logger.info(f"web_search: using Tavily for '{query}'")
        return _tavily_search(query, tavily_key)

    if exa_key:
        logger.info(f"web_search: using Exa for '{query}'")
        return _exa_search(query, exa_key)

    logger.warning(f"web_search: No API key found (TAVILY_API_KEY or EXA_API_KEY). Using MOCK for '{query}'")
    return (
        f"[MOCK] Résultats de recherche pour '{query}':\n"
        f"1. Résultat pertinent A\n"
        f"2. Résultat pertinent B\n"
        f"3. Résultat pertinent C\n"
        f"\n(Note: Set TAVILY_API_KEY or EXA_API_KEY to enable real search)"
    )


@skill("get_public_apis", "Liste des catégories d'APIs publiques disponibles")
def get_public_apis_list() -> str:
    """Retourne les catégories d'APIs publiques disponibles.

    Sert de point d'entrée rapide avant d'utiliser le skill
    `discover_public_apis` pour une recherche détaillée.

    Returns:
        Liste des catégories disponibles.
    """
    categories = [
        "Animals", "Anime", "Anti-Malware", "Art & Design",
        "Authentication", "Blockchain", "Books", "Business",
        "Calendar", "Cloud Storage", "Cryptocurrency", "Currency Exchange",
        "Data Validation", "Development", "Entertainment", "Finance",
        "Food & Drink", "Games & Comics", "Geocoding", "Government",
        "Health", "Jobs", "Machine Learning", "Music",
        "News", "Open Data", "Science & Math", "Security",
        "Social", "Sports & Fitness", "Transportation", "Weather",
    ]
    return "Catégories d'APIs publiques:\n" + ", ".join(categories)
