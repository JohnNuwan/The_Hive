"""
OpenClaw Basic Skills
Part of Sovereign Stack V3.0

Outils fondamentaux pour l'interaction avec le système de fichiers
et le web. Ces skills sont les briques de base utilisées par les
agents OpenClaw pour lire, lister et rechercher.

Skills disponibles :
    - fs_read        : Lecture sécurisée d'un fichier (read-only).
    - fs_list        : Listing du contenu d'un répertoire.
    - web_search     : Recherche web (Tavily API).
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


@skill("web_search", "Recherche sur le web (Tavily API)")
def web_search(query: str) -> str:
    """Effectue une recherche web et retourne les résultats.

    Utilise Tavily API si la clé est présente, sinon retourne un mock.

    Args:
        query: La requête de recherche.

    Returns:
        Résultats de recherche formatés.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        logger.warning("TAVILY_API_KEY not found. Using MOCK results.")
        return (
            f"[MOCK] Résultats de recherche pour '{query}' (API Key missing):\n"
            f"1. Résultat pertinent A\n"
            f"2. Résultat pertinent B\n"
            f"3. Résultat pertinent C"
        )

    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 5
            },
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return f"No results found for '{query}'."

        formatted_results = [f"Search results for '{query}':"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "No Title")
            url = result.get("url", "#")
            # Truncate content specifically to keep output concise
            content = result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", "")
            formatted_results.append(f"{i}. {title}\n   URL: {url}\n   Content: {content}\n")

        return "\n".join(formatted_results)

    except httpx.RequestError as e:
        logger.error(f"Tavily API request failed: {e}")
        return f"Error performing web search: {e}"
    except httpx.HTTPStatusError as e:
        logger.error(f"Tavily API returned error: {e.response.status_code} - {e.response.text}")
        return f"Error: Web search API returned {e.response.status_code}."
    except Exception as e:
        logger.error(f"Unexpected error in web_search: {e}")
        return f"Unexpected error during search: {e}"


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
