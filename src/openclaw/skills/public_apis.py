"""
OpenClaw Skill: Public APIs Crawler
Part of Sovereign Stack V3.0

Permet à l'agent de découvrir des outils gratuits via le repository
public-apis (https://github.com/public-apis/public-apis).

Ce skill utilise l'API publique publicapis.org quand disponible,
avec un fallback sur une base statique intégrée.

Skills disponibles :
    - discover_public_apis : Recherche d'APIs par catégorie.
"""

import httpx
import logging
from typing import Optional, List, Dict
from .registry import skill

logger = logging.getLogger(__name__)

# URL de l'API publique (peut être instable)
PUBLIC_APIS_URL = "https://api.publicapis.org/entries"


# ═══════════════════════════════════════════════════════════════════════════════
# BASE STATIQUE (Fallback quand l'API est down)
# ═══════════════════════════════════════════════════════════════════════════════

OFFLINE_DATABASE: List[Dict[str, str]] = [
    {"API": "CoinGecko", "Category": "Cryptocurrency", "Link": "https://www.coingecko.com/en/api", "Description": "Crypto market data"},
    {"API": "Binance", "Category": "Cryptocurrency", "Link": "https://binance-docs.github.io/apidocs/", "Description": "Exchange data & trading"},
    {"API": "OpenWeather", "Category": "Weather", "Link": "https://openweathermap.org/api", "Description": "Weather forecasts & current data"},
    {"API": "Cat Facts", "Category": "Animals", "Link": "https://alexwohlbruck.github.io/cat-facts/", "Description": "Daily random cat facts"},
    {"API": "REST Countries", "Category": "Geocoding", "Link": "https://restcountries.com", "Description": "Country information"},
    {"API": "JSONPlaceholder", "Category": "Development", "Link": "https://jsonplaceholder.typicode.com", "Description": "Fake API for testing"},
    {"API": "NewsAPI", "Category": "News", "Link": "https://newsapi.org", "Description": "News headlines & articles"},
    {"API": "PokeAPI", "Category": "Games & Comics", "Link": "https://pokeapi.co", "Description": "Pokémon data"},
    {"API": "ExchangeRate-API", "Category": "Finance", "Link": "https://www.exchangerate-api.com", "Description": "Currency exchange rates"},
    {"API": "GitHub", "Category": "Development", "Link": "https://docs.github.com/en/rest", "Description": "GitHub repository data"},
]


@skill("discover_public_apis", "Cherche une API publique gratuite par catégorie (ex: 'Finance', 'Animals')")
async def discover_public_apis(category: Optional[str] = None) -> str:
    """Récupère une liste d'APIs publiques gratuites.

    Tente d'abord l'API en ligne (publicapis.org), puis bascule
    sur la base statique intégrée en cas d'échec.

    Args:
        category: Filtre optionnel par catégorie (ex: "Finance", "Weather").
                  Si None, retourne toutes les APIs disponibles.

    Returns:
        Liste formatée des APIs trouvées avec nom, lien et description.
    """
    try:
        return await _fetch_online(category)
    except Exception as e:
        logger.warning(f"Public APIs online fetch failed: {e}. Using offline backup.")
        return _offline_backup(category)


async def _fetch_online(category: Optional[str] = None) -> str:
    """Requête l'API publicapis.org pour les données en ligne.

    Args:
        category: Filtre optionnel par catégorie.

    Returns:
        Liste formatée des APIs.

    Raises:
        Exception: Si la requête HTTP échoue.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(PUBLIC_APIS_URL, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        entries = data.get("entries", [])

        if category:
            entries = [
                e for e in entries
                if category.lower() in e.get("Category", "").lower()
            ]

        # Limiter pour ne pas flooder le contexte LLM
        top_entries = entries[:10]

        result = f"Found {len(entries)} APIs (showing top {len(top_entries)}):\n"
        for api in top_entries:
            result += f"  - {api['API']} ({api['Link']}): {api['Description']}\n"

        return result


def _offline_backup(category: Optional[str] = None) -> str:
    """Retourne les APIs depuis la base statique intégrée.

    Utilisé comme fallback quand l'API publique est indisponible.

    Args:
        category: Filtre optionnel par catégorie.

    Returns:
        Liste formatée des APIs du backup.
    """
    entries = OFFLINE_DATABASE

    if category:
        entries = [
            e for e in entries
            if category.lower() in e.get("Category", "").lower()
        ]

    result = f"[OFFLINE MODE] {len(entries)} APIs found:\n"
    for api in entries:
        result += f"  - {api['API']} ({api['Link']}): {api['Description']}\n"
    return result
