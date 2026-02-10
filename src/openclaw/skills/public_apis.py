"""
OpenClaw Skill: Public APIs Crawler
Part of Sovereign Stack V3.0

Permet à l'agent de découvrir des outils gratuits via le repository public-apis.
"""

import httpx
import logging
from .registry import skill

logger = logging.getLogger(__name__)

PUBLIC_APIS_URL = "https://api.publicapis.org/entries" # Fallback
# Note: L'API officielle est souvent down, on pourrait parser le README du repo GitHub si besoin.
# Pour ce MVP on tente l'API standard ou une liste statique de secours.

@skill("discover_public_apis", "Cherche une API publique gratuite par catégorie (ex: 'Finance', 'Animals')")
async def discover_public_apis(category: str = None) -> str:
    """
    Récupère une liste d'APIs gratuites.
    Si category est spécifié, filtre les résultats.
    """
    async with httpx.AsyncClient() as client:
        try:
            # On tente l'API (souvent instable, timeout court)
            response = await client.get(PUBLIC_APIS_URL, timeout=5.0)
            response.raise_for_status()
            data = response.json()
            entries = data.get("entries", [])
            
            if category:
                entries = [e for e in entries if category.lower() in e.get("Category", "").lower()]
            
            # Limite pour ne pas flooder le contexte
            top_entries = entries[:10] 
            
            result = f"Found {len(entries)} APIs (showing top {len(top_entries)}):\n"
            for api in top_entries:
                result += f"- {api['API']} ({api['Link']}): {api['Description']}\n"
                
            return result

        except Exception as e:
            logger.warning(f"Public APIs fetch failed: {e}. Using offline backup.")
            return _offline_backup(category)

def _offline_backup(category: str = None) -> str:
    """Backup statique quand l'API est down"""
    backup_db = [
        {"API": "CoinGecko", "Category": "Cryptocurrency", "Link": "https://www.coingecko.com/en/api", "Description": "Crypto prices"},
        {"API": "Binance", "Category": "Cryptocurrency", "Link": "https://binance-docs.github.io/apidocs/", "Description": "Exchange data"},
        {"API": "OpenWeather", "Category": "Weather", "Link": "https://openweathermap.org/api", "Description": "Weather data"},
        {"API": "Cat Facts", "Category": "Animals", "Link": "https://alexwohlbruck.github.io/cat-facts/", "Description": "Daily cat facts"},
    ]
    
    if category:
        backup_db = [e for e in backup_db if category.lower() in e.get("Category", "").lower()]
        
    result = "[OFFLINE MODE] Public APIs:\n"
    for api in backup_db:
        result += f"- {api['API']} ({api['Link']}): {api['Description']}\n"
    return result
