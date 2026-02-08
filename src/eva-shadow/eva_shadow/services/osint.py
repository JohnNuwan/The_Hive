"""
Service OSINT — Recherche Web et Scraping pour The Shadow.

Implémente les capacités de renseignement de l'Expert C :
- Recherche web rapide via DuckDuckGo (HTML parsing).
- Reconnaissance d'entités (Entity Recon) avec Threat Intel simulé.
- En production, peut se connecter à Brave Search API, Shodan,
  VirusTotal et AlienVault OTX pour des données enrichies.
"""

import logging
from datetime import datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class OSINTService:
    """
    Service de recherche et d'investigation web.

    Utilise httpx pour les requêtes HTTP asynchrones et BeautifulSoup
    pour le parsing HTML. En mode lite, scrape DuckDuckGo directement.

    Attributes:
        _client: Client HTTP asynchrone avec User-Agent personnalisé.
    """

    def __init__(self):
        """Initialise le client HTTP avec un User-Agent spécifique."""
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) THE-HIVE/1.0"
            },
        )

    async def quick_search(self, query: str) -> list[dict[str, Any]]:
        """
        Recherche web rapide via DuckDuckGo (HTML parsing).

        En production, remplacer par Brave Search API ou Google Serper
        pour des résultats plus fiables et structurés.

        Args:
            query: Requête de recherche textuelle.

        Returns:
            list[dict]: Liste de résultats avec title, url et snippet.
        """
        logger.info(f"Recherche rapide: {query}")

        try:
            url = f"https://html.duckduckgo.com/html/?q={query}"
            response = await self._client.get(url)

            if response.status_code != 200:
                logger.warning(f"DDG error: {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            for link in soup.find_all("a", class_="result__a")[:5]:
                results.append(
                    {
                        "title": link.get_text().strip(),
                        "url": link.get("href"),
                        "snippet": "",
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Erreur recherche: {e}")
            return []

    async def entity_recon(self, target: str) -> dict[str, Any]:
        """
        Effectue une reconnaissance complète sur une entité.

        Combine une recherche web avec une analyse Threat Intel
        simulée (en production, utiliser les APIs de sécurité).

        Args:
            target: Cible de la reconnaissance (nom, domaine, IP).

        Returns:
            dict: Rapport complet avec web_findings et threat_intel.
        """
        logger.info(f"Reconnaissance cible: {target}")

        # 1. Recherche web
        web_results = await self.quick_search(target)

        # 2. Analyse Threat Intel (simulée — remplacer par API réelle en prod)
        threat_intel = (
            {
                "is_malicious": False,
                "reputation_score": 85,
                "sources": ["AlienVault", "VirusTotal"],
            }
            if "." in target
            else {}
        )

        return {
            "target": target,
            "summary": f"Rapport d'investigation pour {target}",
            "web_findings": web_results,
            "threat_intel": threat_intel,
            "timestamp": datetime.now().isoformat(),
        }
