"""
Search Service — Moteur de recherche et synthèse.

Fournit les fonctionnalités de base de recherche :
- Recherche web via DuckDuckGo HTML.
- Synthèse des résultats via LLM.
- Veille de tendances via RSS.
"""

import logging
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field
from shared import get_settings

logger = logging.getLogger(__name__)


class ResearchQuery(BaseModel):
    """Requête de recherche."""
    query: str = Field(..., min_length=3)
    domain: str = Field(default="general", description="Domain: finance, tech, science, crypto, general")
    depth: str = Field(default="quick", description="Depth: quick, deep")
    max_results: int = Field(default=5, ge=1, le=20)


class ResearchService:
    """Service de recherche et veille."""

    RSS_SOURCES: dict[str, list[dict[str, str]]] = {
        "finance": [
            {"name": "Bloomberg", "url": "https://www.bloomberg.com"},
            {"name": "Reuters", "url": "https://www.reuters.com"},
        ],
        "tech": [
            {"name": "TechCrunch", "url": "https://techcrunch.com"},
            {"name": "Ars Technica", "url": "https://arstechnica.com"},
        ],
        "crypto": [
            {"name": "CoinDesk", "url": "https://www.coindesk.com"},
            {"name": "Cointelegraph", "url": "https://cointelegraph.com"},
        ],
        "science": [
            {"name": "Nature", "url": "https://www.nature.com"},
            {"name": "arXiv", "url": "https://arxiv.org"},
        ],
    }

    def __init__(self):
        self.settings = get_settings()
        self.search_count = 0
        logger.info("🔬 ResearchService initialisé")

    async def search(self, request: ResearchQuery) -> dict[str, Any]:
        """Effectue une recherche et synthétise les résultats."""
        import time
        start = time.time()
        self.search_count += 1

        results = await self._web_search(request.query, request.max_results)
        synthesis = await self._synthesize(request.query, results)

        elapsed = int((time.time() - start) * 1000)
        return {
            "query": request.query,
            "domain": request.domain,
            "results": results,
            "synthesis": synthesis,
            "search_time_ms": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

    async def _web_search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Recherche web via DuckDuckGo HTML."""
        results = []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for i, result_div in enumerate(soup.select(".result")):
                        if i >= max_results:
                            break
                        link = result_div.select_one(".result__a")
                        snippet = result_div.select_one(".result__snippet")
                        if link:
                            results.append({
                                "title": link.get_text(strip=True),
                                "url": link.get("href", ""),
                                "summary": snippet.get_text(strip=True) if snippet else "",
                                "source": "DuckDuckGo",
                                "relevance_score": round(1.0 - (i * 0.1), 2),
                            })
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
        return results

    async def _synthesize(self, query: str, results: list[dict[str, Any]]) -> str:
        """Synthétise les résultats via LLM."""
        if not results:
            return "Aucun résultat trouvé."
        try:
            settings = self.settings
            ollama_host = getattr(settings, "OLLAMA_HOST", "localhost")
            ollama_port = getattr(settings, "OLLAMA_PORT", 11434)
            model = getattr(settings, "DEFAULT_EXPERT_MODEL", "gemma3:4b")

            summaries = "\n".join(
                f"- {r.get('title', 'N/A')}: {r.get('summary', 'N/A')}"
                for r in results[:5]
            )
            prompt = f"Synthesize these search results for the query '{query}':\n{summaries}\n\nProvide a concise synthesis in 3-4 sentences."

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"http://{ollama_host}:{ollama_port}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "Synthèse non disponible.")
        except Exception as e:
            logger.debug(f"LLM synthesis failed: {e}")
        return "Synthèse automatique non disponible (LLM offline)."

    async def get_trends(self, domain: str = "tech") -> dict[str, Any]:
        """Récupère les tendances actuelles pour un domaine."""
        sources = self.RSS_SOURCES.get(domain, self.RSS_SOURCES.get("tech", []))
        return {
            "domain": domain,
            "sources": sources,
            "message": f"Veille {domain} — {len(sources)} sources surveillées",
            "timestamp": datetime.now().isoformat(),
        }
