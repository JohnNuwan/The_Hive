"""
Search Service â€” Moteur de recherche et synthÃ¨se.

Fournit les fonctionnalitÃ©s de base de recherche :
- Recherche web via DuckDuckGo HTML.
- SynthÃ¨se des rÃ©sultats via LLM.
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
    """RequÃªte de recherche."""
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
        logger.info("ðŸ”¬ ResearchService initialisÃ©")

    async def search(self, request: ResearchQuery) -> dict[str, Any]:
        """Effectue une recherche et synthÃ©tise les rÃ©sultats."""
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
            logger.warning(f"Echec de recherche web: {e}")
        return results

    async def _synthesize(self, query: str, results: list[dict[str, Any]]) -> str:
        """Synthesise les resultats via le backend LLM configure."""
        if not results:
            return "Aucun resultat trouve."

        summaries = "\n".join(
            f"- {r.get('title', 'N/A')}: {r.get('summary', 'N/A')}"
            for r in results[:5]
        )
        prompt = (
            f"Synthese de recherche pour '{query}'.\n\n"
            f"Resultats:\n{summaries}\n\n"
            "Produis une synthese concise en francais (3-4 phrases) avec les idees cles."
        )

        try:
            llm_backend = getattr(self.settings, "llm_backend", "vllm").strip().lower()

            if llm_backend == "vllm":
                host = getattr(self.settings, "vllm_host", "localhost")
                port = getattr(self.settings, "vllm_port", 8000)
                model = getattr(
                    self.settings,
                    "council_model_research",
                    getattr(self.settings, "vllm_model", "Qwen/Qwen2.5-1.5B-Instruct"),
                )

                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 384,
                }
                endpoint = f"http://{host}:{port}/v1/chat/completions"

                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(endpoint, json=payload)
                    if resp.status_code == 200:
                        choices = resp.json().get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            return content.strip() or "Synthese non disponible."
            else:
                host = getattr(self.settings, "ollama_host", "localhost")
                port = getattr(self.settings, "ollama_port", 11434)
                model = getattr(self.settings, "ollama_model", "gemma3:4b")
                endpoint = f"http://{host}:{port}/api/generate"

                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        endpoint,
                        json={"model": model, "prompt": prompt, "stream": False},
                    )
                    if resp.status_code == 200:
                        return resp.json().get("response", "Synthese non disponible.")
        except Exception as exc:
            logger.debug("Echec synthese LLM: %s", exc)

        return "Synthese automatique non disponible (LLM indisponible)."

    async def get_trends(self, domain: str = "tech") -> dict[str, Any]:
        """RÃ©cupÃ¨re les tendances actuelles pour un domaine."""
        sources = self.RSS_SOURCES.get(domain, self.RSS_SOURCES.get("tech", []))
        return {
            "domain": domain,
            "sources": sources,
            "message": f"Veille {domain} â€” {len(sources)} sources surveillÃ©es",
            "timestamp": datetime.now().isoformat(),
        }


