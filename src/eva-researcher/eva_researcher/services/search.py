"""
Service de recherche et de veille pour EVA Researcher.

Ce service conserve la recherche a la demande et ajoute une integration
systematique avec la file de revue des connaissances collectees.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel, Field
from shared import get_settings

from eva_researcher.services.ingestion import (
    AutoApproveRequest,
    KnowledgeIngestionService,
    ReviewDecisionRequest,
    SyncSourcesRequest,
)

logger = logging.getLogger(__name__)


class ResearchQuery(BaseModel):
    """Represente une recherche a la demande."""

    query: str = Field(..., min_length=3)
    domain: str = Field(default="general", description="Domaine logique de la recherche")
    depth: str = Field(default="quick", description="Profondeur: quick ou deep")
    max_results: int = Field(default=5, ge=1, le=20)


class ResearchService:
    """Fournit la recherche web, arXiv et la collecte de connaissance."""

    def __init__(self, ingestion_service: KnowledgeIngestionService | None = None) -> None:
        self.settings = get_settings()
        self.ingestion = ingestion_service or KnowledgeIngestionService()
        self.search_count = 0
        logger.info("ResearchService initialise")

    async def start_background_tasks(self) -> None:
        """Demarre les collectes planifiees de connaissance."""
        await self.ingestion.start_background_tasks()

    async def stop_background_tasks(self) -> None:
        """Arrete les collectes planifiees de connaissance."""
        await self.ingestion.stop_background_tasks()

    async def search(self, request: ResearchQuery) -> dict[str, Any]:
        """Effectue une recherche web puis synthese les resultats."""
        import time

        start = time.time()
        self.search_count += 1
        results = await self._web_search(request.query, request.max_results)
        synthesis = await self._synthesize(request.query, results)
        queue_summary = await self.ingestion.queue_manual_search(
            query=request.query,
            domain=request.domain,
            results=results,
            synthesis=synthesis,
        )
        elapsed = int((time.time() - start) * 1000)
        return {
            "query": request.query,
            "domain": request.domain,
            "results": results,
            "synthesis": synthesis,
            "search_time_ms": elapsed,
            "timestamp": datetime.now().isoformat(),
            "review_queue": queue_summary,
        }

    async def search_papers(self, query: str, category: str, max_results: int) -> dict[str, Any]:
        """Interroge arXiv et met les papiers en file de revue."""
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            response = await client.get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": f"cat:{category} AND all:{query}",
                    "max_results": max_results,
                    "sortBy": "relevance",
                },
            )
            response.raise_for_status()

        root = ET.fromstring(response.text)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        papers: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", namespace):
            title = entry.findtext("atom:title", default="", namespaces=namespace).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=namespace).strip()
            papers.append(
                {
                    "title": title,
                    "summary": (summary[:500] + "...") if len(summary) > 500 else summary,
                    "url": entry.findtext("atom:id", default="", namespaces=namespace),
                    "published": entry.findtext("atom:published", default="", namespaces=namespace),
                    "authors": [
                        author.text.strip()
                        for author in entry.findall("atom:author/atom:name", namespace)
                        if author.text
                    ],
                }
            )
        queue_summary = await self.ingestion.queue_arxiv_results(query=query, category=category, papers=papers)
        return {
            "query": query,
            "category": category,
            "papers": papers,
            "total": len(papers),
            "review_queue": queue_summary,
        }

    async def get_trends(self, domain: str = "tech") -> dict[str, Any]:
        """Retourne les sources de veille utiles par domaine."""
        sources = {
            "finance": ["Reuters", "Bloomberg", "Google News Marches"],
            "tech": ["TechCrunch", "Google News IA", "arXiv"],
            "crypto": ["CoinDesk", "Cointelegraph", "Google News Marches"],
            "science": ["arXiv", "Nature", "Google News IA"],
        }
        selected = sources.get(domain, sources["tech"])
        ingest_sources = await self.ingestion.get_sources()
        return {
            "domain": domain,
            "sources": selected,
            "ingest_sources": ingest_sources.get("sources", []),
            "message": f"Veille {domain}: {len(selected)} sources de reference.",
            "timestamp": datetime.now().isoformat(),
        }

    async def sync_sources(self, request: SyncSourcesRequest | None = None) -> dict[str, Any]:
        """Declenche une synchronisation manuelle des sources."""
        return await self.ingestion.sync_sources(request)

    async def get_ingestion_status(self, tail: int = 30) -> dict[str, Any]:
        """Retourne l'etat global de la pipeline d'ingestion."""
        return await self.ingestion.get_status(tail=tail)

    async def list_review_items(
        self,
        review_status: str = "pending",
        limit: int = 50,
        offset: int = 0,
        source_key: str | None = None,
        family: str | None = None,
        trust_level: str | None = None,
        review_mode: str | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        """Retourne les candidats en revue selon un statut donne."""
        return await self.ingestion.list_review_items(
            review_status=review_status,
            limit=limit,
            offset=offset,
            source_key=source_key,
            family=family,
            trust_level=trust_level,
            review_mode=review_mode,
            search=search,
        )

    async def approve_review_item(self, item_id: str, decision: ReviewDecisionRequest) -> dict[str, Any]:
        """Approuve un candidat puis l'ingere durablement."""
        return await self.ingestion.approve_item(item_id, decision)

    async def reject_review_item(self, item_id: str, decision: ReviewDecisionRequest) -> dict[str, Any]:
        """Rejette un candidat de la file de revue."""
        return await self.ingestion.reject_item(item_id, decision)

    async def retry_review_item_ingestion(self, item_id: str, reviewed_by: str = "manual:retry") -> dict[str, Any]:
        """Relance l'ingestion durable d'un candidat en erreur."""

        return await self.ingestion.retry_failed_ingestion(item_id, reviewed_by=reviewed_by)

    async def auto_approve_review_items(self, request: AutoApproveRequest | None = None) -> dict[str, Any]:
        """Applique les politiques d'auto-approbation sur la file de revue."""

        return await self.ingestion.auto_approve_pending_items(request)

    async def list_approved_items(self, limit: int = 50) -> dict[str, Any]:
        """Retourne les derniers elements approuves et ingeres."""
        return await self.ingestion.list_approved_items(limit=limit)

    async def get_sources(self) -> dict[str, Any]:
        """Retourne la configuration des sources et leurs compteurs."""
        return await self.ingestion.get_sources()

    async def _web_search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Recherche web via DuckDuckGo HTML."""
        results: list[dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if response.status_code == 200:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(response.text, "html.parser")
                    for index, result_div in enumerate(soup.select(".result")):
                        if index >= max_results:
                            break
                        link = result_div.select_one(".result__a")
                        snippet = result_div.select_one(".result__snippet")
                        if link:
                            results.append(
                                {
                                    "title": link.get_text(strip=True),
                                    "url": link.get("href", ""),
                                    "summary": snippet.get_text(strip=True) if snippet else "",
                                    "source": "DuckDuckGo",
                                    "relevance_score": round(1.0 - (index * 0.1), 2),
                                }
                            )
        except Exception as exc:
            logger.warning("Echec de recherche web: %s", exc)
        return results

    async def _synthesize(self, query: str, results: list[dict[str, Any]]) -> str:
        """Synthesise les resultats via le backend LLM configure."""
        if not results:
            return "Aucun resultat trouve."

        summaries = "\n".join(
            f"- {result.get('title', 'N/A')}: {result.get('summary', 'N/A')}"
            for result in results[:5]
        )
        prompt = (
            f"Synthese de recherche pour '{query}'.\n\n"
            f"Resultats:\n{summaries}\n\n"
            "Produis une synthese concise en francais avec les idees cles et les risques."
        )

        try:
            backend = getattr(self.settings, "llm_backend", "vllm").strip().lower()
            if backend == "vllm":
                endpoint = f"http://{self.settings.vllm_host}:{self.settings.vllm_port}/v1/chat/completions"
                payload = {
                    "model": getattr(
                        self.settings,
                        "council_model_research",
                        getattr(self.settings, "vllm_model", "Qwen/Qwen2.5-1.5B-Instruct"),
                    ),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 384,
                }
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(endpoint, json=payload)
                    if response.status_code == 200:
                        choices = response.json().get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            return content.strip() or "Synthese non disponible."
            endpoint = f"http://{self.settings.ollama_host}:{self.settings.ollama_port}/api/generate"
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    endpoint,
                    json={"model": self.settings.ollama_model, "prompt": prompt, "stream": False},
                )
                if response.status_code == 200:
                    return response.json().get("response", "Synthese non disponible.")
        except Exception as exc:
            logger.debug("Echec synthese LLM: %s", exc)
        return "Synthese automatique non disponible (LLM indisponible)."
