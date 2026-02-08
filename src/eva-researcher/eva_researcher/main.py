"""
The Researcher - Agent de Recherche & Veille
Expert I: Recherche académique, veille technologique, analyse de papers.

En mode Lite, Researcher utilise le LLM + web scraping pour :
- Rechercher et résumer des articles/papers
- Veille technologique automatisée
- Analyse de tendances marché
- Fact-checking basique
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchQuery(BaseModel):
    """Requête de recherche"""
    query: str = Field(..., min_length=3)
    domain: str = Field(default="general", description="Domain: finance, tech, science, crypto, general")
    depth: str = Field(default="quick", description="Depth: quick, deep")
    max_results: int = Field(default=5, ge=1, le=20)


class ResearchResult(BaseModel):
    """Résultat de recherche"""
    title: str
    source: str
    url: str | None = None
    summary: str
    relevance_score: float = 0.0


class ResearchReport(BaseModel):
    """Rapport de recherche complet"""
    query: str
    domain: str
    results: list[ResearchResult]
    synthesis: str
    timestamp: datetime = Field(default_factory=datetime.now)
    search_time_ms: int = 0


class TrendReport(BaseModel):
    """Rapport de tendances"""
    domain: str
    trends: list[dict[str, Any]]
    analysis: str
    timestamp: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchService:
    """Service de recherche et veille"""

    # Sources de veille par domaine
    RSS_SOURCES: dict[str, list[dict[str, str]]] = {
        "finance": [
            {"name": "Bloomberg", "url": "https://www.bloomberg.com"},
            {"name": "Reuters", "url": "https://www.reuters.com"},
            {"name": "Financial Times", "url": "https://www.ft.com"},
        ],
        "tech": [
            {"name": "Hacker News", "url": "https://news.ycombinator.com"},
            {"name": "TechCrunch", "url": "https://techcrunch.com"},
            {"name": "ArXiv CS", "url": "https://arxiv.org/list/cs.AI/recent"},
        ],
        "crypto": [
            {"name": "CoinDesk", "url": "https://www.coindesk.com"},
            {"name": "The Block", "url": "https://www.theblock.co"},
            {"name": "DeFi Llama", "url": "https://defillama.com"},
        ],
    }

    def __init__(self):
        self.settings = get_settings()
        self.search_count = 0
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0 THE-HIVE-Researcher/1.0"}
        )

    async def search(self, request: ResearchQuery) -> ResearchReport:
        """Effectue une recherche et synthétise les résultats"""
        start = datetime.now()
        self.search_count += 1

        # 1. Recherche web via DuckDuckGo
        web_results = await self._web_search(request.query, request.max_results)

        # 2. Synthèse via LLM si résultats trouvés
        synthesis = await self._synthesize(request.query, web_results)

        elapsed = int((datetime.now() - start).total_seconds() * 1000)

        return ResearchReport(
            query=request.query,
            domain=request.domain,
            results=web_results,
            synthesis=synthesis,
            search_time_ms=elapsed
        )

    async def _web_search(self, query: str, max_results: int = 5) -> list[ResearchResult]:
        """Recherche web via DuckDuckGo HTML"""
        try:
            url = f"https://html.duckduckgo.com/html/?q={query}"
            response = await self._client.get(url)

            if response.status_code != 200:
                return []

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            for link in soup.find_all("a", class_="result__a")[:max_results]:
                snippet_el = link.find_next("a", class_="result__snippet")
                snippet = snippet_el.get_text() if snippet_el else ""

                results.append(ResearchResult(
                    title=link.get_text().strip(),
                    source="DuckDuckGo",
                    url=link.get("href"),
                    summary=snippet.strip(),
                    relevance_score=0.8
                ))

            return results
        except Exception as e:
            logger.error(f"Erreur recherche web: {e}")
            return []

    async def _synthesize(self, query: str, results: list[ResearchResult]) -> str:
        """Synthétise les résultats via LLM"""
        if not results:
            return "Aucun résultat trouvé pour cette recherche."

        context = "\n".join([
            f"- {r.title}: {r.summary}" for r in results
        ])

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{self.settings.ollama_host}:{self.settings.ollama_port}/api/generate",
                    json={
                        "model": self.settings.ollama_model,
                        "prompt": f"Synthétise les résultats de recherche suivants sur '{query}':\n\n{context}\n\nFais une synthèse concise et actionnable en 3-5 phrases.",
                        "stream": False
                    }
                )
                data = response.json()
                return data.get("response", "Synthèse indisponible")
        except Exception as e:
            logger.warning(f"LLM non disponible pour synthèse: {e}")
            return f"Synthèse automatique indisponible. {len(results)} résultats trouvés pour '{query}'."

    async def get_trends(self, domain: str = "tech") -> TrendReport:
        """Récupère les tendances actuelles pour un domaine"""
        sources = self.RSS_SOURCES.get(domain, self.RSS_SOURCES["tech"])

        # En mode lite, on fait une recherche des trending topics
        results = await self._web_search(f"trending {domain} news today 2026", 5)

        trends = [
            {"title": r.title, "source": r.source, "url": r.url}
            for r in results
        ]

        analysis = await self._synthesize(f"tendances {domain} actuelles", results)

        return TrendReport(
            domain=domain,
            trends=trends,
            analysis=analysis
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Researcher"""
    logger.info("🔬 Démarrage The Researcher (Veille & Analyse)...")

    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    app.state.research = ResearchService()
    asyncio.create_task(hard_heartbeat())

    logger.info("✅ The Researcher est en veille active")
    yield
    logger.info("🛑 Arrêt The Researcher")


async def hard_heartbeat():
    """Signal de présence"""
    redis = get_redis_client()
    while True:
        try:
            payload = {"status": "online", "ts": datetime.now().timestamp(), "expert": "researcher"}
            await redis.cache_set("eva.researcher.status", payload, ttl_seconds=10)
        except Exception:
            pass
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="The Researcher API",
    description="Agent de Recherche & Veille - THE HIVE",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "service": "researcher"}


@app.post("/search", response_model=ResearchReport)
async def search(request: ResearchQuery):
    """Lance une recherche et synthétise les résultats"""
    service: ResearchService = app.state.research
    return await service.search(request)


@app.get("/trends", response_model=TrendReport)
async def get_trends(domain: str = Query(default="tech")):
    """Récupère les tendances actuelles pour un domaine"""
    service: ResearchService = app.state.research
    return await service.get_trends(domain)


@app.get("/stats")
async def get_stats():
    """Statistiques de recherche"""
    service: ResearchService = app.state.research
    return {
        "total_searches": service.search_count,
        "available_domains": list(service.RSS_SOURCES.keys()),
    }
