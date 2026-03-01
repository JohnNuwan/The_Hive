"""
The Researcher - Agent de Recherche & Veille.
Expert I: Recherche académique, veille technologique, analyse de papers.

Fonctionnalités :
- Recherche web multi-sources (DuckDuckGo HTML).
- Synthèse de résultats via LLM.
- Veille de tendances par domaine (RSS).
- Analyse de papers ArXiv.
- Veille concurrentielle.
- Base de connaissances progressive.
- State-of-the-art tracking.

En mode Lite, Researcher utilise le LLM + web scraping.
"""

import asyncio
import hashlib
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client

from eva_researcher.services.search import ResearchService, ResearchQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════


class ResearchResult(BaseModel):
    """Résultat de recherche."""
    title: str
    source: str
    url: str | None = None
    summary: str
    relevance_score: float = 0.0


class ResearchReport(BaseModel):
    """Rapport de recherche complet."""
    query: str
    domain: str
    results: list[dict[str, Any]]
    synthesis: str
    search_time_ms: int = 0


class ArxivSearchRequest(BaseModel):
    """Requête de recherche ArXiv."""
    query: str = Field(..., min_length=3)
    max_results: int = Field(default=5, ge=1, le=20)
    category: str = Field(default="cs.AI", description="Catégorie ArXiv: cs.AI, cs.LG, q-fin, stat.ML...")


class CompetitiveIntelRequest(BaseModel):
    """Requête de veille concurrentielle."""
    target: str = Field(..., min_length=2, description="Nom de l'entreprise/produit")
    aspects: list[str] = Field(
        default=["pricing", "features", "news"],
        description="Aspects: pricing, features, news, funding, technology"
    )


class KnowledgeEntry(BaseModel):
    """Entrée dans la base de connaissances."""
    topic: str = Field(..., min_length=2)
    content: str = Field(..., min_length=10)
    domain: str = Field(default="general")
    source: str = ""
    tags: list[str] = []


class SotaEntry(BaseModel):
    """Entrée state-of-the-art."""
    task: str = Field(..., description="Tâche ML: object_detection, nlp_translation, trading_prediction...")
    method: str
    score: float
    benchmark: str = ""
    paper_url: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🔬 Démarrage The Researcher...")
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    app.state.service = ResearchService()
    app.state.knowledge_base: list[dict[str, Any]] = []
    app.state.sota_tracker: dict[str, dict[str, Any]] = {}
    app.state.research_history: deque[dict[str, Any]] = deque(maxlen=200)
    app.state.competitive_cache: dict[str, dict[str, Any]] = {}

    asyncio.create_task(hard_heartbeat())
    logger.info("✅ The Researcher prêt à explorer")
    yield
    logger.info("🛑 Arrêt The Researcher")


async def hard_heartbeat():
    try:
        redis = get_redis_client()
    except Exception:
        redis = None
    while True:
        try:
            if redis:
                payload = {
                    "status": "online",
                    "ts": datetime.now().timestamp(),
                    "expert": "researcher",
                    "knowledge_entries": len(app.state.knowledge_base),
                }
                await redis.cache_set("eva.researcher.status", payload, ttl_seconds=10)
        except Exception:
            pass
        await asyncio.sleep(2.0)


app = FastAPI(
    title="The Researcher API",
    description="Agent de Recherche, Veille & Knowledge Base - THE HIVE",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — RECHERCHE
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", tags=["Système"])
async def health():
    return {
        "status": "ok",
        "service": "researcher",
        "knowledge_entries": len(app.state.knowledge_base),
        "sota_tasks_tracked": len(app.state.sota_tracker),
    }


@app.post("/search", tags=["Recherche"])
async def search(request: ResearchQuery):
    """Lance une recherche et synthétise les résultats."""
    service: ResearchService = app.state.service
    result = await service.search(request)
    app.state.research_history.append({
        "query": request.query,
        "results_count": len(result.get("results", [])) if isinstance(result, dict) else 0,
        "timestamp": datetime.now().isoformat(),
    })
    return result


@app.get("/trends", tags=["Recherche"])
async def get_trends(domain: str = Query(default="tech")):
    """Récupère les tendances actuelles pour un domaine."""
    service: ResearchService = app.state.service
    return await service.get_trends(domain)


@app.get("/stats", tags=["Recherche"])
async def get_stats():
    """Statistiques de recherche."""
    return {
        "total_searches": len(app.state.research_history),
        "knowledge_entries": len(app.state.knowledge_base),
        "sota_tasks": len(app.state.sota_tracker),
        "competitive_reports": len(app.state.competitive_cache),
    }


@app.get("/history", tags=["Recherche"])
async def get_history(limit: int = Query(default=50, ge=1, le=200)):
    """Historique des recherches."""
    return {"history": list(app.state.research_history)[-limit:]}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — ARXIV / PAPERS
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/papers", tags=["ArXiv"])
async def search_papers(request: ArxivSearchRequest):
    """
    Recherche et résume des papers ArXiv.

    Interroge l'API ArXiv et retourne les papers les plus pertinents
    avec un résumé synthétique.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": f"cat:{request.category} AND all:{request.query}",
                    "max_results": request.max_results,
                    "sortBy": "relevance",
                }
            )
            if resp.status_code != 200:
                return {"status": "error", "message": "ArXiv API not available"}

            # Parse XML simplifié
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            papers = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                link = entry.find("atom:id", ns)
                published = entry.find("atom:published", ns)
                authors = entry.findall("atom:author/atom:name", ns)

                papers.append({
                    "title": title.text.strip() if title is not None else "",
                    "summary": (summary.text.strip()[:500] + "...") if summary is not None else "",
                    "url": link.text if link is not None else "",
                    "published": published.text if published is not None else "",
                    "authors": [a.text for a in authors[:3]],
                })

            return {
                "query": request.query,
                "category": request.category,
                "papers": papers,
                "total": len(papers),
            }
    except Exception as e:
        logger.error(f"ArXiv search error: {e}")
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — VEILLE CONCURRENTIELLE
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/competitive", tags=["Intelligence"])
async def competitive_intel(request: CompetitiveIntelRequest):
    """
    Veille concurrentielle sur une entreprise/produit.

    Combine recherche web et analyse pour produire un rapport.
    """
    service: ResearchService = app.state.service

    results = {}
    for aspect in request.aspects:
        query = f"{request.target} {aspect} 2026"
        search_result = await service.search(ResearchQuery(query=query, max_results=3))
        results[aspect] = search_result

    report = {
        "target": request.target,
        "aspects_analyzed": request.aspects,
        "findings": results,
        "generated_at": datetime.now().isoformat(),
    }

    app.state.competitive_cache[request.target] = report
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/knowledge", tags=["Knowledge"])
async def add_knowledge(entry: KnowledgeEntry):
    """Ajoute une entrée à la base de connaissances."""
    knowledge = {
        "id": f"KB-{uuid4().hex[:8].upper()}",
        **entry.model_dump(),
        "created_at": datetime.now().isoformat(),
    }
    app.state.knowledge_base.append(knowledge)
    return {"status": "added", "entry": knowledge}


@app.get("/knowledge", tags=["Knowledge"])
async def search_knowledge(q: str = Query(default=""), domain: str = Query(default="")):
    """Recherche dans la base de connaissances."""
    results = app.state.knowledge_base
    if q:
        q_lower = q.lower()
        results = [e for e in results if q_lower in e.get("topic", "").lower() or q_lower in e.get("content", "").lower()]
    if domain:
        results = [e for e in results if e.get("domain") == domain]
    return {"results": results, "total": len(results)}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — SOTA TRACKING
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/sota", tags=["SOTA"])
async def track_sota(entry: SotaEntry):
    """Enregistre un état de l'art pour une tâche donnée."""
    current = app.state.sota_tracker.get(entry.task)
    is_new_best = current is None or entry.score > current.get("score", 0)

    data = {
        **entry.model_dump(),
        "is_current_best": is_new_best,
        "updated_at": datetime.now().isoformat(),
    }

    if is_new_best:
        app.state.sota_tracker[entry.task] = data

    return {"status": "new_best" if is_new_best else "recorded", "entry": data}


@app.get("/sota", tags=["SOTA"])
async def get_sota():
    """Liste les state-of-the-art connus pour chaque tâche."""
    return {"tasks": app.state.sota_tracker, "total": len(app.state.sota_tracker)}


@app.get("/pea-analysis", tags=["Finance"])
async def get_pea_analysis():
    """Analyse fondamentale macro-économique des actions PEA ciblées."""
    return {
        "status": "info",
        "message": "Analyse PEA en cours de développement — intégration avec Researcher search",
        "targets": ["TotalEnergies", "LVMH", "Air Liquide", "Saint-Gobain"],
    }
