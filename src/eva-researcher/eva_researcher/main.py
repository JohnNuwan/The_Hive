"""
API principale de The Researcher.

Ce service centralise:
- la recherche a la demande;
- la veille academique et actualite;
- la file de revue de connaissance;
- quelques endpoints historiques de knowledge base et de SOTA.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared.redis_client import get_redis_client, init_redis

from eva_researcher.services.ingestion import ReviewDecisionRequest, SyncSourcesRequest
from eva_researcher.services.search import ResearchQuery, ResearchService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivSearchRequest(BaseModel):
    """Represente une recherche arXiv a la demande."""

    query: str = Field(..., min_length=3)
    max_results: int = Field(default=5, ge=1, le=20)
    category: str = Field(default="cs.AI", description="Categorie arXiv ciblee")


class CompetitiveIntelRequest(BaseModel):
    """Represente une demande de veille concurrentielle."""

    target: str = Field(..., min_length=2, description="Nom de l'entreprise ou du produit")
    aspects: list[str] = Field(default_factory=lambda: ["pricing", "features", "news"])


class KnowledgeEntry(BaseModel):
    """Represente une entree ajoutee manuellement a la base interne."""

    topic: str = Field(..., min_length=2)
    content: str = Field(..., min_length=10)
    domain: str = Field(default="general")
    source: str = ""
    tags: list[str] = Field(default_factory=list)


class SotaEntry(BaseModel):
    """Represente une entree state of the art."""

    task: str = Field(..., description="Tache ciblee")
    method: str
    score: float
    benchmark: str = ""
    paper_url: str = ""


async def _heartbeat_loop(app: FastAPI) -> None:
    """Publie un heartbeat leger dans Redis tant que le service est actif."""
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
                await redis.set("eva.researcher.status", payload)
        except Exception:
            pass
        await asyncio.sleep(2.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise le service, Redis et les collectes planifiees."""
    logger.info("Demarrage The Researcher...")
    try:
        await init_redis()
        logger.info("Redis connecte")
    except Exception as exc:
        logger.warning("Redis non disponible au demarrage: %s", exc)

    service = ResearchService()
    app.state.service = service
    app.state.knowledge_base: list[dict[str, Any]] = []
    app.state.sota_tracker: dict[str, dict[str, Any]] = {}
    app.state.research_history: deque[dict[str, Any]] = deque(maxlen=200)
    app.state.competitive_cache: dict[str, dict[str, Any]] = {}
    app.state.heartbeat_task = asyncio.create_task(_heartbeat_loop(app))
    await service.start_background_tasks()
    logger.info("The Researcher pret")
    try:
        yield
    finally:
        await service.stop_background_tasks()
        heartbeat_task = getattr(app.state, "heartbeat_task", None)
        if heartbeat_task:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        logger.info("Arret The Researcher")


app = FastAPI(
    title="The Researcher API",
    description="Agent de recherche, veille et revue de connaissance - THE HIVE",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/health", tags=["Systeme"])
async def health() -> dict[str, Any]:
    """Retourne l'etat global du service."""
    ingestion_status = await app.state.service.get_ingestion_status(tail=5)
    return {
        "status": "ok",
        "service": "researcher",
        "knowledge_entries": len(app.state.knowledge_base),
        "sota_tasks_tracked": len(app.state.sota_tracker),
        "review_counts": ingestion_status.get("counts", {}),
    }


@app.post("/search", tags=["Recherche"])
async def search(request: ResearchQuery) -> dict[str, Any]:
    """Lance une recherche web et synthese les resultats."""
    result = await app.state.service.search(request)
    app.state.research_history.append(
        {
            "query": request.query,
            "results_count": len(result.get("results", [])),
            "timestamp": datetime.now().isoformat(),
        }
    )
    return result


@app.get("/trends", tags=["Recherche"])
async def get_trends(domain: str = Query(default="tech")) -> dict[str, Any]:
    """Retourne les sources de veille pour un domaine."""
    return await app.state.service.get_trends(domain)


@app.get("/stats", tags=["Recherche"])
async def get_stats() -> dict[str, Any]:
    """Retourne des statistiques de recherche et d'ingestion."""
    ingestion_status = await app.state.service.get_ingestion_status(tail=5)
    return {
        "total_searches": len(app.state.research_history),
        "knowledge_entries": len(app.state.knowledge_base),
        "sota_tasks": len(app.state.sota_tracker),
        "competitive_reports": len(app.state.competitive_cache),
        "review_counts": ingestion_status.get("counts", {}),
    }


@app.get("/history", tags=["Recherche"])
async def get_history(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    """Retourne l'historique des recherches a la demande."""
    return {"history": list(app.state.research_history)[-limit:]}


@app.post("/papers", tags=["ArXiv"])
async def search_papers(request: ArxivSearchRequest) -> dict[str, Any]:
    """Recherche et resume des papers arXiv."""
    try:
        return await app.state.service.search_papers(
            query=request.query,
            category=request.category,
            max_results=request.max_results,
        )
    except Exception as exc:
        logger.error("Erreur de recherche arXiv: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.post("/competitive", tags=["Intelligence"])
async def competitive_intel(request: CompetitiveIntelRequest) -> dict[str, Any]:
    """Produit un mini rapport de veille concurrentielle."""
    findings: dict[str, Any] = {}
    for aspect in request.aspects:
        query = f"{request.target} {aspect} 2026"
        findings[aspect] = await app.state.service.search(ResearchQuery(query=query, max_results=3))
    report = {
        "target": request.target,
        "aspects_analyzed": request.aspects,
        "findings": findings,
        "generated_at": datetime.now().isoformat(),
    }
    app.state.competitive_cache[request.target] = report
    return report


@app.post("/knowledge", tags=["Knowledge"])
async def add_knowledge(entry: KnowledgeEntry) -> dict[str, Any]:
    """Ajoute une entree manuelle a la base interne."""
    knowledge = {"id": f"KB-{uuid4().hex[:8].upper()}", **entry.model_dump(), "created_at": datetime.now().isoformat()}
    app.state.knowledge_base.append(knowledge)
    return {"status": "added", "entry": knowledge}


@app.get("/knowledge", tags=["Knowledge"])
async def search_knowledge(q: str = Query(default=""), domain: str = Query(default="")) -> dict[str, Any]:
    """Recherche dans la base de connaissances locale."""
    results = app.state.knowledge_base
    if q:
        q_lower = q.lower()
        results = [
            entry
            for entry in results
            if q_lower in entry.get("topic", "").lower() or q_lower in entry.get("content", "").lower()
        ]
    if domain:
        results = [entry for entry in results if entry.get("domain") == domain]
    return {"results": results, "total": len(results)}


@app.post("/sota", tags=["SOTA"])
async def track_sota(entry: SotaEntry) -> dict[str, Any]:
    """Enregistre un etat de l'art pour une tache."""
    current = app.state.sota_tracker.get(entry.task)
    is_new_best = current is None or entry.score > current.get("score", 0)
    data = {**entry.model_dump(), "is_current_best": is_new_best, "updated_at": datetime.now().isoformat()}
    if is_new_best:
        app.state.sota_tracker[entry.task] = data
    return {"status": "new_best" if is_new_best else "recorded", "entry": data}


@app.get("/sota", tags=["SOTA"])
async def get_sota() -> dict[str, Any]:
    """Liste les etats de l'art connus."""
    return {"tasks": app.state.sota_tracker, "total": len(app.state.sota_tracker)}


@app.get("/pea-analysis", tags=["Finance"])
async def get_pea_analysis() -> dict[str, Any]:
    """Expose l'etat actuel du module PEA."""
    return {
        "status": "info",
        "message": "Analyse PEA en cours de developpement via Researcher.",
        "targets": ["TotalEnergies", "LVMH", "Air Liquide", "Saint-Gobain"],
    }


@app.post("/ingest/sources/sync", tags=["Ingestion"])
async def sync_ingest_sources(request: SyncSourcesRequest) -> dict[str, Any]:
    """Lance une collecte immediate des sources configurees."""
    return await app.state.service.sync_sources(request)


@app.get("/ingest/status", tags=["Ingestion"])
async def get_ingest_status(tail: int = Query(default=30, ge=1, le=100)) -> dict[str, Any]:
    """Retourne l'etat global de la pipeline d'ingestion."""
    return await app.state.service.get_ingestion_status(tail=tail)


@app.get("/ingest/review", tags=["Ingestion"])
async def list_ingest_review(
    review_status: str = Query(default="pending", pattern="^(pending|approved|rejected|ingested)$"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Liste les candidats en revue par statut."""
    return await app.state.service.list_review_items(review_status=review_status, limit=limit, offset=offset)


@app.post("/ingest/review/{item_id}/approve", tags=["Ingestion"])
async def approve_ingest_item(item_id: str, decision: ReviewDecisionRequest) -> dict[str, Any]:
    """Approuve un candidat et lance son ingestion durable."""
    try:
        return await app.state.service.approve_review_item(item_id, decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Candidat introuvable: {exc.args[0]}") from exc


@app.post("/ingest/review/{item_id}/reject", tags=["Ingestion"])
async def reject_ingest_item(item_id: str, decision: ReviewDecisionRequest) -> dict[str, Any]:
    """Rejette un candidat de la file de revue."""
    try:
        return await app.state.service.reject_review_item(item_id, decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Candidat introuvable: {exc.args[0]}") from exc


@app.get("/ingest/approved", tags=["Ingestion"])
async def list_ingested_items(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    """Retourne les derniers candidats approuves et ingeres."""
    return await app.state.service.list_approved_items(limit=limit)


@app.get("/ingest/sources", tags=["Ingestion"])
async def list_ingest_sources() -> dict[str, Any]:
    """Retourne les sources actives et leurs dernieres synchronisations."""
    return await app.state.service.get_sources()
