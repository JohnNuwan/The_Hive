"""
Pipeline d'ingestion de connaissance pour EVA Researcher.

Cette v1 couvre deux familles:
- academique via arXiv;
- actualite via des flux RSS/news curies.

Les collectes alimentent une file de revue obligatoire. Seuls les elements
approuves sont ecrits durablement via MemoryBridge vers Qdrant et Neo4j.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from hashlib import sha256
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from pydantic import BaseModel, ConfigDict, Field
from shared import get_settings
from shared.memory_bridge import MemoryBridge, get_memory_bridge
from shared.redis_client import RedisClient, get_redis_client

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """Retourne l'heure UTC au format ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int = 600) -> str:
    """Nettoie et tronque un texte pour stockage ou affichage."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3].rstrip()}..."


def _normalize_title(title: str) -> str:
    """Normalise un titre pour la deduplication."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _normalize_url(url: str) -> str:
    """Normalise une URL pour la deduplication."""
    if not url:
        return ""
    parsed = urlparse(url.strip())
    query_items = sorted(parse_qsl(parsed.query, keep_blank_values=False))
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",
            urlencode(query_items),
            "",
        )
    )


def _safe_int(value: Any, default: int = 0) -> int:
    """Convertit une valeur en entier sans lever d'exception."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class KnowledgeCandidate(BaseModel):
    """Represente un candidat de connaissance en attente de revue."""

    model_config = ConfigDict(extra="allow")

    id: str
    source_type: str
    source_name: str
    title: str
    url: str = ""
    published_at: str | None = None
    authors: list[str] = Field(default_factory=list)
    origin: str = ""
    summary_raw: str = ""
    summary_curated: str = ""
    tags: list[str] = Field(default_factory=list)
    family: str
    confidence_score: float = 0.0
    priority_score: float = 0.0
    content_hash: str
    review_status: str = "pending"
    metadata: dict[str, Any] = Field(default_factory=dict)
    collected_at: str = Field(default_factory=_utc_now_iso)
    reviewed_at: str | None = None
    reviewed_by: str | None = None
    rejection_reason: str | None = None
    ingested_at: str | None = None


class SyncSourcesRequest(BaseModel):
    """Controle une synchronisation immediate des sources configurees."""

    include_arxiv: bool = True
    include_news: bool = True
    max_items_per_source: int | None = Field(default=None, ge=1, le=50)
    trigger: str = "manual"


class ReviewDecisionRequest(BaseModel):
    """Represente une decision de revue."""

    reviewed_by: str = "manual"
    reason: str = ""


@dataclass(slots=True)
class QueueResult:
    """Represente le resultat normalise d'une mise en file."""

    item_id: str | None
    status: str
    reason: str = ""


class KnowledgeIngestionService:
    """Pilote la collecte, la revue et l'ingestion durable des connaissances."""

    STATE_KEY = "eva.researcher.ingest.state"
    ITEM_KEY_PREFIX = "eva.researcher.ingest.item:"
    MAX_LOG_ENTRIES = 80
    MAX_FUZZY_TITLES = 200

    def __init__(
        self,
        redis_client: RedisClient | None = None,
        memory_bridge: MemoryBridge | None = None,
    ) -> None:
        self.settings = get_settings()
        self.redis = redis_client or get_redis_client()
        self.memory_bridge = memory_bridge or get_memory_bridge()
        self._state_lock = asyncio.Lock()
        self._sync_lock = asyncio.Lock()
        self._background_tasks: list[asyncio.Task[Any]] = []
        self._stop_event = asyncio.Event()
        self._local_state = self._build_default_state()
        self._local_items: dict[str, dict[str, Any]] = {}

    async def start_background_tasks(self) -> None:
        """Demarre les collectes planifiees configurees."""
        if not self.settings.researcher_ingestion_enabled:
            self._append_log("info", "Collecte de connaissance desactivee.")
            return
        if self._background_tasks:
            return
        self._stop_event.clear()
        if self.settings.researcher_ingestion_arxiv_enabled:
            self._background_tasks.append(
                asyncio.create_task(
                    self._run_source_loop(
                        source_label="arxiv",
                        interval_minutes=self.settings.researcher_ingestion_arxiv_interval_minutes,
                        include_arxiv=True,
                        include_news=False,
                    )
                )
            )
        if self.settings.researcher_ingestion_news_enabled:
            self._background_tasks.append(
                asyncio.create_task(
                    self._run_source_loop(
                        source_label="actualites",
                        interval_minutes=self.settings.researcher_ingestion_news_interval_minutes,
                        include_arxiv=False,
                        include_news=True,
                    )
                )
            )
        self._append_log("info", "Collectes planifiees demarrees.")

    async def stop_background_tasks(self) -> None:
        """Arrete proprement les taches de collecte planifiees."""
        self._stop_event.set()
        tasks = list(self._background_tasks)
        self._background_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._append_log("info", "Collectes planifiees arretees.")

    async def sync_sources(self, request: SyncSourcesRequest | None = None) -> dict[str, Any]:
        """Synchronise immediatement les sources configurees."""
        payload = request or SyncSourcesRequest()
        async with self._sync_lock:
            started_at = _utc_now_iso()
            await self._update_run_state(
                active=True,
                status="running",
                trigger=payload.trigger,
                strategy="hybrid_review",
                reason="sync_requested",
                started_at=started_at,
                updated_at=started_at,
                current_source="initialisation",
            )
            self._append_log("info", f"Demarrage d'une synchronisation ({payload.trigger}).")

            stats = {"queued": 0, "duplicates": 0, "errors": 0, "sources": []}
            max_items = payload.max_items_per_source or self.settings.researcher_ingestion_max_items_per_source
            try:
                if payload.include_arxiv:
                    arxiv_candidates = await self._collect_arxiv_candidates(max_items=max_items)
                    result = await self._queue_candidates(arxiv_candidates)
                    stats["queued"] += result["queued"]
                    stats["duplicates"] += result["duplicates"]
                    stats["errors"] += result["errors"]
                    stats["sources"].append({"name": "arxiv", **result})
                if payload.include_news:
                    news_candidates = await self._collect_news_candidates(max_items=max_items)
                    result = await self._queue_candidates(news_candidates)
                    stats["queued"] += result["queued"]
                    stats["duplicates"] += result["duplicates"]
                    stats["errors"] += result["errors"]
                    stats["sources"].append({"name": "news", **result})

                finished_at = _utc_now_iso()
                await self._update_run_state(
                    active=False,
                    status="ok",
                    updated_at=finished_at,
                    finished_at=finished_at,
                    current_source=None,
                    reason="sync_completed",
                    last_run={
                        "trigger": payload.trigger,
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "summary": stats,
                    },
                )
                self._append_log(
                    "info",
                    f"Synchronisation terminee: {stats['queued']} candidats, "
                    f"{stats['duplicates']} doublons, {stats['errors']} erreurs.",
                )
                return {
                    "status": "ok",
                    "trigger": payload.trigger,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    **stats,
                }
            except Exception as exc:
                finished_at = _utc_now_iso()
                await self._update_run_state(
                    active=False,
                    status="error",
                    updated_at=finished_at,
                    finished_at=finished_at,
                    reason=str(exc),
                )
                self._append_log("error", f"Echec de synchronisation: {exc}")
                raise

    async def queue_manual_search(
        self,
        query: str,
        domain: str,
        results: list[dict[str, Any]],
        synthesis: str,
    ) -> dict[str, Any]:
        """Alimente la file de revue depuis une recherche a la demande."""
        candidates: list[KnowledgeCandidate] = []
        for index, result in enumerate(results[:5], start=1):
            title = result.get("title") or f"Resultat {index}"
            summary = result.get("summary") or synthesis or ""
            candidate = self._build_candidate(
                source_type="manual_search",
                source_name=result.get("source", "manual_search"),
                family="actualite",
                title=title,
                url=result.get("url", ""),
                summary_raw=summary,
                summary_curated=summary,
                tags=[domain, "manual_search"],
                metadata={
                    "query": query,
                    "domain": domain,
                    "relevance_score": result.get("relevance_score", 0.0),
                    "source_key": "manual_search",
                },
                origin=f"Recherche manuelle: {query}",
                confidence_score=min(0.95, 0.4 + (result.get("relevance_score", 0.0) / 2)),
            )
            candidates.append(candidate)
        return await self._queue_candidates(candidates)

    async def queue_arxiv_results(
        self,
        query: str,
        category: str,
        papers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Alimente la file de revue depuis des resultats arXiv a la demande."""
        candidates: list[KnowledgeCandidate] = []
        for paper in papers:
            summary = paper.get("summary", "")
            candidate = self._build_candidate(
                source_type="arxiv",
                source_name=f"arXiv {category}",
                family="academique",
                title=paper.get("title", ""),
                url=paper.get("url", ""),
                published_at=paper.get("published"),
                authors=paper.get("authors", []),
                summary_raw=summary,
                summary_curated=summary,
                tags=[category, "arxiv", query],
                metadata={"query": query, "category": category, "source_key": f"arxiv:{category}"},
                origin="API arXiv",
                confidence_score=0.92,
            )
            candidates.append(candidate)
        return await self._queue_candidates(candidates)

    async def list_review_items(
        self,
        review_status: str = "pending",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Retourne une liste paginee d'elements de revue."""
        state = await self._load_state()
        status_to_ids = {
            "pending": state["pending_ids"],
            "approved": state["approved_ids"],
            "rejected": state["rejected_ids"],
            "ingested": state["ingested_ids"],
        }
        selected_ids = list(status_to_ids.get(review_status, []))
        items = await self._load_items(selected_ids)
        items.sort(
            key=lambda item: (
                float(item.get("priority_score", 0.0)),
                item.get("collected_at", ""),
            ),
            reverse=True,
        )
        return {
            "status": "ok",
            "review_status": review_status,
            "total": len(selected_ids),
            "items": items[offset : offset + limit],
        }

    async def approve_item(self, item_id: str, decision: ReviewDecisionRequest) -> dict[str, Any]:
        """Approuve un candidat puis l'envoie en memoire durable."""
        async with self._state_lock:
            item = await self._load_item(item_id)
            if item is None:
                raise KeyError(item_id)
            previous_status = item.get("review_status", "pending")
            item["review_status"] = "approved"
            item["reviewed_at"] = _utc_now_iso()
            item["reviewed_by"] = decision.reviewed_by
            item["rejection_reason"] = None
            await self._move_item_state(item_id, previous_status, "approved")
            await self._save_item(item)
        await self._ingest_candidate(item_id)
        ingested = await self._load_item(item_id)
        self._append_log("info", f"Candidat approuve et ingere: {item_id}")
        return {"status": "ok", "item": ingested}

    async def reject_item(self, item_id: str, decision: ReviewDecisionRequest) -> dict[str, Any]:
        """Rejette un candidat sans ecriture durable."""
        async with self._state_lock:
            item = await self._load_item(item_id)
            if item is None:
                raise KeyError(item_id)
            previous_status = item.get("review_status", "pending")
            item["review_status"] = "rejected"
            item["reviewed_at"] = _utc_now_iso()
            item["reviewed_by"] = decision.reviewed_by
            item["rejection_reason"] = decision.reason or "Rejet manuel"
            await self._move_item_state(item_id, previous_status, "rejected")
            await self._save_item(item)
        self._append_log("info", f"Candidat rejete: {item_id}")
        return {"status": "ok", "item": item}

    async def list_approved_items(self, limit: int = 50) -> dict[str, Any]:
        """Retourne les derniers elements valides."""
        state = await self._load_state()
        item_ids = list(reversed(state["ingested_ids"]))[:limit]
        return {"status": "ok", "total": len(item_ids), "items": await self._load_items(item_ids)}

    async def get_sources(self) -> dict[str, Any]:
        """Expose les sources configurees et leur dernier etat."""
        state = await self._load_state()
        source_stats = state.get("source_stats", {})
        sources = []
        for source in self._configured_sources():
            if source["source_type"] == "arxiv":
                stats = self._aggregate_source_stats(source_stats, prefix="arxiv:")
            else:
                stats = source_stats.get(source["key"], {})
            sources.append(
                {
                    **source,
                    "last_sync": stats.get("last_sync"),
                    "queued": _safe_int(stats.get("queued")),
                    "duplicates": _safe_int(stats.get("duplicates")),
                    "errors": _safe_int(stats.get("errors")),
                }
            )
        return {"status": "ok", "sources": sources}

    async def get_status(self, tail: int = 30) -> dict[str, Any]:
        """Retourne l'etat global de la pipeline d'ingestion."""
        state = await self._load_state()
        return {
            "status": "ok",
            "counts": {
                "pending": len(state["pending_ids"]),
                "approved": len(state["approved_ids"]),
                "rejected": len(state["rejected_ids"]),
                "ingested": len(state["ingested_ids"]),
            },
            "active_run": state.get("active_run"),
            "last_run": state.get("last_run"),
            "source_stats": state.get("source_stats", {}),
            "duplicate_rate": self._compute_duplicate_rate(state.get("source_stats", {})),
            "dependencies": await self._dependency_status(),
            "logs": list(state.get("logs", []))[-max(1, min(tail, 100)) :],
        }

    def _build_default_state(self) -> dict[str, Any]:
        return {
            "pending_ids": [],
            "approved_ids": [],
            "rejected_ids": [],
            "ingested_ids": [],
            "hash_index": {},
            "url_index": {},
            "title_index": {},
            "source_stats": {},
            "active_run": {
                "run_id": None,
                "active": False,
                "status": "idle",
                "trigger": None,
                "strategy": "hybrid_review",
                "reason": None,
                "started_at": None,
                "updated_at": None,
                "finished_at": None,
                "current_source": None,
            },
            "last_run": None,
            "logs": [],
        }

    def _configured_sources(self) -> list[dict[str, Any]]:
        """Retourne les sources curiees de la v1."""
        return [
            {
                "key": "arxiv",
                "source_type": "arxiv",
                "source_name": "arXiv",
                "family": "academique",
                "categories": [
                    category.strip()
                    for category in self.settings.researcher_arxiv_categories.split(",")
                    if category.strip()
                ],
            },
            {
                "key": "google_news_ai",
                "source_type": "news",
                "source_name": "Google News IA",
                "family": "actualite",
                "url": "https://news.google.com/rss/search?q=artificial+intelligence&hl=fr&gl=FR&ceid=FR:fr",
            },
            {
                "key": "google_news_cyber",
                "source_type": "news",
                "source_name": "Google News Cyber",
                "family": "actualite",
                "url": "https://news.google.com/rss/search?q=cybersecurity&hl=fr&gl=FR&ceid=FR:fr",
            },
            {
                "key": "google_news_markets",
                "source_type": "news",
                "source_name": "Google News Marches",
                "family": "actualite",
                "url": "https://news.google.com/rss/search?q=markets+trading&hl=fr&gl=FR&ceid=FR:fr",
            },
            {
                "key": "techcrunch_ai",
                "source_type": "rss",
                "source_name": "TechCrunch IA",
                "family": "actualite",
                "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
            },
        ]

    async def _run_source_loop(
        self,
        source_label: str,
        interval_minutes: int,
        include_arxiv: bool,
        include_news: bool,
    ) -> None:
        """Execute une boucle de collecte planifiee."""
        while not self._stop_event.is_set():
            try:
                await self.sync_sources(
                    SyncSourcesRequest(
                        include_arxiv=include_arxiv,
                        include_news=include_news,
                        trigger=f"scheduled:{source_label}",
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._append_log("error", f"Echec de collecte planifiee {source_label}: {exc}")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=max(60, interval_minutes * 60))
            except asyncio.TimeoutError:
                continue

    async def _queue_candidates(self, candidates: list[KnowledgeCandidate]) -> dict[str, Any]:
        """Ajoute des candidats a la file apres deduplication stricte."""
        summary = {"queued": 0, "duplicates": 0, "errors": 0, "items": []}
        if not candidates:
            return summary
        async with self._state_lock:
            state = await self._load_state()
            for candidate in candidates:
                try:
                    duplicate = self._detect_duplicate(state, candidate)
                    if duplicate.status == "duplicate":
                        summary["duplicates"] += 1
                        summary["items"].append(
                            {"id": duplicate.item_id, "status": "duplicate", "reason": duplicate.reason}
                        )
                        source_key = candidate.metadata.get("source_key", candidate.source_name)
                        self._bump_source_stats(state, source_key, duplicate=True)
                        continue
                    await self._save_item(candidate.model_dump())
                    state["pending_ids"].append(candidate.id)
                    state["hash_index"][candidate.content_hash] = candidate.id
                    if candidate.url:
                        state["url_index"][_normalize_url(candidate.url)] = candidate.id
                    state["title_index"][_normalize_title(candidate.title)] = candidate.id
                    source_key = candidate.metadata.get("source_key", candidate.source_name)
                    self._bump_source_stats(state, source_key, queued=True)
                    summary["queued"] += 1
                    summary["items"].append({"id": candidate.id, "status": "queued"})
                except Exception as exc:
                    summary["errors"] += 1
                    source_key = candidate.metadata.get("source_key", candidate.source_name)
                    self._bump_source_stats(state, source_key, error=True)
                    self._append_log("error", f"Echec de mise en file {candidate.source_name}: {exc}")
            await self._save_state(state)
        return summary

    def _detect_duplicate(self, state: dict[str, Any], candidate: KnowledgeCandidate) -> QueueResult:
        """Detecte un doublon via hash, URL puis similarite forte de titre."""
        item_id = state["hash_index"].get(candidate.content_hash)
        if item_id:
            return QueueResult(item_id, "duplicate", "hash")
        normalized_url = _normalize_url(candidate.url)
        if normalized_url:
            item_id = state["url_index"].get(normalized_url)
            if item_id:
                return QueueResult(item_id, "duplicate", "url")
        normalized_title = _normalize_title(candidate.title)
        item_id = state["title_index"].get(normalized_title)
        if item_id:
            return QueueResult(item_id, "duplicate", "title_exact")
        for indexed_title, existing_id in list(state["title_index"].items())[-self.MAX_FUZZY_TITLES :]:
            if indexed_title and SequenceMatcher(None, indexed_title, normalized_title).ratio() >= 0.94:
                return QueueResult(existing_id, "duplicate", "title_fuzzy")
        return QueueResult(None, "queued")

    async def _ingest_candidate(self, item_id: str) -> None:
        """Ecrit un candidat approuve dans la memoire durable."""
        async with self._state_lock:
            item = await self._load_item(item_id)
            if item is None:
                raise KeyError(item_id)
            if item.get("review_status") == "ingested":
                return
            content = self._build_memory_content(item)
            metadata = {
                "provenance": item.get("source_name"),
                "source_type": item.get("source_type"),
                "family": item.get("family"),
                "tags": item.get("tags", []),
                "confidence_score": item.get("confidence_score", 0.0),
                "priority_score": item.get("priority_score", 0.0),
                "url": item.get("url"),
                "published_at": item.get("published_at"),
                "review_status": "approved",
                "content_type": "knowledge_candidate",
                **item.get("metadata", {}),
            }
        await self.memory_bridge.add(content, user_id="researcher", metadata=metadata)
        async with self._state_lock:
            refreshed = await self._load_item(item_id)
            if refreshed is None:
                return
            previous_status = refreshed.get("review_status", "approved")
            refreshed["review_status"] = "ingested"
            refreshed["ingested_at"] = _utc_now_iso()
            await self._move_item_state(item_id, previous_status, "ingested")
            await self._save_item(refreshed)

    def _build_memory_content(self, item: dict[str, Any]) -> str:
        """Construit le contenu propre a injecter en memoire."""
        parts = [
            f"Titre: {item.get('title', '').strip()}",
            f"Famille: {item.get('family', '').strip()}",
            f"Source: {item.get('source_name', '').strip()}",
        ]
        if item.get("authors"):
            parts.append(f"Auteurs: {', '.join(item['authors'])}")
        if item.get("published_at"):
            parts.append(f"Publie le: {item['published_at']}")
        if item.get("summary_curated"):
            parts.append(f"Resume: {item['summary_curated']}")
        if item.get("tags"):
            parts.append(f"Tags: {', '.join(item['tags'])}")
        if item.get("url"):
            parts.append(f"URL: {item['url']}")
        return "\n".join(parts)

    async def _collect_arxiv_candidates(self, max_items: int) -> list[KnowledgeCandidate]:
        """Collecte les derniers papiers arXiv selon les categories configurees."""
        collected: list[KnowledgeCandidate] = []
        categories = [
            category.strip()
            for category in self.settings.researcher_arxiv_categories.split(",")
            if category.strip()
        ]
        if not categories:
            return collected
        async with httpx.AsyncClient(timeout=25) as client:
            for category in categories:
                await self._update_run_state(current_source=f"arxiv:{category}", updated_at=_utc_now_iso())
                response = await client.get(
                    "http://export.arxiv.org/api/query",
                    params={
                        "search_query": f"cat:{category}",
                        "max_results": max_items,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                    },
                )
                response.raise_for_status()
                root = ET.fromstring(response.text)
                namespace = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", namespace):
                    summary = entry.findtext("atom:summary", default="", namespaces=namespace)
                    collected.append(
                        self._build_candidate(
                            source_type="arxiv",
                            source_name=f"arXiv {category}",
                            family="academique",
                            title=_truncate(entry.findtext("atom:title", default="", namespaces=namespace), 240),
                            url=entry.findtext("atom:id", default="", namespaces=namespace),
                            published_at=entry.findtext("atom:published", default="", namespaces=namespace),
                            authors=[
                                author.text.strip()
                                for author in entry.findall("atom:author/atom:name", namespace)
                                if author.text
                            ],
                            summary_raw=_truncate(summary, 1400),
                            summary_curated=_truncate(summary, 900),
                            tags=[category, "arxiv"],
                            origin="API arXiv",
                            confidence_score=0.92,
                            metadata={"category": category, "source_key": f"arxiv:{category}"},
                        )
                    )
        return collected

    async def _collect_news_candidates(self, max_items: int) -> list[KnowledgeCandidate]:
        """Collecte les derniers articles a partir des flux news/RSS configures."""
        collected: list[KnowledgeCandidate] = []
        async with httpx.AsyncClient(timeout=25, headers={"User-Agent": "THE-HIVE-Researcher/1.0"}) as client:
            for source in self._configured_sources():
                if source["source_type"] == "arxiv":
                    continue
                await self._update_run_state(current_source=source["key"], updated_at=_utc_now_iso())
                try:
                    response = await client.get(source["url"])
                    response.raise_for_status()
                    collected.extend(
                        self._parse_feed(
                            xml_payload=response.text,
                            source_key=source["key"],
                            source_name=source["source_name"],
                            source_type=source["source_type"],
                            family=source["family"],
                            max_items=max_items,
                        )
                    )
                except Exception as exc:
                    self._append_log("error", f"Flux {source['source_name']} indisponible: {exc}")
        return collected

    def _parse_feed(
        self,
        xml_payload: str,
        source_key: str,
        source_name: str,
        source_type: str,
        family: str,
        max_items: int,
    ) -> list[KnowledgeCandidate]:
        """Parse un flux RSS ou Atom sans dependance supplementaire."""
        root = ET.fromstring(xml_payload)
        items: list[KnowledgeCandidate] = []
        if root.tag.endswith("rss"):
            entries = root.findall("./channel/item")
            for entry in entries[:max_items]:
                summary = (
                    entry.findtext("description", default="")
                    or entry.findtext("{http://purl.org/rss/1.0/modules/content/}encoded", default="")
                )
                items.append(
                    self._build_candidate(
                        source_type=source_type,
                        source_name=source_name,
                        family=family,
                        title=_truncate(entry.findtext("title", default=""), 240),
                        url=entry.findtext("link", default=""),
                        published_at=entry.findtext("pubDate", default=None),
                        summary_raw=_truncate(summary, 1400),
                        summary_curated=_truncate(summary, 900),
                        tags=[source_type, family],
                        origin=source_name,
                        confidence_score=0.78,
                        metadata={"source_key": source_key},
                    )
                )
        else:
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", namespace)
            for entry in entries[:max_items]:
                link = ""
                for link_node in entry.findall("atom:link", namespace):
                    href = link_node.attrib.get("href", "")
                    if href:
                        link = href
                        break
                summary = (
                    entry.findtext("atom:summary", default="", namespaces=namespace)
                    or entry.findtext("atom:content", default="", namespaces=namespace)
                )
                items.append(
                    self._build_candidate(
                        source_type=source_type,
                        source_name=source_name,
                        family=family,
                        title=_truncate(entry.findtext("atom:title", default="", namespaces=namespace), 240),
                        url=link,
                        published_at=entry.findtext("atom:updated", default="", namespaces=namespace),
                        summary_raw=_truncate(summary, 1400),
                        summary_curated=_truncate(summary, 900),
                        tags=[source_type, family],
                        origin=source_name,
                        confidence_score=0.78,
                        metadata={"source_key": source_key},
                    )
                )
        return items

    def _build_candidate(
        self,
        *,
        source_type: str,
        source_name: str,
        family: str,
        title: str,
        url: str,
        summary_raw: str,
        summary_curated: str,
        tags: list[str],
        origin: str,
        confidence_score: float,
        published_at: str | None = None,
        authors: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeCandidate:
        """Construit un candidat normalise avec hash et score priorise."""
        normalized_url = _normalize_url(url)
        content_hash = sha256(
            "||".join(
                [
                    _normalize_title(title),
                    normalized_url,
                    _truncate(summary_curated or summary_raw, 800),
                ]
            ).encode("utf-8")
        ).hexdigest()
        return KnowledgeCandidate(
            id=f"KNO-{content_hash[:12].upper()}",
            source_type=source_type,
            source_name=source_name,
            title=_truncate(title, 240),
            url=normalized_url,
            published_at=published_at,
            authors=authors or [],
            origin=origin,
            summary_raw=_truncate(summary_raw, 1400),
            summary_curated=_truncate(summary_curated, 900),
            tags=list(dict.fromkeys(tag for tag in tags if tag)),
            family=family,
            confidence_score=round(confidence_score, 4),
            priority_score=round(self._compute_priority_score(family, confidence_score, published_at, tags), 4),
            content_hash=content_hash,
            metadata=metadata or {},
        )

    def _compute_priority_score(
        self,
        family: str,
        confidence_score: float,
        published_at: str | None,
        tags: list[str],
    ) -> float:
        """Calcule un score simple de priorisation."""
        score = confidence_score * 0.6
        if family == "academique":
            score += 0.2
        if any(tag.lower().startswith("q-fin") for tag in tags):
            score += 0.1
        if published_at:
            score += 0.1
        return min(1.0, score)

    def _compute_duplicate_rate(self, source_stats: dict[str, Any]) -> float:
        """Calcule un taux de duplication global."""
        queued = sum(_safe_int(stats.get("queued")) for stats in source_stats.values())
        duplicates = sum(_safe_int(stats.get("duplicates")) for stats in source_stats.values())
        total = queued + duplicates
        if total == 0:
            return 0.0
        return round(duplicates / total, 4)

    def _aggregate_source_stats(self, source_stats: dict[str, Any], prefix: str) -> dict[str, Any]:
        """Agrege les compteurs d'un groupe de sources partageant un prefixe."""
        aggregate = {"queued": 0, "duplicates": 0, "errors": 0, "last_sync": None}
        for key, stats in source_stats.items():
            if not key.startswith(prefix):
                continue
            aggregate["queued"] += _safe_int(stats.get("queued"))
            aggregate["duplicates"] += _safe_int(stats.get("duplicates"))
            aggregate["errors"] += _safe_int(stats.get("errors"))
            last_sync = stats.get("last_sync")
            if last_sync and (aggregate["last_sync"] is None or last_sync > aggregate["last_sync"]):
                aggregate["last_sync"] = last_sync
        return aggregate

    async def _dependency_status(self) -> dict[str, Any]:
        """Retourne l'etat operationnel des dependances utiles."""
        redis_status = {"ok": False, "detail": "indisponible"}
        try:
            await self.redis.connect()
            redis_status = {"ok": True, "detail": "connecte"}
        except Exception as exc:
            redis_status = {"ok": False, "detail": str(exc)}
        qdrant_status = {"ok": False, "detail": "indisponible"}
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(
                    f"http://{self.settings.qdrant_host}:{self.settings.qdrant_port}/collections"
                )
                qdrant_status = {"ok": response.status_code == 200, "detail": f"http:{response.status_code}"}
        except Exception as exc:
            qdrant_status = {"ok": False, "detail": str(exc)}
        neo4j_status = {"ok": False, "detail": "indisponible"}
        try:
            await self.memory_bridge.graph.connect()
            neo4j_status = {
                "ok": self.memory_bridge.graph.driver is not None,
                "detail": "connecte" if self.memory_bridge.graph.driver is not None else "non_connecte",
            }
        except Exception as exc:
            neo4j_status = {"ok": False, "detail": str(exc)}
        return {"redis": redis_status, "qdrant": qdrant_status, "neo4j": neo4j_status}

    def _bump_source_stats(
        self,
        state: dict[str, Any],
        source_name: str,
        *,
        queued: bool = False,
        duplicate: bool = False,
        error: bool = False,
    ) -> None:
        """Met a jour les compteurs par source."""
        stats = state["source_stats"].setdefault(
            source_name,
            {"queued": 0, "duplicates": 0, "errors": 0, "last_sync": None},
        )
        if queued:
            stats["queued"] += 1
        if duplicate:
            stats["duplicates"] += 1
        if error:
            stats["errors"] += 1
        stats["last_sync"] = _utc_now_iso()

    async def _update_run_state(self, **updates: Any) -> None:
        """Met a jour l'etat du run actif."""
        async with self._state_lock:
            state = await self._load_state()
            active_run = deepcopy(state.get("active_run", {}))
            for key, value in updates.items():
                if key == "last_run":
                    continue
                active_run[key] = value
            state["active_run"] = active_run
            if "last_run" in updates and updates["last_run"] is not None:
                state["last_run"] = updates["last_run"]
            await self._save_state(state)

    async def _load_state(self) -> dict[str, Any]:
        """Charge l'etat global depuis Redis ou la memoire locale."""
        try:
            await self.redis.connect()
            payload = await self.redis.get(self.STATE_KEY)
            if payload:
                state = json.loads(payload)
                local_logs = list(self._local_state.get("logs", []))
                remote_logs = list(state.get("logs", []))
                if len(local_logs) > len(remote_logs):
                    state["logs"] = local_logs[-self.MAX_LOG_ENTRIES :]
                return state
        except Exception:
            pass
        return deepcopy(self._local_state)

    async def _save_state(self, state: dict[str, Any]) -> None:
        """Sauvegarde l'etat global."""
        local_logs = list(self._local_state.get("logs", []))
        remote_logs = list(state.get("logs", []))
        if len(local_logs) > len(remote_logs):
            state["logs"] = local_logs[-self.MAX_LOG_ENTRIES :]
        self._local_state = deepcopy(state)
        try:
            await self.redis.connect()
            await self.redis.set(self.STATE_KEY, state)
        except Exception:
            pass

    async def _load_item(self, item_id: str) -> dict[str, Any] | None:
        """Charge un item de revue par identifiant."""
        key = f"{self.ITEM_KEY_PREFIX}{item_id}"
        try:
            await self.redis.connect()
            payload = await self.redis.get(key)
            if payload:
                return json.loads(payload)
        except Exception:
            pass
        item = self._local_items.get(item_id)
        return deepcopy(item) if item else None

    async def _load_items(self, item_ids: list[str]) -> list[dict[str, Any]]:
        """Charge une liste d'items dans l'ordre demande."""
        items: list[dict[str, Any]] = []
        for item_id in item_ids:
            item = await self._load_item(item_id)
            if item is not None:
                items.append(item)
        return items

    async def _save_item(self, item: dict[str, Any]) -> None:
        """Sauvegarde un item de revue."""
        item_id = item["id"]
        self._local_items[item_id] = deepcopy(item)
        try:
            await self.redis.connect()
            await self.redis.set(f"{self.ITEM_KEY_PREFIX}{item_id}", item)
        except Exception:
            pass

    async def _move_item_state(self, item_id: str, from_status: str | None, to_status: str) -> None:
        """Deplace un item entre listes d'etats."""
        state = await self._load_state()
        mappings = {
            "pending": list(state["pending_ids"]),
            "approved": list(state["approved_ids"]),
            "rejected": list(state["rejected_ids"]),
            "ingested": list(state["ingested_ids"]),
        }
        if from_status in mappings and item_id in mappings[from_status]:
            mappings[from_status] = [existing for existing in mappings[from_status] if existing != item_id]
        if item_id not in mappings[to_status]:
            mappings[to_status].append(item_id)
        state.update(
            {
                "pending_ids": mappings["pending"],
                "approved_ids": mappings["approved"],
                "rejected_ids": mappings["rejected"],
                "ingested_ids": mappings["ingested"],
            }
        )
        await self._save_state(state)

    def _append_log(self, level: str, message: str) -> None:
        """Ajoute un log exploitable a l'etat partage."""
        logs = list(self._local_state.get("logs", []))
        logs.append({"ts": _utc_now_iso(), "level": level, "message": message})
        self._local_state["logs"] = logs[-self.MAX_LOG_ENTRIES :]
