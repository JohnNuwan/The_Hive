"""Tests API minimaux pour la v1 d'ingestion Researcher."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = Path(__file__).resolve().parents[2] / "shared"
for candidate in (ROOT, SHARED):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

if "neo4j" not in sys.modules:
    neo4j_module = types.ModuleType("neo4j")

    class _FakeNeo4jDriver:
        async def verify_connectivity(self):
            return None

        async def execute_query(self, query, parameters=None):
            return [], None, None

        async def close(self):
            return None

    class _FakeAsyncGraphDatabase:
        @staticmethod
        def driver(uri, auth=None):
            return _FakeNeo4jDriver()

    class _FakeGraphDatabase:
        @staticmethod
        def driver(uri, auth=None):
            return _FakeNeo4jDriver()

    neo4j_module.AsyncGraphDatabase = _FakeAsyncGraphDatabase
    neo4j_module.GraphDatabase = _FakeGraphDatabase
    sys.modules["neo4j"] = neo4j_module

from fastapi.testclient import TestClient

import eva_researcher.main as researcher_main


def _build_mock_service():
    service = SimpleNamespace()
    service.start_background_tasks = AsyncMock()
    service.stop_background_tasks = AsyncMock()
    service.search = AsyncMock(
        return_value={
            "query": "test",
            "domain": "general",
            "results": [],
            "synthesis": "Synthese",
            "search_time_ms": 1,
            "timestamp": "2026-03-14T00:00:00",
            "review_queue": {"queued": 0, "duplicates": 0, "errors": 0, "items": []},
        }
    )
    service.get_trends = AsyncMock(return_value={"domain": "tech", "sources": []})
    service.get_ingestion_status = AsyncMock(
        return_value={
            "status": "ok",
            "counts": {"pending": 1, "approved": 0, "rejected": 0, "ingested": 0},
            "active_run": None,
            "last_run": None,
            "source_stats": {},
            "duplicate_rate": 0.0,
            "dependencies": {},
            "logs": [],
        }
    )
    service.sync_sources = AsyncMock(return_value={"status": "ok", "queued": 2, "duplicates": 0, "errors": 0, "sources": []})
    service.list_review_items = AsyncMock(return_value={"status": "ok", "total": 1, "items": []})
    service.approve_review_item = AsyncMock(return_value={"status": "ok", "item": {"id": "KNO-1", "review_status": "ingested"}})
    service.reject_review_item = AsyncMock(side_effect=KeyError("KNO-404"))
    service.list_approved_items = AsyncMock(return_value={"status": "ok", "total": 0, "items": []})
    service.get_sources = AsyncMock(return_value={"status": "ok", "sources": []})
    service.search_papers = AsyncMock(return_value={"query": "q", "category": "cs.AI", "papers": [], "total": 0, "review_queue": {}})
    return service


def test_ingest_status_endpoint(monkeypatch):
    """Verifie l'endpoint /ingest/status."""
    mock_service = _build_mock_service()
    monkeypatch.setattr(researcher_main, "init_redis", AsyncMock())
    monkeypatch.setattr(researcher_main, "ResearchService", lambda: mock_service)

    with TestClient(researcher_main.app) as client:
        response = client.get("/ingest/status")
    assert response.status_code == 200
    assert response.json()["counts"]["pending"] == 1


def test_sync_sources_endpoint(monkeypatch):
    """Verifie l'endpoint /ingest/sources/sync."""
    mock_service = _build_mock_service()
    monkeypatch.setattr(researcher_main, "init_redis", AsyncMock())
    monkeypatch.setattr(researcher_main, "ResearchService", lambda: mock_service)

    with TestClient(researcher_main.app) as client:
        response = client.post("/ingest/sources/sync", json={"include_arxiv": True, "include_news": True})
    assert response.status_code == 200
    assert response.json()["queued"] == 2


def test_approve_missing_item_returns_404(monkeypatch):
    """Verifie le 404 sur rejet d'un item absent."""
    mock_service = _build_mock_service()
    monkeypatch.setattr(researcher_main, "init_redis", AsyncMock())
    monkeypatch.setattr(researcher_main, "ResearchService", lambda: mock_service)

    with TestClient(researcher_main.app) as client:
        response = client.post("/ingest/review/KNO-404/reject", json={"reviewed_by": "qa", "reason": "absent"})
    assert response.status_code == 404


def test_search_endpoint_keeps_existing_shape(monkeypatch):
    """Verifie que /search garde sa forme de reponse historique."""
    mock_service = _build_mock_service()
    monkeypatch.setattr(researcher_main, "init_redis", AsyncMock())
    monkeypatch.setattr(researcher_main, "ResearchService", lambda: mock_service)

    with TestClient(researcher_main.app) as client:
        response = client.post("/search", json={"query": "test", "domain": "general", "max_results": 3})
    payload = response.json()
    assert response.status_code == 200
    assert "results" in payload
    assert "synthesis" in payload
    assert "review_queue" in payload
