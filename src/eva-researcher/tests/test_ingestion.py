"""Tests de la file de revue et de l'ingestion de connaissance."""

import asyncio
import json
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

from eva_researcher.services.ingestion import KnowledgeIngestionService, ReviewDecisionRequest


class FakeRedisClient:
    """Client Redis minimal pour tests unitaires."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def connect(self) -> None:
        return None

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value, ex=None):
        if isinstance(value, dict):
            self.store[key] = json.dumps(value)
        else:
            self.store[key] = value
        return True


class FakeGraph:
    """Double minimal du graphe Neo4j."""

    def __init__(self):
        self.driver = object()

    async def connect(self):
        return None


class FakeMemoryBridge:
    """Double minimal du bridge memoire."""

    def __init__(self):
        self.items = []
        self.graph = FakeGraph()

    async def add(self, content: str, user_id: str = "user", metadata=None):
        self.items.append({"content": content, "user_id": user_id, "metadata": metadata or {}})


def test_deduplication_pending_queue():
    """Verifie qu'un meme candidat n'entre qu'une seule fois en revue."""

    async def scenario():
        service = KnowledgeIngestionService(redis_client=FakeRedisClient(), memory_bridge=FakeMemoryBridge())
        first = await service.queue_manual_search(
            query="agentic ai",
            domain="tech",
            results=[
                {
                    "title": "Agentic AI systems",
                    "url": "https://example.com/a",
                    "summary": "Resume stable",
                    "source": "DuckDuckGo",
                    "relevance_score": 0.9,
                }
            ],
            synthesis="Synthese courte",
        )
        second = await service.queue_manual_search(
            query="agentic ai",
            domain="tech",
            results=[
                {
                    "title": "Agentic AI systems",
                    "url": "https://example.com/a",
                    "summary": "Resume stable",
                    "source": "DuckDuckGo",
                    "relevance_score": 0.9,
                }
            ],
            synthesis="Synthese courte",
        )
        review = await service.list_review_items()
        assert first["queued"] == 1
        assert second["duplicates"] == 1
        assert review["total"] == 1

    asyncio.run(scenario())


def test_approve_moves_item_to_ingested_memory():
    """Verifie qu'une approbation ecrit en memoire durable."""

    async def scenario():
        memory = FakeMemoryBridge()
        service = KnowledgeIngestionService(redis_client=FakeRedisClient(), memory_bridge=memory)
        queued = await service.queue_manual_search(
            query="risk management",
            domain="finance",
            results=[
                {
                    "title": "Risk management in trading",
                    "url": "https://example.com/risk",
                    "summary": "Article sur le risque.",
                    "source": "DuckDuckGo",
                    "relevance_score": 0.8,
                }
            ],
            synthesis="Synthese risque",
        )
        item_id = queued["items"][0]["id"]
        result = await service.approve_item(item_id, ReviewDecisionRequest(reviewed_by="qa"))
        assert result["item"]["review_status"] == "ingested"
        assert len(memory.items) == 1
        approved = await service.list_approved_items()
        assert approved["total"] == 1

    asyncio.run(scenario())


def test_reject_keeps_memory_clean():
    """Verifie qu'un rejet n'ecrit rien en memoire durable."""

    async def scenario():
        memory = FakeMemoryBridge()
        service = KnowledgeIngestionService(redis_client=FakeRedisClient(), memory_bridge=memory)
        queued = await service.queue_manual_search(
            query="macro news",
            domain="finance",
            results=[
                {
                    "title": "Macro market news",
                    "url": "https://example.com/news",
                    "summary": "Resume macro.",
                    "source": "DuckDuckGo",
                    "relevance_score": 0.6,
                }
            ],
            synthesis="Synthese macro",
        )
        item_id = queued["items"][0]["id"]
        result = await service.reject_item(item_id, ReviewDecisionRequest(reviewed_by="qa", reason="Hors sujet"))
        assert result["item"]["review_status"] == "rejected"
        assert len(memory.items) == 0

    asyncio.run(scenario())


def test_sync_sources_aggregates_candidates():
    """Verifie qu'une sync combine les collectes arXiv et news."""

    async def scenario():
        service = KnowledgeIngestionService(redis_client=FakeRedisClient(), memory_bridge=FakeMemoryBridge())
        arxiv_candidate = service._build_candidate(
            source_type="arxiv",
            source_name="arXiv cs.AI",
            family="academique",
            title="Paper A",
            url="https://arxiv.org/abs/1",
            summary_raw="Resume A",
            summary_curated="Resume A",
            tags=["cs.AI", "arxiv"],
            origin="API arXiv",
            confidence_score=0.9,
            metadata={"source_key": "arxiv:cs.AI"},
        )
        news_candidate = service._build_candidate(
            source_type="news",
            source_name="Google News IA",
            family="actualite",
            title="Article B",
            url="https://example.com/b",
            summary_raw="Resume B",
            summary_curated="Resume B",
            tags=["news", "actualite"],
            origin="Google News IA",
            confidence_score=0.8,
            metadata={"source_key": "google_news_ai"},
        )

        async def fake_arxiv(max_items: int):
            return [arxiv_candidate]

        async def fake_news(max_items: int):
            return [news_candidate]

        service._collect_arxiv_candidates = fake_arxiv
        service._collect_news_candidates = fake_news

        result = await service.sync_sources()
        assert result["queued"] == 2
        status = await service.get_status()
        assert status["counts"]["pending"] == 2

    asyncio.run(scenario())
