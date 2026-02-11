"""
Memory Bridge Module — HippoRAG 2 Integration
Part of Sovereign Stack V3.0

Orchestrates interactions between:
1. Mem0 (Vector Memory — semantic retrieval via Qdrant)
2. Neo4j (Graph Memory — HippoRAG 2 associative reasoning via PPR)
3. Redis (Short-term context)

HippoRAG 2 Upgrade:
- add() now extracts triples and stores in Neo4j knowledge graph
- search() now runs hybrid: Mem0 vectors + Neo4j PPR expansion
- pattern_complete() finds complex concepts from vague queries
"""

import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache

try:
    from mem0 import Memory
except ImportError:
    Memory = None  # Fallback if mem0ai not installed

from shared.config import get_settings
from shared.memory_graph import get_graph_memory

logger = logging.getLogger(__name__)


class MemoryBridge:
    """
    Unified memory bridge for E.V.A. — HippoRAG 2 Edition.

    Combines:
    - Mem0 (Qdrant vectors) for fast semantic search
    - Neo4j (knowledge graph) for associative multi-hop reasoning
    - Pattern Completion for "hippocampal" recall from vague cues
    """

    def __init__(self):
        self.settings = get_settings()
        self.graph = get_graph_memory()

        # Configuration Mem0 (Qdrant backend)
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": self.settings.qdrant_host,
                    "port": self.settings.qdrant_port,
                    "collection_name": self.settings.qdrant_collection_conversations,
                }
            },
            # Same LLM as core (Ollama/vLLM)
            "llm": {
                "provider": "openai",  # Compatible vLLM
                "config": {
                    "model": self.settings.vllm_model if self.settings.llm_backend == "vllm" else self.settings.ollama_model,
                    "openai_base_url": f"http://{self.settings.vllm_host}:{self.settings.vllm_port}/v1"
                                       if self.settings.llm_backend == "vllm"
                                       else f"http://{self.settings.ollama_host}:{self.settings.ollama_port}/v1"
                }
            }
        }

        if Memory:
            try:
                self.mem0 = Memory.from_config(config)
                logger.info("Mem0 initialized with Qdrant & vLLM/Ollama")
            except Exception as e:
                logger.error(f"Failed to initialize Mem0: {e}")
                self.mem0 = None
        else:
            logger.warning("mem0ai package not found. Memory Bridge running in degraded mode.")
            self.mem0 = None

    # ── Write: Add Memory ─────────────────────────────────────

    async def add(self, content: str, user_id: str = "user", metadata: Dict[str, Any] = None):
        """
        Add a memory (Vector + Graph sync via HippoRAG 2).

        1. Store in Mem0 (Qdrant vectors) for semantic retrieval
        2. Extract triples from content (HippoRAG 2 parahippocampal)
        3. Store triples in Neo4j knowledge graph
        """
        # 1. Vector Memory (Mem0 → Qdrant)
        if self.mem0:
            self.mem0.add(content, user_id=user_id, metadata=metadata)

        # 2. HippoRAG 2: Triple extraction + Graph ingestion
        triples = await self.graph.ingest_text(content, source=f"mem0_sync:{user_id}")

        logger.debug(
            f"Memory added: {len(content)} chars, "
            f"{len(triples)} triples extracted"
        )

    # ── Read: Hybrid Search ───────────────────────────────────

    async def search(self, query: str, user_id: str = "user", limit: int = 5) -> List[str]:
        """
        Hybrid search: Mem0 vectors + HippoRAG 2 PPR expansion.

        1. Semantic vector search via Mem0 (fast, high recall)
        2. Pattern Completion via Neo4j PPR (deep, associative)
        3. Merge and deduplicate results
        """
        results = []

        # 1. Vector Search (Mem0)
        if self.mem0:
            try:
                memories = self.mem0.search(query, user_id=user_id, limit=limit)
                results.extend([m.get("memory", "") for m in memories if m.get("memory")])
            except Exception as e:
                logger.warning(f"Mem0 search failed: {e}")

        # 2. HippoRAG 2: Pattern Completion (graph propagation)
        try:
            graph_results = await self.graph.pattern_complete(query, max_results=limit)
            for r in graph_results:
                # Add entity name + associated facts
                name = r.get("name", "")
                facts = r.get("associated_facts", [])
                score = r.get("score", 0)

                # Add associated facts as results
                for fact in facts:
                    if fact and fact not in results:
                        results.append(fact)

                # Add entity name if it's informative
                if name and len(name) > 3 and name not in results:
                    relations = r.get("relations", [])
                    rel_str = ", ".join(relations[:3]) if relations else "related"
                    results.append(f"{name} ({rel_str}, score={score:.2f})")
        except Exception as e:
            logger.warning(f"HippoRAG 2 search failed: {e}")

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique.append(r)

        return unique[:limit]

    # ── Pattern Complete (CDcs: "mot-clé vague") ──────────────

    async def pattern_complete(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        HippoRAG 2 Pattern Completion — the "hippocampal" recall.

        Given a vague keyword or phrase, discovers complex related
        strategies and concepts through knowledge graph propagation.

        This is the CDcs v3.0 requirement:
        "E.V.A. doit pouvoir retrouver une stratégie de trading complexe
         à partir d'un simple mot-clé vague."

        Returns:
            List of dicts with {name, score, relations, associated_facts}.
        """
        return await self.graph.pattern_complete(query, max_results=limit)

    # ── Read All ──────────────────────────────────────────────

    async def get_all(self, user_id: str = "user") -> List[str]:
        """Retrieve all memories."""
        if self.mem0:
            memories = self.mem0.get_all(user_id=user_id)
            return [m.get("memory", "") for m in memories]
        return []

    # ── Stats ─────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """Return unified memory statistics."""
        graph_stats = await self.graph.get_stats()
        mem0_count = 0
        if self.mem0:
            try:
                all_mems = self.mem0.get_all()
                mem0_count = len(all_mems) if all_mems else 0
            except Exception:
                pass

        return {
            "vector_memories": mem0_count,
            "graph": graph_stats,
            "hipporag2_enabled": True,
        }


@lru_cache
def get_memory_bridge() -> MemoryBridge:
    return MemoryBridge()
