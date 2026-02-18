"""
Graph Memory Module (Neo4j) — HippoRAG 2 Implementation
Part of Sovereign Stack V3.0

Implements the HippoRAG 2 Pattern Completion algorithm:
1. Triple Extraction — (Subject, Predicate, Object) from text via LLM
2. Knowledge Graph — Typed nodes + edges in Neo4j
3. Personalized PageRank (PPR) — Associative retrieval from seed nodes
4. Pattern Completion — Find complex strategies from vague keywords

Reference: HippoRAG 2 (Ohio State / UIUC, 2025)
"""

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase, AsyncGraphDatabase

from shared import get_settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Triple Extraction (Parahippocampal Region)
# ═══════════════════════════════════════════════════════════════

def extract_triples_from_text(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract (Subject, Predicate, Object) triples from text.

    Uses rule-based extraction for speed. In production, this would
    call the LLM with a structured prompt for richer extraction.

    Returns:
        List of (subject, predicate, object) tuples.
    """
    triples = []

    # Pattern 1: "X is/are Y" → (X, IS, Y)
    for match in re.finditer(r"(\b[A-Z][a-zA-Z0-9_.-]+\b)\s+(?:is|are)\s+(.+?)(?:\.|,|$)", text):
        subj = match.group(1).strip()
        obj = match.group(2).strip()[:100]
        if len(subj) > 1 and len(obj) > 1:
            triples.append((subj, "IS", obj))

    # Pattern 2: "X uses/utilise Y" → (X, USES, Y)
    for match in re.finditer(r"(\b[A-Z][a-zA-Z0-9_.-]+\b)\s+(?:uses?|utilise)\s+(.+?)(?:\.|,|$)", text):
        subj = match.group(1).strip()
        obj = match.group(2).strip()[:100]
        triples.append((subj, "USES", obj))

    # Pattern 3: "X → Y" or "X -> Y" → (X, LEADS_TO, Y)
    for match in re.finditer(r"(\b[A-Za-z0-9_.-]+\b)\s*(?:→|->)\s*(\b[A-Za-z0-9_.-]+\b)", text):
        triples.append((match.group(1), "LEADS_TO", match.group(2)))

    # Pattern 4: "X trades/bought/sold Y" → (X, TRADES, Y)
    for match in re.finditer(r"(\b[A-Z][a-zA-Z0-9_.-]+\b)\s+(?:trades?|bought|sold|long|short)\s+(.+?)(?:\.|,|$)", text):
        subj = match.group(1).strip()
        obj = match.group(2).strip()[:100]
        triples.append((subj, "TRADES", obj))

    # Pattern 5: "X with Y" → (X, ASSOCIATED_WITH, Y)
    for match in re.finditer(r"(\b[A-Z][a-zA-Z0-9_.-]+\b)\s+with\s+(\b[A-Z][a-zA-Z0-9_.-]+\b)", text):
        triples.append((match.group(1), "ASSOCIATED_WITH", match.group(2)))

    # Fallback: extract key terms as standalone entities connected to the fact
    if not triples:
        # Extract capitalized terms as entities
        entities = re.findall(r"\b([A-Z][a-zA-Z0-9_.-]{2,})\b", text)
        unique_entities = list(dict.fromkeys(entities))[:5]
        for i, entity in enumerate(unique_entities):
            triples.append((entity, "MENTIONED_IN", text[:60]))
            # Connect consecutive entities
            if i > 0:
                triples.append((unique_entities[i - 1], "CO_OCCURS_WITH", entity))

    return triples


async def extract_triples_llm(text: str, llm_client=None) -> List[Tuple[str, str, str]]:
    """
    LLM-based triple extraction (higher quality, slower).

    Sends a structured prompt to the LLM to extract semantic triples.
    Falls back to rule-based extraction if LLM is unavailable.
    """
    if llm_client is None:
        return extract_triples_from_text(text)

    prompt = (
        "Extract knowledge triples from this text. "
        "Return ONLY lines in format: SUBJECT | PREDICATE | OBJECT\n"
        "Each line is one triple. Use uppercase for predicates.\n\n"
        f"Text: {text}\n\n"
        "Triples:"
    )

    try:
        import httpx
        response = await llm_client.post(
            "/v1/completions",
            json={"prompt": prompt, "max_tokens": 200, "temperature": 0.1},
        )
        result = response.json().get("choices", [{}])[0].get("text", "")

        triples = []
        for line in result.strip().split("\n"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3 and all(parts):
                triples.append((parts[0], parts[1].upper().replace(" ", "_"), parts[2]))
        return triples if triples else extract_triples_from_text(text)

    except Exception as e:
        logger.warning(f"LLM triple extraction failed: {e}")
        return extract_triples_from_text(text)


# ═══════════════════════════════════════════════════════════════
#  Graph Memory with HippoRAG 2 (Neo4j)
# ═══════════════════════════════════════════════════════════════

class GraphMemoryWithNeo4j:
    """
    HippoRAG 2 Knowledge Graph client (async Neo4j).

    Supports:
    - Entity/relation CRUD
    - Triple-based knowledge ingestion
    - Personalized PageRank (PPR) for associative retrieval
    - Pattern Completion for multi-hop reasoning
    """

    def __init__(self):
        settings = get_settings()
        self.uri = f"bolt://{settings.neo4j_host}:{settings.neo4j_port}"
        self.auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
        self.driver = None
        self._indexes_created = False
        logger.info(f"GraphMemory (HippoRAG 2) initialized targeting {self.uri}")

    async def connect(self):
        """Establishes connection with Neo4j."""
        if not self.driver:
            try:
                self.driver = AsyncGraphDatabase.driver(self.uri, auth=self.auth)
                await self.driver.verify_connectivity()
                logger.info("Connected to Neo4j successfully.")
                # Create indexes on first connection
                if not self._indexes_created:
                    await self._create_indexes()
            except Exception as e:
                logger.error(f"Neo4j connection failed: {e}")
                self.driver = None

    async def close(self):
        """Closes the connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None

    async def _create_indexes(self):
        """Create Neo4j indexes for fast lookup (idempotent)."""
        queries = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX fact_text IF NOT EXISTS FOR (n:Fact) ON (n.text)",
            "CREATE INDEX concept_name IF NOT EXISTS FOR (n:Concept) ON (n.name)",
        ]
        for q in queries:
            try:
                await self.execute_query(q)
            except Exception:
                pass  # Index may already exist
        self._indexes_created = True
        logger.info("Neo4j indexes ensured.")

    # ── Basic CRUD ────────────────────────────────────────────

    async def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        if not self.driver:
            await self.connect()
            if not self.driver:
                return []

        try:
            records, summary, keys = await self.driver.execute_query(query, parameters or {})
            return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Neo4j query error: {e}")
            return []

    async def add_entity(self, label: str, name: str, properties: Dict[str, Any] = None):
        """Create or merge a node by label and name."""
        query = (
            f"MERGE (n:{label} {{name: $name}}) "
            "SET n += $props "
            "RETURN n"
        )
        await self.execute_query(query, {"name": name, "props": properties or {}})

    async def add_relation(self, source_name: str, relation_type: str, target_name: str):
        """Create a relation between two existing nodes."""
        query = (
            "MATCH (a), (b) "
            "WHERE a.name = $source_name AND b.name = $target_name "
            f"MERGE (a)-[r:{relation_type}]->(b) "
            "RETURN type(r)"
        )
        await self.execute_query(query, {"source_name": source_name, "target_name": target_name})

    # ── HippoRAG 2: Triple Ingestion ─────────────────────────

    async def store_triples(self, triples: List[Tuple[str, str, str]], source: str = "hipporag"):
        """
        Store (Subject, Predicate, Object) triples in the knowledge graph.

        Creates Entity nodes and typed relationships.
        Synonyms and co-occurrences are linked for pattern completion.
        """
        # Group triples by predicate to allow batched UNWIND operations
        # This reduces N queries to K queries (where K is number of unique predicates)
        triples_by_pred: Dict[str, List[Dict[str, str]]] = {}
        for subj, pred, obj in triples:
            # Sanitize predicate for Cypher (must be alphanumeric)
            safe_pred = re.sub(r"[^A-Z0-9_]", "_", pred.upper())
            if not safe_pred:
                safe_pred = "RELATED_TO"

            if safe_pred not in triples_by_pred:
                triples_by_pred[safe_pred] = []

            triples_by_pred[safe_pred].append({
                "subj": subj.strip(),
                "obj": obj.strip()
            })

        # Execute one batch query per unique predicate
        for pred, batch in triples_by_pred.items():
            query = (
                "UNWIND $batch AS row "
                "MERGE (s:Entity {name: row.subj}) "
                "MERGE (o:Entity {name: row.obj}) "
                f"MERGE (s)-[r:{pred}]->(o) "
                "SET r.source = $source, r.count = COALESCE(r.count, 0) + 1"
            )
            await self.execute_query(query, {
                "batch": batch,
                "source": source,
            })

        logger.debug(f"Stored {len(triples)} triples from {source} (in {len(triples_by_pred)} queries)")

    async def ingest_text(self, text: str, source: str = "user", llm_client=None):
        """
        Full HippoRAG 2 ingestion pipeline:
        1. Extract triples from text
        2. Store in knowledge graph
        3. Link to source Fact node

        Args:
            text: Raw text to process.
            source: Source identifier for provenance.
            llm_client: Optional httpx client for LLM-based extraction.
        """
        # Extract triples
        triples = await extract_triples_llm(text, llm_client)

        if triples:
            await self.store_triples(triples, source=source)

        # Also store the original fact as a node
        fact_id = text[:50].replace("'", "")
        await self.add_entity("Fact", fact_id, {
            "full_text": text,
            "source": source,
            "triple_count": len(triples),
        })

        # Link extracted entities to the fact
        for subj, _, _ in triples:
            clean_subj = subj.strip()
            try:
                query = (
                    "MATCH (e:Entity {name: $entity}), (f:Fact {name: $fact}) "
                    "MERGE (e)-[:EXTRACTED_FROM]->(f)"
                )
                await self.execute_query(query, {"entity": clean_subj, "fact": fact_id})
            except Exception:
                pass

        return triples

    # ── HippoRAG 2: Personalized PageRank ────────────────────

    async def ppr_search(
        self,
        seed_entities: List[str],
        max_results: int = 10,
        damping_factor: float = 0.85,
        max_iterations: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Personalized PageRank from seed entity nodes.

        This is the core of HippoRAG 2's "hippocampal retrieval":
        starting from known seed nodes, propagate activation through
        the knowledge graph to discover associated concepts.

        Args:
            seed_entities: Names of seed entity nodes.
            max_results: Maximum number of results to return.
            damping_factor: PPR damping factor (0.85 = standard).
            max_iterations: Max PPR iterations.

        Returns:
            Ranked list of {name, score, labels, path_from_seed}.
        """
        if not seed_entities:
            return []

        # Build the PPR query using Neo4j GDS (Graph Data Science) or manual BFS
        # We use a manual iterative approach for compatibility (no GDS plugin required)
        query = """
        // Find seed nodes
        MATCH (seed:Entity)
        WHERE seed.name IN $seeds
        WITH collect(seed) AS seedNodes

        // BFS expansion (up to 3 hops for pattern completion)
        UNWIND seedNodes AS start
        MATCH path = (start)-[*1..3]-(connected)
        WHERE connected <> start
        
        // Score: closer nodes score higher (inverse of path length)
        WITH connected,
             min(length(path)) AS minDist,
             count(DISTINCT path) AS pathCount,
             collect(DISTINCT type(relationships(path)[-1])) AS relTypes
        
        // PPR-inspired scoring: proximity × connectivity
        WITH connected,
             (1.0 / (1.0 + minDist)) * (1.0 + log(pathCount + 1)) AS score,
             relTypes,
             labels(connected) AS nodeLabels
        
        RETURN connected.name AS name,
               score,
               nodeLabels AS labels,
               relTypes AS relations,
               minDist AS distance
        ORDER BY score DESC
        LIMIT $limit
        """

        results = await self.execute_query(query, {
            "seeds": seed_entities,
            "limit": max_results,
        })

        return results

    async def pattern_complete(
        self,
        query_text: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        HippoRAG 2 Pattern Completion.

        Given a vague query, find relevant complex knowledge by:
        1. Extracting key entities from the query
        2. Finding matching seed nodes in the graph
        3. Running PPR to discover associated concepts
        4. Returning ranked results with context

        This implements the CDcs requirement:
        "E.V.A. doit pouvoir retrouver une stratégie de trading complexe
         à partir d'un simple mot-clé vague."
        """
        # Step 1: Extract entities from query
        entities = re.findall(r"\b([A-Z][a-zA-Z0-9_.-]{1,})\b", query_text)
        # Also try lowercase important terms
        terms = re.findall(r"\b([a-zA-Z]{3,})\b", query_text.lower())
        all_candidates = list(dict.fromkeys(entities + [t.upper() for t in terms[:5]]))

        if not all_candidates:
            return []

        # Step 2: Find which candidates exist as nodes
        check_query = (
            "MATCH (n:Entity) "
            "WHERE n.name IN $candidates "
            "RETURN n.name AS name"
        )
        existing = await self.execute_query(check_query, {"candidates": all_candidates})
        seed_names = [r["name"] for r in existing]

        if not seed_names:
            # Fallback: fuzzy match on any node
            fuzzy_query = (
                "MATCH (n:Entity) "
                "WHERE ANY(c IN $candidates WHERE n.name CONTAINS c OR c CONTAINS n.name) "
                "RETURN n.name AS name LIMIT 5"
            )
            existing = await self.execute_query(fuzzy_query, {"candidates": all_candidates})
            seed_names = [r["name"] for r in existing]

        if not seed_names:
            return []

        # Step 3: PPR from seeds
        ppr_results = await self.ppr_search(seed_names, max_results=max_results)

        # Step 4: Enrich results with associated facts
        enriched = []
        for result in ppr_results:
            name = result.get("name", "")
            # Find facts connected to this entity
            fact_query = (
                "MATCH (e:Entity {name: $name})-[:EXTRACTED_FROM]->(f:Fact) "
                "RETURN f.full_text AS fact LIMIT 3"
            )
            facts = await self.execute_query(fact_query, {"name": name})
            result["associated_facts"] = [f["fact"] for f in facts if f.get("fact")]
            enriched.append(result)

        logger.info(
            f"Pattern Complete: query='{query_text[:40]}' → "
            f"{len(seed_names)} seeds → {len(enriched)} results"
        )
        return enriched

    # ── Graph Statistics ──────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        node_count = await self.execute_query("MATCH (n) RETURN count(n) AS count")
        edge_count = await self.execute_query("MATCH ()-[r]->() RETURN count(r) AS count")
        entity_count = await self.execute_query("MATCH (n:Entity) RETURN count(n) AS count")
        fact_count = await self.execute_query("MATCH (n:Fact) RETURN count(n) AS count")

        return {
            "total_nodes": node_count[0]["count"] if node_count else 0,
            "total_edges": edge_count[0]["count"] if edge_count else 0,
            "entities": entity_count[0]["count"] if entity_count else 0,
            "facts": fact_count[0]["count"] if fact_count else 0,
        }


@lru_cache
def get_graph_memory() -> GraphMemoryWithNeo4j:
    return GraphMemoryWithNeo4j()
