"""
Graph Memory Module (Neo4j)
Part of Sovereign Stack V3.0 (HippoRAG 2)

Gère la connexion à Neo4j et les opérations de base sur le graphe de connaissances.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, AsyncGraphDatabase

from shared import get_settings

logger = logging.getLogger(__name__)

class GraphMemoryWithNeo4j:
    """
    Client asynchrone pour Neo4j.
    Permet de stocker et récupérer des faits et des entités.
    """
    
    def __init__(self):
        settings = get_settings()
        self.uri = f"bolt://{settings.neo4j_host}:{settings.neo4j_port}"
        # Auth: neo4j / password
        self.auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
        self.driver = None
        logger.info(f"GraphMemory initialized targeting {self.uri}")

    async def connect(self):
        """Etablit la connexion avec Neo4j"""
        if not self.driver:
            try:
                self.driver = AsyncGraphDatabase.driver(self.uri, auth=self.auth)
                await self.driver.verify_connectivity()
                logger.info("Connecté à Neo4j avec succès.")
            except Exception as e:
                logger.error(f"Echec connexion Neo4j: {e}")
                self.driver = None

    async def close(self):
        """Ferme la connexion"""
        if self.driver:
            await self.driver.close()
            self.driver = None

    async def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Exécute une requête Cypher et retourne les résultats"""
        if not self.driver:
            await self.connect()
            if not self.driver:
                return []

        try:
            records, summary, keys = await self.driver.execute_query(query, parameters or {})
            # records est une liste de neo4j.Record, on peut les convertir en dict si besoin
            return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Erreur requête Neo4j: {e}")
            return []

    async def add_entity(self, label: str, name: str, properties: Dict[str, Any] = None):
        """Crée ou merge un noeud identifié par son label et son nom"""
        query = (
            f"MERGE (n:{label} {{name: $name}}) "
            "SET n += $props "
            "RETURN n"
        )
        await self.execute_query(query, {"name": name, "props": properties or {}})

    async def add_relation(self, source_name: str, relation_type: str, target_name: str):
        """Crée une relation entre deux noeuds existants (basé sur 'name')"""
        # Note: Supposons que les noeuds ont la propriété 'name' unique pour simplifier
        query = (
            "MATCH (a), (b) "
            "WHERE a.name = $source_name AND b.name = $target_name "
            f"MERGE (a)-[r:{relation_type}]->(b) "
            "RETURN type(r)"
        )
        await self.execute_query(query, {"source_name": source_name, "target_name": target_name})

@lru_cache
def get_graph_memory() -> GraphMemoryWithNeo4j:
    return GraphMemoryWithNeo4j()
