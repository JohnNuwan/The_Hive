"""
Memory Bridge Module (Mem0)
Part of Sovereign Stack V3.0 (HippoRAG 2)

Orchestre les interactions entre:
1. Mem0 (Vector Memory pour le retrieval sémantique)
2. Neo4j (Graph Memory pour le raisonnement associatif)
3. Redis (Short-term context)
"""

import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache

try:
    from mem0 import Memory
except ImportError:
    Memory = None # Fallback si mem0ai n'est pas installé

from shared.config import get_settings
from shared.memory_graph import get_graph_memory

logger = logging.getLogger(__name__)

class MemoryBridge:
    """
    Pont unifié pour la mémoire E.V.A.
    Utilise Mem0 comme interface principale pour le stockage vectoriel (Qdrant).
    Synchronise les faits importants vers Neo4j.
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
            # On utilise le même LLM que le core (Ollama/vLLM)
            "llm": {
                "provider": "openai", # Compatible vLLM
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

    async def add(self, content: str, user_id: str = "user", metadata: Dict[str, Any] = None):
        """Ajoute un souvenir (Vector + Graph sync)"""
        if self.mem0:
            # 1. Stockage Vectoriel (Mem0 -> Qdrant)
            self.mem0.add(content, user_id=user_id, metadata=metadata)
            
            # 2. Synchronisation Graph (Neo4j) - Extraction basique pour l'instant
            # TODO: Utiliser un LLM pour extraire (Sujet, Prédicat, Objet)
            # Pour l'instant, on loggue juste l'interaction comme un fait brut
            await self.graph.add_entity("Fact", content[:50], {"full_text": content, "source": "mem0_sync"})
            await self.graph.add_relation("User_Generic", "KNOWS", content[:50])

    async def search(self, query: str, user_id: str = "user", limit: int = 5) -> List[str]:
        """Recherche hybride (Mem0 + Graph exploration prévue)"""
        results = []
        
        # 1. Recherche Vectorielle (Mem0)
        if self.mem0:
            memories = self.mem0.search(query, user_id=user_id, limit=limit)
            # Mem0 retourne une liste de dicts
            results.extend([m.get("memory", "") for m in memories])
            
        return results

    async def get_all(self, user_id: str = "user") -> List[str]:
        """Récupère tous les souvenirs"""
        if self.mem0:
             memories = self.mem0.get_all(user_id=user_id)
             return [m.get("memory", "") for m in memories]
        return []

@lru_cache
def get_memory_bridge() -> MemoryBridge:
    return MemoryBridge()
