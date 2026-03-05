try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

import os

class MemoryLayer:
    """
    LE LOBE TEMPORAL (Memory Layer)
    ------------------------------
    Gère la mémoire à long terme via Mem0.
    Permet à E.V.A. de se souvenir des préférences et de l'histoire de l'utilisateur.
    """

    def __init__(self, user_id="admin"):
        self.user_id = user_id
        if MEM0_AVAILABLE:
            # Configuration Mem0 (peut utiliser Qdrant ou Chroma en backend)
            ollama_host = os.getenv("OLLAMA_HOST", "host.docker.internal")
            ollama_port = os.getenv("OLLAMA_PORT", "11434")
            ollama_base_url = f"http://{ollama_host}:{ollama_port}"

            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": os.getenv("QDRANT_HOST", "host.docker.internal"),
                        "port": int(os.getenv("QDRANT_PORT", 6333)),
                    }
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": "nomic-embed-text:latest",
                        "ollama_base_url": ollama_base_url
                    }
                },
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": "gemma3", # Match current server models
                        "temperature": 0,
                        "ollama_base_url": ollama_base_url
                    }
                }
            }
            self.memory = Memory.from_config(config)
            print("🧠 Mem0 persistent memory online.")
        else:
            print("⚠️ Mem0 not installed. Long-term memory is disabled.")

    def store_event(self, text, metadata=None):
        """Stocke une information importante."""
        if MEM0_AVAILABLE:
            self.memory.add(text, user_id=self.user_id, metadata=metadata)
            return True
        return False

    def recall(self, query):
        """Récupère les souvenirs pertinents pour une requête."""
        if MEM0_AVAILABLE:
            return self.memory.search(query, user_id=self.user_id)
        return []

    def get_user_profile(self):
        """Extrait le profil psychologique/préférences construit par Mem0."""
        if MEM0_AVAILABLE:
            return self.memory.get_all(user_id=self.user_id)
        return []

if __name__ == "__main__":
    # Test à blanc
    ml = MemoryLayer()
    ml.store_event("L'utilisateur préfère un risque conservateur de 2% le weekend.")
    print(f"Recall test: {ml.recall('préférences risque')}")
