import logging
import httpx
from typing import Optional
from shared import get_settings

logger = logging.getLogger(__name__)

class CouncilService:
    """
    Gestionnaire du 'Conseil des Experts' (Model Swapping).
    Permet de charger/décharger dynamiquement les modèles dans Ollama 
    en fonction du rôle requis, optimisé pour la VRAM limitée (RTX 2060).
    """

    def __init__(self):
        self.settings = get_settings()
        self.base_url = f"http://{self.settings.ollama_host}:{self.settings.ollama_port}"
        self.current_model: Optional[str] = None
        self._client = httpx.AsyncClient(timeout=300.0) # Long timeout for pulls/loads

        # Mapping des rôles vers les modèles configurés
        self.role_map = {
            "general": self.settings.council_model_general,
            "research": self.settings.council_model_research,
            "banker": self.settings.council_model_banker,
        }

    async def prepare_model(self, role: str = "general") -> str:
        """
        Prépare le modèle pour un rôle spécifique.
        Si le modèle n'est pas celui chargé, il effectue le swap.
        """
        target_model = self.role_map.get(role, self.settings.council_model_general)

        if self.current_model == target_model:
            return target_model

        logger.info(f"🔮 COUNCIL: Switching model to '{target_model}' for role '{role}'")
        
        try:
            # 1. Charger le nouveau modèle (Ollama charge automatiquement au premier call ou via /api/generate vide)
            # On utilise l'API /api/generate avec un prompt vide pour forcer le chargement
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": target_model, "prompt": "", "keep_alive": "5m"}
            )
            response.raise_for_status()
            
            self.current_model = target_model
            logger.info(f"✅ COUNCIL: Model '{target_model}' ready.")
            return target_model
            
        except Exception as e:
            logger.error(f"❌ COUNCIL Error: Failed to load model '{target_model}': {e}")
            # Fallback sur le modèle général par défaut si possible
            return self.settings.council_model_general

    async def unload_current(self):
        """Décharge le modèle actuel de la VRAM"""
        if self.current_model:
            logger.info(f"🧹 COUNCIL: Unloading '{self.current_model}'")
            try:
                await self._client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.current_model, "keep_alive": 0}
                )
                self.current_model = None
            except Exception as e:
                logger.warning(f"⚠️ COUNCIL: Failed to unload model: {e}")

_council_service: Optional[CouncilService] = None

def get_council_service() -> CouncilService:
    global _council_service
    if _council_service is None:
        _council_service = CouncilService()
    return _council_service
