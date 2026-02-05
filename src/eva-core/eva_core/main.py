"""
Application FastAPI Principale pour l'Orchestrateur EVA (Core).

Ce module est le point d'entrée de l'API REST centrale de THE HIVE.
Il gère :
- L'orchestration des messages utilisateurs.
- Le routage des intentions (Intents) vers les Experts appropriés via Redis.
- La gestion de la mémoire à court et long terme (RAG).
- Le cycle de vie de l'application (connexions BDD, Redis).

Architecture :
    - FastAPI pour le serveur web asynchrone.
    - Redis pour la communication inter-agents (Pub/Sub).
    - TimescaleDB/Qdrant pour le stockage.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared import (
    ChatMessage,
    Intent,
    MessageRole,
    Settings,
    get_settings,
)
from shared.redis_client import get_redis_client, init_redis

from eva_core.router.intent import IntentRouter
from eva_core.services.llm import LLMService, get_llm_service
from eva_core.services.memory import MemoryService, get_memory_service

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES API
# ═══════════════════════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    """Requête de chat"""
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: UUID | None = None


class ChatResponse(BaseModel):
    """Réponse de chat"""
    message: str
    session_id: UUID
    intent: Intent | None = None
    metadata: dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Réponse de santé"""
    status: str = "ok"
    version: str = "0.1.0"
    environment: str
    timestamp: datetime = Field(default_factory=datetime.now)


class SessionResponse(BaseModel):
    """Réponse création session"""
    session_id: UUID
    created_at: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application (Démarrage et Arrêt).

    Cette fonction est exécutée au lancement et à l'arrêt du serveur FastAPI.
    Elle est responsable de :
    1. Charger la configuration et les secrets.
    2. Initialiser les connexions aux bases de données (Redis, Postgres, Qdrant).
    3. Instancier les services singletons (LLM, Router, Memory).
    4. Nettoyer les ressources proprement lors de l'arrêt (Graceful Shutdown).

    Args:
        app (FastAPI): L'instance de l'application en cours.

    Yields:
        None: Rend la main à l'application une fois l'initialisation terminée.
    """
    # Startup
    logger.info("🚀 Démarrage EVA Core...")
    settings = get_settings()
    logger.info(f"Environnement: {settings.environment}")

    # Initialiser Redis
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Initialiser les services
    app.state.settings = settings
    app.state.intent_router = IntentRouter()
    app.state.llm_service = get_llm_service()
    app.state.memory_service = get_memory_service()

    logger.info("✅ EVA Core prêt")

    yield

    # Shutdown
    logger.info("🛑 Arrêt EVA Core...")
    redis_client = get_redis_client()
    await redis_client.disconnect()


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="EVA Core API",
    description="API principale de l'Orchestrateur EVA - THE HIVE",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", response_model=HealthResponse, tags=["Système"])
async def health_check() -> HealthResponse:
    """
    Vérifie l'état de santé opérationnel de l'API (Health Check).

    Utilisé par Docker, K8s ou le Watchdog (ESP32) pour vérifier si le service
    est en vie et réactif. Ne vérifie pas nécessairement toutes les dépendances
    profondes pour éviter les temps de latence, mais confirme que le thread
    principal tourne.

    Returns:
        HealthResponse: Objet contenant le statut 'ok', la version et l'environnement.
    """
    settings: Settings = app.state.settings
    return HealthResponse(
        status="ok",
        environment=settings.environment,
    )


@app.post("/session", response_model=SessionResponse, tags=["Chat"])
async def create_session() -> SessionResponse:
    """
    Initialise une nouvelle session de conversation.

    Génère un identifiant unique (UUID) pour tracer le contexte d'une discussion
    entre l'utilisateur et EVA. Cet ID doit être fourni dans les requêtes /chat
    suivantes pour maintenir l'historique-mémoire.

    Returns:
        SessionResponse: Contient le nouvel UUID de session généré.
    """
    session_id = uuid4()
    logger.info(f"Nouvelle session créée: {session_id}")
    return SessionResponse(session_id=session_id)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Traite un message utilisateur et génère une réponse orchestrée.

    C'est le cœur réactif du système. Le flux de traitement est le suivant :
    1. **Réception** : Validation du payload et récupération de la session.
    2. **Classification** : Le Router analyse l'intention (Intent) du message.
    3. **Routage** :
        - Si l'intent concerne le CORE (Chat général), le LLM répond directement.
        - Si l'intent est spécialisé (ex: Trading), le message est publié sur Redis
          pour l'Expert concerné (ex: Banker).
    4. **Mémorisation** : Le message utilisateur est archivé dans la mémoire vectorielle.

    Args:
        request (ChatRequest): Le message de l'utilisateur et l'ID de session.

    Returns:
        ChatResponse: La réponse textuelle (ou confirmation de dispatch) et les métadonnées.

    Raises:
        HTTPException(500): En cas d'erreur critique de traitement ou de connexion Redis.
    """
    session_id = request.session_id or uuid4()
    
    try:
        # Créer le message utilisateur
        user_message = ChatMessage(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message,
        )

        # Classification de l'intent
        intent_router: IntentRouter = app.state.intent_router
        intent = await intent_router.classify(request.message)
        logger.info(f"Intent classifié: {intent.intent_type} (confiance: {intent.confidence:.2f})")

        # Générer la réponse selon l'intent
        llm_service: LLMService = app.state.llm_service
        
        if intent.target_expert == "core":
            # Le Core répond directement
            response_text = await llm_service.generate_response(
                messages=[user_message],
                system_prompt="Tu es EVA, une IA assistante personnelle intelligente et bienveillante.",
            )
        else:
            # Router vers un autre expert via Redis
            redis_client = get_redis_client()
            await redis_client.send_to_agent(
                source="core",
                target=intent.target_expert,
                action=intent.intent_type.value,
                payload={
                    "session_id": str(session_id),
                    "message": request.message,
                    "entities": intent.entities,
                },
            )
            response_text = f"J'ai transmis ta demande à l'expert {intent.target_expert}. Réponse en cours..."

        # Sauvegarder en mémoire
        memory_service: MemoryService = app.state.memory_service
        await memory_service.store_message(user_message)

        return ChatResponse(
            message=response_text,
            session_id=session_id,
            intent=intent,
            metadata={
                "expert": intent.target_expert,
                "confidence": intent.confidence,
            },
        )

    except Exception as e:
        logger.exception(f"Erreur chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/search", tags=["Mémoire"])
async def search_memory(
    query: str,
    session_id: UUID | None = None,
    limit: int = 5,
) -> list[dict]:
    """
    Effectue une recherche sémantique dans la mémoire vectorielle (RAG).

    Permet de retrouver des fragments de conversations passées ou des documents
    ingérés pertinents par rapport à la requête `query`. Utilise Qdrant en backend.

    Args:
        query (str): Le texte ou le concept à rechercher.
        session_id (UUID | None, optional): Filtrer par session spécifique. Defaults to None.
        limit (int, optional): Nombre maximum de résultats à retourner. Defaults to 5.

    Returns:
        list[dict]: Liste des documents trouvés avec leur score de similarité.
    """
    memory_service: MemoryService = app.state.memory_service
    results = await memory_service.search(
        query=query,
        session_id=session_id,
        limit=limit,
    )
    return results


@app.get("/agents/status", tags=["Agents"])
async def agents_status() -> dict[str, Any]:
    """
    Récupère l'état de connexion de tous les Experts du Conseil.

    Interroge le registre (Redis ou Heartbeat) pour savoir quels services sont
    actuellement en ligne et prêts à recevoir des ordres.

    Returns:
        dict[str, Any]: Dictionnaire {nom_expert: {status: 'online'|'offline', ...}}
    """
    # TODO: Implémenter la découverte des agents via Redis
    return {
        "core": {"status": "online", "version": "0.1.0"},
        "banker": {"status": "unknown"},
        "sentinel": {"status": "unknown"},
    }
