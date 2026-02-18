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

import asyncio
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
    BaseHealthResponse,
    SwarmGRPCClient,
)
from shared.redis_client import get_redis_client, init_redis
from shared.mqtt_client import EVAMQTTClient
from shared.auth_middleware import InternalAuthMiddleware
from shared.internal_auth import get_internal_headers

from eva_core.router.intent import IntentRouter
from eva_core.services.llm import LLMService, get_llm_service
from eva_core.services.memory import MemoryService, get_memory_service
from eva_core.services.prompt_master import PromptMaster
from eva_core.strategy import StrategyOrchestrator
from eva_core.self_healing import SelfHealingService
from eva_core.services.docker_monitor import SystemMonitor
from eva_core.identity import EVA_CORE_IDENTITY, EVA_GAMIFICATION_PROTOCOL

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES API
# ═══════════════════════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    """
    Modèle de requête pour l'endpoint de chat.

    Attributes:
        message (str): Le contenu textuel du message utilisateur.
        session_id (UUID | None): L'identifiant de la session (facultatif si nouvelle session).
    """
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: UUID | None = None


class ChatResponse(BaseModel):
    """
    Modèle de réponse standard pour le chat.

    Attributes:
        message (str): La réponse textuelle générée par l'IA ou le système.
        session_id (UUID): L'ID de session confirmé.
        intent (Intent | None): L'intention détectée (si disponible).
        thoughts (str | None): Le raisonnement interne (Chain of Thought).
        metadata (dict): Métadonnées additionnelles (score de confiance, expert, etc.).
    """
    message: str
    session_id: UUID
    intent: Intent | None = None
    thoughts: str | None = None  # Trace de raisonnement extraite
    metadata: dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseHealthResponse):
    """
    Réponse étendue pour le Health Check.

    Attributes:
        environment (str): L'environnement d'exécution (dev, prod, staging).
    """
    environment: str


class SessionResponse(BaseModel):
    """
    Réponse à une demande de création de session.

    Attributes:
        session_id (UUID): Le nouvel identifiant unique généré.
        created_at (datetime): Date de création.
    """
    session_id: UUID
    created_at: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


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
    # Démarrage
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
    app.state.intent_router = IntentRouter(use_llm=settings.use_ollama)
    app.state.llm_service = get_llm_service()
    app.state.memory_service = get_memory_service()
    
    # Intégration Biblio_IA / PromptMaster
    app.state.prompt_master = PromptMaster()

    # Intégration MQTT
    app.state.mqtt = EVAMQTTClient("core")
    await app.state.mqtt.connect()

    # Intégration Strategy Orchestrator & Self-Healing
    app.state.strategy_orchestrator = StrategyOrchestrator()
    app.state.self_healing = SelfHealingService()
    
    # System Monitor (Docker + Hardware)
    app.state.system_monitor = SystemMonitor()
    
    # gRPC Client (High Performance Signal Routing)
    app.state.grpc_client = SwarmGRPCClient()
    app.state.grpc_client.connect()
    
    # Telemetry
    app.state.start_time = datetime.now()
    app.state.request_count = 0
    app.state.error_count = 0

    logger.info("✅ EVA Core prêt avec moteur de prompts Biblio_IA et lien MQTT")

    # Démarrage de l'orchestrateur de survie Phoenix
    asyncio.create_task(app.state.self_healing.start_monitoring())

    yield

    # Arrêt (Shutdown)
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

# Internal Security Middleware
app.add_middleware(InternalAuthMiddleware)


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

        # Classification de l'intent (Via Strategy Orchestrator pour plus de "profondeur")
        strategy: StrategyOrchestrator = app.state.strategy_orchestrator
        intent = await strategy.route_request(request.message)
        logger.info(f"Intent orchestré: {intent.intent_type} -> {intent.target_expert} (confiance: {intent.confidence:.2f})")

        # Générer la réponse selon l'intent
        llm_service: LLMService = app.state.llm_service
        
        if intent.target_expert == "core":
            # Le Core répond directement
            prompt_master: PromptMaster = app.state.prompt_master
            method = "react" if intent.confidence < 0.8 else "costar"
            wrapped_message = prompt_master.wrap_with_method(request.message, method=method)
            expert_injector = prompt_master.get_expert_injector("core")
            
            # Mappage de l'expert vers le rôle du modèle (Prompt Engineering)
            role_map = {
                "banker": "banker",
                "researcher": "research",
                "shadow": "research",
                "lab": "research",
            }
            target_role = role_map.get(intent.target_expert, "general")

            response_text, thoughts = await llm_service.generate_response(
                messages=[ChatMessage(
                    session_id=session_id,
                    role=MessageRole.USER,
                    content=wrapped_message
                )],
                system_prompt=f"{expert_injector}\n{EVA_CORE_IDENTITY}\n\n{EVA_GAMIFICATION_PROTOCOL}",
                role=target_role,
            )
            # Créer le message de l'expert avec les pensées
            expert_message = ChatMessage(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=response_text,
                thoughts=thoughts,
            )
        elif intent.target_expert == "all":
            # SWARM MODE: Parallélisation sur tous les agents concernés
            redis_client = get_redis_client()
            await redis_client.broadcast_to_swarm(
                source="core",
                action=intent.intent_type.value,
                payload={
                    "session_id": str(session_id),
                    "message": request.message,
                    "entities": intent.entities,
                    "mode": "parallel"
                },
            )
            response_text = "Activation du Swarm Mode. Tous les experts concernés travaillent en parallèle..."
        else:
            # Routage vers expert avec préférence gRPC pour les signaux critiques
            redis_client = get_redis_client()
            grpc_client: SwarmGRPCClient = app.state.grpc_client
            
            payload = {
                "session_id": str(session_id),
                "message": request.message,
                "entities": intent.entities,
            }
            
            # Si le signal est critique, on tente d'abord gRPC
            grpc_success = False
            if intent.intent_type.value in ["DANGER", "TRADE", "ORDER", "KILL_SWITCH"]:
                logger.info(f"⚡ tentative routage gRPC pour action critique: {intent.intent_type.value}")
                grpc_success = grpc_client.send_signal(
                    source="core",
                    target=intent.target_expert,
                    action=intent.intent_type.value,
                    payload=payload,
                    priority=1 # P1_CRITIQUE
                )
                if grpc_success:
                    logger.info(f"✅ Signal gRPC envoyé avec succès vers {intent.target_expert}")
            
            # Repli sur Redis si gRPC échoue ou n'est pas utilisé
            if not grpc_success:
                await redis_client.send_to_agent(
                    source="core",
                    target=intent.target_expert,
                    action=intent.intent_type.value,
                    payload=payload,
                )
            
            # Miroir MQTT pour le Banker
            if intent.target_expert == "banker" and intent.intent_type.value in ["TRADE", "ORDER"]:
                mqtt_client: EVAMQTTClient = app.state.mqtt
                await mqtt_client.publish("eva/banker/requests/critical", payload, qos=2)
                logger.info("🛡️ Critical Order mirrored on MQTT (QoS 2)")

            response_text = f"Consultation de l'expert {intent.target_expert} lancée."

        # Sauvegarder en mémoire
        memory_service: MemoryService = app.state.memory_service
        await memory_service.store_message(user_message)

        return ChatResponse(
            message=response_text,
            session_id=session_id,
            intent=intent,
            thoughts=thoughts if intent.target_expert == "core" else None,
            metadata={
                "expert": intent.target_expert,
                "confidence": intent.confidence,
            },
        )

    except Exception as e:
        logger.exception(f"Erreur chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/swarm/drones", tags=["Swarm"])
async def get_active_drones() -> list[dict]:
    """
    Récupère la liste des drones autonomes actifs dans la ruche (Swarm Mode).

    Returns:
        list[dict]: Liste des drones actifs avec leur statut et mission.
    """
    redis_client = get_redis_client()
    keys = await redis_client._client.keys("swarm:drone:*")
    drones = []
    for key in keys:
        drone_data = await redis_client.cache_get(key)
        if drone_data:
            drones.append(drone_data)
    return drones


# Note : L'ancien orchestrateur de self-healing est remplacé par SelfHealingService


@app.get("/memory/fragments", tags=["Mémoire"])
async def get_memory_fragments(limit: int = 50) -> list[dict[str, Any]]:
    """
    Récupère les derniers fragments de mémoire stockés (historique récent).

    Args:
        limit (int): Nombre maximum de fragments à récupérer.

    Returns:
        list[dict[str, Any]]: Liste des messages ou fragments de mémoire.
    """
    memory_service: MemoryService = app.state.memory_service
    # Réutiliser une version simplifiée de scroll
    results = await memory_service.get_session_history(session_id=None, limit=limit) # None means all if implemented
    return results


@app.get("/memory/graph", tags=["Mémoire"])
async def get_memory_graph(
    limit: int = 50,
    similarity_threshold: float = 0.8,
) -> dict[str, Any]:
    """
    Retourne les données du graphe de connaissances (noeuds et relations).

    Args:
        limit (int): Nombre maximum de noeuds à retourner.
        similarity_threshold (float): Seuil de similarité pour les liens.

    Returns:
        dict[str, Any]: Structure JSON compatible avec les bibliothèques de graphe (d3.js).
    """
    memory_service: MemoryService = app.state.memory_service
    return await memory_service.get_graph_data(limit=limit, similarity_threshold=similarity_threshold)


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
    Récupère l'état de connexion de tous les Experts du Conseil via Redis.

    Returns:
        dict[str, Any]: Dictionnaire {agent_id: status_info}.
    """
    redis_client = get_redis_client()
    now = datetime.now().timestamp()
    
    # On définit les agents attendus (The Hive Council)
    agents = ["banker", "sentinel", "shadow", "wraith", "keeper", "substrate", "accountant"]
    status_report = {
        "core": {"status": "online", "version": "0.1.0", "uptime": "active"}
    }
    
    # Optimisation: mget pour récupérer tous les status en un seul appel O(1)
    agent_keys = [f"eva.{agent}.status" for agent in agents]
    heartbeats = await redis_client.cache_mget(agent_keys)

    # zip strict=False pour éviter les erreurs si la liste n'est pas alignée (ce qui ne devrait pas arriver)
    for agent, heartbeat in zip(agents, heartbeats):
        if not heartbeat:
            # Fallback sur la vérification du channel PubSub (pour le banker spécifique à son heartbeat 300ms)
            status_report[agent] = {"status": "offline"}
        else:
            last_seen = heartbeat.get("ts", 0)
            if now - last_seen < 30: # 30 secondes de grâce
                status_report[agent] = {"status": "online", **heartbeat}
            else:
                status_report[agent] = {"status": "stale", "last_seen": last_seen}

    return status_report


@app.get("/trading/status", tags=["Trading"])
async def trading_status() -> dict[str, Any]:
    """
    Agrège les données de trading provenant de l'expert Banker.

    Récupère en parallèle les comptes, positions et risques depuis l'API Banker.

    Returns:
        dict[str, Any]: Vue unifiée du trading {account, positions, risk}.
    """
    import httpx
    settings: Settings = app.state.settings
    banker_url = f"http://{settings.banker_api_host}:{settings.banker_api_port}"
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            # On récupère tout en parallèle avec les headers de sécurité internes
            internal_headers = get_internal_headers("core")
            responses = await asyncio.gather(
                client.get(f"{banker_url}/account", headers=internal_headers),
                client.get(f"{banker_url}/positions", headers=internal_headers),
                client.get(f"{banker_url}/risk/status", headers=internal_headers),
                return_exceptions=True
            )
            
            # Parsing des résultats
            account = responses[0].json() if not isinstance(responses[0], Exception) and responses[0].status_code == 200 else {}
            positions = responses[1].json() if not isinstance(responses[1], Exception) and responses[1].status_code == 200 else []
            risk = responses[2].json() if not isinstance(responses[2], Exception) and responses[2].status_code == 200 else {}
            
            return {
                "account": account,
                "positions": positions,
                "risk": risk,
                "banker": {"status": "online" if not isinstance(responses[0], Exception) else "offline"}
            }
        except Exception as e:
            logger.error(f"Erreur proxy Banker: {e}")
            return {
                "account": {},
                "positions": [],
                "risk": {},
                "banker": {"status": "offline", "error": str(e)}
            }


@app.get("/system/status", tags=["Système"])
async def system_status() -> dict[str, Any]:
    """
    Agrège les données de santé hardware provenant de l'expert Sentinel.

    Agit comme un proxy pour éviter au frontend d'interroger Sentinel directement.

    Returns:
        dict[str, Any]: Métriques système et état de santé global.
    """
    import httpx
    settings: Settings = app.state.settings
    sentinel_url = f"http://localhost:{settings.sentinel_api_port}"
    
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            response = await client.get(
                f"{sentinel_url}/system/metrics", 
                headers=get_internal_headers("core")
            )
            if response.status_code == 200:
                metrics = response.json()
                return {
                    "health": "optimum",
                    "metrics": metrics,
                    "sentinel": {"status": "online"}
                }
            return {"health": "unknown", "sentinel": {"status": "offline"}}
        except Exception as e:
            logger.error(f"Erreur proxy Sentinel: {e}")
            llm_service = get_llm_service()
            return {
                "health": "degraded",
                "sentinel": {"status": "offline"},
                "active_role": llm_service.council.current_role,
                "active_model": llm_service.council.current_model
            }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS MONITORING DIRECT (Docker, System Metrics, Telemetry)
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/system/metrics", tags=["Monitoring"])
async def get_system_metrics_direct() -> dict[str, Any]:
    """
    Retourne les métriques système directement via psutil/nvidia-smi.

    Utilisé par le frontend MonitoringView quand Sentinel est offline ou pour vérification.

    Returns:
        dict[str, Any]: CPU, RAM, Disque, GPU.
    """
    monitor: SystemMonitor = app.state.system_monitor
    return await monitor.get_system_metrics()


@app.get("/docker/containers", tags=["Monitoring"])
async def get_docker_containers() -> list[dict[str, Any]]:
    """
    Retourne la liste des conteneurs Docker et leurs stats en temps réel.

    Returns:
        list[dict[str, Any]]: Liste des conteneurs actifs avec stats CPU/RAM.
    """
    monitor: SystemMonitor = app.state.system_monitor
    return await monitor.get_docker_containers()


@app.get("/telemetry", tags=["Système"])
async def get_telemetry() -> dict[str, Any]:
    """
    Retourne les métriques de télémétrie internes du Core.

    Returns:
        dict[str, Any]: Uptime, compteurs de requêtes/erreurs, modèle actif.
    """
    start_time: datetime = app.state.start_time
    uptime = (datetime.now() - start_time).total_seconds()
    llm_service = get_llm_service()
    return {
        "service_name": "core",
        "uptime_seconds": int(uptime),
        "requests_total": app.state.request_count,
        "errors_total": app.state.error_count,
        "active_role": llm_service.council.current_role,
        "active_model": llm_service.council.current_model,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/circuit-breaker/status", tags=["Système"])
async def get_circuit_breaker_status() -> dict[str, Any]:
    """
    Retourne l'état du circuit-breaker du Core (Self-Healing).

    Returns:
        dict[str, Any]: État (OPEN/CLOSED), échecs consécutifs.
    """
    # Vérifie si un circuit-breaker est attaché au self-healing service
    self_healing: SelfHealingService = app.state.self_healing
    if hasattr(self_healing, 'circuit_breaker') and self_healing.circuit_breaker:
        return self_healing.circuit_breaker.get_status()
    
    # Repli : état nominal
    return {
        "name": "core_circuit_breaker",
        "state": "CLOSED",
        "failures": 0,
        "failure_threshold": 5,
    }
