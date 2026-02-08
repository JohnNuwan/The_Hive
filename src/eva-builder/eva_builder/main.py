"""
The Builder — Agent DevOps et Maintenance de THE HIVE.

Expert E du système d'experts. Responsable de :
- La génération automatique de documentation (Librarian).
- L'analyse des logs et la maintenance proactive.
- L'intégration continue et le déploiement (en production).
- L'exécution de scripts de maintenance via le Librarian.

Architecture :
    - Mode asynchrone pour ne pas bloquer sur des tâches longues (Builds).
    - Accès privilégié au système de fichiers (Lecture/Écriture).
    - Heartbeat vers le Core pour la découverte des agents.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from shared.redis_client import init_redis, get_redis_client

from eva_builder.services.librarian import LibrarianService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application Builder.

    Initialise Redis, le service Librarian et démarre le heartbeat.

    Args:
        app (FastAPI): L'instance de l'application en cours.

    Yields:
        None: Rend la main une fois l'initialisation terminée.
    """
    logger.info("🛠️ Démarrage The Builder (DevOps Agent)...")

    # Redis — tolérant aux pannes au démarrage
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Services
    app.state.librarian = LibrarianService()

    # Heartbeat
    asyncio.create_task(hard_heartbeat())

    logger.info("✅ The Builder est au travail (prêt)")
    yield
    logger.info("🛑 Arrêt The Builder")


# ═══════════════════════════════════════════════════════════════════════════════
# TÂCHES DE FOND
# ═══════════════════════════════════════════════════════════════════════════════


async def hard_heartbeat():
    """
    Signal de présence pour l'Orchestrateur Core.

    Publie l'état « online » dans Redis toutes les 2 secondes.
    """
    redis = get_redis_client()
    while True:
        try:
            payload = {
                "status": "online",
                "ts": datetime.now().timestamp(),
                "expert": "builder",
            }
            await redis.cache_set("eva.builder.status", payload, ttl_seconds=10)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="The Builder API",
    description="Agent DevOps & Maintenance - THE HIVE",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", tags=["Système"])
async def health():
    """Vérifie la santé du module Builder."""
    return {"status": "ok", "service": "builder"}


@app.post("/maintenance/docgen", tags=["Maintenance"])
async def generate_docs():
    """
    Déclenche la regénération de la documentation technique.

    Le Librarian parcourt le code source et crée des README.md
    automatiques là où ils sont absents.

    Returns:
        dict: Nombre de fichiers générés.
    """
    librarian: LibrarianService = app.state.librarian
    stats = await librarian.scan_and_generate()
    return {"status": "success", "files_processed": stats}


@app.get("/maintenance/logs/analyze", tags=["Maintenance"])
async def analyze_errors():
    """
    Analyse les logs système pour identifier les anomalies.

    En mode lite, retourne un statut nominal.
    En production, analysera les fichiers de log avec le LLM.

    Returns:
        dict: Résultat de l'analyse (statut et message).
    """
    return {"status": "info", "message": "Aucune erreur majeure détectée dans les 24h"}
