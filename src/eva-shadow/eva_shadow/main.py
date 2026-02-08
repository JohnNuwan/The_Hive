"""
The Shadow — Agent OSINT et Renseignement de THE HIVE.

Expert C du système d'experts. Responsable de :
- La recherche web et le scraping (DuckDuckGo, Brave Search).
- La reconnaissance d'entités (Entity Recon / Threat Intel).
- La veille sur les menaces et les opportunités.

Architecture :
    - Utilise httpx + BeautifulSoup pour le scraping web.
    - En production, peut se connecter à des API payantes
      (Brave Search, Shodan, VirusTotal, AlienVault OTX).
    - Heartbeat vers le Core pour la découverte des agents.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client

from eva_shadow.services.osint import OSINTService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application Shadow.

    Initialise Redis, instancie le service OSINT et démarre le heartbeat.

    Args:
        app (FastAPI): L'instance de l'application en cours.

    Yields:
        None: Rend la main une fois l'initialisation terminée.
    """
    logger.info("🌑 Démarrage The Shadow (OSINT Agent)...")

    # Redis — tolérant aux pannes au démarrage
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Service OSINT
    app.state.osint = OSINTService()

    # Heartbeat — MANQUAIT DANS LA VERSION PRÉCÉDENTE
    asyncio.create_task(hard_heartbeat())

    logger.info("✅ The Shadow dans les ténèbres (prêt)")

    yield

    logger.info("🛑 Arrêt The Shadow")


# ═══════════════════════════════════════════════════════════════════════════════
# TÂCHES DE FOND
# ═══════════════════════════════════════════════════════════════════════════════


async def hard_heartbeat():
    """
    Signal de présence pour l'Orchestrateur Core.

    Sans ce heartbeat, le Shadow apparaissait comme « offline »
    dans le dashboard du Core (/agents/status).
    """
    redis = get_redis_client()
    while True:
        try:
            payload = {
                "status": "online",
                "ts": datetime.now().timestamp(),
                "expert": "shadow",
            }
            await redis.cache_set("eva.shadow.status", payload, ttl_seconds=10)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="The Shadow API",
    description="Agent OSINT & Renseignement - THE HIVE",
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
    """Vérifie la santé du module Shadow."""
    return {"status": "ok", "service": "shadow"}


@app.get("/search", tags=["OSINT"])
async def search(q: str = Query(..., min_length=2)):
    """
    Recherche OSINT rapide via DuckDuckGo.

    Args:
        q (str): Requête de recherche (min 2 caractères).

    Returns:
        dict: Requête et liste des résultats trouvés.
    """
    osint_service: OSINTService = app.state.osint
    results = await osint_service.quick_search(q)
    return {"query": q, "results": results}


@app.get("/recon", tags=["OSINT"])
async def recon(target: str):
    """
    Recherche approfondie sur une cible (Entity Recon).

    Combine recherche web et analyse Threat Intel pour
    produire un rapport complet.

    Args:
        target (str): Cible de la reconnaissance (nom, domaine, IP).

    Returns:
        dict: Rapport d'investigation complet.
    """
    osint_service: OSINTService = app.state.osint
    report = await osint_service.entity_recon(target)
    return report
