"""
EVA Substrate — Le Corps Biologique de THE HIVE.

Ce module gère les « fonctions vitales » de l'infrastructure :
- Monitoring de la consommation énergétique (CPU, GPU, réseau).
- Rythme circadien : adaptation jour/nuit des ressources.
- Allocation dynamique des accélérateurs (TPU/GPU).

Architecture :
    - Passif : collecte et expose les métriques.
    - Communique avec le Core via Redis heartbeat.
    - Peut réduire les charges la nuit (mode éco).
"""

import logging
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from eva_substrate.energy_monitor import EnergyMonitor
from eva_substrate.circadian_rhythm import CircadianRhythm
from eva_substrate.resource_allocator import ResourceAllocator
from shared.redis_client import init_redis, get_redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application Substrate.

    Initialise la connexion Redis, instancie les services de monitoring
    énergétique, rythme circadien et allocation de ressources.

    Args:
        app (FastAPI): L'instance de l'application en cours.

    Yields:
        None: Rend la main une fois l'initialisation terminée.
    """
    logger.info("🌿 Démarrage EVA Substrate (Le Corps)...")

    # Redis — tolérant aux pannes au démarrage
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Services
    app.state.monitor = EnergyMonitor()
    app.state.rhythm = CircadianRhythm()
    app.state.allocator = ResourceAllocator()

    # Heartbeat
    asyncio.create_task(hard_heartbeat(app.state.rhythm))

    logger.info("✅ EVA Substrate actif")
    yield
    logger.info("🛑 Arrêt EVA Substrate")


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="EVA Substrate API",
    description="Le Corps de THE HIVE — Énergie & Rythme Circadien",
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
# TÂCHES DE FOND
# ═══════════════════════════════════════════════════════════════════════════════


async def hard_heartbeat(rhythm: CircadianRhythm):
    """
    Signal de présence pour l'Orchestrateur Core.

    Inclut le mode circadien courant (jour/nuit) dans le payload
    pour que le Core puisse adapter le comportement global.

    Args:
        rhythm (CircadianRhythm): Service de rythme circadien.
    """
    redis = get_redis_client()
    while True:
        try:
            mode_info = rhythm.get_current_mode()
            payload = {
                "status": "online",
                "ts": datetime.now().timestamp(),
                "expert": "substrate",
                "mode": mode_info["mode"],
                "is_night": mode_info["is_night"],
            }
            await redis.cache_set("eva.substrate.status", payload, ttl_seconds=10)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", tags=["Système"])
async def health():
    """Vérifie la santé du module Substrate."""
    return {"status": "online", "service": "substrate"}


@app.get("/metrics", tags=["Énergie"])
async def get_metrics():
    """
    Retourne les métriques énergétiques et hardware.

    Returns:
        dict: Consommation CPU, GPU, réseau et température.
    """
    monitor: EnergyMonitor = app.state.monitor
    return monitor.get_current_consumption()


@app.get("/mode", tags=["Circadien"])
async def get_mode():
    """
    Retourne le mode circadien actuel (Jour/Nuit).

    Returns:
        dict: Mode courant, heure, état is_night.
    """
    rhythm: CircadianRhythm = app.state.rhythm
    return rhythm.get_current_mode()


@app.post("/allocate", tags=["Ressources"])
async def allocate_tpus(profile: str):
    """
    Alloue les accélérateurs (TPU/GPU) selon un profil spécifique.

    Args:
        profile (str): Profil d'allocation ('trading', 'analysis', 'sleep').

    Returns:
        dict: Résultat de l'allocation.
    """
    allocator: ResourceAllocator = app.state.allocator
    return allocator.set_profile(profile)
