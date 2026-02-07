"""
The Sentinel - Agent de Sécurité et Monitoring THE HIVE
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from shared import Settings, get_settings
from shared.redis_client import init_redis

from eva_sentinel.services.monitor import SystemMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Sentinel"""
    logger.info("🛡️ Démarrage The Sentinel...")
    
    # Redis
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Monitor
    app.state.monitor = SystemMonitor()
    await app.state.monitor.start()
    
    # Heartbeat
    import asyncio
    app.state.heartbeat_task = asyncio.create_task(hard_heartbeat())
    
    logger.info("✅ The Sentinel actif")
    
    yield
    
    # Shutdown
    app.state.heartbeat_task.cancel()
    await app.state.monitor.stop()
    logger.info("🛑 Arrêt The Sentinel")


async def hard_heartbeat():
    """
    Signal haute fréquence pour l'Orchestrateur Core.
    Persiste l'état dans Redis pour la découverte des agents.
    """
    from shared.redis_client import get_redis_client
    from datetime import datetime
    import asyncio
    
    redis = get_redis_client()
    while True:
        try:
            payload = {"status": "online", "ts": datetime.now().timestamp(), "expert": "sentinel"}
            await redis.cache_set("eva.sentinel.status", payload, ttl_seconds=10)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
        await asyncio.sleep(1.0)

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="The Sentinel API",
    description="Agent de Sécurité - THE HIVE",
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

@app.get("/health")
async def health():
    return {"status": "ok", "service": "sentinel"}

@app.get("/system/metrics")
async def get_metrics():
    """Retourne les métriques hardware actuelles"""
    return await app.state.monitor.get_current_metrics()

@app.get("/security/alerts")
async def get_alerts():
    """Retourne les alertes de sécurité récentes"""
    # TODO: Intégration OSINT/Wazuh
    return [
        {
            "id": "alert-001",
            "type": "INTEGRITY_CHECK",
            "severity": "info",
            "message": "Kernel hashing OK",
            "timestamp": "2026-02-05T11:55:00Z"
        }
    ]
