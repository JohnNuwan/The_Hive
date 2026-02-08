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
from shared.auth_middleware import InternalAuthMiddleware

from eva_sentinel.services.monitor import SystemMonitor
from eva_sentinel.services.notifier import TelegramNotifier

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
    
    # Notifier
    app.state.notifier = TelegramNotifier()
    
    # Heartbeat
    import asyncio
    app.state.heartbeat_task = asyncio.create_task(hard_heartbeat())
    
    # Listeners de notifications
    app.state.notif_task = asyncio.create_task(notif_listener(app.state.notifier))
    
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


async def notif_listener(notifier: TelegramNotifier):
    """
    Écoute les canaux de la ruche et envoie des notifications Telegram.
    """
    from shared.redis_client import get_redis_client
    redis = get_redis_client()
    
    async def handle_alert(channel, message):
        # Dispatcher les alertes selon le canal
        if channel == "danger_signal":
            await notifier.notify_emergency("Nervous System", f"Signal critique détecté: {message}")
        
        elif channel == "eva.banker.trades":
            # Format attendu {ticket, symbol, profit}
            try:
                await notifier.notify_trade(
                    symbol=message.get("symbol", "UNKNOWN"),
                    profit=float(message.get("profit", 0.0)),
                    ticket=int(message.get("ticket_id", 0))
                )
            except Exception as e:
                logger.error(f"Failed to process trade notification: {e}")
        
        elif channel == "eva.swarm.healing":
            await notifier.notify_self_healing(
                service=message.get("service", "unknown"),
                event=message.get("event", "restart")
            )

    await redis.subscribe([
        "danger_signal", 
        "eva.banker.trades", 
        "eva.swarm.healing"
    ], handle_alert)
    
    logger.info("📡 Listener de notifications opérationnel")
    await redis.listen()

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

# Sécurité Inter-Agents
app.add_middleware(InternalAuthMiddleware)

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
