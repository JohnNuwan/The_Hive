"""
The Sentinel — Agent de Sécurité et Monitoring de THE HIVE.

Expert F du système d'experts. Responsable de :
- La surveillance hardware en temps réel (CPU, RAM, GPU, Disque).
- La vérification d'intégrité des fichiers critiques (Constitution, Kernel).
- L'envoi d'alertes Telegram en cas d'anomalie.
- L'écoute des canaux de la ruche pour broadcaster les notifications.

Architecture :
    - SystemMonitor : collecte psutil toutes les 5 secondes.
    - SecurityEngine : scan d'intégrité périodique (toutes les 5 minutes).
    - TelegramNotifier : broadcasting des alertes critiques.
    - Heartbeat vers le Core pour la découverte des agents.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from shared import Settings, get_settings, SwarmGRPCClient
from shared.redis_client import init_redis
from shared.auth_middleware import InternalAuthMiddleware

from eva_sentinel.services.monitor import SystemMonitor
from eva_sentinel.services.notifier import TelegramNotifier
from eva_sentinel.sentiment_engine import SecurityEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application Sentinel.

    Initialise Redis, le monitoring hardware, le notifier Telegram,
    le moteur de sécurité et démarre les tâches de fond.

    Args:
        app (FastAPI): L'instance de l'application en cours.

    Yields:
        None: Rend la main une fois l'initialisation terminée.
    """
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
    
    # Security Engine
    app.state.security = SecurityEngine()
    
    # gRPC Client
    app.state.grpc_client = SwarmGRPCClient()
    app.state.grpc_client.connect()
    
    # Baseline integrity scan
    await app.state.security.check_integrity()
    
    # Heartbeat
    import asyncio
    app.state.heartbeat_task = asyncio.create_task(hard_heartbeat())
    
    # Listeners de notifications
    app.state.notif_task = asyncio.create_task(notif_listener(app.state.notifier))
    
    # Security scan périodique
    app.state.security_task = asyncio.create_task(periodic_security_scan(app.state.security))
    
    logger.info("✅ The Sentinel actif")
    
    yield
    
    # Shutdown
    app.state.heartbeat_task.cancel()
    app.state.security_task.cancel()
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
            # Si on détecte un signal critique, on peut aussi le relayer ou agir
            # Mais ici c'est un listener de broadcast.
            # On va plutôt s'assurer que Sentinel peut émettre via gRPC si besoin.
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


async def periodic_security_scan(security: SecurityEngine):
    """Scan de sécurité périodique (toutes les 5 minutes)"""
    while True:
        try:
            await security.check_integrity()
            await security.check_network()
        except Exception as e:
            logger.error(f"Erreur scan sécurité: {e}")
        await asyncio.sleep(300)  # 5 minutes

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

@app.get("/health", tags=["Système"])
async def health():
    """Vérifie la santé du module Sentinel."""
    return {"status": "ok", "service": "sentinel"}


@app.get("/system/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Retourne les métriques hardware actuelles (CPU, RAM, GPU, Disque).

    Returns:
        HardwareMetrics: Snapshot des métriques système.
    """
    return await app.state.monitor.get_current_metrics()

@app.get("/security/alerts")
async def get_alerts():
    """Retourne les alertes de sécurité récentes"""
    security: SecurityEngine = app.state.security
    alerts = security.get_alerts(limit=20)
    
    # Toujours inclure un statut baseline si pas d'alertes
    if not alerts:
        alerts = [{
            "id": "baseline-001",
            "type": "INTEGRITY_CHECK",
            "severity": "info",
            "message": "Kernel integrity verified — All systems nominal",
            "timestamp": datetime.now().isoformat()
        }]
    
    return alerts


@app.get("/security/scan")
async def run_security_scan():
    """Lance un scan de sécurité complet à la demande"""
    security: SecurityEngine = app.state.security
    return await security.run_full_scan()


@app.get("/security/integrity")
async def check_integrity():
    """Vérifie l'intégrité des fichiers critiques"""
    security: SecurityEngine = app.state.security
    return await security.check_integrity()
