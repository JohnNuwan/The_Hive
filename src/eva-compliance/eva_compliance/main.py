"""
EVA Compliance — Agent Juridique & Fiscal de THE HIVE.

Ce module implémente le « Keeper » (Expert L) du système d'experts.
Il est responsable de :
- L'écoute des trades profitables via Redis Pub/Sub.
- Le provisionnement automatique des taxes (URSSAF, 25% BNC).
- La gestion du compte escrow (fonds bloqués pour l'État).
- L'exposition de l'identité juridique de l'entité.

Architecture :
    - Passif : écoute les événements du Banker et provisionne.
    - Aucune action de trading, uniquement comptable.
    - Persistance sur disque (escrow_ledger.json).
"""

import logging
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from eva_compliance.legal_wrapper import LegalWrapper
from eva_compliance.tax_manager import TaxManager
from shared.redis_client import init_redis, get_redis_client
from shared.auth_middleware import InternalAuthMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application Compliance.

    Initialise la connexion Redis, démarre les tâches de fond
    (écoute des trades, heartbeat) et instancie les services
    juridiques et fiscaux.

    Args:
        app (FastAPI): L'instance de l'application en cours.

    Yields:
        None: Rend la main à l'application une fois l'initialisation terminée.
    """
    logger.info("⚖️ Démarrage EVA Compliance (Le Keeper)...")

    # Redis — tolérant aux pannes au démarrage
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Services
    app.state.legal = LegalWrapper()
    app.state.tax_manager = TaxManager()

    # Tâches de fond
    asyncio.create_task(trade_listener(app.state.tax_manager))
    asyncio.create_task(hard_heartbeat())

    logger.info("✅ EVA Compliance actif et à l'écoute du Banker")
    yield
    logger.info("🛑 Arrêt EVA Compliance")


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="EVA Compliance (Keeper) API",
    description="Agent Juridique & Fiscal - THE HIVE",
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

app.add_middleware(InternalAuthMiddleware)


# ═══════════════════════════════════════════════════════════════════════════════
# TÂCHES DE FOND
# ═══════════════════════════════════════════════════════════════════════════════


async def trade_listener(tax_manager: TaxManager):
    """
    Écoute les trades profitables sur le canal Redis `eva.compliance.trades`.

    Chaque signal reçu du Banker est traité pour calculer la part fiscale
    et la provisionner dans le ledger escrow.

    Args:
        tax_manager (TaxManager): Instance du gestionnaire de taxes.
    """
    redis = get_redis_client()

    async def handle_trade(channel: str, message: dict):
        """Callback exécuté à la réception d'un signal de trade."""
        logger.info(f"⚖️ Signal de profit reçu: {message}")
        result = tax_manager.process_trade_result(message)
        logger.info(f"📝 Résultat provision: {result.get('message', result.get('status'))}")

    try:
        await redis.subscribe(["eva.compliance.trades"], handle_trade)
        await redis.listen()
    except Exception as e:
        logger.error(f"Erreur listener trades: {e}")


async def hard_heartbeat():
    """
    Signal haute fréquence pour l'Orchestrateur Core.

    Publie l'état « online » dans Redis sous la clé `eva.keeper.status`
    (le Core attend le nom « keeper » dans sa découverte d'agents).
    """
    redis = get_redis_client()
    while True:
        try:
            payload = {
                "status": "online",
                "ts": datetime.now().timestamp(),
                "expert": "keeper",
            }
            await redis.cache_set("eva.keeper.status", payload, ttl_seconds=10)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
        await asyncio.sleep(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", tags=["Système"])
async def health():
    """Vérifie la santé du module Compliance / Keeper."""
    return {"status": "online", "service": "compliance"}


@app.get("/ledger", tags=["Fiscal"])
async def get_ledger():
    """
    Récupère l'état du compte escrow (fonds bloqués pour l'URSSAF).

    Returns:
        dict: Total bloqué et nombre de transactions enregistrées.
    """
    tax_manager: TaxManager = app.state.tax_manager
    return tax_manager.get_escrow_status()


@app.get("/identity", tags=["Juridique"])
async def get_identity():
    """
    Retourne l'identité juridique publique de l'entité (SIRET, propriétaire).

    Returns:
        dict: Informations d'identité de la corporation.
    """
    legal: LegalWrapper = app.state.legal
    return legal.get_public_identity()
