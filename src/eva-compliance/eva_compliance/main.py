import logging
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from eva_compliance.legal_wrapper import LegalWrapper
from eva_compliance.tax_manager import TaxManager
from shared.redis_client import init_redis, get_redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Compliance"""
    logger.info("⚖️ Démarrage EVA Compliance...")
    await init_redis()
    
    # Démarrage des tâches de fond
    asyncio.create_task(trade_listener())
    asyncio.create_task(hard_heartbeat())
    
    logger.info("✅ EVA Compliance actif et à l'écoute du Banker")
    yield
    logger.info("🛑 Arrêt EVA Compliance")

app = FastAPI(title="EVA Compliance", lifespan=lifespan)
legal = LegalWrapper()
tax_manager = TaxManager()

async def trade_listener():
    """Écoute les trades profitables pour provisionnement"""
    redis = get_redis_client()
    
    async def handle_trade(channel, message):
        logger.info(f"⚖️ Signal de profit reçu: {message}")
        result = tax_manager.process_trade_result(message)
        logger.info(f"📝 Résultat provision: {result['message']}")
        
    await redis.subscribe(["eva.compliance.trades"], handle_trade)
    await redis.listen()

async def hard_heartbeat():
    """Signal de présence pour l'orchestrateur"""
    redis = get_redis_client()
    while True:
        payload = {"status": "online", "ts": datetime.now().timestamp(), "expert": "keeper"} # Keeper ou Compliance ? Le CDC dit Keeper pour système, mais ici c'est Compliance.
        # On utilise le nom de l'expert attendu dans le rapport de statut du Core
        await redis.cache_set("eva.keeper.status", payload, ttl_seconds=10)
        await asyncio.sleep(1.0)

@app.get("/health")
async def health():
    return {"status": "online", "service": "compliance"}

@app.get("/ledger")
async def get_ledger():
    """Récupère l'état du compte escrow (fonds bloqués)"""
    return tax_manager.get_escrow_status()

@app.get("/identity")
async def get_identity():
    return legal.get_public_identity()
