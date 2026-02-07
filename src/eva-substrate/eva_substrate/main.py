import logging
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from eva_substrate.energy_monitor import EnergyMonitor
from eva_substrate.circadian_rhythm import CircadianRhythm
from eva_substrate.resource_allocator import ResourceAllocator
from shared.redis_client import init_redis, get_redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Substrate"""
    logger.info("🌿 Démarrage EVA Substrate (Le Corps)...")
    await init_redis()
    
    # Heartbeat
    asyncio.create_task(hard_heartbeat())
    
    logger.info("✅ EVA Substrate actif")
    yield
    logger.info("🛑 Arrêt EVA Substrate")

app = FastAPI(title="EVA Substrate", lifespan=lifespan)
monitor = EnergyMonitor()
rhythm = CircadianRhythm()
allocator = ResourceAllocator()

async def hard_heartbeat():
    """Signal de présence pour l'orchestrateur"""
    redis = get_redis_client()
    while True:
        mode_info = rhythm.get_current_mode()
        payload = {
            "status": "online", 
            "ts": datetime.now().timestamp(), 
            "expert": "substrate",
            "mode": mode_info["mode"],
            "is_night": mode_info["is_night"]
        }
        # Publication sur eva.substrate.status pour la découverte
        await redis.cache_set("eva.substrate.status", payload, ttl_seconds=10)
        await asyncio.sleep(2.0)

@app.get("/health")
async def health():
    return {"status": "online", "service": "substrate"}

@app.get("/metrics")
async def get_metrics():
    """Retourne les métriques énergétiques et hardware"""
    return monitor.get_current_consumption()

@app.get("/mode")
async def get_mode():
    """Retourne le mode actuel (Jour/Nuit)"""
    return rhythm.get_current_mode()

@app.post("/allocate")
async def allocate_tpus(profile: str):
    """Alloue les TPUs selon un profil spécifique"""
    return allocator.set_profile(profile)
