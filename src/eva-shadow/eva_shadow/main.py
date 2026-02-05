"""
The Shadow - Agent OSINT et Recherche Web
Expert C: Enquêteur et Threat Intel
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from shared import get_settings
from shared.redis_client import init_redis

from eva_shadow.services.osint import OSINTService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Shadow"""
    logger.info("🌑 Démarrage The Shadow...")
    
    # Redis
    try:
        await init_redis()
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Service
    app.state.osint = OSINTService()
    
    logger.info("✅ The Shadow dans les ténèbres (prêt)")
    
    yield
    
    logger.info("🛑 Arrêt The Shadow")

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="The Shadow API",
    description="Agent OSINT - THE HIVE",
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
    return {"status": "ok", "service": "shadow"}

@app.get("/search")
async def search(q: str = Query(..., min_length=2)):
    """Recherche OSINT rapide"""
    osint_service: OSINTService = app.state.osint
    results = await osint_service.quick_search(q)
    return {"query": q, "results": results}

@app.get("/recon")
async def recon(target: str):
    """Recherche approfondie sur une cible (Entity Recon)"""
    osint_service: OSINTService = app.state.osint
    report = await osint_service.entity_recon(target)
    return report
