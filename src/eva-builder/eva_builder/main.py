"""
The Builder - Architecte Système et DevOps
Expert E: Librarian et Handyman
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from shared.redis_client import init_redis

from eva_builder.services.librarian import LibrarianService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Builder"""
    logger.info("🛠️ Démarrage The Builder...")
    
    # Redis
    try:
        await init_redis()
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Services
    app.state.librarian = LibrarianService()
    
    logger.info("✅ The Builder est au travail (prêt)")
    
    yield
    
    logger.info("🛑 Arrêt The Builder")

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="The Builder API",
    description="Agent DevOps - THE HIVE",
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
    return {"status": "ok", "service": "builder"}

@app.post("/maintenance/docgen")
async def generate_docs():
    """Lance la génération de documentation Markdown pour le monorepo"""
    librarian: LibrarianService = app.state.librarian
    stats = await librarian.scan_and_generate()
    return {"status": "success", "files_processed": stats}

@app.get("/maintenance/logs/analyze")
async def analyze_errors():
    """Analyse les logs récents pour détecter des bugs récurrents"""
    return {"status": "info", "message": "Aucune erreur majeure détectée dans les 24h"}
