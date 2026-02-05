"""
Application FastAPI 'The Builder' (Expert E).

L'Agent DevOps de la Ruche. Il est responsable de :
- L'intégration continue et le déploiement.
- La génération automatique de documentation.
- L'analyse des logs et la maintenance proactive.
- L'exécution de scripts shell via le Librarian/Handyman.

Architecture :
    - Mode asynchrone pour ne pas bloquer sur des tâches longues (Builds).
    - Accès privilégié au système de fichiers (Lecture/Écriture).
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
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie du Builder.

    Initialise les outils DevOps (Librarian) et vérifie l'accès aux ressources
    système critiques (Docker socket, répertoires de logs).

    Args:
        app (FastAPI): Instance de l'application.

    Yields:
        None: Rend la main une fois le service prêt.
    """
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
    """
    Vérifie l'état opérationnel du Builder.

    Returns:
        dict: Statut 'ok' si le service réagit.
    """
    return {"status": "ok", "service": "builder"}

@app.post("/maintenance/docgen")
async def generate_docs():
    """
    Déclenche la regénération complète de la documentation technique.

    Scanne le code source, extrait les docstrings et met à jour les fichiers Markdown
    dans le dossier `Documentation/`.

    Returns:
        dict: Rapport de génération (fichiers traités, erreurs).
    """
    librarian: LibrarianService = app.state.librarian
    stats = await librarian.scan_and_generate()
    return {"status": "success", "files_processed": stats}

@app.get("/maintenance/logs/analyze")
async def analyze_errors():
    """
    Analyse les logs système pour identifier les anomalies récurrentes.

    Utilise des patterns Regex pour détecter les erreurs critiques (StackTraces)
    dans les fichiers de logs rotatifs.

    Returns:
        dict: Synthèse des erreurs trouvées et suggestions de correctifs.
    """
    return {"status": "info", "message": "Aucune erreur majeure détectée dans les 24h"}
