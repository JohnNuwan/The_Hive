"""
The Muse - Agent Media & Création de Contenu
Expert G: Génération de contenu textuel, scripts, articles, posts sociaux.

En mode Lite (sans SDXL/GPU dédié), Muse utilise le LLM local (Ollama)
pour la génération textuelle : copywriting, articles, scripts YouTube, etc.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

class ContentRequest(BaseModel):
    """Requête de génération de contenu"""
    content_type: str = Field(..., description="Type: article, tweet, youtube_script, linkedin, email, ad_copy")
    topic: str = Field(..., min_length=3)
    tone: str = Field(default="professional", description="Ton: professional, casual, persuasive, technical, creative")
    language: str = Field(default="fr")
    max_length: int = Field(default=500, ge=50, le=5000)
    context: str | None = None


class ContentResponse(BaseModel):
    """Réponse avec contenu généré"""
    content: str
    content_type: str
    word_count: int
    generation_time_ms: int
    model_used: str


class ContentTemplate(BaseModel):
    """Template de contenu pré-défini"""
    name: str
    description: str
    content_type: str
    prompt_template: str


# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class MuseService:
    """Service de génération de contenu via LLM"""

    TEMPLATES: dict[str, ContentTemplate] = {
        "tweet_thread": ContentTemplate(
            name="Twitter Thread",
            description="Thread Twitter engageant sur un sujet tech/finance",
            content_type="tweet",
            prompt_template="Écris un thread Twitter de 5 tweets sur : {topic}. Ton: {tone}. Chaque tweet doit être percutant et < 280 caractères. Utilise des emojis stratégiquement."
        ),
        "linkedin_post": ContentTemplate(
            name="LinkedIn Post",
            description="Post LinkedIn professionnel avec hook",
            content_type="linkedin",
            prompt_template="Rédige un post LinkedIn engageant sur : {topic}. Commence par un hook accrocheur. Ton: {tone}. Inclus des bullet points et un CTA."
        ),
        "youtube_script": ContentTemplate(
            name="YouTube Script",
            description="Script vidéo YouTube structuré",
            content_type="youtube_script",
            prompt_template="Écris un script YouTube de 3-5 minutes sur : {topic}. Structure: Hook (10s) → Intro → 3 points clés → Conclusion + CTA. Ton: {tone}."
        ),
        "blog_article": ContentTemplate(
            name="Blog Article",
            description="Article de blog SEO-optimisé",
            content_type="article",
            prompt_template="Rédige un article de blog optimisé SEO sur : {topic}. Inclus: titre H1, sous-titres H2, introduction, développement, conclusion. Ton: {tone}. Langue: {language}."
        ),
        "ad_copy": ContentTemplate(
            name="Ad Copy",
            description="Copywriting publicitaire (Facebook Ads, Google Ads)",
            content_type="ad_copy",
            prompt_template="Écris 3 variations de copy publicitaire pour : {topic}. Chaque variation: Headline (< 30 car), Description (< 90 car), CTA. Ton: {tone}."
        ),
    }

    def __init__(self):
        self.settings = get_settings()
        self.generation_count = 0

    async def generate_content(self, request: ContentRequest) -> ContentResponse:
        """Génère du contenu via le LLM local (Ollama)"""
        import httpx
        start = datetime.now()

        # Construire le prompt
        template = self.TEMPLATES.get(request.content_type)
        if template:
            prompt = template.prompt_template.format(
                topic=request.topic,
                tone=request.tone,
                language=request.language
            )
        else:
            prompt = f"Génère du contenu de type '{request.content_type}' sur le sujet: {request.topic}. Ton: {request.tone}. Langue: {request.language}."

        if request.context:
            prompt += f"\n\nContexte additionnel: {request.context}"

        prompt += f"\n\nLongueur maximale: {request.max_length} mots."

        # Appel Ollama
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"http://{self.settings.ollama_host}:{self.settings.ollama_port}/api/generate",
                    json={
                        "model": self.settings.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.8, "top_p": 0.9}
                    }
                )
                data = response.json()
                content = data.get("response", "Erreur de génération")
        except Exception as e:
            logger.error(f"Erreur Ollama: {e}")
            content = f"[Mode Offline] Contenu placeholder pour '{request.topic}'. Connectez Ollama pour la génération réelle."

        elapsed = int((datetime.now() - start).total_seconds() * 1000)
        self.generation_count += 1

        return ContentResponse(
            content=content.strip(),
            content_type=request.content_type,
            word_count=len(content.split()),
            generation_time_ms=elapsed,
            model_used=self.settings.ollama_model
        )

    def get_templates(self) -> list[dict[str, str]]:
        """Retourne la liste des templates disponibles"""
        return [
            {"id": k, "name": v.name, "description": v.description, "type": v.content_type}
            for k, v in self.TEMPLATES.items()
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Muse"""
    logger.info("🎨 Démarrage The Muse (Media Factory)...")

    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    app.state.muse_service = MuseService()
    asyncio.create_task(hard_heartbeat())

    logger.info("✅ The Muse est inspirée (prête)")
    yield
    logger.info("🛑 Arrêt The Muse")


async def hard_heartbeat():
    """Signal de présence"""
    redis = get_redis_client()
    while True:
        try:
            payload = {"status": "online", "ts": datetime.now().timestamp(), "expert": "muse"}
            await redis.cache_set("eva.muse.status", payload, ttl_seconds=10)
        except Exception:
            pass
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="The Muse API",
    description="Agent Media & Création de Contenu - THE HIVE",
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
    return {"status": "ok", "service": "muse", "mode": "lite_text_only"}


@app.get("/templates")
async def list_templates():
    """Liste les templates de contenu disponibles"""
    service: MuseService = app.state.muse_service
    return {"templates": service.get_templates()}


@app.post("/generate", response_model=ContentResponse)
async def generate_content(request: ContentRequest):
    """Génère du contenu textuel via le LLM"""
    service: MuseService = app.state.muse_service
    return await service.generate_content(request)


@app.get("/stats")
async def get_stats():
    """Statistiques de génération"""
    service: MuseService = app.state.muse_service
    return {
        "total_generations": service.generation_count,
        "available_templates": len(service.TEMPLATES),
        "model": service.settings.ollama_model,
        "mode": "text_only_lite"
    }
