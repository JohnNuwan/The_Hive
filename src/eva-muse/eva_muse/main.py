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

from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client
from shared.telegram_client import TelegramClient

from eva_muse.services.comfy_client import ComfyUIClient

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

class MediaRequest(BaseModel):
    """Requête de génération de média (Image/Vidéo)"""
    prompt: str = Field(..., description="Description de l'image (ex: Cyberpunk hacker, glowing green screens)")
    width: int = Field(default=1024)
    height: int = Field(default=1024)
    media_type: str = Field(default="image", description="Type: image, video")

class TradeResult(BaseModel):
    """Résultat d'un trade à viraliser"""
    symbol: str = Field(..., description="Asset symbol (ex: BTCUSD, XAUUSD)")
    action: str = Field(..., description="BUY or SELL")
    pnl: float = Field(..., description="Trade Profit & Loss")


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

@app.post("/generate/media")
async def generate_media(request: MediaRequest):
    """Génère une image (Media Factory) via l'API locale ComfyUI"""
    client = ComfyUIClient()
    
    # Workflow API "Standard" (SDXL API format minimaliste). 
    # Pour FLUX.1 ou d'autres complexités, il faudra charger un JSON externe.
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 8,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": 8566257,
                "steps": 20
            }
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": request.height, "width": request.width}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": request.prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "text, watermark, ugly, low quality"}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Hive_Muse", "images": ["8", 0]}}
    }
    
    try:
        images = await client.generate_from_workflow(workflow)
        if not images:
            raise HTTPException(status_code=500, detail="ComfyUI n'a renvoyé aucune image.")
        
        # On retourne l'image binaire avec le content-type image/png (prend le premier blob)
        return Response(content=images[0], media_type="image/png")
    
    except Exception as e:
        logger.error(f"Erreur Génération Media: {e}")
        # Message d'erreur formaté si ComfyUI n'est pas encore installé ou si le modèle manque
        raise HTTPException(
            status_code=503, 
            detail=f"Moteur Media Factory injoignable ou erreur de rendu. Assurez-vous que ComfyUI tourne sur le port 8188. ({str(e)})"
        )


@app.post("/viralize/trade")
async def viralize_trade(trade: TradeResult):
    """Viralize un trade gagnant: Génère une image hype et la poste sur Telegram"""
    
    # 1. Construire le prompt
    import random
    styles = [
        "cinematic lighting, hyperrealistic, 8k resolution, octane render",
        "anime style, neon genesis evangelion, high quality, masterpiece",
        "cyberpunk street, neon signs, rainy night, highly detailed"
    ]
    style = random.choice(styles)
    
    prompt = f"Cyberpunk hacker celebrating a successful {trade.action} trade on {trade.symbol} for ${trade.pnl:.2f}, glowing neon green holograms, {style}"
    
    # 2. Appeler ComfyUI
    client = ComfyUIClient()
    workflow = {
        "3": {"class_type": "KSampler", "inputs": {"cfg": 8, "denoise": 1, "latent_image": ["5", 0], "model": ["4", 0], "negative": ["7", 0], "positive": ["6", 0], "sampler_name": "euler", "scheduler": "normal", "seed": 8566257, "steps": 20}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": 1024, "width": 1024}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "text, watermark, ugly, low quality"}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Hive_Muse_Trade", "images": ["8", 0]}}
    }
    
    try:
        images = await client.generate_from_workflow(workflow)
        if not images:
            raise HTTPException(status_code=500, detail="ComfyUI n'a renvoyé aucune image.")
            
        img_bytes = images[0]
        
        # 3. Envoyer sur Telegram
        caption = f"🚀 *E.V.A Win !*\n\nAsset: `{trade.symbol}`\nAction: `{trade.action}`\nProfit: `+${trade.pnl:.2f}`\n\n*The Hive Automata | Muse Media Factory*"
        telegram = TelegramClient()
        await telegram.send_photo(img_bytes, caption)
        
        return {"status": "success", "message": "Trade viralized successfully!"}
        
    except Exception as e:
        logger.error(f"Erreur Viralization Trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
