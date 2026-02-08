"""
The Wraith - Agent Vision & Perception
Expert D: Analyse d'images, OCR, screenshot analysis.

En mode Lite (sans Coral TPU), Wraith offre :
- Capture et analyse de screenshots
- OCR basique via Tesseract (si installé)
- Analyse de graphiques de trading (chart reading)
- Description d'images via LLM multimodal (si disponible)
"""

import asyncio
import logging
import base64
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from typing import Any

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

class ScreenshotAnalysis(BaseModel):
    """Résultat d'analyse de screenshot"""
    description: str
    detected_elements: list[str] = []
    ocr_text: str | None = None
    chart_data: dict[str, Any] | None = None
    confidence: float = 0.0


class VisionStatus(BaseModel):
    """Statut du système de vision"""
    tpu_available: bool = False
    ocr_available: bool = False
    llm_vision_available: bool = False
    mode: str = "lite"


# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class VisionService:
    """Service de vision en mode lite"""

    def __init__(self):
        self.settings = get_settings()
        self.ocr_available = self._check_ocr()
        self.analysis_count = 0

    def _check_ocr(self) -> bool:
        """Vérifie si Tesseract OCR est disponible"""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            logger.info("Tesseract OCR non disponible — mode dégradé")
            return False

    async def analyze_screenshot(self, image_bytes: bytes) -> ScreenshotAnalysis:
        """Analyse un screenshot/image"""
        self.analysis_count += 1
        ocr_text = None
        detected_elements = []

        # OCR si disponible
        if self.ocr_available:
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(img)
                detected_elements.append("text_extracted")
            except Exception as e:
                logger.error(f"Erreur OCR: {e}")

        # Tentative d'analyse via LLM multimodal (Ollama avec llava)
        description = await self._llm_describe(image_bytes)

        return ScreenshotAnalysis(
            description=description,
            detected_elements=detected_elements,
            ocr_text=ocr_text,
            confidence=0.7 if description != "Mode offline" else 0.0
        )

    async def _llm_describe(self, image_bytes: bytes) -> str:
        """Tente une description via LLM multimodal"""
        import httpx
        try:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{self.settings.ollama_host}:{self.settings.ollama_port}/api/generate",
                    json={
                        "model": "llava",  # Modèle multimodal
                        "prompt": "Décris cette image en détail. Si c'est un graphique de trading, identifie la tendance, les niveaux clés et les patterns.",
                        "images": [b64_image],
                        "stream": False
                    }
                )
                data = response.json()
                return data.get("response", "Analyse impossible")
        except Exception as e:
            logger.warning(f"LLM Vision non disponible: {e}")
            return "Mode lite: Installez llava via 'ollama pull llava' pour l'analyse visuelle."

    async def capture_screen(self) -> bytes | None:
        """Capture l'écran actuel (Windows)"""
        try:
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            buffer = BytesIO()
            screenshot.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Capture d'écran impossible: {e}")
            return None

    def get_status(self) -> VisionStatus:
        return VisionStatus(
            tpu_available=False,
            ocr_available=self.ocr_available,
            llm_vision_available=False,  # Vrai seulement si llava est pull
            mode="lite"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Wraith"""
    logger.info("👁️ Démarrage The Wraith (Vision Lite)...")

    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    app.state.vision = VisionService()
    asyncio.create_task(hard_heartbeat())

    logger.info("✅ The Wraith observe (mode lite)")
    yield
    logger.info("🛑 Arrêt The Wraith")


async def hard_heartbeat():
    """Signal de présence"""
    redis = get_redis_client()
    while True:
        try:
            payload = {"status": "online", "ts": datetime.now().timestamp(), "expert": "wraith"}
            await redis.cache_set("eva.wraith.status", payload, ttl_seconds=10)
        except Exception:
            pass
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="The Wraith API",
    description="Agent Vision - THE HIVE (Lite Mode)",
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
    vision: VisionService = app.state.vision
    status = vision.get_status()
    return {"status": "ok", "service": "wraith", "mode": status.mode, "ocr": status.ocr_available}


@app.post("/analyze", response_model=ScreenshotAnalysis)
async def analyze_image(file: UploadFile = File(...)):
    """Analyse une image uploadée"""
    vision: VisionService = app.state.vision
    image_bytes = await file.read()
    return await vision.analyze_screenshot(image_bytes)


@app.get("/capture")
async def capture_and_analyze():
    """Capture l'écran et l'analyse"""
    vision: VisionService = app.state.vision
    screen_data = await vision.capture_screen()
    if screen_data is None:
        return {"error": "Capture d'écran impossible"}
    analysis = await vision.analyze_screenshot(screen_data)
    return analysis


@app.get("/status", response_model=VisionStatus)
async def get_vision_status():
    """Statut détaillé du système de vision"""
    vision: VisionService = app.state.vision
    return vision.get_status()
