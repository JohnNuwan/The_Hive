"""
The Sage - Agent Santé, Bien-être & Conseil Environnemental
Expert H: Monitoring de la santé de l'opérateur, conseils ergonomiques,
analyse du rythme circadien, rappels d'hydratation et pauses.

En mode Lite, Sage fonctionne comme un coach bien-être intégré
qui utilise les données système (uptime, heure) pour donner des conseils.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

class WellnessReport(BaseModel):
    """Rapport de bien-être"""
    session_duration_minutes: int
    break_recommended: bool
    hydration_reminder: bool
    ergonomic_tip: str
    circadian_status: str  # optimal, warning, critical
    recommendations: list[str]
    timestamp: datetime = Field(default_factory=datetime.now)


class SessionStats(BaseModel):
    """Statistiques de la session de travail"""
    started_at: datetime
    duration_minutes: int
    breaks_taken: int
    last_break: datetime | None
    productivity_score: float  # 0-100


# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class WellnessService:
    """Service de bien-être et coaching santé"""

    ERGONOMIC_TIPS = [
        "Vérifiez votre posture : dos droit, épaules détendues, pieds à plat.",
        "Règle 20-20-20 : toutes les 20 min, regardez à 20 pieds pendant 20 sec.",
        "Ajustez la luminosité de votre écran à l'ambiance lumineuse de la pièce.",
        "Gardez vos poignets neutres quand vous tapez — pas d'angle excessif.",
        "Votre écran devrait être à distance d'un bras, le haut au niveau des yeux.",
        "Étirez votre cou doucement : inclinez la tête à gauche, puis à droite.",
        "Faites des rotations d'épaules : 10 vers l'avant, 10 vers l'arrière.",
        "Levez-vous et marchez 2 minutes toutes les 45 minutes.",
        "Respirez profondément : inspirez 4s, retenez 7s, expirez 8s (méthode 4-7-8).",
        "Vérifiez la température de la pièce : 20-22°C est optimal pour la concentration.",
    ]

    def __init__(self):
        self.session_start = datetime.now()
        self.breaks_taken = 0
        self.last_break: datetime | None = None
        self.tip_index = 0

    def get_wellness_report(self) -> WellnessReport:
        """Génère un rapport de bien-être basé sur le contexte actuel"""
        now = datetime.now()
        duration = int((now - self.session_start).total_seconds() / 60)
        hour = now.hour

        # Déterminer le statut circadien
        if 6 <= hour <= 10:
            circadian = "optimal"
            phase = "Phase matinale — pic de cortisol, idéal pour le travail analytique."
        elif 10 <= hour <= 14:
            circadian = "optimal"
            phase = "Phase productive — concentration maximale."
        elif 14 <= hour <= 16:
            circadian = "warning"
            phase = "Creux post-prandial — ralentissement naturel, tâches légères recommandées."
        elif 16 <= hour <= 20:
            circadian = "optimal"
            phase = "Second pic — bon moment pour la créativité et les meetings."
        elif 20 <= hour <= 23:
            circadian = "warning"
            phase = "Phase de transition — mélatonine en hausse, réduire la lumière bleue."
        else:
            circadian = "critical"
            phase = "Nuit profonde — le repos est essentiel pour la performance cognitive."

        # Recommandations
        recommendations = [phase]
        break_recommended = False
        hydration = False

        if duration > 45 and (not self.last_break or (now - self.last_break).total_seconds() > 2700):
            break_recommended = True
            recommendations.append("Pause recommandée : vous travaillez depuis plus de 45 minutes sans pause.")

        if duration > 0 and duration % 30 < 5:
            hydration = True
            recommendations.append("Hydratation : buvez un verre d'eau (objectif 2L/jour).")

        if hour >= 22:
            recommendations.append("Il est tard. Considérez arrêter pour maintenir votre rythme circadien.")

        # Tip ergonomique rotatif
        tip = self.ERGONOMIC_TIPS[self.tip_index % len(self.ERGONOMIC_TIPS)]
        self.tip_index += 1

        return WellnessReport(
            session_duration_minutes=duration,
            break_recommended=break_recommended,
            hydration_reminder=hydration,
            ergonomic_tip=tip,
            circadian_status=circadian,
            recommendations=recommendations,
        )

    def take_break(self) -> dict[str, Any]:
        """Enregistre une pause"""
        self.breaks_taken += 1
        self.last_break = datetime.now()
        return {
            "message": "Pause enregistrée. Bravo pour prendre soin de vous !",
            "breaks_today": self.breaks_taken,
            "break_time": self.last_break.isoformat()
        }

    def get_session_stats(self) -> SessionStats:
        """Statistiques de session"""
        now = datetime.now()
        duration = int((now - self.session_start).total_seconds() / 60)

        # Score de productivité basé sur les pauses (optimal: 1 pause / 45 min)
        expected_breaks = max(1, duration // 45)
        break_ratio = min(1.0, self.breaks_taken / expected_breaks) if expected_breaks > 0 else 1.0
        score = min(100, break_ratio * 80 + 20)  # 20 points de base

        return SessionStats(
            started_at=self.session_start,
            duration_minutes=duration,
            breaks_taken=self.breaks_taken,
            last_break=self.last_break,
            productivity_score=round(score, 1)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Sage"""
    logger.info("🧘 Démarrage The Sage (Wellness Coach)...")

    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    app.state.wellness = WellnessService()
    asyncio.create_task(hard_heartbeat())

    logger.info("✅ The Sage veille sur votre bien-être")
    yield
    logger.info("🛑 Arrêt The Sage")


async def hard_heartbeat():
    """Signal de présence"""
    redis = get_redis_client()
    while True:
        try:
            payload = {"status": "online", "ts": datetime.now().timestamp(), "expert": "sage"}
            await redis.cache_set("eva.sage.status", payload, ttl_seconds=10)
        except Exception:
            pass
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="The Sage API",
    description="Agent Bien-être & Santé - THE HIVE",
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
    return {"status": "ok", "service": "sage"}


@app.get("/wellness", response_model=WellnessReport)
async def get_wellness():
    """Rapport de bien-être actuel avec recommandations"""
    service: WellnessService = app.state.wellness
    return service.get_wellness_report()


@app.post("/break")
async def register_break():
    """Enregistre une pause de l'opérateur"""
    service: WellnessService = app.state.wellness
    return service.take_break()


@app.get("/session", response_model=SessionStats)
async def get_session():
    """Statistiques de la session de travail"""
    service: WellnessService = app.state.wellness
    return service.get_session_stats()
