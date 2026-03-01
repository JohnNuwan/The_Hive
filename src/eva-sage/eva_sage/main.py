"""
The Sage - Agent Santé, Bien-être & Conseil Environnemental.
Expert H: Monitoring de la santé de l'opérateur, conseils ergonomiques,
analyse du rythme circadien, rappels d'hydratation, nutrition et pauses.

Architecture :
    - Suivi de session de travail en temps réel.
    - Intégration du rythme circadien (via Substrate).
    - Recommandations personnalisées basées sur le contexte.
    - Dashboard santé consolidé.
"""

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

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


class WellnessReport(BaseModel):
    """Rapport de bien-être."""
    session_duration_minutes: int
    break_recommended: bool
    hydration_reminder: bool
    ergonomic_tip: str
    circadian_status: str
    recommendations: list[str]
    overall_score: float = Field(ge=0, le=10)
    timestamp: datetime = Field(default_factory=datetime.now)


class NutritionEntry(BaseModel):
    """Entrée de suivi nutritionnel."""
    type: str = Field(..., description="Type: water, meal, snack, coffee, supplement")
    amount: float = Field(default=1.0, description="Quantité (verres d'eau, portions...)")
    notes: str = ""


class MoodEntry(BaseModel):
    """Entrée de suivi d'humeur."""
    mood: str = Field(..., description="Mood: excellent, good, neutral, tired, stressed")
    energy_level: int = Field(default=5, ge=1, le=10)
    notes: str = ""


class SleepEntry(BaseModel):
    """Entrée de sommeil."""
    hours: float = Field(..., gt=0, le=24)
    quality: str = Field(default="good", description="Quality: excellent, good, average, poor")
    notes: str = ""


ERGONOMIC_TIPS = [
    "Vérifiez votre posture : dos droit, épaules détendues, pieds à plat.",
    "L'écran doit être à hauteur des yeux, à 50-70cm de distance.",
    "Étirez vos poignets et vos doigts toutes les 30 minutes.",
    "Faites 20 secondes de regard au loin toutes les 20 minutes (règle 20-20-20).",
    "Levez-vous et marchez 2 minutes toutes les heures.",
    "Détendez vos trapèzes : laissez tomber vos épaules du bout des doigts.",
    "Respirez profondément 5 fois : inspirer 4s, retenir 4s, expirer 6s.",
    "Hydratation : buvez au moins 200ml d'eau maintenant.",
    "Vérifiez la luminosité de votre écran (adapter au contexte).",
    "Detox digitale : fermez les yeux 30 secondes et concentrez-vous sur vos sensations.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE
# ═══════════════════════════════════════════════════════════════════════════════


class WellnessService:
    """Service de bien-être et coaching santé."""

    def __init__(self):
        self.session_start = datetime.now()
        self.breaks_taken = 0
        self.last_break = None
        self.tip_index = 0
        self.water_count = 0
        self.meal_count = 0
        self.coffee_count = 0
        self.nutrition_log: deque[dict[str, Any]] = deque(maxlen=100)
        self.mood_log: deque[dict[str, Any]] = deque(maxlen=100)
        self.sleep_log: deque[dict[str, Any]] = deque(maxlen=30)

    def get_session_minutes(self) -> int:
        return int((datetime.now() - self.session_start).total_seconds() / 60)

    def get_wellness_report(self) -> dict[str, Any]:
        """Génère un rapport de bien-être complet."""
        session_min = self.get_session_minutes()
        now = datetime.now()
        hour = now.hour

        # Recommandation de pause
        minutes_since_break = (
            int((now - self.last_break).total_seconds() / 60)
            if self.last_break else session_min
        )
        break_recommended = minutes_since_break > 45

        # Rappel hydratation
        expected_water = max(1, session_min // 45)
        hydration_reminder = self.water_count < expected_water

        # Status circadien
        if 6 <= hour < 10:
            circadian = "🌅 Matin — Phase d'éveil (cortisol élevé, idéal pour tâches créatives)"
        elif 10 <= hour < 14:
            circadian = "☀️ Milieu de journée — Pic de productivité"
        elif 14 <= hour < 16:
            circadian = "😴 Post-déjeuner — Baisse d'énergie (micro-sieste recommandée)"
        elif 16 <= hour < 19:
            circadian = "🔥 Regain d'énergie — Bon pour les tâches analytiques"
        elif 19 <= hour < 22:
            circadian = "🌙 Soirée — Ralentir, éviter les écrans bleus"
        else:
            circadian = "🌑 Nuit — Repos fortement recommandé"

        # Tip ergonomique
        tip = ERGONOMIC_TIPS[self.tip_index % len(ERGONOMIC_TIPS)]
        self.tip_index += 1

        # Recommandations dynamiques
        recommendations = []
        if break_recommended:
            recommendations.append("⏸️ Vous travaillez depuis trop longtemps. Prenez une pause de 5 min.")
        if hydration_reminder:
            recommendations.append(f"💧 Hydratation : vous n'avez bu que {self.water_count} verre(s). Objectif : {expected_water}.")
        if self.coffee_count > 3:
            recommendations.append("☕ Attention : vous avez dépassé 3 cafés. Privilégiez l'eau ou le thé.")
        if hour >= 22:
            recommendations.append("🛏️ Il est tard. Envisagez de vous reposer pour un meilleur lendemain.")
        if session_min > 480:
            recommendations.append("🏃 Session de 8h+. Votre corps a besoin d'exercice physique.")
        if not recommendations:
            recommendations.append("✅ Tout va bien ! Continuez à ce rythme.")

        # Score global
        score = 8.0
        if break_recommended: score -= 1.5
        if hydration_reminder: score -= 1.0
        if hour >= 23: score -= 2.0
        if self.coffee_count > 4: score -= 0.5
        score = max(1.0, min(10.0, score))

        return {
            "session_duration_minutes": session_min,
            "break_recommended": break_recommended,
            "hydration_reminder": hydration_reminder,
            "ergonomic_tip": tip,
            "circadian_status": circadian,
            "recommendations": recommendations,
            "overall_score": round(score, 1),
            "timestamp": now.isoformat(),
        }

    def take_break(self) -> dict[str, Any]:
        self.breaks_taken += 1
        self.last_break = datetime.now()
        return {
            "breaks_taken": self.breaks_taken,
            "duration_since_start": self.get_session_minutes(),
            "message": "✅ Pause enregistrée. Bougez, hydratez-vous !"
        }

    def log_nutrition(self, entry: dict) -> dict[str, Any]:
        entry["timestamp"] = datetime.now().isoformat()
        self.nutrition_log.append(entry)
        if entry.get("type") == "water":
            self.water_count += int(entry.get("amount", 1))
        elif entry.get("type") == "coffee":
            self.coffee_count += int(entry.get("amount", 1))
        elif entry.get("type") == "meal":
            self.meal_count += 1
        return {"status": "logged", "entry": entry}

    def get_dashboard(self) -> dict[str, Any]:
        session_min = self.get_session_minutes()
        return {
            "session": {
                "started_at": self.session_start.isoformat(),
                "duration_minutes": session_min,
                "breaks_taken": self.breaks_taken,
            },
            "nutrition": {
                "water_glasses": self.water_count,
                "meals": self.meal_count,
                "coffees": self.coffee_count,
            },
            "mood_entries": len(self.mood_log),
            "sleep_entries": len(self.sleep_log),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌿 Démarrage The Sage (Santé & Bien-être)...")
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    app.state.wellness = WellnessService()
    asyncio.create_task(hard_heartbeat())
    logger.info("✅ The Sage veille sur votre santé")
    yield
    logger.info("🛑 Arrêt The Sage")


async def hard_heartbeat():
    try:
        redis = get_redis_client()
    except Exception:
        redis = None
    while True:
        try:
            if redis:
                payload = {"status": "online", "ts": datetime.now().timestamp(), "expert": "sage"}
                await redis.cache_set("eva.sage.status", payload, ttl_seconds=10)
        except Exception:
            pass
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(title="The Sage API", description="Agent Santé & Bien-être - THE HIVE", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", tags=["Système"])
async def health():
    return {"status": "ok", "service": "sage"}


@app.get("/wellness", tags=["Bien-être"])
async def get_wellness():
    """Rapport de bien-être actuel avec recommandations."""
    w: WellnessService = app.state.wellness
    return w.get_wellness_report()


@app.post("/break", tags=["Bien-être"])
async def register_break():
    """Enregistre une pause de l'opérateur."""
    w: WellnessService = app.state.wellness
    return w.take_break()


@app.get("/session", tags=["Bien-être"])
async def get_session():
    """Statistiques de la session de travail."""
    w: WellnessService = app.state.wellness
    return w.get_dashboard()


@app.post("/nutrition", tags=["Nutrition"])
async def log_nutrition(entry: NutritionEntry):
    """Enregistre une entrée nutritionnelle (eau, repas, café...)."""
    w: WellnessService = app.state.wellness
    return w.log_nutrition(entry.model_dump())


@app.get("/nutrition/history", tags=["Nutrition"])
async def nutrition_history():
    """Historique des entrées nutritionnelles."""
    w: WellnessService = app.state.wellness
    return {"log": list(w.nutrition_log), "total": len(w.nutrition_log)}


@app.post("/mood", tags=["Humeur"])
async def log_mood(entry: MoodEntry):
    """Enregistre l'état d'humeur et le niveau d'énergie."""
    w: WellnessService = app.state.wellness
    data = {**entry.model_dump(), "timestamp": datetime.now().isoformat()}
    w.mood_log.append(data)
    return {"status": "logged", "entry": data}


@app.get("/mood/history", tags=["Humeur"])
async def mood_history():
    """Historique des entrées d'humeur."""
    w: WellnessService = app.state.wellness
    return {"log": list(w.mood_log), "total": len(w.mood_log)}


@app.post("/sleep", tags=["Sommeil"])
async def log_sleep(entry: SleepEntry):
    """Enregistre une nuit de sommeil."""
    w: WellnessService = app.state.wellness
    data = {**entry.model_dump(), "timestamp": datetime.now().isoformat()}
    w.sleep_log.append(data)
    return {"status": "logged", "entry": data}


@app.get("/sleep/history", tags=["Sommeil"])
async def sleep_history():
    """Historique du sommeil."""
    w: WellnessService = app.state.wellness
    return {"log": list(w.sleep_log), "total": len(w.sleep_log)}


@app.get("/circadian", tags=["Circadien"])
async def get_circadian():
    """État circadien actuel et recommandations associées."""
    hour = datetime.now().hour
    phases = [
        (range(6, 10), "morning", "🌅 Phase d'éveil", "Tâches créatives, planification"),
        (range(10, 14), "peak", "☀️ Pic de productivité", "Tâches complexes, décisions importantes"),
        (range(14, 16), "dip", "😴 Creux post-déjeuner", "Tâches routinières, micro-sieste 20min"),
        (range(16, 19), "rebound", "🔥 Regain d'énergie", "Tâches analytiques, sport"),
        (range(19, 22), "wind_down", "🌙 Ralentissement", "Lecture, détente, pas d'écran bleu"),
    ]
    for hours, phase_id, label, tip in phases:
        if hour in hours:
            return {"phase": phase_id, "label": label, "tip": tip, "hour": hour}
    return {"phase": "sleep", "label": "🌑 Phase de repos", "tip": "Dormez !", "hour": hour}


@app.get("/dashboard", tags=["Dashboard"])
async def get_dashboard():
    """Dashboard santé consolidé."""
    w: WellnessService = app.state.wellness
    dashboard = w.get_dashboard()
    wellness = w.get_wellness_report()
    return {**dashboard, "wellness": wellness}
