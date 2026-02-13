"""
API du Service de Gamification EVA.
Expose les endpoints pour le traitement des trades, le statut et le tableau de bord RPG.
"""

from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from eva_gamification.engine import GamificationEngine
from eva_gamification.models import TradeResult, GameState
from shared import BaseHealthResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie : Initialise le moteur et la connexion Redis."""
    app.state.engine = GamificationEngine()
    await app.state.engine.initialize()
    yield
    # Nettoyage si nécessaire

app = FastAPI(
    title="Service de Gamification EVA",
    description="Microservice pour les mécaniques RPG et l'ajustement dynamique de la difficulté",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=BaseHealthResponse, tags=["Système"])
async def health_check():
    """Vérifie la santé du service."""
    return BaseHealthResponse(status="ok", version="1.0.0")

@app.post("/process-trade")
async def process_trade(trade: TradeResult, request: Request):
    """
    Reçoit un trade clôturé, calcule les SP (Points de Souveraineté) et met à jour l'état du jeu.
    Déclenché par EVA-Banker.
    """
    engine: GamificationEngine = request.app.state.engine
    try:
        result = await engine.process_trade_result(trade)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status(request: Request):
    """Retourne l'état interne complet (Debug)."""
    engine: GamificationEngine = request.app.state.engine
    return {
        "state": engine.state,
        "mab": engine.mab.state
    }

@app.get("/dashboard")
async def get_dashboard(request: Request):
    """
    Retourne les données formatées pour le tableau de bord RPG (Masque les Euros, affiche les SP/Progrès).
    """
    engine: GamificationEngine = request.app.state.engine
    quest_percent = engine.state.quest_progress * 100

    return {
        "sovereignty_points": int(engine.state.sp),
        "level": engine.state.level,
        "current_quest": {
            "name": engine.state.current_quest,
            "progress_percent": f"{quest_percent:.1f}%",
            "visual_bar": "#" * int(quest_percent / 5) + "-" * (20 - int(quest_percent / 5))
        },
        "inventory": engine.state.unlocked_techs,
        "difficulty_mode": engine.mab.state.current_arm.upper()
    }
