"""
EVA Gamification Service API.
Exposes endpoints for Trade Processing, Status, and RPG Dashboard.
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from eva_gamification.engine import GamificationEngine
from eva_gamification.models import TradeResult, GameState

# Global Engine Instance
engine = GamificationEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Initialize Engine and Redis connection."""
    await engine.initialize()
    yield
    # Cleanup if needed

app = FastAPI(
    title="EVA Gamification Service",
    description="Microservice for RPG Mechanics and Dynamic Difficulty Adjustment",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/process-trade")
async def process_trade(trade: TradeResult):
    """
    Receives a closed trade, calculates SP, and updates game state.
    Triggered by EVA-Banker.
    """
    try:
        result = await engine.process_trade_result(trade)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Returns the full internal state (Debug)."""
    return {
        "state": engine.state,
        "mab": engine.mab.state
    }

@app.get("/dashboard")
async def get_dashboard():
    """
    Returns data formatted for the RPG Dashboard (Hides Euros, shows SP/Progress).
    """
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
