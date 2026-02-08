"""
EVA Lab - Laboratoire d'Expérimentation & Backtesting
Expert Lab: Arena de combat, backtesting, évolution génétique, World Model.

C'est ici que les stratégies naissent, combattent et évoluent.
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

from eva_lab.arena import Arena
from eva_lab.backtester import Backtester
from eva_lab.dreamer_model import DreamerModel
from eva_lab.genetic_updater import GeneticUpdater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES API
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestRequest(BaseModel):
    """Requête de backtest"""
    strategy_name: str = Field(..., min_length=1)
    symbol: str = Field(default="XAUUSD")
    period_months: int = Field(default=6, ge=1, le=36)
    initial_balance: float = Field(default=10000.0, gt=0)


class ArenaRequest(BaseModel):
    """Requête de combat dans l'Arena"""
    challenger_id: str
    champion_id: str = "CURRENT_PROD"


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Lab"""
    logger.info("🧪 Démarrage EVA Lab (Le Colisée)...")

    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    app.state.arena = Arena()
    app.state.backtester = Backtester()
    app.state.dreamer = DreamerModel()
    app.state.genetic = GeneticUpdater()

    asyncio.create_task(hard_heartbeat())

    logger.info("✅ EVA Lab opérationnel — les stratégies peuvent combattre")
    yield
    logger.info("🛑 Arrêt EVA Lab")


async def hard_heartbeat():
    """Signal de présence"""
    redis = get_redis_client()
    while True:
        try:
            payload = {"status": "online", "ts": datetime.now().timestamp(), "expert": "lab"}
            await redis.cache_set("eva.lab.status", payload, ttl_seconds=10)
        except Exception:
            pass
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="EVA Lab API",
    description="Laboratoire d'Expérimentation - THE HIVE",
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
    return {"status": "online", "service": "lab"}


@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Lance un backtest sur données historiques"""
    backtester: Backtester = app.state.backtester
    result = await backtester.run_backtest(
        strategy_name=request.strategy_name,
        symbol=request.symbol,
        period_months=request.period_months,
        initial_balance=request.initial_balance
    )
    return result.to_dict()


@app.get("/backtest/history")
async def get_backtest_history():
    """Historique des backtests exécutés"""
    backtester: Backtester = app.state.backtester
    return {"backtests": backtester.get_history()}


@app.post("/arena/battle")
async def arena_battle(request: ArenaRequest):
    """Lance un combat de stratégies dans l'Arena"""
    arena: Arena = app.state.arena
    return arena.battle(request.challenger_id, request.champion_id)


@app.get("/arena/history")
async def arena_history():
    """Historique des combats de l'Arena"""
    arena: Arena = app.state.arena
    return {"battles": arena.history}


@app.get("/insights")
async def get_insights():
    """Prédictions du World Model (DreamerV3)"""
    dreamer: DreamerModel = app.state.dreamer
    return dreamer.predict_future_market()


@app.post("/evolve")
async def trigger_evolution():
    """Déclenche la boucle génétique d'amélioration"""
    genetic: GeneticUpdater = app.state.genetic
    return genetic.check_for_updates()


@app.get("/stats")
async def get_lab_stats():
    """Statistiques globales du Lab"""
    backtester: Backtester = app.state.backtester
    arena: Arena = app.state.arena
    return {
        "backtests_run": len(backtester.results_history),
        "arena_battles": len(arena.history),
        "active_experiments": 0,
        "best_strategy": backtester.results_history[-1].strategy_name if backtester.results_history else None
    }
