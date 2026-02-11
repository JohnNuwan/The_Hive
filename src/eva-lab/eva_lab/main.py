"""
EVA Lab - Laboratoire d'Expérimentation & Backtesting
Expert Lab: Arena de combat, backtesting, évolution génétique, World Model.

Sprint 5 : Shadow Learning + Feature Flag DreamerV3.
C'est ici que les stratégies naissent, combattent et évoluent.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client

from eva_lab.arena import Arena
from eva_lab.backtester import Backtester
from eva_lab.dreamer_model import DreamerModel
from eva_lab.genetic_updater import GeneticUpdater
from eva_lab.shadow_learning import ShadowLearningService
from eva_lab.dreamer_gate import DreamerGate

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


class TradeRecordRequest(BaseModel):
    """Requête d'enregistrement d'un trade pour le Shadow Learning"""
    symbol: str = "XAUUSD"
    action: str = "BUY"
    price: float = 0.0
    volume: float = 0.01
    pnl: float = 0.0
    indicators: Optional[dict] = None
    done: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Lab — avec Feature Flags Sprint 5"""
    settings = get_settings()
    logger.info("🧪 Démarrage EVA Lab (Le Colisée)...")

    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # ─── Modules classiques ───
    app.state.arena = Arena()
    app.state.backtester = Backtester()
    app.state.dreamer = DreamerModel()
    app.state.genetic = GeneticUpdater()

    # ─── Sprint 5 : Feature Flags ───
    app.state.dreamer_gate = DreamerGate(
        enable_training=settings.enable_dreamer_training,
    )

    # ─── Sprint 5 : Shadow Learning ───
    if settings.enable_shadow_learning:
        app.state.shadow = ShadowLearningService(
            data_dir="data/shadow_learning",
            buffer_size=settings.shadow_learning_buffer_size,
            dreamer_enabled=settings.enable_dreamer_training,
        )
        # Lancer le flush automatique en tâche de fond
        asyncio.create_task(
            app.state.shadow.start_auto_flush(
                interval_seconds=settings.shadow_learning_flush_interval
            )
        )
        logger.info("📡 Shadow Learning actif — collecte passive DreamerV3")
    else:
        app.state.shadow = None
        logger.info("💤 Shadow Learning désactivé")

    asyncio.create_task(hard_heartbeat())

    logger.info("✅ EVA Lab opérationnel — les stratégies peuvent combattre")
    yield
    
    # Flush final avant arrêt
    if app.state.shadow:
        count = app.state.shadow.manual_flush()
        logger.info(f"💾 Shadow Learning: {count} transitions saved on shutdown")
    
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
    description="Laboratoire d'Expérimentation - THE HIVE (Sprint 5: Shadow Learning)",
    version="0.2.0",
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


# ═══════════════════════════════════════════════════════════════════════════════
# SPRINT 5 ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/shadow/record")
async def record_trade(request: TradeRecordRequest):
    """Enregistre un trade dans le Shadow Learning buffer.

    Les données collectées seront utilisées pour entraîner DreamerV3
    quand ENABLE_DREAMER_TRAINING=True.
    """
    shadow: ShadowLearningService = app.state.shadow
    if not shadow:
        return {"status": "disabled", "reason": "ENABLE_SHADOW_LEARNING=False"}

    shadow.record_trade(
        symbol=request.symbol,
        action=request.action,
        price=request.price,
        volume=request.volume,
        pnl=request.pnl,
        indicators=request.indicators,
        done=request.done,
    )
    return {"status": "recorded", "buffer_size": shadow.buffer.size}


@app.post("/shadow/flush")
async def flush_shadow():
    """Force un flush immédiat du buffer Shadow Learning sur disque."""
    shadow: ShadowLearningService = app.state.shadow
    if not shadow:
        return {"status": "disabled"}
    count = shadow.manual_flush()
    return {"status": "flushed", "transitions_written": count}


@app.get("/shadow/stats")
async def shadow_stats():
    """Statistiques du Shadow Learning."""
    shadow: ShadowLearningService = app.state.shadow
    if not shadow:
        return {"status": "disabled"}
    return shadow.get_stats()


@app.get("/dreamer/status")
async def dreamer_status():
    """Statut du DreamerV3 Gate (Feature Flag)."""
    gate: DreamerGate = app.state.dreamer_gate
    return gate.get_status()


@app.post("/dreamer/predict")
async def dreamer_predict(observation: dict):
    """Exécute une prédiction via le World Model (inference-only ou training)."""
    gate: DreamerGate = app.state.dreamer_gate
    return gate.run_inference(observation)


@app.post("/dreamer/train")
async def dreamer_train():
    """Tente de lancer l'entraînement DreamerV3 (bloqué si Flag=False)."""
    gate: DreamerGate = app.state.dreamer_gate
    return gate.start_training(data_dir="data/shadow_learning")


# ═══════════════════════════════════════════════════════════════════════════════
# STATS (UPDATED)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/stats")
async def get_lab_stats():
    """Statistiques globales du Lab (incluant Sprint 5)"""
    backtester: Backtester = app.state.backtester
    arena: Arena = app.state.arena
    gate: DreamerGate = app.state.dreamer_gate
    shadow: ShadowLearningService = app.state.shadow

    stats = {
        "backtests_run": len(backtester.results_history),
        "arena_battles": len(arena.history),
        "active_experiments": 0,
        "best_strategy": backtester.results_history[-1].strategy_name if backtester.results_history else None,
        "dreamer": gate.get_status(),
    }
    if shadow:
        stats["shadow_learning"] = shadow.get_stats()

    return stats

