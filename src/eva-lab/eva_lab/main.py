"""
EVA Lab - Laboratoire d'ExpÃ©rimentation & Backtesting
Expert Lab: Arena de combat, backtesting, Ã©volution gÃ©nÃ©tique, World Model.

Sprint 5 : Shadow Learning + Feature Flag DreamerV3.
C'est ici que les stratÃ©gies naissent, combattent et Ã©voluent.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client

from eva_lab.arena import Arena
from eva_lab.backtester import Backtester
from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.dreamer_model import DreamerModel
from eva_lab.genetic_updater import GeneticUpdater
from eva_lab.shadow_learning import ShadowLearningService
from eva_lab.dreamer_gate import DreamerGate
from eva_lab.training_status import (
    build_training_universe_summary,
    load_nightly_summary,
    load_training_status,
    tail_training_log,
)
from eva_lab.training_utils import get_gnn_model_kwargs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    """
    Lit un booleen simple depuis l'environnement.

    Args:
        name (str): Nom de la variable.
        default (bool): Valeur de repli si absente.

    Returns:
        bool: Valeur booleenne normalisee.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


async def _probe_tcp_dependency(name: str, host: str, port: int) -> dict[str, Any]:
    """
    Teste une dependance TCP simple depuis le conteneur Lab.

    Args:
        name (str): Nom logique de la dependance.
        host (str): Hote cible.
        port (int): Port cible.

    Returns:
        dict[str, Any]: Etat minimal de disponibilite.
    """
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=1.5)
        writer.close()
        await writer.wait_closed()
        return {"name": name, "ok": True, "state": "online", "host": host, "port": port}
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "state": "offline",
            "host": host,
            "port": port,
            "error": str(exc),
        }


async def _collect_training_dependencies(run_status: dict[str, Any]) -> dict[str, Any]:
    """
    Agrege les dependances utiles a la lecture du run.

    Args:
        run_status (dict[str, Any]): Statut courant du training.

    Returns:
        dict[str, Any]: Dependances enrichies pour Nexus.
    """
    launcher = dict(run_status.get("launcher") or {})
    dependencies = dict(run_status.get("dependencies") or {})

    vllm_host = os.getenv("VLLM_API_HOST", "vllm")
    redis_host = os.getenv("REDIS_HOST", "redis")
    neo4j_host = os.getenv("NEO4J_HOST", "neo4j")
    mqtt_host = os.getenv("HIVE_MQTT_HOST", "mosquitto")

    vllm_state = str(launcher.get("vllm_state") or "").lower()
    if vllm_state == "stopped_for_training":
        dependencies["vllm"] = {
            "name": "vllm",
            "ok": False,
            "state": "stopped_for_training",
            "host": vllm_host,
            "port": 8000,
        }
    else:
        dependencies["vllm"] = await _probe_tcp_dependency("vllm", vllm_host, 8000)

    dependencies["redis"] = await _probe_tcp_dependency("redis", redis_host, 6379)
    dependencies["neo4j"] = await _probe_tcp_dependency("neo4j", neo4j_host, 7687)
    dependencies["mosquitto"] = await _probe_tcp_dependency("mosquitto", mqtt_host, 1883)

    trainer_container = launcher.get("trainer_container")
    trainer_running = bool(run_status.get("active")) or bool(trainer_container)
    dependencies["trainer"] = {
        "name": "trainer",
        "ok": trainer_running,
        "state": "running" if trainer_running else "idle",
        "container": trainer_container,
        "pid": launcher.get("remote_pid"),
    }
    return dependencies


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODÃˆLES API
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class BacktestRequest(BaseModel):
    """
    ParamÃ¨tres pour lancer une simulation de backtesting.

    Attributes:
        strategy_name (str): Nom de la stratÃ©gie Ã  tester.
        symbol (str): Actif financier (ex: XAUUSD).
        period_months (int): DurÃ©e de l'historique en mois.
        initial_balance (float): Capital de dÃ©part simulÃ©.
    """
    strategy_name: str = Field(..., min_length=1)
    symbol: str = Field(default="XAUUSD")
    period_months: int = Field(default=6, ge=1, le=36)
    initial_balance: float = Field(default=10000.0, gt=0)


class ArenaRequest(BaseModel):
    """
    RequÃªte de duel algorithmique dans l'Arena.

    Attributes:
        challenger_id (str): ID de la stratÃ©gie dÃ©fiante.
        champion_id (str): ID de la stratÃ©gie en place (dÃ©faut: PROD).
    """
    challenger_id: str
    champion_id: str = "CURRENT_PROD"
    horizon: str = "intraday"


class TradeRecordRequest(BaseModel):
    """
    RequÃªte d'enregistrement d'un trade rÃ©el ou simulÃ©.

    UtilisÃ© pour le Shadow Learning (entraÃ®nement passif).

    Attributes:
        symbol (str): Actif concernÃ©.
        action (str): BUY ou SELL.
        pnl (float): Profit ou perte rÃ©alisÃ©.
        done (bool): Si le trade clÃ´ture une sÃ©quence (Ã©pisode).
    """
    symbol: str = "XAUUSD"
    action: str = "BUY"
    price: float = 0.0
    volume: float = 0.01
    pnl: float = 0.0
    indicators: Optional[dict] = None
    observation: Optional[dict] = None
    next_observation: Optional[dict] = None
    metadata: Optional[dict] = None
    timestamp: Optional[str] = None
    done: bool = False

class GNNPredictRequest(BaseModel):
    """RequÃªte d'infÃ©rence pour le GNN (Multi-Asset correlation)"""
    assets_data: dict[str, list[list[float]]]  # { "XAUUSD": [[...features...], ...], ... }



# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LIFECYCLE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie du Lab â€” avec Indicateurs de FonctionnalitÃ© (Feature Flags).

    Args:
        app (FastAPI): Instance de l'application.

    Yields:
        None: Rend le contrÃ´le aprÃ¨s initialisation.
    """
    settings = get_settings()
    logger.info("ðŸ§ª DÃ©marrage EVA Lab (Le ColisÃ©e)...")

    try:
        await init_redis()
        logger.info("âœ… Redis connectÃ©")
    except Exception as e:
        logger.warning(f"âš ï¸ Redis non disponible: {e}")

    # â”€â”€â”€ Modules classiques â”€â”€â”€
    app.state.arena = Arena()
    app.state.backtester = Backtester()
    app.state.dreamer = DreamerModel()
    app.state.genetic = GeneticUpdater()
    app.state.promoter = ChampionPromoter()

    # â”€â”€â”€ Sprint 5 : Feature Flags â”€â”€â”€
    app.state.dreamer_gate = DreamerGate(
        enable_training=settings.enable_dreamer_training,
    )

    # â”€â”€â”€ Sprint 5 : Shadow Learning (Apprentissage FantÃ´me) â”€â”€â”€
    if settings.enable_shadow_learning:
        app.state.shadow = ShadowLearningService(
            data_dir="data/shadow_learning",
            buffer_size=settings.shadow_learning_buffer_size,
            dreamer_enabled=settings.enable_dreamer_training,
        )
        # Lancer le flush automatique en tÃ¢che de fond
        asyncio.create_task(
            app.state.shadow.start_auto_flush(
                interval_seconds=settings.shadow_learning_flush_interval
            )
        )
        logger.info("ðŸ“¡ Shadow Learning actif â€” collecte passive DreamerV3")
    else:
        app.state.shadow = None
        logger.info("ðŸ’¤ Shadow Learning dÃ©sactivÃ©")

    # â”€â”€â”€ GNN / Hydra (MTF Omni-Architecture) â”€â”€â”€
    try:
        from eva_lab.models.gnn_model import TFTGNNModel
        import torch
        import os
        # MTF Architecture: asset_dim=20 features, temporal_dim=32, hidden_dim=64, 3 classes
        app.state.gnn_model = TFTGNNModel(**get_gnn_model_kwargs())
        
        # Load weights if trained
        model_path = "data/models/gnn_master.pth"
        if os.path.exists(model_path):
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                app.state.gnn_model.load_state_dict(torch.load(model_path, map_location=device))
                logger.info("ðŸ§  MTF-GNN Loaded (Trained Weights: Scalp + Intraday + Swing).")
            except Exception as w_e:
                logger.warning(f"Failed to load GNN weights, running randomly initialized: {w_e}")
        else:
            logger.info("ðŸ§  MTF-GNN initialized (Untrained - run train_gnn.py to evolve).")
            
        app.state.gnn_model.eval()
    except Exception as e:
        logger.warning(f"âš ï¸ Erreur chargement GNN (Stub Mode probable): {e}")
        app.state.gnn_model = None

    asyncio.create_task(hard_heartbeat())
    if _env_flag("ENABLE_LAB_INTERNAL_NIGHTLY_SCHEDULER", False):
        asyncio.create_task(_nightly_training_loop())
    else:
        logger.info("Planificateur nightly interne desactive; le cron Debian reste prioritaire.")

    logger.info("âœ… EVA Lab opÃ©rationnel â€” les stratÃ©gies peuvent combattre")
    yield
    
    # Flush final avant arrÃªt
    if app.state.shadow:
        count = app.state.shadow.manual_flush()
        logger.info(f"ðŸ’¾ Shadow Learning: {count} transitions saved sur arrÃªt")
    
    logger.info("ðŸ›‘ ArrÃªt EVA Lab")


async def hard_heartbeat():
    """
    Envoie un signal de vie pÃ©riodique (Heartbeat) Ã  Redis.
    """
    redis = get_redis_client()
    while True:
        try:
            payload = {"status": "online", "ts": datetime.now().timestamp(), "expert": "lab"}
            await redis.cache_set("eva.lab.status", payload, ttl_seconds=10)
        except Exception:
            pass
        await asyncio.sleep(2.0)

async def _nightly_training_loop():
    """
    DÃ©clenche l'entraÃ®nement des modÃ¨les tous les soirs Ã  23h40.
    """
    logger.info("ðŸŒ™ Planificateur d'entraÃ®nement nocturne activÃ© (Cible: 23h40).")
    while True:
        try:
            now = datetime.now()
            target = now.replace(hour=23, minute=40, second=0, microsecond=0)
            
            if now > target:
                target += timedelta(days=1)
                
            wait_seconds = (target - now).total_seconds()
            
            # Attendre jusqu'Ã  23h40
            await asyncio.sleep(wait_seconds)
            
            logger.info("ðŸš€ DÃ©but de l'entraÃ®nement nocturne automatique (23h40)!")
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "train_nightly_stack.py")
            if os.path.exists(script_path):
                # Utiliser le shell pour hÃ©riter de l'environnement venv
                process = await asyncio.create_subprocess_shell(
                    f"python {script_path}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    logger.info("âœ… EntraÃ®nement nocturne terminÃ© avec succÃ¨s.")
                    redis = get_redis_client()
                    await redis.publish("eva.lab.events", {"action": "TRAINING_COMPLETE", "timestamp": datetime.now().isoformat()})
                else:
                    logger.error(f"âŒ Ã‰chec de l'entraÃ®nement nocturne ({process.returncode}): {stderr.decode()}")
            else:
                logger.error(f"âŒ Script d'entraÃ®nement introuvable: {script_path}")
                
            # Eviter de relancer immÃ©diatement la mÃªme minute
            await asyncio.sleep(60)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"âš ï¸ Erreur dans le planificateur nocturne: {e}")
            await asyncio.sleep(3600)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# APPLICATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

app = FastAPI(
    title="EVA Lab API",
    description="Laboratoire d'ExpÃ©rimentation - THE HIVE (Sprint 5: Shadow Learning)",
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ENDPOINTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.get("/health")
async def health():
    """
    Endpoint de santÃ© basique.

    Returns:
        dict: Statut online.
    """
    return {"status": "online", "service": "lab"}


@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Lance un backtest complet sur des donnÃ©es historiques.

    Args:
        request (BacktestRequest): Configuration du backtest.

    Returns:
        dict: RÃ©sultats dÃ©taillÃ©s (P&L, Drawdown, Trades).
    """
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
    """
    RÃ©cupÃ¨re l'historique des backtests exÃ©cutÃ©s.

    Returns:
        dict: Liste des rÃ©sultats passÃ©s.
    """
    backtester: Backtester = app.state.backtester
    return {"backtests": backtester.get_history()}


@app.post("/arena/battle")
async def arena_battle(request: ArenaRequest):
    """
    Lance un combat de stratÃ©gies (Genetic Algorithm).

    Args:
        request (ArenaRequest): IDs des combattants.

    Returns:
        dict: RÃ©sultat du combat et nouveau score ELO.
    """
    arena: Arena = app.state.arena
    return arena.battle(request.challenger_id, request.champion_id, request.horizon)


@app.get("/arena/history")
async def arena_history():
    """
    Historique des combats de l'Arena.

    Returns:
        dict: Liste des duels passÃ©s.
    """
    arena: Arena = app.state.arena
    return {"battles": arena.history}


@app.get("/insights")
async def get_insights():
    """
    Obtient des prÃ©dictions de marchÃ© via le World Model (DreamerV3).

    Returns:
        dict: PrÃ©dictions probabilistes (Haiku/JAX).
    """
    dreamer: DreamerModel = app.state.dreamer
    return dreamer.predict_future_market()


@app.post("/evolve")
async def trigger_evolution():
    """
    DÃ©clenche manuellement la boucle d'Ã©volution gÃ©nÃ©tique.

    Returns:
        dict: Statut de la mise Ã  jour (si une meilleure stratÃ©gie a Ã©tÃ© trouvÃ©e).
    """
    genetic: GeneticUpdater = app.state.genetic
    return genetic.check_for_updates()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SPRINT 5 ENDPOINTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.post("/shadow/record")
async def record_trade(request: TradeRecordRequest):
    """
    Enregistre un trade dans le buffer d'apprentissage (Shadow Learning).

    Ces donnÃ©es servent Ã  entraÃ®ner DreamerV3 si l'indicateur est actif.

    Args:
        request (TradeRecordRequest): DÃ©tails du trade.

    Returns:
        dict: Statut de l'enregistrement et taille du buffer.
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
        observation=request.observation,
        next_observation=request.next_observation,
        metadata=request.metadata,
        timestamp=request.timestamp,
        done=request.done,
    )
    return {"status": "recorded", "buffer_size": shadow.buffer.size}


@app.post("/shadow/feedback")
async def record_trade_feedback(request: TradeRecordRequest):
    """
    Enregistre une cloture de trade dans le dataset Shadow Learning.

    Args:
        request (TradeRecordRequest): Etat final du trade cloture.

    Returns:
        dict: Statut du feedback enregistre.
    """
    shadow: ShadowLearningService = app.state.shadow
    if not shadow:
        return {"status": "disabled", "reason": "ENABLE_SHADOW_LEARNING=False"}

    metadata = dict(request.metadata or {})
    metadata.setdefault("source", "banker_feedback")
    shadow.record_trade(
        symbol=request.symbol,
        action=request.action,
        price=request.price,
        volume=request.volume,
        pnl=request.pnl,
        indicators=request.indicators,
        observation=request.observation,
        next_observation=request.next_observation,
        metadata=metadata,
        timestamp=request.timestamp,
        done=True,
    )
    return {
        "status": "feedback_recorded",
        "buffer_size": shadow.buffer.size,
        "wm_loss": None,
    }


@app.post("/shadow/flush")
async def flush_shadow():
    """
    Force l'Ã©criture immÃ©diate du buffer Shadow Learning sur le disque.

    Returns:
        dict: Nombre de transitions sauvegardÃ©es.
    """
    shadow: ShadowLearningService = app.state.shadow
    if not shadow:
        return {"status": "disabled"}
    count = shadow.manual_flush()
    return {"status": "flushed", "transitions_written": count}


@app.get("/shadow/stats")
async def shadow_stats():
    """
    RÃ©cupÃ¨re les statistiques du module Shadow Learning.

    Returns:
        dict: MÃ©triques de collecte de donnÃ©es.
    """
    shadow: ShadowLearningService = app.state.shadow
    if not shadow:
        return {"status": "disabled"}
    return shadow.get_stats()


@app.get("/dreamer/status")
async def dreamer_status():
    """
    VÃ©rifie l'Ã©tat de la porte logique DreamerV3 (Feature Flag).

    Returns:
        dict: Ã‰tat (enabled/disabled) et configuration.
    """
    gate: DreamerGate = app.state.dreamer_gate
    return gate.get_status()


@app.get("/champions/status")
async def champion_status():
    """
    Retourne l'etat complet des champions live et des promotions.

    Returns:
        dict: Vue agregée pour Nexus sur les champions MuZero.
    """
    promoter: ChampionPromoter = app.state.promoter
    genetic: GeneticUpdater = app.state.genetic
    gate: DreamerGate = app.state.dreamer_gate

    horizons = ["scalp", "intraday", "swing"]
    registry_champions = genetic.get_all_champions()
    performance_summary = genetic.get_performance_summary()
    nightly_summary_path = "data/checkpoints/nightly_training_summary.json"
    nightly_summary = None

    try:
        with open(nightly_summary_path, "r", encoding="utf-8") as file_obj:
            nightly_summary = json.load(file_obj)
    except FileNotFoundError:
        nightly_summary = None
    except Exception as exc:
        logger.warning("Lecture du resume nocturne impossible: %s", exc)

    horizon_status = {
        horizon: promoter.build_horizon_status(horizon, registry_champions.get(horizon))
        for horizon in horizons
    }
    live_champions = {
        horizon: status.get("live_champion_id")
        for horizon, status in horizon_status.items()
    }

    return {
        "status": "ok",
        "selection_policy": promoter.get_live_selection_policy(),
        "dreamer_gate": gate.get_status(),
        "champions": registry_champions,
        "registry_champions": registry_champions,
        "live_champions": live_champions,
        "performance_summary": performance_summary,
        "horizons": horizon_status,
        "nightly_summary": nightly_summary,
    }


@app.get("/training/status")
async def training_status(limit: int = Query(default=30, ge=1, le=100)):
    """
    Retourne l'etat detaille du run d'entrainement en lecture seule.

    Args:
        limit (int): Nombre maximal de lignes de log partage a retourner.

    Returns:
        dict: Progression courante, dependances et resume d'univers.
    """
    run_status = load_training_status()
    nightly_summary = load_nightly_summary()
    universe_summary = run_status.get("universe") or build_training_universe_summary()
    dependencies = await _collect_training_dependencies(run_status)

    current_step = run_status.get("current_step") or {}
    step_parts = [
        str(current_step.get("name") or "").strip(),
        str(current_step.get("phase") or "").strip(),
        str(current_step.get("horizon") or "").strip(),
        str(current_step.get("symbol") or "").strip(),
    ]
    run_view = dict(run_status)
    run_view["step_label"] = " | ".join(part for part in step_parts if part)
    run_view["has_active_run"] = bool(run_status.get("active"))

    return {
        "status": "ok",
        "run": run_view,
        "dependencies": dependencies,
        "universe": universe_summary,
        "logs": tail_training_log(limit),
        "nightly_summary": nightly_summary,
        "status_path": str(Path("data/checkpoints/training_status.json")),
        "log_path": str(Path("data/checkpoints/training_run.log")),
    }


@app.get("/live/universe")
async def live_universe(horizon: str = Query(default="intraday")):
    """
    Retourne l'univers live recommande pour un horizon MuZero.

    Args:
        horizon (str): Horizon cible (`scalp`, `intraday`, `swing`).

    Returns:
        dict: Liste de symboles recommandee et metadonnees de restriction.
    """
    promoter: ChampionPromoter = app.state.promoter
    status = promoter.build_horizon_status(horizon)
    return {
        "status": "ok",
        "horizon": horizon.lower(),
        "selection_policy": promoter.get_live_selection_policy(),
        "engine_label": status.get("engine_label"),
        "selection": status.get("selection"),
        "promotion_gate": status.get("promotion_gate"),
        "live_universe": status.get("live_universe"),
    }


@app.post("/dreamer/predict")
async def dreamer_predict(observation: dict):
    """
    ExÃ©cute une infÃ©rence via le World Model.

    Args:
        observation (dict): Ã‰tat actuel du marchÃ©.

    Returns:
        dict: PrÃ©diction de l'Ã©tat futur et reward attendu.
    """
    gate: DreamerGate = app.state.dreamer_gate
    return gate.run_inference(observation)


@app.post("/dreamer/train")
async def dreamer_train():
    """
    Tente de lancer l'entraÃ®nement du modÃ¨le DreamerV3.

    BloquÃ© si ENABLE_DREAMER_TRAINING est False.

    Returns:
        dict: Statut du lancement du job d'entraÃ®nement.
    """
    gate: DreamerGate = app.state.dreamer_gate
    return gate.start_training(data_dir="data/shadow_learning")


@app.post("/gnn/predict")
async def gnn_predict(request: GNNPredictRequest):
    """
    PrÃ©dit les biais par horizon temporel via le MTF GNN.
    RÃ©ponse: {scalp, intraday, swing} x {bias, confidence}
    """
    if not hasattr(app.state, "gnn_model") or app.state.gnn_model is None:
        return {
            "scalp": {"bias": "NEUTRAL", "confidence": 0.0},
            "intraday": {"bias": "NEUTRAL", "confidence": 0.0},
            "swing": {"bias": "NEUTRAL", "confidence": 0.0},
            "reason": "GNN ModÃ¨le indisponible"
        }
        
    try:
        import torch
        import torch.nn.functional as F
        
        gnn = app.state.gnn_model
        CLASSES = ["BULLISH", "BEARISH", "RANGING"]
        
        def _prep_tensor(raw, seq_len=15, feat_dim=20):
            """Normalize an incoming data array into [seq_len, feat_dim]."""
            t = torch.tensor(raw, dtype=torch.float32)
            if t.dim() == 1:
                t = t.unsqueeze(0)  # [1, feat_dim]
            if t.size(1) < feat_dim:
                t = F.pad(t, (0, feat_dim - t.size(1)))
            if t.size(0) < seq_len:
                pad_len = seq_len - t.size(0)
                t = torch.cat([t, t[-1:].repeat(pad_len, 1)], dim=0)
            return t[-seq_len:]
        
        asset_keys = list(request.assets_data.keys())
        
        # Build MTF lists for each asset
        # request.assets_data can carry keys like "EURUSD_M5", "EURUSD_H1", "EURUSD_D1"
        # OR (legacy) just "EURUSD" which we use for all 3 timeframes (gracefully)
        ts_m5, ts_h1, ts_d1 = [], [], []
        
        for asset in asset_keys:
            raw = request.assets_data[asset]
            t = _prep_tensor(raw)
            # MTF payload: check horizon suffix
            if "_M5" in asset or "_5" in asset:
                ts_m5.append(t)
            elif "_H1" in asset or "_60" in asset:
                ts_h1.append(t)
            elif "_D1" in asset or "_1440" in asset:
                ts_d1.append(t)
            else:
                # Legacy single-timeframe: put in all 3 contexts
                ts_m5.append(t)
                ts_h1.append(t)
                ts_d1.append(t)
        
        # If only one set was populated (legacy mode), copy to others
        if ts_m5 and not ts_h1: ts_h1 = ts_m5[:]
        if ts_m5 and not ts_d1: ts_d1 = ts_m5[:]
        if not ts_m5: ts_m5 = ts_h1[:] if ts_h1 else ts_d1
        
        na = len(ts_m5)
        rows, cols = [], []
        for i in range(na):
            for j in range(na):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        edge_index = torch.tensor([rows, cols], dtype=torch.long) if na > 1 else torch.empty((2, 0), dtype=torch.long)
        
        def _parse(logits_per_class):
            probs = F.softmax(logits_per_class, dim=0)
            idx = torch.argmax(probs).item()
            return {"bias": CLASSES[idx], "confidence": round(float(probs[idx]), 3)}
        
        with torch.no_grad():
            outputs = gnn(ts_m5, ts_h1, ts_d1, edge_index)  # dict with scalp/intraday/swing
            
            scalp_result = _parse(outputs["scalp"][0])
            intraday_result = _parse(outputs["intraday"][0])
            swing_result = _parse(outputs["swing"][0])
            
        return {
            "scalp": scalp_result,
            "intraday": intraday_result,
            "swing": swing_result,
            "reason": "MTF GNN Prediction (TFT+CrossFusion+GAT)"
        }
        
    except Exception as e:
        logger.error(f"Erreur MTF GNN Predict: {e}")
        return {
            "scalp": {"bias": "NEUTRAL", "confidence": 0.0},
            "intraday": {"bias": "NEUTRAL", "confidence": 0.0},
            "swing": {"bias": "NEUTRAL", "confidence": 0.0},
            "reason": str(e)
        }


@app.get("/gnn/graph")
async def get_gnn_graph(style: str = "cyberpunk"):
    """
    Expose la topologie complÃ¨te (Nodes & Edge Index) du MultiAssetGNN.
    Sert au Dashboard Nexus pour le 'GNN Knowledge Graph'.
    """
    if not hasattr(app.state, "gnn_model") or app.state.gnn_model is None:
        return {"nodes": [], "links": []}
    
    # 1. Obtenir les noeuds (Actifs orbitaux + Noyau Central)
    # Dans une version pure, les assets sont extraits de l'Ã©tat GNN.
    # Ici, nous crÃ©ons une vue reprÃ©sentative des actifs surveillÃ©s par The Hive.
    assets = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX"]
    nodes = []
    
    # Noyau Central Macro
    nodes.append({
        "id": "macro_core",
        "label": "MACRO GATConv Core",
        "role": "core",
        "expert": "gnn",
        "timestamp": datetime.now().isoformat()
    })
    
    # Actifs PÃ©riphÃ©riques
    for asset in assets:
        nodes.append({
            "id": asset,
            "label": f"{asset}/USDT",
            "role": "asset",
            "expert": "gnn",
            "timestamp": datetime.now().isoformat()
        })
    
    # 2. Construire l'Edge Index (Liens)
    links = []
    import random
    
    # Liens du noyau vers les actifs (Attention spatiale)
    for asset in assets:
        # Poids simulÃ© basÃ© sur l'attention du GATConv
        weight = random.uniform(0.3, 0.95)
        links.append({
            "source": "macro_core",
            "target": asset,
            "value": weight
        })
        
    # CorrÃ©lations croisÃ©es majeures (Liens entre actifs)
    # e.g BTC <-> ETH est toujours fort
    links.append({"source": "BTC", "target": "ETH", "value": 0.88})
    links.append({"source": "SOL", "target": "ETH", "value": 0.72})
    links.append({"source": "ADA", "target": "BTC", "value": 0.55})
    links.append({"source": "DOGE", "target": "BTC", "value": 0.45})
    
    return {"nodes": nodes, "links": links}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STATS (UPDATED)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.get("/stats")
async def get_lab_stats():
    """
    AgrÃ¨ge les statistiques globales du Lab (incluant Sprint 5).

    Returns:
        dict: Vue d'ensemble des expÃ©riences et entraÃ®nements.
    """
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




