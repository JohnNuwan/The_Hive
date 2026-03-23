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
from shared import PromotionReportEnvelope, TrainingRunEnvelope, get_settings
from shared.redis_client import init_redis, get_redis_client

from eva_lab.arena import Arena
from eva_lab.backtester import Backtester
from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.dreamer_model import DreamerModel
from eva_lab.genetic_updater import GeneticUpdater
from eva_lab.gnn_registry import (
    build_market_gnn_graph_snapshot,
    load_market_gnn_registry,
)
from eva_lab.live_inference_models import LivePredictRequest
from eva_lab.shadow_learning import ShadowLearningService
from eva_lab.dreamer_gate import DreamerGate
from eva_lab.timescale_store import describe_timescale_source
from eva_lab.training_status import (
    build_training_universe_summary,
    classify_training_symbol,
    derive_observed_training_step,
    format_training_step_label,
    load_nightly_summary,
    select_effective_training_step,
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
    timescale_info = describe_timescale_source()

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
    dependencies["timescaledb"] = await _probe_tcp_dependency(
        "timescaledb",
        str(timescale_info.get("host") or "timescaledb"),
        int(timescale_info.get("port") or 5432),
    )
    dependencies["timescaledb"]["enabled"] = bool(timescale_info.get("enabled", False))
    dependencies["timescaledb"]["source"] = str(timescale_info.get("source") or "csv")

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


async def _publish_training_run_snapshot(
    run_view: dict[str, Any],
    dependencies: dict[str, Any],
    universe: dict[str, Any],
    nightly_summary: dict[str, Any] | None,
) -> None:
    """
    Publie un instantane structure du run courant pour EVA Core.

    Args:
        run_view (dict[str, Any]): Vue courante du run.
        dependencies (dict[str, Any]): Dependances observees.
        universe (dict[str, Any]): Resume de l'univers.
        nightly_summary (dict[str, Any] | None): Resume nightly en lecture seule.
    """
    try:
        current_step = dict(run_view.get("current_step") or {})
        symbol = str(current_step.get("symbol") or "") or None
        family = str(run_view.get("family") or "").strip()
        if not family and symbol:
            family = classify_training_symbol(symbol)
        envelope = TrainingRunEnvelope(
            engine=str(run_view.get("engine") or "") or None,
            run_id=str(run_view.get("run_id") or "") or None,
            horizon=str(current_step.get("horizon") or "") or None,
            family=family or None,
            feature_profile=str(run_view.get("feature_profile") or "") or None,
            dataset_id=str(run_view.get("dataset_id") or "") or None,
            dataset_source=str(run_view.get("dataset_source") or "") or None,
            mechanics_profile_version=str(run_view.get("mechanics_profile_version") or "") or None,
            ga_status=str(run_view.get("ga_status") or "") or None,
            ga_generation=run_view.get("ga_generation"),
            ga_trial=str(run_view.get("ga_trial") or "") or None,
            trial_mode=str(run_view.get("trial_mode") or "") or None,
            trial_cost_profile=str(run_view.get("trial_cost_profile") or "") or None,
            replay_cache_status=str(run_view.get("replay_cache_status") or "") or None,
            replay_cache_key=str(run_view.get("replay_cache_key") or "") or None,
            replay_cache_entries=run_view.get("replay_cache_entries"),
            replay_cache_source=str(run_view.get("replay_cache_source") or "") or None,
            shadow_buffer_size=run_view.get("shadow_buffer_size"),
            sequence_length=run_view.get("sequence_length"),
            sequence_stride=run_view.get("sequence_stride"),
            world_model_steps=run_view.get("world_model_steps"),
            dataset_coverage=dict(run_view.get("dataset_coverage") or {}),
            phase=str(current_step.get("phase") or "") or None,
            current_symbol=symbol,
            status=str(run_view.get("status") or "idle"),
            arena_progress=run_view.get("arena_progress"),
            dependencies=dependencies,
            universe=universe,
            payload={
                "run": run_view,
                "nightly_summary": nightly_summary,
            },
            metadata={"source": "training_status_endpoint"},
        )
        redis = get_redis_client()
        payload = envelope.model_dump()
        await redis.cache_set("eva:state:training:run", payload, ttl_seconds=60)
        await redis.publish("eva.training.run", payload)
    except Exception as exc:
        logger.debug("Publication du run training ignoree: %s", exc)


async def _publish_champion_status_snapshot(payload: dict[str, Any]) -> None:
    """
    Publie un instantane agrege des champions et promotions.

    Args:
        payload (dict[str, Any]): Charge utile complete de l'endpoint champions.
    """
    try:
        redis = get_redis_client()
        await redis.cache_set("eva:state:champions:status", payload, ttl_seconds=120)
        await redis.publish("eva.training.champions", payload)
        engine_payloads = dict(payload.get("engines") or {})
        if not engine_payloads:
            engine_payloads = {"muzero": dict(payload.get("horizons") or {})}
        for engine_name, horizons in engine_payloads.items():
            for horizon, status in dict(horizons or {}).items():
                envelope = PromotionReportEnvelope(
                    engine=str(engine_name or "muzero"),
                    horizon=str(horizon),
                    family=str(status.get("family") or "") or None,
                    live_champion_id=str(status.get("live_champion_id") or "") or None,
                    challenger_id=str(
                        status.get("candidate_id")
                        or (status.get("manifest") or {}).get("challenger_id")
                        or ""
                    ) or None,
                    promotion_gate=dict(status.get("promotion_gate") or {}),
                    promotion_checks=dict(status.get("promotion_checks") or {}),
                    metrics_by_symbol=dict(status.get("metrics_by_symbol") or {}),
                    metrics_by_position_mechanics=dict(status.get("metrics_by_position_mechanics") or {}),
                    feature_profile=str(status.get("feature_profile") or "") or None,
                    dataset_id=str(status.get("dataset_id") or "") or None,
                    failure_mode=str(status.get("failure_mode") or "") or None,
                    top_live_symbols=list(status.get("top_live_symbols") or []),
                    payload=status,
                    metadata={"source": "champions_status_endpoint"},
                )
                serialized = envelope.model_dump()
                await redis.cache_set(
                    f"eva:state:promotion:{str(engine_name).lower()}:{str(horizon).lower()}",
                    serialized,
                    ttl_seconds=120,
                )
                await redis.publish("eva.training.promotion", serialized)
    except Exception as exc:
        logger.debug("Publication du statut champions ignoree: %s", exc)


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
    Retourne l'etat exploitable du pipeline DreamerV3.

    Returns:
        dict: Etat du gate, du pipeline et des derniers artefacts Dreamer.
    """
    gate: DreamerGate = app.state.dreamer_gate
    promoter: ChampionPromoter = app.state.promoter
    horizons = ["scalp", "intraday", "swing"]
    training_run = load_training_status()
    active_run = training_run if str(training_run.get("engine") or "").lower() == "dreamer" else {}
    pipeline = {
        "active": bool(active_run.get("active")),
        "run_id": active_run.get("run_id"),
        "status": active_run.get("status"),
        "family": active_run.get("family"),
        "feature_profile": active_run.get("feature_profile"),
        "mechanics_profile_version": active_run.get("mechanics_profile_version"),
        "ga_status": active_run.get("ga_status"),
        "ga_generation": active_run.get("ga_generation"),
        "ga_trial": active_run.get("ga_trial"),
        "trial_mode": active_run.get("trial_mode"),
        "trial_cost_profile": active_run.get("trial_cost_profile"),
        "replay_cache_status": active_run.get("replay_cache_status"),
        "replay_cache_key": active_run.get("replay_cache_key"),
        "replay_cache_entries": active_run.get("replay_cache_entries"),
        "shadow_buffer_size": active_run.get("shadow_buffer_size"),
        "sequence_length": active_run.get("sequence_length"),
        "sequence_stride": active_run.get("sequence_stride"),
        "world_model_steps": active_run.get("world_model_steps"),
        "dataset_id": active_run.get("dataset_id"),
        "dataset_source": active_run.get("dataset_source"),
        "dataset_coverage": active_run.get("dataset_coverage", {}),
    }
    engine_horizons = {
        horizon: promoter.build_engine_horizon_status("dreamer", horizon)
        for horizon in horizons
    }
    latest_candidate = None
    latest_verdict = None
    for horizon in horizons:
        horizon_status = dict(engine_horizons.get(horizon) or {})
        if latest_candidate is None and horizon_status.get("candidate_id"):
            latest_candidate = {
                "engine": "dreamer",
                "horizon": horizon,
                "candidate_id": horizon_status.get("candidate_id"),
                "failure_mode": horizon_status.get("failure_mode"),
            }
        if latest_verdict is None and horizon_status.get("promotion_gate"):
            latest_verdict = {
                "engine": "dreamer",
                "horizon": horizon,
                "status": horizon_status.get("promotion_gate", {}).get("status"),
                "reason": horizon_status.get("gate_reason"),
                "failure_mode": horizon_status.get("failure_mode"),
            }
    return {
        **gate.get_status(),
        "pipeline": pipeline,
        "horizons": engine_horizons,
        "latest_candidate": latest_candidate,
        "latest_verdict": latest_verdict,
    }


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

    engine_status = promoter.build_engine_matrix_status(horizons, registry_champions)
    horizon_status = dict(engine_status.get("muzero") or {})
    live_champions = {
        horizon: status.get("live_champion_id")
        for horizon, status in horizon_status.items()
    }
    live_champions_by_engine = {
        engine: {
            horizon: status.get("live_champion_id")
            for horizon, status in statuses.items()
        }
        for engine, statuses in engine_status.items()
    }

    payload = {
        "status": "ok",
        "selection_policy": promoter.get_live_selection_policy(),
        "dreamer_gate": gate.get_status(),
        "champions": registry_champions,
        "registry_champions": registry_champions,
        "live_champions": live_champions,
        "live_champions_by_engine": live_champions_by_engine,
        "performance_summary": performance_summary,
        "horizons": horizon_status,
        "engines": engine_status,
        "nightly_summary": nightly_summary,
    }
    await _publish_champion_status_snapshot(payload)
    return payload


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
    logs = tail_training_log(limit)

    current_step = run_status.get("current_step") or {}
    arena_progress = run_status.get("arena_progress") or None
    observed_step = derive_observed_training_step(logs)
    effective_step = select_effective_training_step(current_step, observed_step)
    run_view = dict(run_status)
    run_view["current_step"] = effective_step
    run_view["arena_progress"] = arena_progress
    run_view["reported_step"] = current_step or None
    run_view["observed_step"] = observed_step
    run_view["effective_step"] = effective_step
    run_view["step_label"] = format_training_step_label(effective_step)
    run_view["reported_step_label"] = format_training_step_label(current_step)
    run_view["observed_step_label"] = format_training_step_label(observed_step)
    run_view["has_active_run"] = bool(run_status.get("active"))
    if arena_progress and isinstance(arena_progress, dict):
        challenger_metrics = dict((arena_progress.get("challenger") or {}).get("metrics") or {})
        if challenger_metrics.get("metrics_by_position_mechanics"):
            run_view["metrics_by_position_mechanics"] = challenger_metrics.get("metrics_by_position_mechanics")

    payload = {
        "status": "ok",
        "run": run_view,
        "dependencies": dependencies,
        "universe": universe_summary,
        "logs": logs,
        "nightly_summary": nightly_summary,
        "status_path": str(Path("data/checkpoints/training_status.json")),
        "log_path": str(Path("data/checkpoints/training_run.log")),
        "engine": run_view.get("engine"),
        "dataset_id": run_view.get("dataset_id"),
        "feature_profile": run_view.get("feature_profile"),
        "family": run_view.get("family"),
        "dataset_source": run_view.get("dataset_source"),
        "mechanics_profile_version": run_view.get("mechanics_profile_version"),
        "ga_status": run_view.get("ga_status"),
        "ga_generation": run_view.get("ga_generation"),
        "ga_trial": run_view.get("ga_trial"),
        "trial_mode": run_view.get("trial_mode"),
        "trial_cost_profile": run_view.get("trial_cost_profile"),
        "replay_cache_status": run_view.get("replay_cache_status"),
        "replay_cache_key": run_view.get("replay_cache_key"),
        "replay_cache_entries": run_view.get("replay_cache_entries"),
        "replay_cache_source": run_view.get("replay_cache_source"),
        "shadow_buffer_size": run_view.get("shadow_buffer_size"),
        "sequence_length": run_view.get("sequence_length"),
        "sequence_stride": run_view.get("sequence_stride"),
        "world_model_steps": run_view.get("world_model_steps"),
        "dataset_coverage": run_view.get("dataset_coverage", {}),
        "metrics_by_position_mechanics": run_view.get("metrics_by_position_mechanics", {}),
    }
    await _publish_training_run_snapshot(
        run_view=run_view,
        dependencies=dependencies,
        universe=universe_summary,
        nightly_summary=nightly_summary,
    )
    return payload


@app.get("/live/universe")
async def live_universe(
    horizon: str = Query(default="intraday"),
    engine: str = Query(default="muzero"),
):
    """
    Retourne l'univers live recommande pour un horizon MuZero.

    Args:
        horizon (str): Horizon cible (`scalp`, `intraday`, `swing`).
        engine (str): Moteur cible (`muzero` ou `dreamer`).

    Returns:
        dict: Liste de symboles recommandee et metadonnees de restriction.
    """
    promoter: ChampionPromoter = app.state.promoter
    normalized_horizon = str(horizon or "intraday").lower()
    normalized_engine = promoter.normalize_engine_name(engine)
    status = promoter.build_engine_horizon_status(normalized_engine, normalized_horizon)
    engine_matrix = promoter.build_engine_matrix_status([normalized_horizon], {})
    live_champions_by_engine = {
        engine_name: {
            item_horizon: item_status.get("live_champion_id")
            for item_horizon, item_status in statuses.items()
        }
        for engine_name, statuses in engine_matrix.items()
    }
    top_live_symbols_by_engine = {
        engine_name: list(
            dict(statuses or {}).get(normalized_horizon, {}).get("top_live_symbols") or []
        )
        for engine_name, statuses in engine_matrix.items()
    }
    return {
        "status": "ok",
        "engine": normalized_engine,
        "horizon": normalized_horizon,
        "family": status.get("family"),
        "feature_profile": status.get("feature_profile"),
        "mechanics_profile_version": status.get("mechanics_profile_version"),
        "dataset_id": status.get("dataset_id"),
        "dataset_source": status.get("dataset_source"),
        "dataset_coverage": status.get("dataset_coverage"),
        "selection_policy": promoter.get_live_selection_policy(),
        "engine_label": status.get("engine_label"),
        "selection": status.get("selection"),
        "live_champion_id": status.get("live_champion_id"),
        "live_champion_id_muzero": live_champions_by_engine.get("muzero", {}).get(normalized_horizon),
        "live_champion_id_dreamer": live_champions_by_engine.get("dreamer", {}).get(normalized_horizon),
        "top_live_symbols_by_engine": top_live_symbols_by_engine,
        "top_live_symbols": status.get("top_live_symbols"),
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


@app.post("/predict/live")
async def predict_live(request: LivePredictRequest):
    """
    Execute une inference live stricte pour le banker.

    Args:
        request (LivePredictRequest): Observation live du banker.

    Returns:
        dict: Action brute du champion scalp live ou blocage explicite.
    """
    gate: DreamerGate = app.state.dreamer_gate
    return gate.run_live_inference(request.model_dump())


@app.post("/predict/ensemble")
async def predict_ensemble(request: LivePredictRequest):
    """
    Execute un arbitrage 50/50 entre MuZero et DreamerV3.

    Args:
        request (LivePredictRequest): Observation live du banker.

    Returns:
        dict: Sous-decisions par moteur et decision finale d'ensemble.
    """
    gate: DreamerGate = app.state.dreamer_gate
    return gate.run_ensemble_inference(request.model_dump())


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


@app.get("/gnn/status")
async def gnn_status():
    """
    Retourne le registre public du Market GNN.

    Returns:
        dict: Version, statut, univers et artefacts du GNN de marche.
    """
    registry = load_market_gnn_registry()
    return {
        "status": "ok",
        "gnn": registry,
    }


@app.get("/gnn/metrics")
async def gnn_metrics():
    """
    Retourne les metriques publiques du Market GNN.

    Returns:
        dict: Metriques consolidees et informations de couverture.
    """
    registry = load_market_gnn_registry()
    return {
        "status": "ok",
        "version": registry.get("version"),
        "model_status": registry.get("status"),
        "trained_at": registry.get("trained_at"),
        "metrics": registry.get("metrics", {}),
        "universe": registry.get("universe", {}),
        "timeframes": registry.get("timeframes", []),
        "artifacts": registry.get("artifacts", {}),
    }


@app.get("/gnn/graph")
async def get_gnn_graph(style: str = "cyberpunk"):
    """
    Expose un graphe reel du Market GNN, derive des historiques.

    Args:
        style (str): Parametre conserve pour compatibilite avec l'UI existante.

    Returns:
        dict: Graphe reel ou etat explicite d'indisponibilite.
    """
    _ = style
    registry = load_market_gnn_registry()
    snapshot = build_market_gnn_graph_snapshot(registry=registry)
    snapshot["version"] = registry.get("version")
    snapshot["model_status"] = registry.get("status")
    snapshot["trained_at"] = registry.get("trained_at")
    snapshot["timeframes"] = registry.get("timeframes", [])
    snapshot["universe"] = registry.get("universe", {})
    snapshot["metrics"] = registry.get("metrics", {})
    return snapshot


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




