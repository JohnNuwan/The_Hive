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
import sys
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
    load_market_gnn_refresh_state,
    persist_market_gnn_refresh_state,
    update_market_gnn_registry,
)
from eva_lab.live_inference_models import LivePredictRequest
from eva_lab.shadow_learning import ShadowLearningService
from eva_lab.dreamer_gate import DreamerGate
from eva_lab.timescale_store import (
    describe_timescale_source,
    load_ga_trials,
    record_ga_trial,
    record_run_window,
)
from eva_lab.training_status import (
    CPU_SCHEDULER_STATE_PATH,
    RUN_LOG_PATH,
    STATUS_DIR,
    build_training_universe_summary,
    classify_training_symbol,
    derive_observed_training_step,
    format_training_step_label,
    load_latest_terminal_summary,
    load_nightly_summary,
    load_cpu_scheduler_state,
    load_seeded_muzero_campaign_state,
    load_sequence_state,
    select_effective_training_step,
    tail_log_file,
    load_training_status,
    tail_training_log,
    persist_seeded_muzero_campaign_state,
    write_terminal_summary,
)
from eva_lab.training_utils import get_gnn_model_kwargs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
HOST_DATA_PREFIX = Path("/home/aza/The_Hive/data")
CONTAINER_DATA_ROOT = Path("/app/eva-lab/data")
LAB_APP_ROOT = Path("/app/eva-lab")
GNN_REFRESH_LOG_PATH = STATUS_DIR / "gnn_refresh.log"


def _resolve_latest_checkpoint_log(*patterns: str) -> Path | None:
    """
    Retourne le journal le plus recent correspondant aux motifs demandes.

    Args:
        *patterns (str): Motifs ``glob`` a appliquer dans le dossier checkpoints.

    Returns:
        Path | None: Chemin du journal retenu ou ``None`` si absent.
    """
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in STATUS_DIR.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _as_path(raw_path: Any) -> Path | None:
    """
    Convertit un chemin brut en chemin lisible depuis le conteneur Lab.

    Args:
        raw_path (Any): Valeur brute potentiellement serialisee dans l'etat.

    Returns:
        Path | None: Chemin resolu dans le conteneur, ou ``None`` si absent.
    """

    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    try:
        relative = candidate.relative_to(HOST_DATA_PREFIX)
    except ValueError:
        return candidate
    mapped = CONTAINER_DATA_ROOT / relative
    return mapped


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
    dependencies["timescaledb"]["state"] = str(timescale_info.get("state") or "disabled")
    dependencies["timescaledb"]["bars_table"] = str(timescale_info.get("bars_table") or "")
    dependencies["timescaledb"]["features_table"] = str(timescale_info.get("features_table") or "")

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
            sequence_id=str(run_view.get("sequence_id") or "") or None,
            sequence_profile=str(run_view.get("sequence_profile") or "") or None,
            window_id=str(run_view.get("window_id") or "") or None,
            trial_id=str(run_view.get("trial_id") or "") or None,
            terminal_summary_path=str(run_view.get("terminal_summary_path") or "") or None,
            supervisor_state=str(run_view.get("supervisor_state") or "") or None,
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


def _current_gnn_refresh_state() -> dict[str, Any]:
    """Charge l'etat persistant du refresh GNN."""

    return load_market_gnn_refresh_state()


def _gnn_refresh_is_running(app: FastAPI) -> bool:
    """Indique si un refresh GNN tourne deja dans le conteneur Lab."""

    process = getattr(app.state, "gnn_refresh_process", None)
    return process is not None and process.returncode is None


def _resolve_gnn_source_run_id() -> str | None:
    """Determine le run source a rattacher au prochain refresh GNN."""

    training_state = load_training_status()
    run_id = str(training_state.get("run_id") or "").strip()
    if run_id:
        return run_id
    latest_summary = load_latest_terminal_summary()
    if latest_summary:
        candidate_run_id = str(latest_summary.get("run_id") or "").strip()
        if candidate_run_id:
            return candidate_run_id
    return None


def _build_gnn_coverage_summary(registry: dict[str, Any]) -> dict[str, Any]:
    """Construit un resume lisible de couverture pour le GNN.

    Args:
        registry (dict[str, Any]): Registre GNN courant.

    Returns:
        dict[str, Any]: Resume synthétique de couverture et fraicheur.
    """

    graph_snapshot = build_market_gnn_graph_snapshot(registry=registry)
    return {
        "graph_status": graph_snapshot.get("status"),
        "graph_reason": graph_snapshot.get("reason"),
        "selected_timeframe": graph_snapshot.get("selected_timeframe"),
        "candidate_timeframes": graph_snapshot.get("candidate_timeframes", []),
        "overlap_points": graph_snapshot.get("overlap_points", 0),
        "displayed_symbol_count": graph_snapshot.get("displayed_symbol_count", 0),
        "universe_symbol_count": graph_snapshot.get("universe_symbol_count", 0),
        "missing_symbols": graph_snapshot.get("missing_symbols", []),
    }


async def _finalize_gnn_refresh(
    app: FastAPI,
    *,
    run_id: str,
    source_run_id: str | None,
    requested_at: str | None,
    started_at: str | None,
    log_path: Path,
    return_code: int,
) -> None:
    """Finalise un refresh GNN avec resume terminal et registre enrichi."""

    finished_at = datetime.utcnow().isoformat() + "Z"
    registry = load_market_gnn_registry()
    coverage_summary = _build_gnn_coverage_summary(registry)
    success = return_code == 0 and bool(registry.get("checkpoint_path"))
    failure_reason = None if success else f"Le script train_gnn.py a quitte avec le code {return_code}."
    refresh_status = "completed" if success else "error"
    status_reason = (
        "Refresh GNN termine avec succes."
        if success
        else (failure_reason or "Le refresh GNN a echoue.")
    )

    persist_market_gnn_refresh_state(
        {
            "status": refresh_status,
            "queued": False,
            "requested_at": requested_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "run_id": run_id,
            "failure_reason": failure_reason,
            "source_run_id": source_run_id,
            "requested_by": "lab",
        }
    )
    update_market_gnn_registry(
        {
            "source_run_id": source_run_id,
            "status_reason": status_reason,
            "last_refresh_requested_at": requested_at,
            "last_refresh_started_at": started_at,
            "last_refresh_finished_at": finished_at,
            "last_refresh_status": refresh_status,
            "coverage_summary": coverage_summary,
        }
    )
    registry = load_market_gnn_registry()
    artifact_state = {
        "checkpoint_present": bool(registry.get("artifacts", {}).get("checkpoint", {}).get("exists")),
        "metrics_present": bool(registry.get("artifacts", {}).get("metrics", {}).get("exists")),
        "log_path": str(log_path),
    }
    write_terminal_summary(
        {
            "engine": "gnn",
            "run_id": run_id,
            "horizon": "market",
            "family": "mixed",
            "trial_id": "refresh",
            "source_run_id": source_run_id,
            "focus_symbols": list(registry.get("focus_symbols") or []),
            "focus_symbol": registry.get("focus_symbol"),
            "context_symbols": list(registry.get("context_symbols") or []),
            "deployment_class": registry.get("deployment_class"),
            "status": refresh_status,
            "terminal_status": refresh_status,
            "failed_step": None if success else "train_gnn",
            "failure_mode": None if success else "refresh_failed",
            "failure_reason": failure_reason,
            "started_at": started_at,
            "finished_at": finished_at,
            "metrics": dict(registry.get("metrics") or {}),
            "coverage_summary": coverage_summary,
            "artifact_state": artifact_state,
        }
    )
    log_stream = getattr(app.state, "gnn_refresh_log_stream", None)
    if log_stream is not None:
        try:
            log_stream.close()
        except Exception:
            pass
        app.state.gnn_refresh_log_stream = None
    app.state.gnn_refresh_process = None
    app.state.gnn_refresh_monitor_task = None
    app.state.gnn_refresh_log_stream = None


async def _wait_for_gnn_refresh_completion(
    app: FastAPI,
    *,
    process: asyncio.subprocess.Process,
    run_id: str,
    source_run_id: str | None,
    requested_at: str | None,
    started_at: str | None,
    log_path: Path,
) -> None:
    """Attend la fin du processus de refresh GNN et publie son resultat."""

    return_code = await process.wait()
    await _finalize_gnn_refresh(
        app,
        run_id=run_id,
        source_run_id=source_run_id,
        requested_at=requested_at,
        started_at=started_at,
        log_path=log_path,
        return_code=return_code,
    )


async def _start_gnn_refresh_process(app: FastAPI, refresh_state: dict[str, Any]) -> dict[str, Any]:
    """Lance effectivement le script de refresh GNN s'il est autorise."""

    if _gnn_refresh_is_running(app):
        return _current_gnn_refresh_state()

    run_id = str(refresh_state.get("run_id") or f"gnn_refresh_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    started_at = datetime.utcnow().isoformat() + "Z"
    source_run_id = str(refresh_state.get("source_run_id") or _resolve_gnn_source_run_id() or "").strip() or None
    GNN_REFRESH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_stream = GNN_REFRESH_LOG_PATH.open("ab")
    gnn_env = os.environ.copy()
    requested_symbols = [
        str(symbol).strip()
        for symbol in list(refresh_state.get("requested_symbols") or [])
        if str(symbol).strip()
    ]
    focus_symbol = str(refresh_state.get("focus_symbol") or "").strip() or None
    context_symbols = [
        str(symbol).strip()
        for symbol in list(refresh_state.get("context_symbols") or [])
        if str(symbol).strip()
    ]
    deployment_class = str(refresh_state.get("deployment_class") or "").strip() or None
    if requested_symbols:
        gnn_env["TRAIN_GNN_SYMBOLS"] = ",".join(requested_symbols)
    if focus_symbol:
        gnn_env["TRAIN_GNN_FOCUS_SYMBOL"] = focus_symbol
    if context_symbols:
        gnn_env["TRAIN_GNN_CONTEXT_SYMBOLS"] = ",".join(context_symbols)
    if deployment_class:
        gnn_env["TRAIN_GNN_DEPLOYMENT_CLASS"] = deployment_class
    for env_name, state_key in (
        ("TRAIN_GNN_EPOCHS", "epochs"),
        ("TRAIN_GNN_BATCH_SIZE", "batch_size"),
        ("TRAIN_GNN_CHECKPOINT_EVERY", "checkpoint_every"),
        ("TRAIN_GNN_MAX_SYMBOLS", "max_symbols"),
    ):
        if refresh_state.get(state_key) is not None:
            gnn_env[env_name] = str(refresh_state.get(state_key))
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(LAB_APP_ROOT / "scripts" / "train_gnn.py"),
            cwd=str(LAB_APP_ROOT),
            env=gnn_env,
            stdout=log_stream,
            stderr=log_stream,
        )
    except Exception:
        log_stream.close()
        raise
    app.state.gnn_refresh_process = process
    app.state.gnn_refresh_log_stream = log_stream
    update_market_gnn_registry(
        {
            "source_run_id": source_run_id,
            "focus_symbols": requested_symbols,
            "focus_symbol": focus_symbol,
            "context_symbols": context_symbols,
            "deployment_class": deployment_class,
            "status_reason": "Refresh GNN en cours.",
            "last_refresh_requested_at": refresh_state.get("requested_at") or started_at,
            "last_refresh_started_at": started_at,
            "last_refresh_status": "running",
        }
    )
    persisted_state = persist_market_gnn_refresh_state(
        {
            **refresh_state,
            "status": "running",
            "queued": False,
            "run_id": run_id,
            "requested_at": refresh_state.get("requested_at") or started_at,
            "started_at": started_at,
            "finished_at": None,
            "failure_reason": None,
            "source_run_id": source_run_id,
        }
    )
    app.state.gnn_refresh_monitor_task = asyncio.create_task(
        _wait_for_gnn_refresh_completion(
            app,
            process=process,
            run_id=run_id,
            source_run_id=source_run_id,
            requested_at=persisted_state.get("requested_at"),
            started_at=started_at,
            log_path=GNN_REFRESH_LOG_PATH,
        )
    )
    return persisted_state


async def _gnn_refresh_queue_loop(app: FastAPI) -> None:
    """Surveille la file de refresh GNN pour demarrer quand le GPU est libre."""

    while True:
        try:
            refresh_state = _current_gnn_refresh_state()
            if refresh_state.get("queued") and not _gnn_refresh_is_running(app):
                training_status = load_training_status()
                if not bool(training_status.get("active")):
                    await _start_gnn_refresh_process(app, refresh_state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Boucle de refresh GNN ignoree: %s", exc)
        await asyncio.sleep(10)


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


class GNNRefreshRequest(BaseModel):
    """Decrit une demande de refresh explicite du GNN."""

    symbols: list[str] = Field(default_factory=list)
    focus_symbol: str | None = None
    context_symbols: list[str] = Field(default_factory=list)
    deployment_class: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    checkpoint_every: int | None = None
    max_symbols: int | None = None



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

    app.state.gnn_refresh_process = None
    app.state.gnn_refresh_monitor_task = None
    refresh_state = _current_gnn_refresh_state()
    if str(refresh_state.get("status") or "").lower() == "running":
        persist_market_gnn_refresh_state(
            {
                **refresh_state,
                "status": "queued",
                "queued": True,
                "failure_reason": "Le service Lab a redemarre avant la fin du refresh precedent.",
            }
        )
    app.state.gnn_refresh_queue_task = asyncio.create_task(_gnn_refresh_queue_loop(app))

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
    queue_task = getattr(app.state, "gnn_refresh_queue_task", None)
    if queue_task:
        queue_task.cancel()
        await asyncio.gather(queue_task, return_exceptions=True)
    monitor_task = getattr(app.state, "gnn_refresh_monitor_task", None)
    if monitor_task:
        monitor_task.cancel()
        await asyncio.gather(monitor_task, return_exceptions=True)
    process = getattr(app.state, "gnn_refresh_process", None)
    if process is not None and process.returncode is None:
        process.terminate()
        await process.wait()
    log_stream = getattr(app.state, "gnn_refresh_log_stream", None)
    if log_stream is not None:
        try:
            log_stream.close()
        except Exception:
            pass
    
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
    sequence_state = load_sequence_state()
    active_run = training_run if str(training_run.get("engine") or "").lower() == "dreamer" else {}
    latest_summary = load_latest_terminal_summary(engine="dreamer")
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
        "dataset_id": active_run.get("dataset_id") or (latest_summary or {}).get("dataset_id"),
        "dataset_source": active_run.get("dataset_source") or (latest_summary or {}).get("dataset_source"),
        "dataset_coverage": active_run.get("dataset_coverage", {}) or (latest_summary or {}).get("dataset_coverage", {}),
        "focus_symbols": active_run.get("focus_symbols", []) or list((latest_summary or {}).get("focus_symbols") or []),
        "gate_profile": active_run.get("gate_profile") or (latest_summary or {}).get("gate_profile"),
        "sequence_id": active_run.get("sequence_id"),
        "window_id": active_run.get("window_id"),
        "trial_id": active_run.get("trial_id"),
        "terminal_summary_path": active_run.get("terminal_summary_path"),
        "supervisor_state": sequence_state.get("state"),
    }
    engine_horizons = {
        horizon: promoter.build_engine_horizon_status("dreamer", horizon)
        for horizon in horizons
    }
    latest_candidate = None
    latest_verdict = None
    latest_run_id = None
    failed_step = None
    artifact_state = None
    if latest_summary:
        latest_run_id = latest_summary.get("run_id")
        failed_step = latest_summary.get("failed_step")
        artifact_state = dict(latest_summary.get("artifact_state") or {})
        if latest_summary.get("latest_candidate"):
            latest_candidate = {
                "engine": "dreamer",
                "horizon": latest_summary.get("horizon"),
                "candidate_id": latest_summary.get("latest_candidate"),
                "failure_mode": latest_summary.get("failure_mode"),
                "run_id": latest_summary.get("run_id"),
            }
        if latest_summary.get("latest_verdict"):
            latest_verdict = {
                "engine": "dreamer",
                "horizon": latest_summary.get("horizon"),
                **dict(latest_summary.get("latest_verdict") or {}),
            }
    for horizon in horizons:
        horizon_status = dict(engine_horizons.get(horizon) or {})
        if latest_candidate is None and horizon_status.get("candidate_id"):
            latest_candidate = {
                "engine": "dreamer",
                "horizon": horizon,
                "candidate_id": horizon_status.get("candidate_id"),
                "failure_mode": horizon_status.get("failure_mode"),
            }
            latest_run_id = latest_run_id or horizon_status.get("latest_run_id")
            failed_step = failed_step or horizon_status.get("failed_step")
            artifact_state = artifact_state or horizon_status.get("artifact_state")
        if latest_verdict is None and horizon_status.get("promotion_gate"):
            latest_verdict = {
                "engine": "dreamer",
                "horizon": horizon,
                "status": horizon_status.get("promotion_gate", {}).get("status"),
                "reason": horizon_status.get("gate_reason"),
                "failure_mode": horizon_status.get("failure_mode"),
            }
    if pipeline.get("active") and not latest_summary:
        latest_run_id = latest_run_id or pipeline.get("run_id")
        failed_step = failed_step or training_run.get("failed_step")
        artifact_state = artifact_state or {"battle_report_present": False}
        if latest_verdict is None:
            latest_verdict = {
                "engine": "dreamer",
                "horizon": str((training_run.get("current_step") or {}).get("horizon") or "").strip() or None,
                "status": "running",
                "reason": "dreamer_run_in_progress",
                "failure_mode": None,
            }
    effective_horizon = str(
        ((latest_verdict or {}).get("horizon"))
        or ((latest_candidate or {}).get("horizon"))
        or ((training_run.get("current_step") or {}).get("horizon"))
        or "scalp"
    ).strip().lower() or "scalp"
    effective_horizon_status = dict(engine_horizons.get(effective_horizon) or {})
    return {
        **gate.get_status(),
        "pipeline": pipeline,
        "horizons": engine_horizons,
        "latest_run_id": latest_run_id,
        "latest_candidate": latest_candidate,
        "latest_verdict": latest_verdict,
        "failed_step": failed_step,
        "artifact_state": artifact_state,
        "focus_symbols": list(pipeline.get("focus_symbols") or []),
        "gate_profile": pipeline.get("gate_profile"),
        "terminal_summary": latest_summary,
        "terminal_summary_path": (
            str((latest_summary or {}).get("path") or "").strip()
            or str(pipeline.get("terminal_summary_path") or "").strip()
            or str(effective_horizon_status.get("terminal_summary_path") or "").strip()
            or None
        ),
        "battle_report_path": effective_horizon_status.get("battle_report_path"),
        "promotion_gate": effective_horizon_status.get("promotion_gate"),
        "promotion_state": effective_horizon_status.get("promotion_state"),
        "can_activate_live": bool(effective_horizon_status.get("can_activate_live", False)),
        "cross_engine_live_comparison": effective_horizon_status.get("cross_engine_live_comparison"),
        "last_successful_step": active_run.get("last_successful_step"),
        "last_successful_step_at": active_run.get("last_successful_step_at"),
        "train_step_phase": active_run.get("train_step_phase"),
        "phase_durations_ms": active_run.get("phase_durations_ms"),
        "stall_detected": bool(active_run.get("stall_detected", False)),
        "stall_reason": active_run.get("stall_reason"),
    }


@app.get("/sequence/status")
async def sequence_status():
    """
    Retourne l'etat persiste du superviseur de sequence V4.

    Returns:
        dict: Etat courant, heartbeat et pointeurs de logs du superviseur.
    """
    state = load_sequence_state()
    return {
        "status": "ok",
        "sequence_id": state.get("sequence_id"),
        "state": state.get("state"),
        "current_window": {
            "profile": state.get("profile"),
            "engine": state.get("engine"),
            "mode": state.get("mode"),
            "window_id": state.get("window_id"),
            "window_index": state.get("window_index"),
        },
        "current_trial": state.get("trial_id"),
        "last_completed_trial": state.get("last_completed_trial"),
        "last_run_id": state.get("last_run_id"),
        "retry_count": state.get("retry_count"),
        "next_step": state.get("next_step"),
        "last_error": state.get("last_error"),
        "continued_after_precheck": state.get("continued_after_precheck"),
        "killed_after_precheck": state.get("killed_after_precheck"),
        "precheck_status": state.get("precheck_status"),
        "precheck_score": state.get("precheck_score"),
        "proxy_terminal_score": state.get("proxy_terminal_score"),
        "supervisor_heartbeat": state.get("supervisor_heartbeat"),
        "stdout_log_path": state.get("stdout_log_path"),
        "stderr_log_path": state.get("stderr_log_path"),
        "payload": state,
    }


@app.get("/dreamer/logs/tail")
async def dreamer_logs_tail(limit: int = Query(default=80, ge=1, le=500)):
    """
    Retourne les dernieres lignes utiles du pipeline Dreamer.

    Args:
        limit (int): Nombre maximal de lignes a retourner.

    Returns:
        dict: Journal filtre Dreamer et dernier verdict structure.
    """
    dreamer_payload = await dreamer_status()
    lines = tail_training_log(limit=limit, source="dreamer")
    return {
        "status": "ok",
        "engine": "dreamer",
        "path": str(RUN_LOG_PATH),
        "line_count": len(lines),
        "lines": lines,
        "pipeline": dreamer_payload.get("pipeline"),
        "latest_candidate": dreamer_payload.get("latest_candidate"),
        "latest_verdict": dreamer_payload.get("latest_verdict"),
    }


def _build_live_governance_snapshot(
    engine_status: dict[str, dict[str, Any]],
    horizon: str = "scalp",
) -> dict[str, Any]:
    """Construit une vue stable des champions enregistres et actifs.

    Args:
        engine_status (dict[str, dict[str, Any]]): Matrice de statut par moteur.
        horizon (str): Horizon live a synthetiser.

    Returns:
        dict[str, Any]: Resume stable pour distinguer enregistre, actif et ensemble.
    """
    normalized_horizon = str(horizon or "scalp").strip().lower() or "scalp"
    muzero_status = dict(engine_status.get("muzero", {}).get(normalized_horizon) or {})
    dreamer_status_payload = dict(engine_status.get("dreamer", {}).get(normalized_horizon) or {})
    registered_live_champion_muzero = (
        str(muzero_status.get("live_champion_id") or "").strip() or None
    )
    registered_live_champion_dreamer = (
        str(dreamer_status_payload.get("live_champion_id") or "").strip() or None
    )
    muzero_can_activate_live = bool(muzero_status.get("can_activate_live", False))
    dreamer_can_activate_live = bool(dreamer_status_payload.get("can_activate_live", False))
    ensemble_enabled = str(
        os.getenv("LIVE_ENSEMBLE_ENABLED", os.getenv("BANKER_ENSEMBLE_ENABLED", "false"))
    ).strip().lower() not in {"0", "false", "no", "off"}
    ensemble_ready = bool(
        registered_live_champion_muzero
        and registered_live_champion_dreamer
        and muzero_can_activate_live
        and dreamer_can_activate_live
    )
    ensemble_active = bool(ensemble_enabled and ensemble_ready)
    return {
        "active_live_engine": (
            "ensemble"
            if ensemble_active
            else ("muzero" if registered_live_champion_muzero and muzero_can_activate_live else None)
        ),
        "registered_live_champion_muzero": registered_live_champion_muzero,
        "registered_live_champion_dreamer": registered_live_champion_dreamer,
        "muzero_promotion_state": muzero_status.get("promotion_state"),
        "dreamer_promotion_state": dreamer_status_payload.get("promotion_state"),
        "muzero_can_activate_live": muzero_can_activate_live,
        "dreamer_can_activate_live": dreamer_can_activate_live,
        "dreamer_live_enabled": bool(
            ensemble_enabled
            and registered_live_champion_dreamer
            and dreamer_can_activate_live
            and ensemble_ready
        ),
        "ensemble_ready": ensemble_ready,
        "ensemble_active": ensemble_active,
    }


def _resolve_runtime_profile(run_status: dict[str, Any] | None = None) -> str:
    """Derive le profil d'exploitation courant du Lab.

    Args:
        run_status (dict[str, Any] | None): Statut training courant si deja charge.

    Returns:
        str: Profil canonique ``day_live_full_stack`` ou ``night_research_training``.
    """

    override = str(os.getenv("HIVE_RUNTIME_PROFILE", "")).strip().lower()
    if override in {"day_live_full_stack", "night_research_training"}:
        return override

    status = dict(run_status or load_training_status() or {})
    launcher = dict(status.get("launcher") or {})
    vllm_state = str(launcher.get("vllm_state") or "").strip().lower()
    if bool(status.get("active")) or vllm_state == "stopped_for_training":
        return "night_research_training"
    return "day_live_full_stack"


async def _build_daytime_activation_snapshot(
    governance_status: dict[str, Any],
    *,
    run_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construit la readiness daytime du full stack trading.

    Args:
        governance_status (dict[str, Any]): Gouvernance live calculee.
        run_status (dict[str, Any] | None): Statut training courant si disponible.

    Returns:
        dict[str, Any]: Readiness des briques actives de jour.
    """

    status = dict(run_status or load_training_status() or {})
    launcher = dict(status.get("launcher") or {})
    gnn_registry = load_market_gnn_registry()
    vllm_host = os.getenv("VLLM_API_HOST", "vllm")
    vllm_state = str(launcher.get("vllm_state") or "").strip().lower()
    if vllm_state == "stopped_for_training":
        vllm_ready = False
        vllm_mode = "stopped_for_training"
    else:
        vllm_probe = await _probe_tcp_dependency("vllm", vllm_host, 8000)
        vllm_ready = bool(vllm_probe.get("ok", False))
        vllm_mode = "online" if vllm_ready else str(vllm_probe.get("state") or "offline")

    muzero_live_ok = bool(
        governance_status.get("registered_live_champion_muzero")
        and governance_status.get("muzero_can_activate_live", False)
    )
    dreamer_live_ok = bool(
        governance_status.get("registered_live_champion_dreamer")
        and governance_status.get("dreamer_can_activate_live", False)
    )
    gnn_consultative_ok = bool(gnn_registry.get("champion_ready", False))
    full_stack_ready = bool(
        governance_status.get("ensemble_ready", False)
        and muzero_live_ok
        and dreamer_live_ok
        and gnn_consultative_ok
        and vllm_ready
    )
    return {
        "muzero_live_ok": muzero_live_ok,
        "dreamer_live_ok": dreamer_live_ok,
        "gnn_consultative_ok": gnn_consultative_ok,
        "vllm_ready": vllm_ready,
        "vllm_mode": vllm_mode,
        "shadow_learning_mode": "shadow_only",
        "intraday_retrain_allowed": False,
        "intraday_promotion_allowed": False,
        "full_stack_ready": full_stack_ready,
    }


def _derive_ga_trial_promotion_state(trial: dict[str, Any]) -> str:
    """Derive un etat de promotion lisible pour un trial GA.

    Args:
        trial (dict[str, Any]): Trial brut charge depuis Timescale.

    Returns:
        str: Etat de promotion stable pour l'API publique.
    """

    payload = dict(trial.get("payload") or {})
    promotion_gate = dict(trial.get("promotion_gate") or payload.get("promotion_gate") or {})
    live_comparison = dict(trial.get("live_comparison") or payload.get("live_comparison") or {})
    gate_status = str(promotion_gate.get("status") or "").strip().lower()
    if gate_status:
        return gate_status
    if bool(live_comparison.get("allowed", False)):
        return "candidate_only"
    if str(trial.get("failure_mode") or "").strip():
        return "blocked"
    return "candidate_only"


def _build_seeded_ga_generation_views(campaign_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Construit une vue agrégée des trials GA par generation.

    Args:
        campaign_state (dict[str, Any]): Etat courant de campagne seedee.

    Returns:
        list[dict[str, Any]]: Generations ordonnees avec leurs trials.
    """

    campaign_id = str(campaign_state.get("campaign_id") or "").strip()
    if not campaign_id:
        return []

    raw_trials = load_ga_trials(campaign_id=campaign_id, limit=512)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for trial in raw_trials:
        generation = int(trial.get("ga_generation") or 0)
        grouped.setdefault(generation, []).append(
            {
                "trial_id": trial.get("trial_id"),
                "run_id": trial.get("run_id"),
                "trial_mode": trial.get("trial_mode"),
                "fitness_score": trial.get("fitness_score"),
                "failure_mode": trial.get("failure_mode"),
                "promotion_state": _derive_ga_trial_promotion_state(trial),
                "finished_at": trial.get("finished_at"),
            }
        )

    generations: list[dict[str, Any]] = []
    for generation in sorted(grouped):
        trials = sorted(
            grouped[generation],
            key=lambda item: (
                str(item.get("finished_at") or ""),
                str(item.get("trial_id") or ""),
            ),
        )
        generations.append(
            {
                "generation": generation,
                "trial_count": len(trials),
                "trials": trials,
            }
        )
    return generations


def _build_ga_status_snapshot(
    engine_status: dict[str, dict[str, Any]],
    genetic: GeneticUpdater,
) -> dict[str, Any]:
    """Construit une vue consultative du champion GA.

    Args:
        engine_status (dict[str, dict[str, Any]]): Matrice de statut des moteurs.
        genetic (GeneticUpdater): Registre genetique courant.

    Returns:
        dict[str, Any]: Vue stable du gagnant de gouvernance par horizon.
    """
    by_horizon: dict[str, dict[str, Any]] = {}
    for horizon in ["scalp", "intraday", "swing"]:
        muzero_status = dict(engine_status.get("muzero", {}).get(horizon) or {})
        by_horizon[horizon] = {
            "champion_id": genetic.get_champion(horizon),
            "source_engine": "muzero",
            "source_run_id": muzero_status.get("latest_run_id"),
            "battle_outcome": ((muzero_status.get("arena_report") or {}).get("outcome")),
            "promotion_gate": muzero_status.get("promotion_gate"),
            "selection_basis": "arena + gate",
            "promotion_state": muzero_status.get("promotion_state"),
            "can_activate_live": bool(muzero_status.get("can_activate_live", False)),
        }
    seeded_campaign = dict(load_seeded_muzero_campaign_state() or {})
    seeded_campaign["generations"] = _build_seeded_ga_generation_views(seeded_campaign)
    return {
        "champion_kind": "governance",
        "horizons": by_horizon,
        "seeded_muzero_campaign": seeded_campaign,
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
    timescale_source = describe_timescale_source()

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
    runtime_profile = _resolve_runtime_profile()
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
    governance_status = _build_live_governance_snapshot(engine_status, horizon="scalp")
    ga_status_payload = _build_ga_status_snapshot(engine_status, genetic)
    daytime_activation = await _build_daytime_activation_snapshot(
        governance_status,
        run_status=load_training_status(),
    )

    payload = {
        "status": "ok",
        "runtime_profile": runtime_profile,
        "selection_policy": promoter.get_live_selection_policy(),
        "dreamer_gate": gate.get_status(),
        "data_source": timescale_source.get("source"),
        "research_context_version": "v1_consultatif",
        "consultative_blockers": {},
        "champions": registry_champions,
        "registry_champions": registry_champions,
        "live_champions": live_champions,
        "live_champions_by_engine": live_champions_by_engine,
        "performance_summary": performance_summary,
        "horizons": horizon_status,
        "engines": engine_status,
        "ga": ga_status_payload,
        "daytime_activation": daytime_activation,
        "nightly_summary": nightly_summary,
        **governance_status,
    }
    await _publish_champion_status_snapshot(payload)
    return payload


@app.get("/ga/status")
async def ga_status():
    """Retourne le champion GA de gouvernance sans l'exposer au live.

    Returns:
        dict: Vue stable du champion GA par horizon.
    """
    promoter: ChampionPromoter = app.state.promoter
    genetic: GeneticUpdater = app.state.genetic
    horizons = ["scalp", "intraday", "swing"]
    engine_status = promoter.build_engine_matrix_status(horizons, genetic.get_all_champions())
    runtime_profile = _resolve_runtime_profile()
    return {
        "status": "ok",
        "runtime_profile": runtime_profile,
        "ga": _build_ga_status_snapshot(engine_status, genetic),
    }


@app.post("/internal/ga-seeded-campaign")
async def persist_seeded_ga_campaign(payload: dict[str, Any]):
    """Persiste l'etat d'une campagne GA seedee depuis un superviseur externe.

    Args:
        payload (dict[str, Any]): Etat partiel ou complet de campagne.

    Returns:
        dict[str, Any]: Etat persiste et identifiant de campagne.
    """

    persisted = persist_seeded_muzero_campaign_state(payload)
    return {
        "status": "ok",
        "campaign_id": str(persisted.get("campaign_id") or "") or None,
        "persisted": persisted,
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
    runtime_profile = _resolve_runtime_profile(run_status)
    nightly_summary = load_nightly_summary()
    sequence_state = load_sequence_state()
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
    run_view["supervisor_state"] = sequence_state.get("state")
    if arena_progress and isinstance(arena_progress, dict):
        challenger_metrics = dict((arena_progress.get("challenger") or {}).get("metrics") or {})
        if challenger_metrics.get("metrics_by_position_mechanics"):
            run_view["metrics_by_position_mechanics"] = challenger_metrics.get("metrics_by_position_mechanics")

    payload = {
        "status": "ok",
        "runtime_profile": runtime_profile,
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
        "focus_symbols": run_view.get("focus_symbols", []),
        "gate_profile": run_view.get("gate_profile"),
        "gold_precheck": run_view.get("gold_precheck"),
        "precheck_status": run_view.get("precheck_status"),
        "precheck_step": run_view.get("precheck_step"),
        "precheck_metrics": run_view.get("precheck_metrics"),
        "precheck_summary_path": run_view.get("precheck_summary_path"),
        "timescaledb_status": ((run_view.get("dataset_coverage") or {}).get("timescaledb") or {}),
        "bars_table": (((run_view.get("dataset_coverage") or {}).get("timescaledb") or {}).get("bars_table")),
        "features_table": (((run_view.get("dataset_coverage") or {}).get("timescaledb") or {}).get("features_table")),
        "ga_status": run_view.get("ga_status"),
        "ga_generation": run_view.get("ga_generation"),
        "ga_trial": run_view.get("ga_trial"),
        "sequence_id": run_view.get("sequence_id"),
        "sequence_profile": run_view.get("sequence_profile"),
        "window_id": run_view.get("window_id"),
        "trial_id": run_view.get("trial_id"),
        "trial_mode": run_view.get("trial_mode"),
        "trial_cost_profile": run_view.get("trial_cost_profile"),
        "terminal_summary_path": run_view.get("terminal_summary_path"),
        "last_successful_step": run_view.get("last_successful_step"),
        "last_successful_step_at": run_view.get("last_successful_step_at"),
        "train_step_phase": run_view.get("train_step_phase"),
        "phase_durations_ms": run_view.get("phase_durations_ms", {}),
        "resume_checkpoint_path": run_view.get("resume_checkpoint_path"),
        "resume_step": run_view.get("resume_step"),
        "stall_detected": run_view.get("stall_detected"),
        "stall_reason": run_view.get("stall_reason"),
        "supervisor_state": run_view.get("supervisor_state"),
        "replay_cache_status": run_view.get("replay_cache_status"),
        "replay_cache_key": run_view.get("replay_cache_key"),
        "replay_cache_entries": run_view.get("replay_cache_entries"),
        "replay_cache_source": run_view.get("replay_cache_source"),
        "shadow_buffer_size": run_view.get("shadow_buffer_size"),
        "sequence_length": run_view.get("sequence_length"),
        "sequence_stride": run_view.get("sequence_stride"),
        "world_model_steps": run_view.get("world_model_steps"),
        "dataset_coverage": run_view.get("dataset_coverage", {}),
        "effective_source_reason": ((run_view.get("dataset_coverage") or {}).get("effective_source_reason")),
        "replay_cache_reuse_ratio": ((run_view.get("dataset_coverage") or {}).get("replay_cache_reuse_ratio")),
        "metrics_by_position_mechanics": run_view.get("metrics_by_position_mechanics", {}),
    }
    await _publish_training_run_snapshot(
        run_view=run_view,
        dependencies=dependencies,
        universe=universe_summary,
        nightly_summary=nightly_summary,
    )
    return payload


@app.get("/training/logs/tail")
async def training_logs_tail(
    limit: int = Query(default=120, ge=1, le=500),
    source: str | None = Query(default=None),
    contains: str | None = Query(default=None),
):
    """
    Retourne un tail filtre du journal partage d'entrainement.

    Args:
        limit (int): Nombre maximal de lignes a retourner.
        source (str | None): Source a filtrer, par exemple ``muzero`` ou ``dreamer``.
        contains (str | None): Motif libre a rechercher dans les lignes.

    Returns:
        dict: Journal filtre et metadonnees de lecture.
    """
    lines = tail_training_log(limit=limit, source=source, contains=contains)
    return {
        "status": "ok",
        "path": str(RUN_LOG_PATH),
        "limit": limit,
        "source": source,
        "contains": contains,
        "line_count": len(lines),
        "lines": lines,
    }


@app.get("/ops/logs")
async def ops_logs(limit: int = Query(default=80, ge=1, le=500)):
    """
    Retourne une vue centralisee des journaux et etats ops partages.

    Args:
        limit (int): Nombre maximal de lignes par journal.

    Returns:
        dict: Tails utiles pour le suivi ops sans lire les logs Docker bruts.
    """
    scheduler_state = load_cpu_scheduler_state()
    sequence_state = load_sequence_state()
    latest_muzero_summary = load_latest_terminal_summary(engine="muzero")
    latest_dreamer_summary = load_latest_terminal_summary(engine="dreamer")
    sequence_stdout_path = _as_path(sequence_state.get("stdout_log_path")) or _resolve_latest_checkpoint_log(
        "*sequence*.out.log"
    )
    sequence_stderr_path = _as_path(sequence_state.get("stderr_log_path")) or _resolve_latest_checkpoint_log(
        "*sequence*.err.log"
    )

    sequence_stdout = (
        tail_log_file(sequence_stdout_path, limit=limit)
        if sequence_stdout_path is not None
        else []
    )
    sequence_stderr = (
        tail_log_file(sequence_stderr_path, limit=limit)
        if sequence_stderr_path is not None
        else []
    )

    return {
        "status": "ok",
        "checkpoints_dir": str(STATUS_DIR),
        "training": {
            "path": str(RUN_LOG_PATH),
            "lines": tail_training_log(limit=limit),
        },
        "dreamer": {
            "path": str(RUN_LOG_PATH),
            "lines": tail_training_log(limit=limit, source="dreamer"),
        },
        "sequence": {
            "state": sequence_state,
            "stdout_path": str(sequence_stdout_path) if sequence_stdout_path else None,
            "stdout_lines": sequence_stdout,
            "stderr_path": str(sequence_stderr_path) if sequence_stderr_path else None,
            "stderr_lines": sequence_stderr,
        },
        "terminal_summaries": {
            "muzero": latest_muzero_summary,
            "dreamer": latest_dreamer_summary,
        },
        "cpu_scheduler": {
            "state_path": str(CPU_SCHEDULER_STATE_PATH),
            "state": scheduler_state,
        },
    }


@app.post("/internal/sequence/window")
async def persist_sequence_window(payload: dict[str, Any]):
    """
    Persiste une fenetre de sequence V4 depuis un superviseur externe.

    Args:
        payload (dict[str, Any]): Etat courant d'une fenetre de sequence.

    Returns:
        dict: Statut de persistance interne.
    """

    persisted = record_run_window(payload)
    return {
        "status": "ok" if persisted else "degraded",
        "persisted": persisted,
        "window_id": str(payload.get("window_id") or "") or None,
    }


@app.post("/internal/ga-trial")
async def persist_ga_trial(payload: dict[str, Any]):
    """
    Persiste un resultat de trial GA depuis un superviseur externe.

    Args:
        payload (dict[str, Any]): Charge utile de scoring d'un trial.

    Returns:
        dict: Statut de persistance interne.
    """

    persisted = record_ga_trial(payload)
    return {
        "status": "ok" if persisted else "degraded",
        "persisted": persisted,
        "trial_id": str(payload.get("trial_id") or "") or None,
    }


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
    governance_status = _build_live_governance_snapshot(
        engine_matrix,
        horizon=normalized_horizon,
    )
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
        **governance_status,
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
    refresh_state = _current_gnn_refresh_state()
    graph_snapshot = build_market_gnn_graph_snapshot(registry=registry)
    return {
        "status": "ok",
        "gnn": registry,
        "champion_id": registry.get("champion_id"),
        "champion_ready": bool(registry.get("champion_ready", False)),
        "champion_kind": registry.get("champion_kind"),
        "focus_symbol": registry.get("focus_symbol"),
        "context_symbols": registry.get("context_symbols", []),
        "deployment_class": registry.get("deployment_class"),
        "graph_readiness": {
            "status": graph_snapshot.get("status"),
            "reason": graph_snapshot.get("reason"),
            "selected_timeframe": graph_snapshot.get("selected_timeframe"),
            "candidate_timeframes": graph_snapshot.get("candidate_timeframes", []),
            "overlap_points": graph_snapshot.get("overlap_points", 0),
            "missing_symbols": graph_snapshot.get("missing_symbols", []),
        },
        "refresh": refresh_state,
    }


@app.post("/gnn/refresh")
async def request_gnn_refresh(request: GNNRefreshRequest | None = None):
    """Declenche ou planifie un refresh explicite du GNN."""

    request_payload = request.model_dump() if request is not None else {}
    current_state = _current_gnn_refresh_state()
    if _gnn_refresh_is_running(app):
        return {"status": "running", "refresh": current_state, "message": "Un refresh GNN est deja en cours."}

    training_state = load_training_status()
    requested_at = datetime.utcnow().isoformat() + "Z"
    source_run_id = _resolve_gnn_source_run_id()
    queued = bool(training_state.get("active"))
    requested_symbols = [
        str(symbol).strip()
        for symbol in list(request_payload.get("symbols") or [])
        if str(symbol).strip()
    ]
    focus_symbol = str(request_payload.get("focus_symbol") or "").strip() or None
    context_symbols = [
        str(symbol).strip()
        for symbol in list(request_payload.get("context_symbols") or [])
        if str(symbol).strip()
    ]
    if focus_symbol and focus_symbol not in requested_symbols:
        requested_symbols.insert(0, focus_symbol)
    refresh_state = persist_market_gnn_refresh_state(
        {
            **current_state,
            "status": "queued" if queued else "pending",
            "queued": queued,
            "requested_at": requested_at,
            "started_at": None if queued else current_state.get("started_at"),
            "finished_at": None,
            "run_id": current_state.get("run_id") or f"gnn_refresh_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "failure_reason": None,
            "source_run_id": source_run_id,
            "requested_by": "api",
            "requested_symbols": requested_symbols,
            "focus_symbol": focus_symbol,
            "context_symbols": context_symbols,
            "deployment_class": str(request_payload.get("deployment_class") or "").strip() or None,
            "epochs": request_payload.get("epochs"),
            "batch_size": request_payload.get("batch_size"),
            "checkpoint_every": request_payload.get("checkpoint_every"),
            "max_symbols": request_payload.get("max_symbols"),
        }
    )
    update_market_gnn_registry(
        {
            "source_run_id": source_run_id,
            "focus_symbols": requested_symbols,
            "focus_symbol": focus_symbol,
            "context_symbols": context_symbols,
            "deployment_class": str(request_payload.get("deployment_class") or "").strip() or None,
            "status_reason": (
                "Refresh GNN planifie: un entrainement GPU est deja actif."
                if queued
                else "Refresh GNN demande manuellement."
            ),
            "last_refresh_requested_at": requested_at,
            "last_refresh_status": "queued" if queued else "pending",
        }
    )
    if not queued:
        refresh_state = await _start_gnn_refresh_process(app, refresh_state)
        return {
            "status": "started",
            "refresh": refresh_state,
            "message": "Refresh GNN demarre.",
        }
    return {
        "status": "queued",
        "refresh": refresh_state,
        "message": "Refresh GNN place en file d'attente jusqu'a liberation du GPU.",
    }


@app.get("/gnn/refresh/status")
async def gnn_refresh_status():
    """Retourne la file de refresh GNN et son etat courant."""

    refresh_state = _current_gnn_refresh_state()
    return {
        "status": "ok",
        "refresh": refresh_state,
        "running": _gnn_refresh_is_running(app),
        "log_path": str(GNN_REFRESH_LOG_PATH),
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
        "champion_id": registry.get("champion_id"),
        "champion_ready": bool(registry.get("champion_ready", False)),
        "champion_kind": registry.get("champion_kind"),
        "status_reason": registry.get("status_reason"),
        "trained_at": registry.get("trained_at"),
        "source_run_id": registry.get("source_run_id"),
        "focus_symbol": registry.get("focus_symbol"),
        "context_symbols": registry.get("context_symbols", []),
        "deployment_class": registry.get("deployment_class"),
        "coverage_summary": registry.get("coverage_summary", {}),
        "last_refresh_requested_at": registry.get("last_refresh_requested_at"),
        "last_refresh_started_at": registry.get("last_refresh_started_at"),
        "last_refresh_finished_at": registry.get("last_refresh_finished_at"),
        "last_refresh_status": registry.get("last_refresh_status"),
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
    snapshot["status_reason"] = registry.get("status_reason")
    snapshot["coverage_summary"] = registry.get("coverage_summary", {})
    snapshot["focus_symbol"] = registry.get("focus_symbol")
    snapshot["context_symbols"] = registry.get("context_symbols", [])
    snapshot["deployment_class"] = registry.get("deployment_class")
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




