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
from datetime import datetime, timedelta, timezone
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
    backfill_timescaledb_from_history,
    build_timescaledb_coverage_report,
    describe_timescale_source,
    load_recent_ga_trials,
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
    load_effective_training_status,
    load_latest_terminal_summary,
    load_nightly_summary,
    load_cpu_scheduler_state,
    load_sequence_state,
    select_effective_training_step,
    set_service_recovery_snapshot,
    tail_log_file,
    load_training_status,
    tail_training_log,
    write_terminal_summary,
)
from eva_lab.training_utils import get_gnn_model_kwargs, parse_symbol_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
HOST_DATA_PREFIX = Path("/home/aza/The_Hive/data")
CONTAINER_DATA_ROOT = Path("/app/eva-lab/data")
LAB_APP_ROOT = Path("/app/eva-lab")
GNN_REFRESH_LOG_PATH = STATUS_DIR / "gnn_refresh.log"
CANONICAL_SCALP_FULL_SYMBOLS = [
    "EURUSD",
    "XAUUSD",
    "GBPUSD",
    "USDJPY",
    "US30.cash",
    "GER40.cash",
    "US500.cash",
]
CANONICAL_COVERAGE_TIMEFRAMES = ["M5", "H1", "D1"]


class TimescaleBackfillRequest(BaseModel):
    """Decrit une demande de backfill CSV vers TimescaleDB.

    Attributes:
        symbols (list[str]): Symboles cibles a recharger.
        timeframes (list[str]): Timeframes a injecter dans TimeDB.
        history_dir (str): Dossier source des historiques CSV.
    """

    symbols: list[str] = Field(default_factory=lambda: list(CANONICAL_SCALP_FULL_SYMBOLS))
    timeframes: list[str] = Field(default_factory=lambda: list(CANONICAL_COVERAGE_TIMEFRAMES))
    history_dir: str = "data/history"


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


def _resolve_runtime_profile(run_status: dict[str, Any] | None = None) -> str:
    """Determine le profil d'exploitation courant du Lab.

    Args:
        run_status (dict[str, Any] | None): Statut training courant si deja charge.

    Returns:
        str: Profil normalise ``day_live_full_stack`` ou ``night_research_training``.
    """

    forced = str(os.getenv("LAB_RUNTIME_PROFILE", "")).strip().lower()
    if forced in {"day_live_full_stack", "night_research_training"}:
        return forced
    current_run = dict(run_status or load_effective_training_status())
    return "night_research_training" if bool(current_run.get("active")) else "day_live_full_stack"


def _normalize_rate_percent(value: Any) -> float:
    """Normalise un taux heterogene vers un pourcentage borné.

    Args:
        value (Any): Valeur brute provenant du registre ou d'un resume.

    Returns:
        float: Taux exprime en pourcentage, borne entre 0 et 100.
    """

    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if numeric <= 1.0:
        numeric *= 100.0
    return round(max(0.0, min(100.0, numeric)), 2)


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convertit une valeur heterogene en flottant robuste.

    Args:
        value (Any): Valeur brute a convertir.
        default (float): Valeur de repli si la conversion echoue.

    Returns:
        float: Valeur convertie ou repli.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_metrics_overview(metrics: dict[str, Any] | None) -> dict[str, float | int]:
    """Construit un bloc compact de metriques de performance.

    Args:
        metrics (dict[str, Any] | None): Metriques brutes du moteur.

    Returns:
        dict[str, float | int]: Vue condensee des metriques de pilotage.
    """

    payload = dict(metrics or {})
    return {
        "profit_factor": round(_to_float(payload.get("profit_factor")), 4),
        "return_pct": round(_to_float(payload.get("return_pct")), 4),
        "net_realized_pct": round(_to_float(payload.get("net_realized_pct")), 4),
        "win_rate": round(_to_float(payload.get("win_rate")), 2),
        "total_trades": int(_to_float(payload.get("total_trades"))),
        "evaluation_games": int(_to_float(payload.get("evaluation_games"))),
        "evaluation_symbols": int(_to_float(payload.get("evaluation_symbols"))),
        "max_drawdown_pct": round(_to_float(payload.get("max_drawdown_pct")), 4),
        "close_quality_score": round(_to_float(payload.get("close_quality_score")), 4),
        "directional_imbalance": round(_to_float(payload.get("directional_imbalance"), 1.0), 4),
        "hold_drag_score": round(_to_float(payload.get("hold_drag_score")), 4),
    }


def _build_symbol_overview(metrics_by_symbol: dict[str, Any] | None, limit: int = 3) -> dict[str, Any]:
    """Trie les meilleurs et pires symboles d'un bloc de metriques.

    Args:
        metrics_by_symbol (dict[str, Any] | None): Metriques consolidees par symbole.
        limit (int): Nombre maximum de symboles a exposer par liste.

    Returns:
        dict[str, Any]: Symboles forts, faibles et couverture globale.
    """

    rows: list[dict[str, Any]] = []
    for symbol, raw_metrics in dict(metrics_by_symbol or {}).items():
        metrics = dict(raw_metrics or {})
        rows.append(
            {
                "symbol": str(symbol),
                "net_realized_pct": round(_to_float(metrics.get("net_realized_pct")), 4),
                "return_pct": round(_to_float(metrics.get("return_pct")), 4),
                "profit_factor": round(_to_float(metrics.get("profit_factor")), 4),
                "score": round(_to_float(metrics.get("score")), 4),
                "evaluation_games": int(_to_float(metrics.get("evaluation_games"))),
            }
        )
    best = sorted(
        rows,
        key=lambda item: (
            item.get("net_realized_pct", 0.0),
            item.get("profit_factor", 0.0),
            item.get("score", 0.0),
        ),
        reverse=True,
    )[:limit]
    worst = sorted(
        rows,
        key=lambda item: (
            item.get("net_realized_pct", 0.0),
            item.get("profit_factor", 0.0),
            item.get("score", 0.0),
        ),
    )[:limit]
    return {
        "count": len(rows),
        "best": best,
        "worst": worst,
    }


def _estimate_training_eta(run_view: dict[str, Any]) -> dict[str, Any]:
    """Estime l'ETA d'un run a partir de la telemetrie de phase.

    Args:
        run_view (dict[str, Any]): Statut courant du run.

    Returns:
        dict[str, Any]: Progression relative et ETA si calculable.
    """

    if not bool(run_view.get("active")):
        return {
            "progress_percent": None,
            "remaining_steps": None,
            "eta_seconds": None,
            "eta_at": None,
        }
    current_step = dict(run_view.get("current_step") or {})
    current = int(_to_float(current_step.get("training_step_current")))
    total = int(_to_float(current_step.get("training_step_total")))
    if total <= 0:
        return {
            "progress_percent": None,
            "remaining_steps": None,
            "eta_seconds": None,
            "eta_at": None,
        }
    progress_percent = round((current / total) * 100.0, 2)
    remaining_steps = max(total - current, 0)
    durations = dict(run_view.get("phase_durations_ms") or {})
    per_step_ms = sum(
        max(_to_float(value), 0.0)
        for value in durations.values()
    )
    if per_step_ms <= 0.0 or remaining_steps <= 0:
        return {
            "progress_percent": progress_percent,
            "remaining_steps": remaining_steps,
            "eta_seconds": 0 if remaining_steps <= 0 else None,
            "eta_at": datetime.now(timezone.utc).isoformat() if remaining_steps <= 0 else None,
        }
    eta_seconds = int((per_step_ms / 1000.0) * remaining_steps)
    eta_at = datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)
    return {
        "progress_percent": progress_percent,
        "remaining_steps": remaining_steps,
        "eta_seconds": eta_seconds,
        "eta_at": eta_at.isoformat(),
    }


def _build_engine_overview_card(engine: str, horizon: str, status: dict[str, Any]) -> dict[str, Any]:
    """Construit une carte compacte d'un moteur et d'un horizon.

    Args:
        engine (str): Moteur cible.
        horizon (str): Horizon de pilotage.
        status (dict[str, Any]): Statut detaille moteur/horizon.

    Returns:
        dict[str, Any]: Carte de synthese pour le pilotage.
    """

    gate = dict(status.get("promotion_gate") or {})
    live_metrics = _build_metrics_overview((gate.get("metrics") or {}))
    candidate_metrics = _build_metrics_overview(status.get("candidate_metrics") or {})
    return {
        "engine": engine,
        "horizon": horizon,
        "live_champion_id": status.get("live_champion_id"),
        "candidate_id": status.get("candidate_id"),
        "can_activate_live": bool(status.get("can_activate_live")),
        "selection": status.get("selection"),
        "promotion_state": status.get("promotion_state") or (
            "promoted"
            if status.get("live_champion_id")
            else ("candidate_only" if status.get("candidate_id") else "none")
        ),
        "gate_allowed": bool(gate.get("allowed")),
        "gate_reason": gate.get("reason") or status.get("gate_reason"),
        "failure_mode": status.get("failure_mode"),
        "feature_profile": status.get("feature_profile"),
        "family": status.get("family"),
        "live_metrics": live_metrics,
        "candidate_metrics": candidate_metrics,
        "top_live_symbols": _build_symbol_overview(status.get("metrics_by_symbol") or {}),
        "top_candidate_symbols": _build_symbol_overview(
            (status.get("candidate_metrics") or {}).get("metrics_by_symbol") or {}
        ),
        "latest_run_id": status.get("latest_run_id"),
        "latest_verdict": status.get("latest_verdict"),
    }


def _build_ga_overview(recent_trials: list[dict[str, Any]], current_run: dict[str, Any]) -> dict[str, Any]:
    """Construit une vue compacte de la campagne GA courante.

    Args:
        recent_trials (list[dict[str, Any]]): Derniers essais persistés.
        current_run (dict[str, Any]): Statut courant du run actif.

    Returns:
        dict[str, Any]: Synthese concise de la campagne GA.
    """

    ranked_trials = sorted(
        [dict(item or {}) for item in recent_trials],
        key=lambda item: float(item.get("fitness_score") or item.get("score") or 0.0),
        reverse=True,
    )
    eta = _estimate_training_eta(current_run)
    return {
        "active_generation": current_run.get("ga_generation"),
        "active_trial": current_run.get("ga_trial"),
        "phase": ((current_run.get("current_step") or {}).get("phase")),
        "progress_percent": eta.get("progress_percent"),
        "eta_seconds": eta.get("eta_seconds"),
        "eta_at": eta.get("eta_at"),
        "recent_best": [
            {
                "trial_id": item.get("trial_id"),
                "generation": item.get("generation"),
                "fitness_score": item.get("fitness_score") or item.get("score"),
                "promotion_state": item.get("promotion_state"),
                "failure_mode": item.get("failure_mode"),
                "finalist_rank": item.get("finalist_rank"),
            }
            for item in ranked_trials[:5]
        ],
    }


def _compute_gnn_champion_payload(registry: dict[str, Any]) -> dict[str, Any]:
    """Construit une vue consultative stable du champion GNN.

    Args:
        registry (dict[str, Any]): Registre GNN brut.

    Returns:
        dict[str, Any]: Vue derivee du champion consultatif.
    """

    status = str(registry.get("status") or "").strip().lower()
    trained_at = str(registry.get("trained_at") or "").strip() or None
    freshness_hours: float | None = None
    if trained_at:
        try:
            trained_dt = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
            freshness_hours = round(
                (datetime.now(tz=timezone.utc) - trained_dt.astimezone(timezone.utc)).total_seconds() / 3600.0,
                2,
            )
        except ValueError:
            freshness_hours = None
    metrics = dict(registry.get("metrics") or {})
    deployment_class = str(registry.get("deployment_class") or "").strip() or "consultative"
    directional_precision = max(
        _normalize_rate_percent(metrics.get("directional_precision")),
        _normalize_rate_percent(metrics.get("scalp_accuracy")),
        _normalize_rate_percent(metrics.get("intraday_accuracy")),
        _normalize_rate_percent(metrics.get("swing_accuracy")),
    )
    agreement_rate = _normalize_rate_percent(metrics.get("agreement_rate"))
    decision_support_metrics = {
        "scalp_accuracy": float(metrics.get("scalp_accuracy", 0.0) or 0.0),
        "intraday_accuracy": float(metrics.get("intraday_accuracy", 0.0) or 0.0),
        "swing_accuracy": float(metrics.get("swing_accuracy", 0.0) or 0.0),
        "directional_precision": directional_precision,
        "agreement_rate": agreement_rate,
        "loss": float(metrics.get("loss", 0.0) or 0.0),
        "samples": int(metrics.get("samples", 0) or 0),
    }
    champion_ready = (
        status in {"validated", "live"}
        and deployment_class == "consultative"
        and not str(registry.get("last_refresh_status") or "").strip().lower() == "error"
        and not bool(registry.get("stale", status == "stale"))
        and bool(str(registry.get("source_run_id") or "").strip())
        and directional_precision >= 55.0
        and (freshness_hours is None or freshness_hours < 72.0)
    )
    return {
        "champion_id": str(registry.get("version") or "").strip() or None,
        "champion_ready": champion_ready,
        "champion_kind": "consultative",
        "source_run_id": str(registry.get("source_run_id") or "").strip() or None,
        "freshness_hours": freshness_hours,
        "decision_support_metrics": decision_support_metrics,
    }


def _group_ga_trials_by_generation(
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Regroupe les essais GA par generation pour l'API publique.

    Args:
        trials (list[dict[str, Any]]): Essais recents lus depuis TimeDB.

    Returns:
        list[dict[str, Any]]: Groupes ordonnes par generation.
    """

    grouped: dict[int, list[dict[str, Any]]] = {}
    for trial in trials:
        generation = int(trial.get("ga_generation") or 0)
        grouped.setdefault(generation, []).append(
            {
                "generation": generation,
                "trial_id": trial.get("trial_id"),
                "phase": trial.get("phase"),
                "early_kill_reason": trial.get("early_kill_reason"),
                "fitness_score": trial.get("fitness_score"),
                "promotion_state": trial.get("promotion_state"),
                "finalist_rank": trial.get("finalist_rank"),
                "failure_mode": trial.get("failure_mode"),
                "run_id": trial.get("run_id"),
                "finished_at": trial.get("finished_at"),
            }
        )
    ordered: list[dict[str, Any]] = []
    for generation in sorted(grouped.keys(), reverse=True):
        ordered.append(
            {
                "generation": generation,
                "trial_count": len(grouped[generation]),
                "trials": grouped[generation],
            }
        )
    return ordered


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


def _pid_is_alive(raw_pid: Any) -> bool | None:
    """
    Indique si un PID local est encore present.

    Args:
        raw_pid (Any): Valeur brute potentiellement serialisee.

    Returns:
        bool | None: ``True`` si le processus existe, ``False`` s'il est
        absent, ``None`` si la valeur n'est pas exploitable.
    """
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _trainer_process_is_alive(run_status: dict[str, Any]) -> bool:
    """
    Indique si le processus d'entrainement annonce existe encore vraiment.

    Args:
        run_status (dict[str, Any]): Statut training courant.

    Returns:
        bool: ``True`` si un processus d'entrainement valide est observe.
    """
    launcher = dict(run_status.get("launcher") or {})
    pid_state = _pid_is_alive(launcher.get("remote_pid"))
    if pid_state is not None:
        return pid_state
    return bool(run_status.get("active"))


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
    trainer_running = _trainer_process_is_alive(run_status)
    dependencies["trainer"] = {
        "name": "trainer",
        "ok": trainer_running,
        "state": "running" if trainer_running else "idle",
        "container": trainer_container,
        "pid": launcher.get("remote_pid"),
    }
    return dependencies


def _build_lab_service_recovery_snapshot(
    app: FastAPI,
    run_status: dict[str, Any],
) -> dict[str, Any]:
    """
    Construit un audit compact de reprise service pour le Lab.

    Args:
        app (FastAPI): Application Lab courante.
        run_status (dict[str, Any]): Statut training courant normalise.

    Returns:
        dict[str, Any]: Resume operateur de reprise apres redemarrage.
    """
    promoter: ChampionPromoter = app.state.promoter
    gnn_payload = _compute_gnn_champion_payload(load_market_gnn_registry())
    dreamer_scalp = promoter.build_engine_horizon_status("dreamer", "scalp")
    dreamer_live_lock = dict(dreamer_scalp.get("live_lock") or {})
    timescale_info = describe_timescale_source()
    return {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "lab_online": True,
        "training_active": bool(run_status.get("active")) and _trainer_process_is_alive(run_status),
        "trainer_process_alive": _trainer_process_is_alive(run_status),
        "gnn_ready": bool(gnn_payload.get("champion_ready")),
        "gnn_freshness_hours": gnn_payload.get("freshness_hours"),
        "dreamer_live_locked": bool(dreamer_live_lock.get("active")),
        "dreamer_live_lock_reason": dreamer_live_lock.get("reason"),
        "timescaledb_enabled": bool(timescale_info.get("enabled")),
        "timescaledb_state": str(timescale_info.get("state") or "disabled"),
        "stale_detected": bool(run_status.get("stale_detected")),
        "stale_reasons": list(run_status.get("stale_reasons") or []),
    }


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

    recovery_snapshot = _build_lab_service_recovery_snapshot(
        app,
        load_effective_training_status(clean_stale=True),
    )
    app.state.service_recovery = recovery_snapshot
    set_service_recovery_snapshot(recovery_snapshot)
    logger.info(
        "Audit de reprise Lab: training_active=%s | gnn_ready=%s | dreamer_live_locked=%s",
        recovery_snapshot.get("training_active"),
        recovery_snapshot.get("gnn_ready"),
        recovery_snapshot.get("dreamer_live_locked"),
    )

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
    sequence_state = load_sequence_state()
    training_run = load_effective_training_status(clean_stale=True, sequence_state=sequence_state)
    active_run = training_run if str(training_run.get("engine") or "").lower() == "dreamer" else {}
    latest_summary = load_latest_terminal_summary(engine="dreamer")
    pipeline = {
        "active": bool(active_run.get("active")),
        "run_id": active_run.get("run_id"),
        "status": active_run.get("status"),
        "terminal_status": active_run.get("terminal_status") or (latest_summary or {}).get("terminal_status"),
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
        "resume_checkpoint_path": active_run.get("resume_checkpoint_path") or (latest_summary or {}).get("resume_checkpoint_path"),
        "resume_step": active_run.get("resume_step") or (latest_summary or {}).get("resume_step"),
        "last_checkpoint_path": active_run.get("last_checkpoint_path") or (latest_summary or {}).get("last_checkpoint_path"),
        "checkpoint_written_at": active_run.get("checkpoint_written_at") or (latest_summary or {}).get("checkpoint_written_at"),
        "resume_available": bool(active_run.get("resume_available")) or bool((latest_summary or {}).get("resume_available")),
        "resume_epoch": active_run.get("resume_epoch") or (latest_summary or {}).get("resume_epoch"),
        "resume_world_model_steps": active_run.get("resume_world_model_steps") or (latest_summary or {}).get("resume_world_model_steps"),
        "slice_budget_seconds": active_run.get("slice_budget_seconds") or (latest_summary or {}).get("slice_budget_seconds"),
        "slice_elapsed_seconds": active_run.get("slice_elapsed_seconds") or (latest_summary or {}).get("slice_elapsed_seconds"),
        "dataset_id": active_run.get("dataset_id") or (latest_summary or {}).get("dataset_id"),
        "dataset_source": active_run.get("dataset_source") or (latest_summary or {}).get("dataset_source"),
        "dataset_coverage": active_run.get("dataset_coverage", {}) or (latest_summary or {}).get("dataset_coverage", {}),
        "focus_symbols": active_run.get("focus_symbols", []) or list((latest_summary or {}).get("focus_symbols") or []),
        "gate_profile": active_run.get("gate_profile") or (latest_summary or {}).get("gate_profile"),
        "sequence_id": active_run.get("sequence_id"),
        "window_id": active_run.get("window_id"),
        "trial_id": active_run.get("trial_id"),
        "terminal_summary_path": active_run.get("terminal_summary_path") or (latest_summary or {}).get("path"),
        "battle_report_path": active_run.get("battle_report_path") or (latest_summary or {}).get("battle_report_path"),
        "promotion_state": active_run.get("promotion_state") or (latest_summary or {}).get("promotion_state"),
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
    primary_live_lock = dict((engine_horizons.get("scalp") or {}).get("live_lock") or {})
    return {
        **gate.get_status(),
        "pipeline": pipeline,
        "horizons": engine_horizons,
        "live_lock": primary_live_lock,
        "live_lock_reason": primary_live_lock.get("reason"),
        "latest_run_id": latest_run_id,
        "latest_candidate": latest_candidate,
        "latest_verdict": latest_verdict,
        "failed_step": failed_step,
        "artifact_state": artifact_state,
        "focus_symbols": list(pipeline.get("focus_symbols") or []),
        "gate_profile": pipeline.get("gate_profile"),
        "resume_available": pipeline.get("resume_available"),
        "resume_checkpoint_path": pipeline.get("resume_checkpoint_path"),
        "resume_step": pipeline.get("resume_step"),
        "last_checkpoint_path": pipeline.get("last_checkpoint_path"),
        "checkpoint_written_at": pipeline.get("checkpoint_written_at"),
        "resume_epoch": pipeline.get("resume_epoch"),
        "resume_world_model_steps": pipeline.get("resume_world_model_steps"),
        "slice_budget_seconds": pipeline.get("slice_budget_seconds"),
        "slice_elapsed_seconds": pipeline.get("slice_elapsed_seconds"),
        "terminal_status": pipeline.get("terminal_status"),
        "terminal_summary_path": pipeline.get("terminal_summary_path"),
        "battle_report_path": pipeline.get("battle_report_path"),
        "promotion_state": pipeline.get("promotion_state"),
        "terminal_summary": latest_summary,
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
        "restart_count": state.get("restart_count"),
        "retry_reason": state.get("retry_reason"),
        "resumed_from_checkpoint": state.get("resumed_from_checkpoint"),
        "resume_step": state.get("resume_step"),
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
    run_status = load_training_status()
    runtime_profile = _resolve_runtime_profile(run_status)

    horizons = ["scalp", "intraday", "swing"]
    registry_champions = genetic.get_all_champions()
    performance_summary = genetic.get_performance_summary()
    gnn_registry = load_market_gnn_registry()
    gnn_champion = _compute_gnn_champion_payload(gnn_registry)
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
    muzero_scalp_status = dict((engine_status.get("muzero") or {}).get("scalp") or {})
    dreamer_scalp_status = dict((engine_status.get("dreamer") or {}).get("scalp") or {})
    gate_status = gate.get_status()
    scalp_coverage = build_timescaledb_coverage_report(
        CANONICAL_SCALP_FULL_SYMBOLS,
        CANONICAL_COVERAGE_TIMEFRAMES,
    )
    daytime_activation = {
        "muzero_live_ok": bool(muzero_scalp_status.get("live_champion_id")),
        "dreamer_live_ok": bool(
            dreamer_scalp_status.get("live_champion_id")
            and dreamer_scalp_status.get("can_activate_live")
            and dreamer_scalp_status.get("promotion_gate", {}).get("allowed")
            and gate_status.get("dreamer_live_enabled", False)
        ),
        "gnn_consultative_ok": bool(gnn_champion.get("champion_ready")),
        "vllm_ready": runtime_profile == "day_live_full_stack",
    }
    daytime_activation["full_stack_ready"] = bool(
        daytime_activation["muzero_live_ok"]
        and daytime_activation["gnn_consultative_ok"]
        and daytime_activation["vllm_ready"]
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
        "daytime_activation": daytime_activation,
        "gnn_consultative": gnn_champion,
        "timescaledb_coverage": scalp_coverage,
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
    sequence_state = load_sequence_state()
    run_status = load_effective_training_status(clean_stale=True, sequence_state=sequence_state)
    nightly_summary = load_nightly_summary()
    universe_summary = run_status.get("universe") or build_training_universe_summary()
    dependencies = await _collect_training_dependencies(run_status)
    logs = tail_training_log(limit)

    current_step = run_status.get("current_step") or {}
    arena_progress = run_status.get("arena_progress") or None
    observed_step = derive_observed_training_step(logs) if run_status.get("active") else None
    effective_step = (
        select_effective_training_step(current_step, observed_step)
        if run_status.get("active")
        else current_step
    )
    run_view = dict(run_status)
    run_view["current_step"] = effective_step
    run_view["arena_progress"] = arena_progress
    run_view["reported_step"] = current_step or None
    run_view["observed_step"] = observed_step
    run_view["effective_step"] = effective_step
    run_view["step_label"] = format_training_step_label(effective_step)
    run_view["reported_step_label"] = format_training_step_label(current_step)
    run_view["observed_step_label"] = format_training_step_label(observed_step)
    run_view["has_active_run"] = bool(run_view.get("active"))
    run_view["supervisor_state"] = sequence_state.get("state")
    if arena_progress and isinstance(arena_progress, dict):
        challenger_metrics = dict((arena_progress.get("challenger") or {}).get("metrics") or {})
        if challenger_metrics.get("metrics_by_position_mechanics"):
            run_view["metrics_by_position_mechanics"] = challenger_metrics.get("metrics_by_position_mechanics")

    runtime_profile = _resolve_runtime_profile(run_status)
    eta_payload = _estimate_training_eta(run_view)
    service_recovery = _build_lab_service_recovery_snapshot(app, run_status)
    app.state.service_recovery = service_recovery
    set_service_recovery_snapshot(service_recovery)
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
        "last_successful_step": run_view.get("last_successful_step"),
        "last_successful_step_at": run_view.get("last_successful_step_at"),
        "train_step_phase": run_view.get("train_step_phase"),
        "phase_durations_ms": run_view.get("phase_durations_ms"),
        "resume_checkpoint_path": run_view.get("resume_checkpoint_path"),
        "resume_step": run_view.get("resume_step"),
        "last_checkpoint_path": run_view.get("last_checkpoint_path"),
        "checkpoint_written_at": run_view.get("checkpoint_written_at"),
        "resume_available": run_view.get("resume_available"),
        "resume_epoch": run_view.get("resume_epoch"),
        "resume_world_model_steps": run_view.get("resume_world_model_steps"),
        "slice_budget_seconds": run_view.get("slice_budget_seconds"),
        "slice_elapsed_seconds": run_view.get("slice_elapsed_seconds"),
        "terminal_status": run_view.get("terminal_status"),
        "battle_report_path": run_view.get("battle_report_path"),
        "promotion_state": run_view.get("promotion_state"),
        "stall_detected": run_view.get("stall_detected"),
        "stall_reason": run_view.get("stall_reason"),
        "stale_detected": run_view.get("stale_detected"),
        "stale_reasons": run_view.get("stale_reasons"),
        "progress_percent": eta_payload.get("progress_percent"),
        "remaining_steps": eta_payload.get("remaining_steps"),
        "eta_seconds": eta_payload.get("eta_seconds"),
        "eta_at": eta_payload.get("eta_at"),
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
        "training_weighting": run_view.get("training_weighting", {}),
        "service_recovery": service_recovery,
    }
    await _publish_training_run_snapshot(
        run_view=run_view,
        dependencies=dependencies,
        universe=universe_summary,
        nightly_summary=nightly_summary,
    )
    return payload


@app.get("/ga/status")
async def ga_status(limit: int = Query(default=40, ge=1, le=200)):
    """Retourne l'etat exploitable de la campagne GA MuZero.

    Args:
        limit (int): Nombre maximal d'essais a retourner.

    Returns:
        dict: Vue agregée des essais, de la generation et du run courant.
    """

    sequence_state = load_sequence_state()
    run_status = load_effective_training_status(clean_stale=True, sequence_state=sequence_state)
    runtime_profile = _resolve_runtime_profile(run_status)
    current_run = dict(run_status)
    recent_trials = load_recent_ga_trials(limit=limit)
    grouped_trials = _group_ga_trials_by_generation(recent_trials)
    active_generation = current_run.get("ga_generation")
    active_trial = current_run.get("ga_trial")
    eta_payload = _estimate_training_eta(current_run)
    current_phase = ((current_run.get("current_step") or {}).get("phase")) if current_run.get("active") else None
    return {
        "status": "ok",
        "runtime_profile": runtime_profile,
        "engine": str(current_run.get("engine") or "muzero"),
        "ga_status": current_run.get("ga_status"),
        "ga_generation": active_generation,
        "ga_trial": active_trial,
        "trial_mode": current_run.get("trial_mode"),
        "trial_cost_profile": current_run.get("trial_cost_profile"),
        "run_id": current_run.get("run_id"),
        "active": bool(current_run.get("active")),
        "stale_detected": bool(current_run.get("stale_detected")),
        "stale_reasons": list(current_run.get("stale_reasons") or []),
        "current_phase": current_phase,
        "current_trial": {
            "generation": active_generation,
            "trial_id": active_trial,
            "phase": current_phase,
            "early_kill_reason": (
                str((current_run.get("gold_precheck") or {}).get("reason") or "").strip() or None
            ),
            "fitness_score": ((current_run.get("gold_precheck") or {}).get("fitness_score")),
            "promotion_state": (
                str((current_run.get("latest_verdict") or {}).get("status") or "").strip() or None
            ),
            "finalist_rank": None,
            "progress_percent": eta_payload.get("progress_percent"),
            "remaining_steps": eta_payload.get("remaining_steps"),
            "eta_seconds": eta_payload.get("eta_seconds"),
            "eta_at": eta_payload.get("eta_at"),
        },
        "trials_by_generation": grouped_trials,
        "trial_count": len(recent_trials),
        "finalists": [
            trial
            for trial in recent_trials
            if trial.get("finalist_rank") is not None
        ],
        "overview": _build_ga_overview(recent_trials, current_run),
    }


@app.get("/factory/overview")
async def factory_overview():
    """Retourne une vue compacte de pilotage des champions et du run actif.

    Returns:
        dict: Tableau de bord concis du live, des challengers et de l'entrainement.
    """

    promoter: ChampionPromoter = app.state.promoter
    gate: DreamerGate = app.state.dreamer_gate
    genetic = GeneticUpdater()
    registry_champions = genetic.get_all_champions()
    horizons = ["scalp", "intraday", "swing"]
    engine_status = promoter.build_engine_matrix_status(horizons, registry_champions)
    sequence_state = load_sequence_state()
    training_state = load_effective_training_status(clean_stale=True, sequence_state=sequence_state)
    observed_step = derive_observed_training_step(tail_training_log(60)) if training_state.get("active") else None
    run_view = dict(training_state)
    run_view["current_step"] = (
        select_effective_training_step(
            run_view.get("current_step") or {},
            observed_step,
        )
        if training_state.get("active")
        else (run_view.get("current_step") or {})
    )
    eta_payload = _estimate_training_eta(run_view)
    gnn_registry = load_market_gnn_registry()
    gnn_champion = _compute_gnn_champion_payload(gnn_registry)
    recent_trials = load_recent_ga_trials(limit=20)
    muzero_scalp = dict((engine_status.get("muzero") or {}).get("scalp") or {})
    dreamer_scalp = dict((engine_status.get("dreamer") or {}).get("scalp") or {})
    gate_status = gate.get_status()
    timescaledb_coverage = build_timescaledb_coverage_report(
        CANONICAL_SCALP_FULL_SYMBOLS,
        CANONICAL_COVERAGE_TIMEFRAMES,
    )
    service_recovery = _build_lab_service_recovery_snapshot(app, training_state)
    app.state.service_recovery = service_recovery
    set_service_recovery_snapshot(service_recovery)
    blockers: list[str] = []
    if not muzero_scalp.get("live_champion_id"):
        blockers.append("Aucun champion MuZero live exploitable.")
    if run_view.get("active") and run_view.get("focus_symbols") != CANONICAL_SCALP_FULL_SYMBOLS:
        blockers.append("Le run actif n'utilise pas encore l'univers canonique 7 symboles.")
    if not dreamer_scalp.get("candidate_id"):
        blockers.append("DreamerV3 n'a pas encore de champion live valide.")
    elif not dreamer_scalp.get("can_activate_live"):
        blockers.append(
            f"DreamerV3 est bloque pour le live: {dreamer_scalp.get('gate_reason') or 'gate_non_validee'}."
        )
    if not gnn_champion.get("champion_ready"):
        blockers.append("Le champion GNN consultatif n'est pas encore pret.")
    missing_timescaledb_symbols = list(
        ((timescaledb_coverage.get("timescaledb") or {}).get("missing_symbols") or [])
    )
    if missing_timescaledb_symbols:
        blockers.append(
            "TimeDB incomplet sur le full 7: "
            + ", ".join(missing_timescaledb_symbols)
            + "."
        )
    return {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_profile": _resolve_runtime_profile(training_state),
        "live": {
            "selection_policy": promoter.get_live_selection_policy(),
            "active_engine": "muzero" if muzero_scalp.get("live_champion_id") else None,
            "muzero_live_champion_id": muzero_scalp.get("live_champion_id"),
            "dreamer_live_champion_id": (
                dreamer_scalp.get("live_champion_id")
                if dreamer_scalp.get("can_activate_live")
                else None
            ),
            "dreamer_candidate_id": dreamer_scalp.get("candidate_id"),
            "ensemble_ready": bool(gate_status.get("ensemble_ready")),
            "ensemble_active": bool(gate_status.get("ensemble_active")),
            "dreamer_live_enabled": bool(gate_status.get("dreamer_live_enabled")),
            "gnn_consultative_ready": bool(gnn_champion.get("champion_ready")),
        },
        "active_training": {
            "run_id": run_view.get("run_id"),
            "engine": run_view.get("engine"),
            "trigger": run_view.get("trigger"),
            "phase": ((run_view.get("current_step") or {}).get("phase")) if run_view.get("active") else None,
            "status": run_view.get("status"),
            "progress_percent": eta_payload.get("progress_percent"),
            "remaining_steps": eta_payload.get("remaining_steps"),
            "eta_seconds": eta_payload.get("eta_seconds"),
            "eta_at": eta_payload.get("eta_at"),
            "focus_symbols": run_view.get("focus_symbols", []),
            "dataset_source": run_view.get("dataset_source"),
            "gate_profile": run_view.get("gate_profile"),
            "feature_profile": run_view.get("feature_profile"),
            "trial_mode": run_view.get("trial_mode"),
            "ga_status": run_view.get("ga_status"),
            "ga_trial": run_view.get("ga_trial"),
        },
        "champions": {
            "muzero_scalp": _build_engine_overview_card("muzero", "scalp", muzero_scalp),
            "dreamer_scalp": _build_engine_overview_card("dreamer", "scalp", dreamer_scalp),
            "gnn_consultative": {
                "champion_id": gnn_champion.get("champion_id"),
                "champion_ready": gnn_champion.get("champion_ready"),
                "champion_kind": gnn_champion.get("champion_kind"),
                "freshness_hours": gnn_champion.get("freshness_hours"),
                "decision_support_metrics": gnn_champion.get("decision_support_metrics"),
            },
        },
        "ga": _build_ga_overview(recent_trials, run_view),
        "timescaledb_coverage": timescaledb_coverage,
        "training_weighting": run_view.get("training_weighting", {}),
        "service_recovery": service_recovery,
        "blockers": blockers,
    }


@app.get("/timescaledb/coverage")
async def timescaledb_coverage(
    symbols: str | None = Query(default=None),
    timeframes: str | None = Query(default=None),
):
    """Expose la couverture CSV et TimescaleDB pour un univers cible.

    Args:
        symbols (str | None): CSV de symboles a verifier.
        timeframes (str | None): CSV de timeframes a verifier.

    Returns:
        dict: Couverture comparee, trous et priorites de backfill.
    """

    requested_symbols = parse_symbol_csv(symbols) or list(CANONICAL_SCALP_FULL_SYMBOLS)
    requested_timeframes = parse_symbol_csv(timeframes) or list(CANONICAL_COVERAGE_TIMEFRAMES)
    return {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **build_timescaledb_coverage_report(
            requested_symbols,
            requested_timeframes,
        ),
    }


@app.post("/timescaledb/backfill")
async def timescaledb_backfill(request: TimescaleBackfillRequest):
    """Backfill TimescaleDB depuis les CSV historiques deja disponibles.

    Args:
        request (TimescaleBackfillRequest): Perimetre du backfill a executer.

    Returns:
        dict: Resume du backfill et couverture apres injection.
    """

    result = backfill_timescaledb_from_history(
        request.symbols,
        request.timeframes,
        history_dir=request.history_dir,
    )
    if result.get("status") == "ok":
        logger.info(
            "Backfill TimeDB termine: %s lignes pour %s symboles / %s timeframes.",
            result.get("inserted_rows"),
            len(request.symbols),
            len(request.timeframes),
        )
    else:
        logger.warning("Backfill TimeDB en echec: %s", result.get("reason"))
    return result


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
    run_status = load_training_status()
    runtime_profile = _resolve_runtime_profile(run_status)
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
        "runtime_profile": runtime_profile,
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
    refresh_state = _current_gnn_refresh_state()
    graph_snapshot = build_market_gnn_graph_snapshot(registry=registry)
    champion_payload = _compute_gnn_champion_payload(registry)
    return {
        "status": "ok",
        "gnn": registry,
        "focus_symbol": registry.get("focus_symbol"),
        "context_symbols": registry.get("context_symbols", []),
        "deployment_class": registry.get("deployment_class"),
        "champion_id": champion_payload.get("champion_id"),
        "champion_ready": champion_payload.get("champion_ready"),
        "champion_kind": champion_payload.get("champion_kind"),
        "source_run_id": champion_payload.get("source_run_id"),
        "freshness_hours": champion_payload.get("freshness_hours"),
        "decision_support_metrics": champion_payload.get("decision_support_metrics"),
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
    champion_payload = _compute_gnn_champion_payload(registry)
    return {
        "status": "ok",
        "version": registry.get("version"),
        "model_status": registry.get("status"),
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
        "champion_id": champion_payload.get("champion_id"),
        "champion_ready": champion_payload.get("champion_ready"),
        "champion_kind": champion_payload.get("champion_kind"),
        "freshness_hours": champion_payload.get("freshness_hours"),
        "decision_support_metrics": champion_payload.get("decision_support_metrics"),
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




