"""Utilitaires partages pour suivre un run d'entrainement EVA Lab."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

STATUS_DIR = Path(os.getenv("TRAINING_CHECKPOINT_DIR", "data/checkpoints"))
STATUS_PATH = STATUS_DIR / "training_status.json"
RUN_LOG_PATH = STATUS_DIR / "training_run.log"
NIGHTLY_SUMMARY_PATH = STATUS_DIR / "nightly_training_summary.json"
CPU_SCHEDULER_STATE_PATH = STATUS_DIR / "cpu_scheduler" / "state.json"
V4_SEQUENCE_DIR = STATUS_DIR / "v4_ga"
V4_SEQUENCE_STATE_PATH = V4_SEQUENCE_DIR / "sequence_state.json"
V4_SEQUENCE_PID_PATH = V4_SEQUENCE_DIR / "sequence_supervisor.pid"
TERMINAL_SUMMARY_DIR = Path(os.getenv("TRAINING_TERMINAL_SUMMARY_DIR", "data/muzero/results"))
MAX_LOG_LINES = int(os.getenv("TRAINING_STATUS_MAX_LOG_LINES", "400"))
_ATOMIC_WRITE_LOCK = threading.Lock()

FOREX_CODES = {
    "AUD",
    "CAD",
    "CHF",
    "CNH",
    "EUR",
    "GBP",
    "JPY",
    "NZD",
    "USD",
}
CRYPTO_BASES = {
    "AAVE",
    "AAV",
    "ADA",
    "ALGO",
    "AVAX",
    "BNB",
    "BTC",
    "DOGE",
    "DOT",
    "ETH",
    "LINK",
    "LTC",
    "SOL",
    "UNI",
    "XRP",
}
CRYPTO_QUOTES = ("USDT", "USDC", "USD", "EUR", "BTC", "ETH")
METAL_CODES = ("XAU", "XAG", "XPT", "XPD")
INDEX_TOKENS = (
    ".CASH",
    "US30",
    "US100",
    "US500",
    "GER40",
    "UK100",
    "NAS100",
    "SPX500",
    "USTEC",
)
CORE_FAMILIES = ("crypto", "forex", "index_cfd", "metal")
SECONDARY_FAMILIES = ("equity_cfd", "cfd_other", "unknown")
HORIZON_TO_TIMEFRAME = {
    "scalp": "M5",
    "intraday": "H1",
    "swing": "D1",
}
PHASE_ORDER = {
    "initialisation": 0,
    "demarrage": 1,
    "collecte": 2,
    "optimisation": 3,
    "arena": 4,
    "termine": 5,
}


def _now_iso() -> str:
    """Retourne l'horodatage courant au format ISO."""

    return datetime.now().isoformat()


def _default_status() -> dict[str, Any]:
    """Construit la structure de base du statut training."""

    return {
        "engine": None,
        "run_id": None,
        "active": False,
        "status": "idle",
        "trigger": None,
        "strategy": None,
        "reason": None,
        "skip_reason": None,
        "started_at": None,
        "updated_at": None,
        "finished_at": None,
        "current_step": None,
        "completed_steps": [],
        "failed_step": None,
        "launcher": {},
        "dependencies": {},
        "universe": {},
        "family": None,
        "dataset_id": None,
        "dataset_source": None,
        "feature_profile": None,
        "mechanics_profile_version": None,
        "focus_symbols": [],
        "gate_profile": None,
        "sequence_id": None,
        "sequence_profile": None,
        "window_id": None,
        "trial_id": None,
        "terminal_summary_path": None,
        "terminal_status": None,
        "supervisor_state": None,
        "ga_status": None,
        "ga_generation": None,
        "ga_trial": None,
        "trial_mode": None,
        "trial_cost_profile": None,
        "replay_cache_status": None,
        "replay_cache_key": None,
        "replay_cache_entries": None,
        "replay_cache_source": None,
        "shadow_buffer_size": None,
        "sequence_length": None,
        "sequence_stride": None,
        "world_model_steps": None,
        "dataset_coverage": {},
        "metrics_by_position_mechanics": {},
        "arena_progress": None,
        "gold_precheck": None,
        "precheck_status": None,
        "precheck_step": None,
        "precheck_metrics": {},
        "precheck_summary_path": None,
        "last_successful_step": None,
        "last_successful_step_at": None,
        "train_step_phase": None,
        "failed_phase": None,
        "exception_type": None,
        "exception_message": None,
        "traceback_tail": [],
        "phase_durations_ms": {},
        "resume_checkpoint_path": None,
        "resume_step": None,
        "last_checkpoint_path": None,
        "checkpoint_written_at": None,
        "resume_available": False,
        "resume_epoch": None,
        "resume_world_model_steps": None,
        "slice_budget_seconds": None,
        "slice_elapsed_seconds": None,
        "battle_report_path": None,
        "promotion_state": None,
        "stall_detected": False,
        "stall_reason": None,
        "last_nonzero_exit": None,
        "training_weighting": {},
        "service_recovery": {},
        "runtime_truth": {},
        "collector_mode": None,
        "collector_workers": None,
        "collector_active_symbols": [],
        "collector_queue_depth": None,
        "inference_batch_profile": {},
        "jax_batch_profile": {},
        "gpu_owner": None,
    }


def _default_sequence_state() -> dict[str, Any]:
    """Construit l'etat minimal du superviseur de sequence V4."""

    return {
        "sequence_id": None,
        "sequence_name": None,
        "state": "idle",
        "profile": None,
        "engine": None,
        "mode": None,
        "trial_id": None,
        "window_id": None,
        "window_index": None,
        "status": "idle",
        "last_run_id": None,
        "last_completed_trial": None,
        "retry_count": 0,
        "next_step": None,
        "last_error": None,
        "supervisor_heartbeat": None,
        "started_at": None,
        "updated_at": None,
        "stdout_log_path": None,
        "stderr_log_path": None,
        "continued_after_precheck": None,
        "killed_after_precheck": None,
        "precheck_status": None,
        "precheck_score": None,
        "proxy_terminal_score": None,
        "retry_reason": None,
        "restart_count": 0,
        "resumed_from_checkpoint": None,
        "resume_step": None,
    }


def _ensure_status_dir() -> None:
    """Cree le dossier de statut si necessaire."""

    STATUS_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ecrit un JSON de facon atomique sans partager le meme fichier temporaire.

    Args:
        path (Path): Chemin cible a ecrire.
        payload (dict[str, Any]): Charge utile JSON a persister.
    """

    with _ATOMIC_WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, indent=2, ensure_ascii=True)
        prefix = f"{path.name}."
        suffix = ".tmp"
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=prefix,
                suffix=suffix,
                delete=False,
            ) as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = Path(handle.name)
            for attempt in range(8):
                try:
                    os.replace(tmp_path, path)
                    tmp_path = None
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.01 * (attempt + 1))
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


def load_nightly_summary() -> dict[str, Any] | None:
    """Charge le dernier resume nightly si disponible."""

    if not NIGHTLY_SUMMARY_PATH.exists():
        return None
    try:
        payload = json.loads(NIGHTLY_SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_training_status() -> dict[str, Any]:
    """Charge le statut courant d'entrainement."""

    if not STATUS_PATH.exists():
        return _default_status()
    try:
        payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _default_status()
    if not isinstance(payload, dict):
        return _default_status()
    status = _default_status()
    status.update(payload)
    return status


def persist_training_status(status: dict[str, Any]) -> dict[str, Any]:
    """Persiste le statut training apres normalisation minimale."""

    snapshot = _default_status()
    snapshot.update(status)
    snapshot["updated_at"] = _now_iso()
    _atomic_write_json(STATUS_PATH, snapshot)
    return snapshot


def load_sequence_state() -> dict[str, Any]:
    """Charge l'etat persiste du superviseur de sequence V4."""

    if not V4_SEQUENCE_STATE_PATH.exists():
        return _default_sequence_state()
    try:
        payload = json.loads(V4_SEQUENCE_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _default_sequence_state()
    if not isinstance(payload, dict):
        return _default_sequence_state()
    state = _default_sequence_state()
    state.update(payload)
    return state


def persist_sequence_state(state: dict[str, Any]) -> dict[str, Any]:
    """Persiste l'etat courant du superviseur de sequence V4."""

    snapshot = _default_sequence_state()
    snapshot.update(state)
    snapshot["updated_at"] = _now_iso()
    _atomic_write_json(V4_SEQUENCE_STATE_PATH, snapshot)
    return snapshot


_FINAL_TRAINING_STATES = {"completed", "paused", "blocked", "error", "ok", "skipped"}
_FINAL_SEQUENCE_STATES = {"completed", "paused", "idle", "blocked", "error"}


def _normalize_runtime_state_label(value: Any) -> str | None:
    """Normalise un libelle de statut pour les heuristiques runtime.

    Args:
        value (Any): Valeur brute a normaliser.

    Returns:
        str | None: Libelle minuscule normalise ou ``None`` si absent.
    """

    label = str(value or "").strip().lower()
    return label or None


def _pid_is_alive(raw_pid: Any) -> bool | None:
    """Indique si un PID local existe encore.

    Args:
        raw_pid (Any): Valeur brute potentiellement serialisee.

    Returns:
        bool | None: ``True`` si le PID repond, ``False`` s'il est absent,
        ``None`` si la valeur n'est pas exploitable.
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


def normalize_runtime_training_status(
    status: dict[str, Any] | None,
    *,
    sequence_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalise un statut training potentiellement stale pour l'API runtime.

    Cette normalisation ne touche pas automatiquement le fichier sur disque.
    Elle sert a presenter une vue fiable quand un ancien run a laisse
    ``active=true`` alors que le terminal et le superviseur sont deja termines.

    Args:
        status (dict[str, Any] | None): Statut training brut.
        sequence_state (dict[str, Any] | None): Etat du superviseur V4 deja charge.

    Returns:
        dict[str, Any]: Statut normalise, enrichi de marqueurs runtime.
    """

    snapshot = _default_status()
    snapshot.update(status or {})

    merged_sequence = _default_sequence_state()
    merged_sequence.update(sequence_state or load_sequence_state())
    sequence_label = _normalize_runtime_state_label(
        snapshot.get("supervisor_state") or merged_sequence.get("state")
    )
    if sequence_label:
        snapshot["supervisor_state"] = sequence_label

    status_label = _normalize_runtime_state_label(snapshot.get("status"))
    terminal_label = _normalize_runtime_state_label(snapshot.get("terminal_status"))
    current_step = dict(snapshot.get("current_step") or {})
    current_step_label = _normalize_runtime_state_label(current_step.get("status"))
    launcher = dict(snapshot.get("launcher") or {})
    pid_alive = _pid_is_alive(launcher.get("remote_pid"))
    launcher_phase = _normalize_runtime_state_label(launcher.get("phase"))
    trainer_container = str(launcher.get("trainer_container") or "").strip()
    stale_reasons: list[str] = []
    resolved_final = None

    if not bool(snapshot.get("active")):
        runtime_running = pid_alive is True or (
            current_step_label == "running"
            and (launcher_phase in {"running", "trainer_running", "container_running"} or bool(trainer_container))
        )
        if runtime_running:
            snapshot["active"] = True
            snapshot["status"] = "running"
            if current_step and not current_step.get("status"):
                current_step["status"] = "running"
                snapshot["current_step"] = current_step

    if bool(snapshot.get("active")):
        if terminal_label in _FINAL_TRAINING_STATES:
            stale_reasons.append("terminal_status_final")
            resolved_final = terminal_label
        elif status_label in _FINAL_TRAINING_STATES and snapshot.get("finished_at"):
            stale_reasons.append("status_final")
            resolved_final = status_label
        elif sequence_label in _FINAL_SEQUENCE_STATES and snapshot.get("finished_at"):
            stale_reasons.append("supervisor_final")
            resolved_final = terminal_label or status_label or current_step_label or "completed"
        elif pid_alive is False:
            stale_reasons.append("launcher_pid_absent")
            resolved_final = terminal_label or status_label or current_step_label or "completed"

    if stale_reasons:
        resolved_status = "completed" if resolved_final == "ok" else (resolved_final or "completed")
        snapshot["active"] = False
        snapshot["status"] = resolved_status
        if current_step and current_step_label == "running":
            current_step["status"] = resolved_status
            current_step["updated_at"] = _now_iso()
            snapshot["current_step"] = current_step
        if not snapshot.get("terminal_status"):
            snapshot["terminal_status"] = resolved_status

    snapshot["stale_detected"] = bool(stale_reasons)
    snapshot["stale_reasons"] = stale_reasons
    return snapshot


def _strip_runtime_training_markers(status: dict[str, Any]) -> dict[str, Any]:
    """Retire les marqueurs temporaires avant persistence.

    Args:
        status (dict[str, Any]): Statut runtime enrichi.

    Returns:
        dict[str, Any]: Charge utile persistable.
    """

    payload = dict(status)
    payload.pop("stale_detected", None)
    payload.pop("stale_reasons", None)
    return payload


def load_effective_training_status(
    *,
    clean_stale: bool = False,
    sequence_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Charge le statut training avec normalisation runtime optionnelle.

    Args:
        clean_stale (bool): Persiste la correction si un stale est detecte.
        sequence_state (dict[str, Any] | None): Etat du superviseur deja charge.

    Returns:
        dict[str, Any]: Statut normalise exploitable par l'API.
    """

    merged_sequence = _default_sequence_state()
    merged_sequence.update(sequence_state or load_sequence_state())
    normalized = normalize_runtime_training_status(
        load_training_status(),
        sequence_state=merged_sequence,
    )
    if clean_stale and normalized.get("stale_detected"):
        persisted = persist_training_status(_strip_runtime_training_markers(normalized))
        persisted["stale_detected"] = True
        persisted["stale_reasons"] = list(normalized.get("stale_reasons") or [])
        return persisted
    return normalized


def merge_sequence_state(patch: dict[str, Any]) -> dict[str, Any]:
    """Fusionne un patch simple dans l'etat du superviseur V4."""

    state = load_sequence_state()
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(state.get(key), dict):
            merged = dict(state.get(key) or {})
            merged.update(value)
            state[key] = merged
        else:
            state[key] = value
    return persist_sequence_state(state)


def _sanitize_summary_token(value: str | None, default: str) -> str:
    """Nettoie un fragment de nom de fichier de resume terminal."""

    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip())
    return token.strip("._-") or default


def get_terminal_summary_path(
    *,
    engine: str,
    run_id: str,
    horizon: str | None = None,
    family: str | None = None,
    trial_id: str | None = None,
) -> Path:
    """Construit le chemin canonique d'un resume terminal par run.

    Args:
        engine (str): Moteur concerne.
        run_id (str): Identifiant unique du run.
        horizon (str | None): Horizon associe si connu.
        family (str | None): Famille d'actifs associee si connue.
        trial_id (str | None): Identifiant de trial si connu.

    Returns:
        Path: Chemin cible du resume terminal.
    """

    safe_engine = _sanitize_summary_token(engine, "unknown_engine")
    safe_horizon = _sanitize_summary_token(horizon, "unknown_horizon")
    safe_family = _sanitize_summary_token(family, "unknown_family")
    safe_trial = _sanitize_summary_token(trial_id, "final")
    safe_run_id = _sanitize_summary_token(run_id, "unknown_run")
    filename = f"terminal_{safe_engine}_{safe_horizon}_{safe_family}_{safe_trial}_{safe_run_id}.json"
    return TERMINAL_SUMMARY_DIR / filename


def get_precheck_summary_path(
    *,
    engine: str,
    run_id: str,
    horizon: str | None = None,
    family: str | None = None,
    trial_id: str | None = None,
) -> Path:
    """Construit le chemin canonique d'un resume de precheck par run.

    Args:
        engine (str): Moteur concerne.
        run_id (str): Identifiant unique du run.
        horizon (str | None): Horizon associe si connu.
        family (str | None): Famille d'actifs associee si connue.
        trial_id (str | None): Identifiant de trial si connu.

    Returns:
        Path: Chemin cible du resume de precheck.
    """

    safe_engine = _sanitize_summary_token(engine, "unknown_engine")
    safe_horizon = _sanitize_summary_token(horizon, "unknown_horizon")
    safe_family = _sanitize_summary_token(family, "unknown_family")
    safe_trial = _sanitize_summary_token(trial_id, "precheck")
    safe_run_id = _sanitize_summary_token(run_id, "unknown_run")
    filename = f"precheck_{safe_engine}_{safe_horizon}_{safe_family}_{safe_trial}_{safe_run_id}.json"
    return TERMINAL_SUMMARY_DIR / filename


def write_terminal_summary(summary: dict[str, Any]) -> Path:
    """Ecrit un resume terminal de run et met a jour le statut courant.

    Args:
        summary (dict[str, Any]): Resume terminal structure.

    Returns:
        Path: Chemin final du resume ecrit.

    Raises:
        ValueError: Si le resume ne contient pas de ``run_id``.
    """

    run_id = str(summary.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Impossible d'ecrire un resume terminal sans run_id.")

    path = get_terminal_summary_path(
        engine=str(summary.get("engine") or "unknown"),
        run_id=run_id,
        horizon=str(summary.get("horizon") or ""),
        family=str(summary.get("family") or ""),
        trial_id=str(summary.get("trial_id") or summary.get("ga_trial") or ""),
    )
    payload = dict(summary)
    payload["path"] = str(path)
    payload.setdefault("terminal_at", _now_iso())
    _atomic_write_json(path, payload)

    persisted = None
    status = load_training_status()
    if str(status.get("run_id") or "").strip() == run_id:
        status["terminal_summary_path"] = str(path)
        status["terminal_status"] = payload.get("terminal_status")
        if payload.get("resume_checkpoint_path") is not None:
            status["resume_checkpoint_path"] = payload.get("resume_checkpoint_path")
        if payload.get("resume_step") is not None:
            status["resume_step"] = payload.get("resume_step")
        if payload.get("last_checkpoint_path") is not None:
            status["last_checkpoint_path"] = payload.get("last_checkpoint_path")
        if payload.get("checkpoint_written_at") is not None:
            status["checkpoint_written_at"] = payload.get("checkpoint_written_at")
        if payload.get("resume_available") is not None:
            status["resume_available"] = bool(payload.get("resume_available"))
        if payload.get("resume_epoch") is not None:
            status["resume_epoch"] = payload.get("resume_epoch")
        if payload.get("resume_world_model_steps") is not None:
            status["resume_world_model_steps"] = payload.get("resume_world_model_steps")
        if payload.get("slice_budget_seconds") is not None:
            status["slice_budget_seconds"] = payload.get("slice_budget_seconds")
        if payload.get("slice_elapsed_seconds") is not None:
            status["slice_elapsed_seconds"] = payload.get("slice_elapsed_seconds")
        if payload.get("battle_report_path") is not None:
            status["battle_report_path"] = payload.get("battle_report_path")
        if payload.get("promotion_state") is not None:
            status["promotion_state"] = payload.get("promotion_state")
        persisted = persist_training_status(status)
    if persisted is not None and str(persisted.get("run_id") or "").strip() == run_id:
        return path
    return path


def write_precheck_summary(summary: dict[str, Any]) -> Path:
    """Ecrit un resume intermediaire de precheck et met a jour le statut courant.

    Args:
        summary (dict[str, Any]): Resume structure du precheck Gold.

    Returns:
        Path: Chemin final du resume ecrit.

    Raises:
        ValueError: Si le resume ne contient pas de ``run_id``.
    """

    run_id = str(summary.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Impossible d'ecrire un resume de precheck sans run_id.")

    path = get_precheck_summary_path(
        engine=str(summary.get("engine") or "unknown"),
        run_id=run_id,
        horizon=str(summary.get("horizon") or ""),
        family=str(summary.get("family") or ""),
        trial_id=str(summary.get("trial_id") or summary.get("ga_trial") or ""),
    )
    payload = dict(summary)
    payload["path"] = str(path)
    payload.setdefault("generated_at", _now_iso())
    _atomic_write_json(path, payload)

    status = load_training_status()
    if str(status.get("run_id") or "").strip() == run_id:
        status["gold_precheck"] = payload
        status["precheck_status"] = payload.get("status")
        status["precheck_step"] = payload.get("step")
        status["precheck_metrics"] = dict(payload.get("metrics") or {})
        status["precheck_summary_path"] = str(path)
        persisted = persist_training_status(status)
        if str(persisted.get("run_id") or "").strip() == run_id:
            return path
    return path


def load_terminal_summary(
    *,
    path: str | os.PathLike[str] | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    """Charge un resume terminal par chemin explicite ou par ``run_id``.

    Args:
        path (str | os.PathLike[str] | Path | None): Chemin explicite du resume.
        run_id (str | None): Identifiant du run a retrouver si le chemin est absent.

    Returns:
        dict[str, Any] | None: Resume charge ou ``None`` si introuvable.
    """

    candidate_path: Path | None = Path(path) if path else None
    if candidate_path is None and run_id:
        matches = sorted(
            TERMINAL_SUMMARY_DIR.glob(f"terminal_*_{_sanitize_summary_token(run_id, 'unknown_run')}.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        candidate_path = matches[0] if matches else None
    if candidate_path is None or not candidate_path.exists():
        return None
    try:
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_latest_terminal_summary(
    *,
    engine: str | None = None,
    horizon: str | None = None,
) -> dict[str, Any] | None:
    """Charge le resume terminal le plus recent avec filtres optionnels.

    Args:
        engine (str | None): Moteur a filtrer.
        horizon (str | None): Horizon a filtrer.

    Returns:
        dict[str, Any] | None: Resume terminal le plus recent ou ``None``.
    """

    engine_token = _sanitize_summary_token(engine, "*") if engine else "*"
    horizon_token = _sanitize_summary_token(horizon, "*") if horizon else "*"
    pattern = f"terminal_{engine_token}_{horizon_token}_*.json"
    candidates = sorted(
        TERMINAL_SUMMARY_DIR.glob(pattern),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        payload = load_terminal_summary(path=candidate)
        if payload is not None:
            return payload
    return None


def merge_training_status(patch: dict[str, Any]) -> dict[str, Any]:
    """Fusionne un patch simple dans le statut training."""

    status = load_training_status()
    for key, value in patch.items():
        if (
            isinstance(value, dict)
            and isinstance(status.get(key), dict)
        ):
            merged = dict(status.get(key) or {})
            merged.update(value)
            status[key] = merged
        else:
            status[key] = value
    return persist_training_status(status)


def set_arena_progress(progress: dict[str, Any] | None) -> dict[str, Any]:
    """Met a jour la progression detaillee de l'Arena.

    Args:
        progress (dict[str, Any] | None): Charge utile de progression ou ``None``.

    Returns:
        dict[str, Any]: Statut training persiste.
    """
    status = load_training_status()
    status["arena_progress"] = progress
    if progress:
        current_step = dict(status.get("current_step") or {})
        if (str(current_step.get("phase") or "").lower() == "arena") or not current_step:
            current_step["phase"] = "arena"
            if progress.get("horizon"):
                current_step["horizon"] = progress.get("horizon")
                current_step.setdefault("name", f"muzero_{progress.get('horizon')}")
            if progress.get("family"):
                current_step["family"] = progress.get("family")
            if progress.get("current_symbol"):
                current_step["symbol"] = progress.get("current_symbol")
            if progress.get("symbol_index") is not None:
                current_step["symbol_index"] = progress.get("symbol_index")
            if progress.get("symbol_total") is not None:
                current_step["symbol_total"] = progress.get("symbol_total")
            current_step["updated_at"] = _now_iso()
            status["current_step"] = current_step
        if progress.get("family") is not None:
            status["family"] = progress.get("family")
        if progress.get("engine") is not None:
            status["engine"] = progress.get("engine")
        if progress.get("dataset_id") is not None:
            status["dataset_id"] = progress.get("dataset_id")
        if progress.get("dataset_source") is not None:
            status["dataset_source"] = progress.get("dataset_source")
        if progress.get("feature_profile") is not None:
            status["feature_profile"] = progress.get("feature_profile")
        if progress.get("mechanics_profile_version") is not None:
            status["mechanics_profile_version"] = progress.get("mechanics_profile_version")
        if progress.get("ga_status") is not None:
            status["ga_status"] = progress.get("ga_status")
        if progress.get("ga_generation") is not None:
            status["ga_generation"] = progress.get("ga_generation")
        if progress.get("ga_trial") is not None:
            status["ga_trial"] = progress.get("ga_trial")
        if progress.get("trial_mode") is not None:
            status["trial_mode"] = progress.get("trial_mode")
        if progress.get("trial_cost_profile") is not None:
            status["trial_cost_profile"] = progress.get("trial_cost_profile")
        if progress.get("replay_cache_status") is not None:
            status["replay_cache_status"] = progress.get("replay_cache_status")
        if progress.get("replay_cache_key") is not None:
            status["replay_cache_key"] = progress.get("replay_cache_key")
        if progress.get("replay_cache_entries") is not None:
            status["replay_cache_entries"] = progress.get("replay_cache_entries")
        if progress.get("replay_cache_source") is not None:
            status["replay_cache_source"] = progress.get("replay_cache_source")
        if progress.get("shadow_buffer_size") is not None:
            status["shadow_buffer_size"] = progress.get("shadow_buffer_size")
        if progress.get("sequence_length") is not None:
            status["sequence_length"] = progress.get("sequence_length")
        if progress.get("sequence_stride") is not None:
            status["sequence_stride"] = progress.get("sequence_stride")
        if progress.get("world_model_steps") is not None:
            status["world_model_steps"] = progress.get("world_model_steps")
        if progress.get("dataset_coverage") is not None:
            status["dataset_coverage"] = dict(progress.get("dataset_coverage") or {})
        current_role = str(progress.get("current_role") or "").lower()
        metrics_payload: dict[str, Any] = {}
        if current_role in {"challenger", "champion"}:
            role_payload = dict(progress.get(current_role) or {})
            metrics_payload = dict(role_payload.get("metrics") or {})
        elif str(progress.get("status") or "").lower() == "completed":
            metrics_payload = dict((progress.get("challenger") or {}).get("metrics") or {})
        mechanics_payload = dict(metrics_payload.get("metrics_by_position_mechanics") or {})
        if mechanics_payload:
            status["metrics_by_position_mechanics"] = mechanics_payload
    return persist_training_status(status)


def set_gold_precheck(progress: dict[str, Any] | None) -> dict[str, Any]:
    """Met a jour le statut detaille du precheck Gold.

    Args:
        progress (dict[str, Any] | None): Charge utile du precheck ou ``None``.

    Returns:
        dict[str, Any]: Statut training persiste.
    """

    status = load_training_status()
    status["gold_precheck"] = progress
    if progress:
        status["precheck_status"] = progress.get("status")
        status["precheck_step"] = progress.get("step")
        status["precheck_metrics"] = dict(progress.get("metrics") or {})
        if progress.get("path"):
            status["precheck_summary_path"] = progress.get("path")
    else:
        status["precheck_status"] = None
        status["precheck_step"] = None
        status["precheck_metrics"] = {}
        status["precheck_summary_path"] = None
    return persist_training_status(status)


def set_training_runtime_state(**payload: Any) -> dict[str, Any]:
    """Met a jour la telemetrie runtime fine d'un run d'entrainement.

    Args:
        **payload (Any): Champs de telemetrie a fusionner dans le statut.

    Returns:
        dict[str, Any]: Statut training persiste.
    """

    allowed_keys = {
        "last_successful_step",
        "last_successful_step_at",
        "train_step_phase",
        "failed_phase",
        "exception_type",
        "exception_message",
        "traceback_tail",
        "phase_durations_ms",
        "resume_checkpoint_path",
        "resume_step",
        "last_checkpoint_path",
        "checkpoint_written_at",
        "resume_available",
        "resume_epoch",
        "resume_world_model_steps",
        "slice_budget_seconds",
        "slice_elapsed_seconds",
        "terminal_status",
        "battle_report_path",
        "promotion_state",
        "stall_detected",
        "stall_reason",
        "last_nonzero_exit",
        "collector_mode",
        "collector_workers",
        "collector_active_symbols",
        "collector_queue_depth",
        "inference_batch_profile",
        "jax_batch_profile",
        "gpu_owner",
    }
    patch = {
        key: value
        for key, value in payload.items()
        if key in allowed_keys
    }
    if not patch:
        return load_training_status()
    return merge_training_status(patch)


def set_training_weighting(payload: dict[str, Any]) -> dict[str, Any]:
    """Met a jour le resume de ponderation utilise par l'entrainement.

    Args:
        payload (dict[str, Any]): Resume compact du profil de ponderation.

    Returns:
        dict[str, Any]: Statut training persiste.
    """

    return merge_training_status({"training_weighting": dict(payload or {})})


def set_service_recovery_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Met a jour le bloc de reprise service pour les APIs runtime.

    Args:
        payload (dict[str, Any]): Instantane courant de reprise.

    Returns:
        dict[str, Any]: Statut training persiste.
    """

    return merge_training_status({"service_recovery": dict(payload or {})})


def set_runtime_truth_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Met a jour la source de verite runtime du trainer.

    Args:
        payload (dict[str, Any]): Resume compact de l'etat runtime observe.

    Returns:
        dict[str, Any]: Statut training persiste.
    """

    return merge_training_status({"runtime_truth": dict(payload or {})})


def set_training_dependency(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Met a jour une dependance dans le statut training."""

    status = load_training_status()
    dependencies = dict(status.get("dependencies") or {})
    dependency = dict(dependencies.get(name) or {})
    dependency.update(payload)
    dependency["updated_at"] = _now_iso()
    dependencies[name] = dependency
    status["dependencies"] = dependencies
    return persist_training_status(status)


def set_training_launcher_state(**payload: Any) -> dict[str, Any]:
    """Met a jour l'etat du lanceur distant."""

    status = load_training_status()
    launcher = dict(status.get("launcher") or {})
    launcher.update({key: value for key, value in payload.items() if value is not None})
    launcher["updated_at"] = _now_iso()
    status["launcher"] = launcher
    return persist_training_status(status)


def append_training_log(message: str, level: str = "INFO", source: str = "training") -> None:
    """Ajoute une ligne courte dans le journal partage du run."""

    if not message:
        return
    _ensure_status_dir()
    line = f"{_now_iso()} [{level.upper()}] [{source}] {message}".strip()
    lines: list[str] = []
    if RUN_LOG_PATH.exists():
        try:
            lines = RUN_LOG_PATH.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []
    lines.append(line)
    RUN_LOG_PATH.write_text("\n".join(lines[-MAX_LOG_LINES:]) + "\n", encoding="utf-8")


def tail_log_file(
    path: str | os.PathLike[str] | Path,
    limit: int = 30,
    source: str | None = None,
    contains: str | None = None,
) -> list[str]:
    """Retourne les dernieres lignes utiles d'un journal texte.

    Args:
        path (str | os.PathLike[str] | Path): Chemin du journal a lire.
        limit (int): Nombre maximal de lignes a retourner.
        source (str | None): Filtre optionnel sur la balise source `[source]`.
        contains (str | None): Motif libre a rechercher dans les lignes.

    Returns:
        list[str]: Lignes filtrees, tronquees a la fin du journal.
    """
    log_path = Path(path)
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    source_filter = str(source or "").strip().lower()
    contains_filter = str(contains or "").strip().lower()
    if source_filter:
        token = f"[{source_filter}]"
        lines = [line for line in lines if token in line.lower()]
    if contains_filter:
        lines = [line for line in lines if contains_filter in line.lower()]

    safe_limit = max(limit, 1)
    return lines[-safe_limit:]


def tail_training_log(
    limit: int = 30,
    source: str | None = None,
    contains: str | None = None,
) -> list[str]:
    """Retourne les dernieres lignes du journal partage.

    Args:
        limit (int): Nombre maximal de lignes a retourner.
        source (str | None): Filtre optionnel sur la balise source.
        contains (str | None): Motif libre a rechercher.

    Returns:
        list[str]: Lignes du journal partage.
    """
    return tail_log_file(RUN_LOG_PATH, limit=limit, source=source, contains=contains)


def load_cpu_scheduler_state() -> dict[str, Any] | None:
    """Charge l'etat persiste du scheduler CPU si disponible.

    Returns:
        dict[str, Any] | None: Etat du scheduler ou ``None`` si absent.
    """
    if not CPU_SCHEDULER_STATE_PATH.exists():
        return None
    try:
        payload = json.loads(CPU_SCHEDULER_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_log_timestamp(line: str) -> str | None:
    """Extrait l'horodatage ISO d'une ligne de log partagee.

    Args:
        line (str): Ligne brute du journal partage.

    Returns:
        str | None: Horodatage ISO si detecte, sinon ``None``.
    """
    prefix = str(line or "").split(" [", 1)[0].strip()
    if not prefix:
        return None
    try:
        datetime.fromisoformat(prefix)
        return prefix
    except ValueError:
        return None


def derive_observed_training_step(log_lines: list[str]) -> dict[str, Any] | None:
    """Derive l'etape la plus recente a partir du journal partage.

    Args:
        log_lines (list[str]): Dernieres lignes du journal d'entrainement.

    Returns:
        dict[str, Any] | None: Etape observee ou ``None`` si aucun motif reconnu.
    """
    patterns: list[tuple[re.Pattern[str], callable]] = [
        (
            re.compile(
                r"MuZero (?P<horizon>\w+): collecte sur (?P<symbol>[A-Za-z0-9._-]+) "
                r"\((?P<symbol_index>\d+)/(?P<symbol_total>\d+)\)\."
            ),
            lambda match, stamp: {
                "name": f"muzero_{match.group('horizon')}",
                "phase": "collecte",
                "horizon": match.group("horizon"),
                "symbol": match.group("symbol"),
                "symbol_index": int(match.group("symbol_index")),
                "symbol_total": int(match.group("symbol_total")),
                "updated_at": stamp,
                "source": "training_log",
            },
        ),
        (
            re.compile(
                r"MuZero (?P<horizon>\w+): optimisation profonde sur (?P<total>\d+) steps\."
            ),
            lambda match, stamp: {
                "name": f"muzero_{match.group('horizon')}",
                "phase": "optimisation",
                "horizon": match.group("horizon"),
                "training_step_total": int(match.group("total")),
                "updated_at": stamp,
                "source": "training_log",
            },
        ),
        (
            re.compile(
                r"MuZero (?P<horizon>\w+): step (?P<step>\d+)/(?P<total>\d+) \|"
            ),
            lambda match, stamp: {
                "name": f"muzero_{match.group('horizon')}",
                "phase": "optimisation",
                "horizon": match.group("horizon"),
                "training_step_current": int(match.group("step")),
                "training_step_total": int(match.group("total")),
                "updated_at": stamp,
                "source": "training_log",
            },
        ),
        (
            re.compile(r"MuZero (?P<horizon>\w+): lancement de l'arena ADN\."),
            lambda match, stamp: {
                "name": f"muzero_{match.group('horizon')}",
                "phase": "arena",
                "horizon": match.group("horizon"),
                "updated_at": stamp,
                "source": "training_log",
            },
        ),
    ]

    for line in reversed(log_lines):
        timestamp = _extract_log_timestamp(line)
        for pattern, factory in patterns:
            match = pattern.search(line)
            if match:
                return factory(match, timestamp)
    return None


def _step_rank(step: dict[str, Any] | None) -> tuple[int, int]:
    """Retourne un rang simple pour comparer deux etapes.

    Args:
        step (dict[str, Any] | None): Etape a classer.

    Returns:
        tuple[int, int]: Rang de phase puis progression numerique.
    """
    if not step:
        return (-1, -1)
    phase = str(step.get("phase") or "").strip().lower()
    progress = int(step.get("training_step_current") or 0)
    return PHASE_ORDER.get(phase, -1), progress


def select_effective_training_step(
    reported_step: dict[str, Any] | None,
    observed_step: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Choisit l'etape la plus credible entre statut et journal.

    Args:
        reported_step (dict[str, Any] | None): Etape issue du statut structure.
        observed_step (dict[str, Any] | None): Etape derivee du journal.

    Returns:
        dict[str, Any] | None: Etape retenue pour l'affichage.
    """
    if not observed_step:
        return reported_step
    if not reported_step:
        return observed_step
    return observed_step if _step_rank(observed_step) > _step_rank(reported_step) else reported_step


def format_training_step_label(step: dict[str, Any] | None) -> str:
    """Construit un libelle lisible d'etape training.

    Args:
        step (dict[str, Any] | None): Etape a formatter.

    Returns:
        str: Libelle compact et lisible.
    """
    if not step:
        return ""

    parts = [
        str(step.get("name") or "").strip(),
        str(step.get("phase") or "").strip(),
        str(step.get("horizon") or "").strip(),
        str(step.get("symbol") or "").strip(),
    ]
    label = " | ".join(part for part in parts if part)

    symbol_index = step.get("symbol_index")
    symbol_total = step.get("symbol_total")
    if symbol_index is not None and symbol_total is not None:
        label = f"{label} | {symbol_index}/{symbol_total}".strip(" |")

    step_current = step.get("training_step_current")
    step_total = step.get("training_step_total")
    if step_current is not None and step_total is not None:
        label = f"{label} | {step_current}/{step_total}".strip(" |")

    return label


def reset_training_status(
    *,
    engine: str | None = None,
    run_id: str,
    trigger: str,
    strategy: str,
    reason: str,
    universe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reinitialise le statut d'un nouveau run en preservant le lanceur."""

    previous = load_training_status()
    focus_symbols = [
        item.strip()
        for item in str(os.getenv("TRAINING_FOCUS_SYMBOLS", "")).split(",")
        if item.strip()
    ]
    status = _default_status()
    status["engine"] = engine
    status["run_id"] = run_id
    status["active"] = True
    status["status"] = "running"
    status["trigger"] = trigger
    status["strategy"] = strategy
    status["reason"] = reason
    status["skip_reason"] = None
    status["started_at"] = _now_iso()
    status["sequence_id"] = str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None
    status["sequence_profile"] = str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None
    status["window_id"] = str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None
    status["trial_id"] = (
        str(os.getenv("TRAINING_TRIAL_ID", "")).strip()
        or str(os.getenv("TRAINING_GA_TRIAL", "")).strip()
        or None
    )
    status["terminal_summary_path"] = None
    status["focus_symbols"] = focus_symbols
    status["gate_profile"] = str(os.getenv("TRAINING_GATE_PROFILE", "")).strip() or None
    status["supervisor_state"] = str(os.getenv("TRAINING_SUPERVISOR_STATE", "")).strip() or None
    status["launcher"] = dict(previous.get("launcher") or {})
    status["dependencies"] = dict(previous.get("dependencies") or {})
    status["universe"] = universe or build_training_universe_summary()
    status["arena_progress"] = None
    persisted = persist_training_status(status)
    append_training_log(
        f"Run {run_id} demarre | strategie={strategy} | trigger={trigger} | raison={reason}",
        source="nightly",
    )
    return persisted


def mark_step_running(
    step_name: str,
    *,
    engine: str | None = None,
    phase: str | None = None,
    horizon: str | None = None,
    family: str | None = None,
    symbol: str | None = None,
    symbol_index: int | None = None,
    symbol_total: int | None = None,
    part_index: int | None = None,
    part_total: int | None = None,
    epoch_current: int | None = None,
    epoch_total: int | None = None,
    training_step_current: int | None = None,
    training_step_total: int | None = None,
    dataset_id: str | None = None,
    dataset_source: str | None = None,
    feature_profile: str | None = None,
    mechanics_profile_version: str | None = None,
    focus_symbols: list[str] | None = None,
    gate_profile: str | None = None,
    ga_status: str | None = None,
    ga_generation: int | None = None,
    ga_trial: str | None = None,
    trial_mode: str | None = None,
    trial_cost_profile: str | None = None,
    replay_cache_status: str | None = None,
    replay_cache_key: str | None = None,
    replay_cache_entries: int | None = None,
    replay_cache_source: str | None = None,
    shadow_buffer_size: int | None = None,
    sequence_length: int | None = None,
    sequence_stride: int | None = None,
    world_model_steps: int | None = None,
    dataset_coverage: dict[str, Any] | None = None,
    metrics_by_position_mechanics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Met a jour l'etape courante d'un run."""

    step = {
        "name": step_name,
        "status": "running",
        "phase": phase,
        "horizon": horizon,
        "family": family,
        "symbol": symbol,
        "symbol_index": symbol_index,
        "symbol_total": symbol_total,
        "part_index": part_index,
        "part_total": part_total,
        "epoch_current": epoch_current,
        "epoch_total": epoch_total,
        "training_step_current": training_step_current,
        "training_step_total": training_step_total,
        "updated_at": _now_iso(),
    }
    status = load_training_status()
    if engine is not None:
        status["engine"] = engine
    status["active"] = True
    status["status"] = "running"
    status["current_step"] = {key: value for key, value in step.items() if value is not None}
    if family is not None:
        status["family"] = family
    if dataset_id is not None:
        status["dataset_id"] = dataset_id
    if dataset_source is not None:
        status["dataset_source"] = dataset_source
    if feature_profile is not None:
        status["feature_profile"] = feature_profile
    if mechanics_profile_version is not None:
        status["mechanics_profile_version"] = mechanics_profile_version
    if focus_symbols is not None:
        status["focus_symbols"] = [str(item).strip() for item in focus_symbols if str(item).strip()]
    elif status.get("focus_symbols") in (None, []):
        status["focus_symbols"] = [
            item.strip()
            for item in str(os.getenv("TRAINING_FOCUS_SYMBOLS", "")).split(",")
            if item.strip()
        ]
    if gate_profile is not None:
        status["gate_profile"] = gate_profile
    elif not status.get("gate_profile"):
        status["gate_profile"] = str(os.getenv("TRAINING_GATE_PROFILE", "")).strip() or None
    if ga_status is not None:
        status["ga_status"] = ga_status
    if ga_generation is not None:
        status["ga_generation"] = ga_generation
    if ga_trial is not None:
        status["ga_trial"] = ga_trial
    if trial_mode is not None:
        status["trial_mode"] = trial_mode
    if trial_cost_profile is not None:
        status["trial_cost_profile"] = trial_cost_profile
    if replay_cache_status is not None:
        status["replay_cache_status"] = replay_cache_status
    if replay_cache_key is not None:
        status["replay_cache_key"] = replay_cache_key
    if replay_cache_entries is not None:
        status["replay_cache_entries"] = replay_cache_entries
    if replay_cache_source is not None:
        status["replay_cache_source"] = replay_cache_source
    if shadow_buffer_size is not None:
        status["shadow_buffer_size"] = shadow_buffer_size
    if sequence_length is not None:
        status["sequence_length"] = sequence_length
    if sequence_stride is not None:
        status["sequence_stride"] = sequence_stride
    if world_model_steps is not None:
        status["world_model_steps"] = world_model_steps
    if dataset_coverage is not None:
        status["dataset_coverage"] = dict(dataset_coverage)
    if metrics_by_position_mechanics is not None:
        status["metrics_by_position_mechanics"] = dict(metrics_by_position_mechanics)
    status["failed_step"] = None
    if (phase or "").lower() != "arena":
        status["arena_progress"] = None
    return persist_training_status(status)


def mark_step_finished(step_name: str, status_value: str, error: str | None = None) -> dict[str, Any]:
    """Marque une etape comme terminee."""

    status = load_training_status()
    completed = list(status.get("completed_steps") or [])
    if status_value == "ok" and step_name not in completed:
        completed.append(step_name)
    status["completed_steps"] = completed
    status["current_step"] = {
        "name": step_name,
        "status": status_value,
        "updated_at": _now_iso(),
    }
    if status_value == "error":
        status["status"] = "error"
        status["failed_step"] = {
            "name": step_name,
            "error": error,
            "updated_at": _now_iso(),
        }
        append_training_log(
            f"Etape {step_name} en erreur: {error or 'inconnue'}",
            level="ERROR",
            source="nightly",
        )
    else:
        append_training_log(
            f"Etape {step_name} terminee avec statut {status_value}.",
            source="nightly",
        )
    return persist_training_status(status)


def finalize_training_status(
    final_status: str,
    *,
    reason: str | None = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    """Finalise le statut d'un run."""

    status = load_training_status()
    status["active"] = False
    status["status"] = final_status
    status["finished_at"] = _now_iso()
    if final_status in {"paused", "completed", "blocked", "error"} or not status.get("terminal_status"):
        status["terminal_status"] = final_status
    if reason is not None:
        status["reason"] = reason
    if skip_reason is not None:
        status["skip_reason"] = skip_reason
    persisted = persist_training_status(status)
    append_training_log(
        f"Run termine avec statut {final_status}" + (f" ({skip_reason})" if skip_reason else ""),
        source="nightly",
    )
    return persisted


def mark_skip_status(reason: str, trigger: str, lock_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Met a jour le statut pour un skip propre du cron.

    Si un run est deja en cours, la fonction preserve la vue du run actif et
    n'ecrase pas son `trigger`, son statut ni son indicateur `active`.

    Args:
        reason (str): Motif du skip.
        trigger (str): Trigger du run qui a ete ignore.
        lock_payload (dict[str, Any] | None): Charge utile du verrou si elle
            est disponible.

    Returns:
        dict[str, Any]: Statut training persiste.
    """

    status = load_training_status()
    launcher = dict(status.get("launcher") or {})
    if lock_payload:
        launcher["skip_lock"] = lock_payload
    launcher["last_skip_reason"] = reason
    launcher["updated_at"] = _now_iso()
    status["launcher"] = launcher
    active_run_present = bool(status.get("active")) and str(status.get("status") or "").lower() == "running"
    if not active_run_present:
        status["active"] = False
        status["status"] = "skipped"
        status["trigger"] = trigger
        status["skip_reason"] = reason
        status["finished_at"] = _now_iso()
        status["arena_progress"] = None
    persisted = persist_training_status(status)
    append_training_log(f"Run ignore: {reason}", level="WARNING", source="launcher")
    return persisted


def _parse_history_filename(path: Path) -> tuple[str, str] | None:
    """Extrait le symbole et le timeframe depuis un nom de CSV."""

    stem = path.stem
    if "_" not in stem:
        return None
    symbol, timeframe = stem.rsplit("_", 1)
    if not symbol or not timeframe:
        return None
    return symbol, timeframe.upper()


def _looks_like_crypto_symbol(symbol: str) -> bool:
    """Retourne vrai si le symbole ressemble a une paire crypto."""

    for quote in CRYPTO_QUOTES:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: -len(quote)]
            if base in CRYPTO_BASES and base not in FOREX_CODES:
                return True
    return False


def _looks_like_forex_symbol(symbol: str) -> bool:
    """Retourne vrai si le symbole ressemble a une paire forex."""

    if len(symbol) < 6:
        return False
    base = symbol[:3]
    quote = symbol[3:6]
    if base in METAL_CODES:
        return False
    return base in FOREX_CODES and quote in FOREX_CODES


def classify_training_symbol(symbol: str) -> str:
    """Classe un symbole dans une famille de marche."""

    symbol_upper = symbol.upper()
    alnum_symbol = "".join(char for char in symbol_upper if char.isalnum())

    if any(alnum_symbol.startswith(code) for code in METAL_CODES):
        return "metal"
    if _looks_like_crypto_symbol(alnum_symbol):
        return "crypto"
    if _looks_like_forex_symbol(alnum_symbol):
        return "forex"
    if any(token in symbol_upper for token in INDEX_TOKENS):
        return "index_cfd"
    if symbol_upper.endswith(".CASH"):
        return "equity_cfd"
    if any(char.isdigit() for char in symbol_upper):
        return "cfd_other"
    if 1 <= len(alnum_symbol) <= 6 and alnum_symbol.isalpha():
        return "equity_cfd"
    return "unknown"


def _can_build_timeframe(available: set[str], timeframe: str) -> bool:
    """Retourne vrai si un timeframe est disponible ou reconstructible."""

    if timeframe in available:
        return True
    if timeframe == "M5":
        return "M1" in available
    if timeframe == "M15":
        return "M5" in available or "M1" in available
    if timeframe == "H1":
        return "M15" in available or "M5" in available or "M1" in available
    if timeframe == "D1":
        return "H1" in available or "M15" in available or "M5" in available or "M1" in available
    if timeframe == "W1":
        return "D1" in available or "H1" in available or "M15" in available or "M5" in available or "M1" in available
    return False


def discover_history_inventory(data_dir: str | os.PathLike[str] | None = None) -> dict[str, set[str]]:
    """Construit l'inventaire brut des historiques disponibles."""

    history_dir = Path(data_dir or os.getenv("TRAINING_DATA_DIR", Path("data") / "history"))
    inventory: dict[str, set[str]] = {}
    if not history_dir.exists():
        return inventory

    for file_path in sorted(history_dir.glob("*.csv")):
        parsed = _parse_history_filename(file_path)
        if parsed is None:
            continue
        symbol, timeframe = parsed
        inventory.setdefault(symbol, set()).add(timeframe)
    return inventory


def build_training_universe_summary(data_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Construit un resume lisible de la diversite d'univers."""

    inventory = discover_history_inventory(data_dir)
    family_counts = {family: 0 for family in (*CORE_FAMILIES, *SECONDARY_FAMILIES)}
    timeframe_counts: dict[str, int] = {}
    family_samples: dict[str, list[str]] = {family: [] for family in (*CORE_FAMILIES, *SECONDARY_FAMILIES)}

    for symbol, timeframes in sorted(inventory.items()):
        family = classify_training_symbol(symbol)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(family_samples.setdefault(family, [])) < 4:
            family_samples[family].append(symbol)
        for timeframe in sorted(timeframes):
            timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1

    sample_symbols: list[str] = []
    for family in (*CORE_FAMILIES, *SECONDARY_FAMILIES):
        for symbol in family_samples.get(family, []):
            if symbol not in sample_symbols:
                sample_symbols.append(symbol)
            if len(sample_symbols) >= 12:
                break
        if len(sample_symbols) >= 12:
            break

    horizon_universe = {}
    for horizon, timeframe in HORIZON_TO_TIMEFRAME.items():
        eligible = sorted(
            symbol
            for symbol, available in inventory.items()
            if _can_build_timeframe(available, timeframe)
        )
        horizon_universe[horizon] = {
            "timeframe": timeframe,
            "count": len(eligible),
            "sample_symbols": eligible[:8],
        }

    return {
        "history_dir": str(Path(data_dir or os.getenv("TRAINING_DATA_DIR", Path("data") / "history"))),
        "total_symbols": len(inventory),
        "family_counts": family_counts,
        "timeframe_counts": dict(sorted(timeframe_counts.items())),
        "family_samples": family_samples,
        "sample_symbols": sample_symbols,
        "horizon_universe": horizon_universe,
    }
