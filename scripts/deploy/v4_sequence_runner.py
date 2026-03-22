"""Superviseur distant de la sequence V4, durable et reprenable.

Ce script s'execute directement sur le serveur Proxmox. Il orchestre les
fenetres MuZero et Dreamer sans dependre du poste local, persiste son etat,
score chaque trial a partir de son resume terminal et continue la sequence tant
que l'echec reste metier et scorables.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src" / "eva-lab", ROOT / "src" / "shared"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from eva_lab.timescale_store import record_ga_trial, record_run_window
from eva_lab.training_status import (
    V4_SEQUENCE_PID_PATH,
    append_training_log,
    load_sequence_state,
    load_terminal_summary,
    load_training_status,
    persist_sequence_state,
)


REMOTE_SCRIPT = ROOT / "scripts" / "run_nightly_training_remote.sh"
RESULTS_DIR = ROOT / "data" / "checkpoints" / "v4_ga"
LAB_INTERNAL_URL = str(os.getenv("TRAINING_LAB_INTERNAL_URL") or "http://127.0.0.1:8600").rstrip("/")


def _now_iso() -> str:
    """Retourne l'horodatage courant au format ISO."""

    return datetime.now().isoformat()


def _load_dotenv(path: Path) -> None:
    """Charge un fichier `.env` simple dans l'environnement du processus."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        normalized = value.strip().strip('"').strip("'")
        os.environ[key] = normalized


def _bootstrap_timescale_env() -> None:
    """Normalise l'environnement TimeDB pour le superviseur distant.

    Le superviseur tourne hors conteneur. Il ne peut donc pas compter sur les
    variables injectees au seul conteneur trainer. Cette routine reconstruit un
    environnement `TRAINING_TIMESCALE_*` exploitable a partir des valeurs
    partagees ou de repli raisonnables.
    """

    defaults = {
        "TRAINING_TIMESCALE_ENABLED": "1",
        "TRAINING_TIMESCALE_HOST": os.getenv("TRAINING_TIMESCALE_HOST", "127.0.0.1"),
        "TRAINING_TIMESCALE_PORT": os.getenv("TIMESCALE_PORT", "5432"),
        "TRAINING_TIMESCALE_DB": os.getenv("TIMESCALE_DB", "thehive"),
        "TRAINING_TIMESCALE_USER": os.getenv("TIMESCALE_USER", "eva"),
        "TRAINING_TIMESCALE_PASSWORD": os.getenv("TIMESCALE_PASSWORD", "devpassword"),
        "TRAINING_TIMESCALE_SSLMODE": "prefer",
        "TRAINING_TIMESCALE_BARS_TABLE": "market.market_bars",
        "TRAINING_TIMESCALE_FEATURES_TABLE": "market.market_features",
        "TRAINING_TIMESCALE_DATASETS_TABLE": "training.training_datasets",
        "TRAINING_TIMESCALE_ARENA_TABLE": "training.arena_results",
        "TRAINING_TIMESCALE_GA_TABLE": "training.ga_trials",
        "TRAINING_TIMESCALE_REPLAY_TABLE": "training.replay_metadata",
        "TRAINING_TIMESCALE_RUN_WINDOWS_TABLE": "training.run_windows",
    }
    for key, value in defaults.items():
        if not str(os.getenv(key) or "").strip():
            os.environ[key] = str(value)


def _load_json(path: Path) -> dict[str, Any]:
    """Charge un JSON dictionnaire ou retourne une structure vide."""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ecrit un JSON de facon atomique."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _post_internal_payload(path: str, payload: dict[str, Any]) -> bool:
    """Tente une persistance via l'API interne de Lab en repli du runtime hote.

    Args:
        path (str): Chemin relatif de l'endpoint interne.
        payload (dict[str, Any]): Charge utile JSON a transmettre.

    Returns:
        bool: ``True`` si Lab confirme la persistance, sinon ``False``.
    """

    url = f"{LAB_INTERNAL_URL}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False
    return bool((body or {}).get("persisted"))


def _results_snapshot_path(profile: str, engine: str, mode: str) -> Path:
    """Retourne le fichier de snapshot correspondant a une fenetre V4."""

    if mode == "proxy_ga":
        suffix = "proxy_results"
    elif mode == "full":
        suffix = "full_results"
    elif mode == "smoke":
        suffix = "smoke_results"
    else:
        suffix = f"{mode}_results"
    return RESULTS_DIR / f"v4_{engine}_{profile}_{suffix}.json"


def _finalists_path(profile: str, engine: str) -> Path:
    """Retourne le fichier des finalistes d'un moteur/profil."""

    return RESULTS_DIR / f"v4_{engine}_{profile}_finalists.json"


def _load_result_entries(path: Path) -> list[dict[str, Any]]:
    """Charge une liste de resultats V4 si elle existe."""

    payload = _load_json(path)
    entries = payload.get("results")
    if not isinstance(entries, list):
        return []
    return [dict(item) for item in entries if isinstance(item, dict)]


def _write_results_snapshot(
    path: Path,
    *,
    sequence_id: str,
    profile: str,
    engine: str,
    mode: str,
    entries: list[dict[str, Any]],
) -> None:
    """Persiste un snapshot de scoring V4 pour une fenetre."""

    _write_json(
        path,
        {
            "sequence_id": sequence_id,
            "profile": profile,
            "engine": engine,
            "mode": mode,
            "generated_at": _now_iso(),
            "results": entries,
        },
    )


def _score_metric_payload(
    metrics: dict[str, Any],
    mechanics: dict[str, Any],
    *,
    failure_mode: str,
) -> float:
    """Calcule un score robuste a partir d'un couple metriques/mecaniques."""

    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    return_pct = float(metrics.get("return_pct", 0.0) or 0.0)
    net_realized_pct = float(metrics.get("net_realized_pct", 0.0) or 0.0)
    positive_episode_rate = float(metrics.get("positive_episode_rate", 0.0) or 0.0)
    long_entry_share = float(metrics.get("long_entry_share", 0.0) or 0.0)
    short_entry_share = float(metrics.get("short_entry_share", 0.0) or 0.0)
    directional_imbalance = float(metrics.get("directional_imbalance", 1.0) or 1.0)
    close_quality_score = float(mechanics.get("close_quality_score", 0.0) or 0.0)
    split_efficiency = float(mechanics.get("split_efficiency", 0.0) or 0.0)
    pyramid_efficiency = float(mechanics.get("pyramid_efficiency", 0.0) or 0.0)
    slbe_capture_rate = float(mechanics.get("slbe_capture_rate", 0.0) or 0.0)
    hold_drag_score = float(mechanics.get("hold_drag_score", 0.0) or 0.0)
    total_trades = int(metrics.get("total_trades", 0) or 0)

    score = (
        return_pct * 180.0
        + net_realized_pct * 140.0
        + max(0.0, profit_factor - 1.0) * 55.0
        + positive_episode_rate * 0.35
        + min(long_entry_share, short_entry_share) * 45.0
        + close_quality_score * 30.0
        + split_efficiency * 18.0
        + pyramid_efficiency * 16.0
        + slbe_capture_rate * 14.0
        - hold_drag_score * 12.0
        - directional_imbalance * 24.0
    )
    if total_trades <= 0:
        score -= 80.0
    if failure_mode in {"inactive", "sell_heavy", "buy_heavy", "bad_exit", "insufficient_sample"}:
        score -= 25.0

    return round(score, 4)


def _score_terminal_summary(summary: dict[str, Any]) -> tuple[float, str, dict[str, Any], dict[str, Any]]:
    """Calcule le fitness terminal d'un trial a partir du resume courant."""

    metrics = dict(summary.get("metrics") or {})
    mechanics = dict(summary.get("metrics_by_position_mechanics") or {})
    failure_mode = str(summary.get("failure_mode") or "unknown")
    return (
        _score_metric_payload(metrics, mechanics, failure_mode=failure_mode),
        failure_mode,
        metrics,
        mechanics,
    )


def _score_precheck_summary(summary: dict[str, Any]) -> tuple[float | None, str | None]:
    """Calcule le score du precheck Gold s'il existe."""

    precheck = dict(summary.get("gold_precheck") or {})
    if not precheck:
        return None, None
    metrics = dict(precheck.get("metrics") or {})
    mechanics = dict(precheck.get("metrics_by_position_mechanics") or {})
    failure_mode = str(precheck.get("failure_mode") or summary.get("failure_mode") or "unknown")
    if not metrics and not mechanics:
        return None, str(precheck.get("status") or "").strip() or None
    return (
        _score_metric_payload(metrics, mechanics, failure_mode=failure_mode),
        str(precheck.get("status") or "").strip() or None,
    )


def _persist_window_state(state: dict[str, Any]) -> dict[str, Any]:
    """Persiste l'etat global et l'historique de la fenetre courante."""

    persisted = persist_sequence_state(state)
    if persisted.get("window_id"):
        window_payload = {
            "window_id": persisted.get("window_id"),
            "sequence_id": persisted.get("sequence_id"),
            "profile": persisted.get("profile"),
            "engine": persisted.get("engine"),
            "mode": persisted.get("mode"),
            "trial_id": persisted.get("trial_id"),
            "window_index": persisted.get("window_index"),
            "status": persisted.get("status") or persisted.get("state"),
            "last_run_id": persisted.get("last_run_id"),
            "retry_count": persisted.get("retry_count", 0),
            "payload": persisted,
            "started_at": persisted.get("started_at"),
            "finished_at": persisted.get("finished_at"),
        }
        if not record_run_window(window_payload):
            _post_internal_payload("/internal/sequence/window", window_payload)
    return persisted


def _build_window_id(sequence_id: str, profile: str, engine: str, mode: str, trial_id: str) -> str:
    """Construit un identifiant stable de fenetre V4."""

    return f"{sequence_id}:{profile}:{engine}:{mode}:{trial_id}"


def _build_trial_env(
    base_overrides: dict[str, Any],
    *,
    sequence_id: str,
    sequence_profile: str,
    window_id: str,
    trial_id: str,
) -> dict[str, str]:
    """Construit l'environnement complet d'un trial supervise."""

    overrides = {str(key): str(value) for key, value in dict(base_overrides or {}).items()}
    overrides["TRAINING_SEQUENCE_ID"] = sequence_id
    overrides["TRAINING_SEQUENCE_PROFILE"] = sequence_profile
    overrides["TRAINING_WINDOW_ID"] = window_id
    overrides["TRAINING_TRIAL_ID"] = trial_id
    overrides["TRAINING_SUPERVISOR_STATE"] = "running"
    return overrides


def _read_run_snapshot() -> dict[str, Any]:
    """Retourne une vue compacte du run courant selon `training_status.json`."""

    payload = load_training_status()
    return {
        "run_id": str(payload.get("run_id") or "").strip() or None,
        "active": bool(payload.get("active")),
        "status": str(payload.get("status") or "").strip() or None,
        "engine": str(payload.get("engine") or "").strip() or None,
        "sequence_id": str(payload.get("sequence_id") or "").strip() or None,
        "window_id": str(payload.get("window_id") or "").strip() or None,
        "trial_id": str(payload.get("trial_id") or "").strip() or None,
        "terminal_summary_path": str(payload.get("terminal_summary_path") or "").strip() or None,
        "failed_step": dict(payload.get("failed_step") or {}),
    }


def _wait_existing_run(window_id: str, state: dict[str, Any], poll_seconds: int = 15) -> dict[str, Any]:
    """Attend la fin d'un run deja actif pour la fenetre courante."""

    while True:
        run_snapshot = _read_run_snapshot()
        state["supervisor_heartbeat"] = _now_iso()
        state["last_run_id"] = run_snapshot.get("run_id")
        _persist_window_state(state)
        if not run_snapshot.get("active") or run_snapshot.get("window_id") != window_id:
            return run_snapshot
        time.sleep(max(5, poll_seconds))


def _run_trial(overrides: dict[str, str], state: dict[str, Any]) -> dict[str, Any]:
    """Execute ou rattache un trial V4 puis retourne son etat terminal."""

    window_id = str(state.get("window_id") or "")
    run_snapshot = _read_run_snapshot()
    if run_snapshot.get("active") and run_snapshot.get("window_id") == window_id:
        print(f"[sequence] Reprise du run actif {run_snapshot.get('run_id')} pour {window_id}.", flush=True)
        final_snapshot = _wait_existing_run(window_id, state)
    elif run_snapshot.get("active"):
        raise RuntimeError(
            f"Un autre run est deja actif ({run_snapshot.get('run_id')}) pour {run_snapshot.get('window_id')}."
        )
    else:
        env = os.environ.copy()
        env.update(overrides)
        print(
            "[sequence] Lancement %s | moteur=%s | mode=%s | trial=%s"
            % (
                state.get("profile"),
                state.get("engine"),
                state.get("mode"),
                state.get("trial_id"),
            ),
            flush=True,
        )
        process = subprocess.Popen(
            ["bash", str(REMOTE_SCRIPT)],
            cwd=ROOT,
            env=env,
        )
        while process.poll() is None:
            run_snapshot = _read_run_snapshot()
            state["supervisor_heartbeat"] = _now_iso()
            if run_snapshot.get("window_id") == window_id and run_snapshot.get("run_id"):
                state["last_run_id"] = run_snapshot.get("run_id")
            _persist_window_state(state)
            time.sleep(15)
        final_snapshot = _read_run_snapshot()
        final_snapshot["returncode"] = process.returncode

    run_id = final_snapshot.get("run_id") or state.get("last_run_id")
    summary = None
    summary_path = str(final_snapshot.get("terminal_summary_path") or "").strip() or None
    if summary_path:
        summary = load_terminal_summary(path=summary_path)
    if summary is None and run_id:
        summary = load_terminal_summary(run_id=str(run_id))

    return {
        "run_id": run_id,
        "status": str(final_snapshot.get("status") or "unknown"),
        "failed_step": dict(final_snapshot.get("failed_step") or {}),
        "returncode": final_snapshot.get("returncode"),
        "terminal_summary": summary,
        "terminal_summary_path": summary_path or (summary or {}).get("path"),
    }


def _handle_terminal_without_summary(
    outcome: dict[str, Any],
    state: dict[str, Any],
    *,
    retry_limit: int,
) -> bool:
    """Determine s'il faut retenter un trial sans resume terminal exploitable."""

    terminal_status = str(outcome.get("status") or "unknown")
    retry_count = int(state.get("retry_count") or 0)
    if terminal_status == "aborted" and retry_count < retry_limit:
        state["retry_count"] = retry_count + 1
        state["status"] = "retrying"
        state["last_error"] = "resume_terminal_absent_apres_aborted"
        _persist_window_state(state)
        append_training_log(
            f"Sequence V4: retry du trial {state.get('trial_id')} apres aborted sans resume terminal.",
            level="WARNING",
            source="sequence",
        )
        return True
    state["status"] = "paused"
    state["state"] = "paused"
    state["last_error"] = (
        f"infra_error:{terminal_status}:resume_terminal_absent"
    )
    _persist_window_state(state)
    return False


def _record_scored_trial(
    *,
    sequence_id: str,
    profile: str,
    engine: str,
    mode: str,
    trial: dict[str, Any],
    generation: int,
    run_id: str | None,
    terminal_summary: dict[str, Any],
    results_path: Path,
) -> dict[str, Any]:
    """Score et persiste un trial termine a partir de son resume terminal."""

    existing = _load_result_entries(results_path)
    score, failure_mode, metrics, mechanics = _score_terminal_summary(terminal_summary)
    precheck_score, precheck_status = _score_precheck_summary(terminal_summary)
    killed_after_precheck = str((terminal_summary.get("latest_verdict") or {}).get("status") or "").strip() == "killed_after_precheck"
    continued_after_precheck = bool(precheck_status) and not killed_after_precheck
    raw_trial_id = str(trial.get("trial_id") or "").strip()
    record_id = ":".join(
        part
        for part in [sequence_id, engine, profile, mode, raw_trial_id, str(run_id or "unknown")]
        if part
    )
    terminal_status = str(terminal_summary.get("terminal_status") or "unknown")
    result_entry = {
        "sequence_id": sequence_id,
        "engine": engine,
        "profile": profile,
        "mode": mode,
        "trial_id": raw_trial_id,
        "generation": generation,
        "run_id": run_id,
        "score": score,
        "proxy_terminal_score": score,
        "precheck_score": precheck_score,
        "precheck_status": precheck_status,
        "continued_after_precheck": continued_after_precheck,
        "killed_after_precheck": killed_after_precheck,
        "failure_mode": failure_mode,
        "terminal_status": terminal_status,
        "summary_path": terminal_summary.get("path"),
        "precheck_summary_path": ((terminal_summary.get("gold_precheck") or {}).get("path")),
        "focus_symbols": list(terminal_summary.get("focus_symbols") or []),
        "gate_profile": terminal_summary.get("gate_profile"),
        "metrics": metrics,
        "mechanics": mechanics,
        "gold_precheck": dict(terminal_summary.get("gold_precheck") or {}),
        "runtime_overrides": dict(trial.get("runtime_overrides") or {}),
        "trial_definition": dict(trial),
        "finished_at": terminal_summary.get("terminal_at") or _now_iso(),
    }
    existing.append(result_entry)
    existing.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    _write_results_snapshot(
        results_path,
        sequence_id=sequence_id,
        profile=profile,
        engine=engine,
        mode=mode,
        entries=existing,
    )
    ga_payload = {
        "trial_id": record_id,
        "engine": engine,
        "sequence_id": sequence_id,
        "profile": profile,
        "horizon": str(terminal_summary.get("horizon") or ""),
        "family": str(terminal_summary.get("family") or ""),
        "feature_profile": str(terminal_summary.get("feature_profile") or ""),
        "mechanics_profile_version": str(terminal_summary.get("mechanics_profile_version") or ""),
        "ga_generation": generation,
        "ga_trial": raw_trial_id,
        "trial_mode": mode,
        "trial_cost_profile": str(terminal_summary.get("trial_cost_profile") or ""),
        "gate_profile": str(terminal_summary.get("gate_profile") or ""),
        "params": dict(trial.get("trial_definition") or trial),
        "fitness_score": score,
        "failure_mode": failure_mode,
        "run_id": run_id,
        "dataset_id": str(terminal_summary.get("dataset_id") or ""),
        "payload": result_entry,
        "finished_at": result_entry["finished_at"],
    }
    if not record_ga_trial(ga_payload):
        _post_internal_payload("/internal/ga-trial", ga_payload)
    return result_entry


def _write_finalists(
    *,
    sequence_id: str,
    profile: str,
    engine: str,
    proxy_results_path: Path,
) -> list[dict[str, Any]]:
    """Construit et persiste les deux meilleurs finalistes proxy."""

    proxy_results = _load_result_entries(proxy_results_path)
    eligible_results = [
        item
        for item in proxy_results
        if not bool(item.get("killed_after_precheck"))
    ]
    finalists = eligible_results[:2]
    _write_json(
        _finalists_path(profile, engine),
        {
            "sequence_id": sequence_id,
            "profile": profile,
            "engine": engine,
            "generated_at": _now_iso(),
            "finalists": finalists,
            "eligible_trial_count": len(eligible_results),
        },
    )
    return finalists


def _resolve_full_runtime_overrides(
    config: dict[str, Any],
    *,
    profile: str,
    engine: str,
    trial_id: str,
    finalist_rank: int,
) -> dict[str, str]:
    """Retourne les surcharges runtime `full` pour un trial finaliste.

    Args:
        config (dict[str, Any]): Configuration globale de sequence.
        profile (str): Profil en cours.
        engine (str): Moteur en cours.
        trial_id (str): Identifiant du trial finaliste.
        finalist_rank (int): Rang du finaliste retenu.

    Returns:
        dict[str, str]: Surcouches `full` a appliquer.
    """

    catalogs = (
        (((config.get("full_catalogs") or {}).get(engine) or {}).get(profile) or {})
    )
    runtime_overrides = {
        str(key): str(value)
        for key, value in dict(catalogs.get(trial_id) or {}).items()
    }
    if runtime_overrides:
        trigger = str(runtime_overrides.get("TRAINING_RUN_TRIGGER") or "").strip()
        if trigger:
            runtime_overrides["TRAINING_RUN_TRIGGER"] = trigger.replace(
                "_finalist_0_",
                f"_finalist_{finalist_rank}_",
            )
        runtime_overrides["TRAINING_GA_GENERATION"] = str(finalist_rank)
    return runtime_overrides


def _execute_window(
    *,
    config: dict[str, Any],
    sequence_state: dict[str, Any],
    profile: str,
    engine: str,
    mode: str,
    window_index: int,
) -> dict[str, Any]:
    """Execute une fenetre V4 complete pour un moteur et un profil."""

    sequence_id = str(config.get("sequence_id") or "")
    retry_limit = int(config.get("retry_limit", 1))
    results_path = _results_snapshot_path(profile, engine, mode)
    completed_entries = _load_result_entries(results_path)
    completed_trial_ids = {
        str(item.get("trial_id") or "").strip()
        for item in completed_entries
        if item.get("trial_id")
    }

    if mode in {"proxy_ga", "smoke"}:
        catalog_key = "smoke_catalogs" if mode == "smoke" else "catalogs"
        trials = list(((((config.get(catalog_key) or {}).get(engine) or {}).get(profile)) or []))
        if not trials:
            raise RuntimeError(f"Aucun catalogue {mode} pour {engine}/{profile}.")
    elif mode == "full":
        finalists_payload = _load_json(_finalists_path(profile, engine))
        trials = list(finalists_payload.get("finalists") or [])
        if not trials:
            append_training_log(
                f"Sequence V4: aucun finaliste disponible pour {engine}/{profile}, fenetre full ignoree.",
                level="WARNING",
                source="sequence",
            )
            sequence_state.update(
                {
                    "state": "running",
                    "profile": profile,
                    "engine": engine,
                    "mode": mode,
                    "trial_id": None,
                    "window_id": _build_window_id(sequence_id, profile, engine, mode, "no_finalist"),
                    "window_index": window_index,
                    "status": "skipped",
                    "next_step": None,
                    "continued_after_precheck": None,
                    "killed_after_precheck": None,
                    "precheck_status": None,
                    "precheck_score": None,
                    "proxy_terminal_score": None,
                }
            )
            _persist_window_state(sequence_state)
            return {
                "profile": profile,
                "engine": engine,
                "mode": mode,
                "status": "skipped",
                "proxy_ready": True,
                "failure_mode": None,
                "terminal_status": "skipped",
            }
    else:
        raise RuntimeError(f"Mode de fenetre non supporte: {mode}")

    smoke_proxy_ready = True
    smoke_last_failure_mode = None
    smoke_last_status = None
    stop_after_current_trial = False

    for generation, trial in enumerate(trials, start=1):
        raw_trial_id = str(trial.get("trial_id") or "").strip()
        if raw_trial_id in completed_trial_ids:
            continue
        window_id = _build_window_id(sequence_id, profile, engine, mode, raw_trial_id)
        sequence_state.update(
            {
                "state": "running",
                "profile": profile,
                "engine": engine,
                "mode": mode,
                "trial_id": raw_trial_id,
                "window_id": window_id,
                "window_index": window_index,
                "status": "running",
                "retry_count": 0,
                "started_at": sequence_state.get("started_at") or _now_iso(),
                "next_step": None,
                "continued_after_precheck": None,
                "killed_after_precheck": None,
                "precheck_status": None,
                "precheck_score": None,
                "proxy_terminal_score": None,
            }
        )
        _persist_window_state(sequence_state)

        runtime_overrides = dict(trial.get("runtime_overrides") or {})
        if mode == "full":
            runtime_overrides = _resolve_full_runtime_overrides(
                config,
                profile=profile,
                engine=engine,
                trial_id=raw_trial_id,
                finalist_rank=generation,
            ) or runtime_overrides
        runtime_overrides["TRAINING_GA_GENERATION"] = str(generation)
        trial_overrides = _build_trial_env(
            runtime_overrides,
            sequence_id=sequence_id,
            sequence_profile=profile,
            window_id=window_id,
            trial_id=raw_trial_id,
        )

        while True:
            outcome = _run_trial(trial_overrides, sequence_state)
            terminal_summary = dict(outcome.get("terminal_summary") or {})
            if not terminal_summary:
                if _handle_terminal_without_summary(outcome, sequence_state, retry_limit=retry_limit):
                    continue
                raise RuntimeError(str(sequence_state.get("last_error") or "resume_terminal_absent"))

            result_entry = _record_scored_trial(
                sequence_id=sequence_id,
                profile=profile,
                engine=engine,
                mode=mode,
                trial=trial,
                generation=generation,
                run_id=outcome.get("run_id"),
                terminal_summary=terminal_summary,
                results_path=results_path,
            )
            sequence_state.update(
                {
                    "status": str(result_entry.get("terminal_status") or "completed"),
                    "last_run_id": outcome.get("run_id"),
                    "last_completed_trial": raw_trial_id,
                    "continued_after_precheck": result_entry.get("continued_after_precheck"),
                    "killed_after_precheck": result_entry.get("killed_after_precheck"),
                    "precheck_status": result_entry.get("precheck_status"),
                    "precheck_score": result_entry.get("precheck_score"),
                    "proxy_terminal_score": result_entry.get("proxy_terminal_score"),
                    "supervisor_heartbeat": _now_iso(),
                    "last_error": None,
                }
            )
            _persist_window_state(sequence_state)
            append_training_log(
                (
                    f"Sequence V4: {profile} | moteur={engine} | mode={mode} | trial={raw_trial_id} "
                    f"| score={result_entry.get('score')} | failure_mode={result_entry.get('failure_mode')}"
                ),
                source="sequence",
            )
            if mode == "smoke":
                smoke_last_failure_mode = str(result_entry.get("failure_mode") or "").strip() or None
                smoke_last_status = str(result_entry.get("terminal_status") or "").strip() or None
                if smoke_last_status == "completed":
                    smoke_proxy_ready = True
                    stop_after_current_trial = True
                    break
                smoke_proxy_ready = False
                if (
                    smoke_last_failure_mode == "insufficient_sample"
                    and generation < len(trials)
                ):
                    append_training_log(
                        (
                            f"Sequence V4: smoke Dreamer {profile} insuffisant, "
                            "tentative de secours autorisee."
                        ),
                        level="WARNING",
                        source="sequence",
                    )
                    stop_after_current_trial = False
                    continue
                stop_after_current_trial = True
                break
            break
        if stop_after_current_trial:
            break

    if mode == "proxy_ga":
        finalists = _write_finalists(
            sequence_id=sequence_id,
            profile=profile,
            engine=engine,
            proxy_results_path=results_path,
        )
        sequence_state["next_step"] = {
            "profile": profile,
            "engine": engine,
            "mode": "full",
            "finalists": [item.get("trial_id") for item in finalists],
        }
    else:
        sequence_state["next_step"] = None
    sequence_state["status"] = "completed"
    _persist_window_state(sequence_state)
    return {
        "profile": profile,
        "engine": engine,
        "mode": mode,
        "status": sequence_state.get("status"),
        "proxy_ready": smoke_proxy_ready if mode == "smoke" else True,
        "failure_mode": smoke_last_failure_mode,
        "terminal_status": smoke_last_status,
    }


def _request_json(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute une requete JSON courte vers Lab.

    Args:
        method (str): Methode HTTP a utiliser.
        path (str): Chemin relatif de l'endpoint.
        payload (dict[str, Any] | None): Charge utile JSON eventuelle.

    Returns:
        dict[str, Any]: Reponse JSON decodee.
    """

    url = f"{LAB_INTERNAL_URL}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None,
        headers={"Content-Type": "application/json"},
        method=method.upper(),
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _execute_gnn_refresh(
    *,
    sequence_state: dict[str, Any],
    refresh_payload: dict[str, Any],
    window_index: int,
) -> dict[str, Any]:
    """Declenche puis attend un refresh GNN.

    Args:
        sequence_state (dict[str, Any]): Etat courant de sequence.
        refresh_payload (dict[str, Any]): Parametres du refresh a lancer.
        window_index (int): Index logique de la fenetre.

    Returns:
        dict[str, Any]: Etat final du refresh GNN.
    """

    sequence_state.update(
        {
            "engine": "gnn",
            "mode": "refresh",
            "profile": str(sequence_state.get("profile") or ""),
            "trial_id": "gnn_refresh",
            "window_id": f"{sequence_state.get('sequence_id')}:gnn:refresh:{window_index}",
            "window_index": window_index,
            "status": "running",
            "next_step": None,
        }
    )
    _persist_window_state(sequence_state)
    append_training_log("Sequence V4: demande de refresh GNN Gold.", source="sequence")
    response = _request_json("POST", "/gnn/refresh", refresh_payload)
    refresh = dict(response.get("refresh") or {})
    run_id = str(refresh.get("run_id") or "").strip() or None
    while True:
        sequence_state["supervisor_heartbeat"] = _now_iso()
        sequence_state["last_run_id"] = run_id
        _persist_window_state(sequence_state)
        current = _request_json("GET", "/gnn/refresh/status")
        refresh = dict(current.get("refresh") or {})
        refresh_status = str(refresh.get("status") or "").strip().lower()
        run_id = str(refresh.get("run_id") or "").strip() or run_id
        if refresh_status in {"completed", "error"}:
            sequence_state["status"] = refresh_status
            sequence_state["last_run_id"] = run_id
            _persist_window_state(sequence_state)
            return {
                "status": refresh_status,
                "run_id": run_id,
                "refresh": refresh,
            }
        time.sleep(15)


def _restart_vllm_service() -> None:
    """Redemarre vLLM a la fin de la sequence Gold."""

    append_training_log(
        "Sequence V4: redemarrage final de vLLM apres les artefacts Gold.",
        source="sequence",
    )
    subprocess.run(
        ["bash", "-lc", "cd /home/aza/The_Hive && docker compose up -d vllm"],
        cwd=ROOT,
        check=True,
    )
    for _attempt in range(18):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5):
                append_training_log("Sequence V4: vLLM de nouveau operationnel.", source="sequence")
                return
        except urllib.error.URLError:
            time.sleep(10)
    raise RuntimeError("vLLM n'a pas retrouve un etat sain apres redemarrage.")


def _run_sequence(config: dict[str, Any]) -> None:
    """Execute l'integralite de la sequence V4."""

    sequence_id = str(config.get("sequence_id") or "").strip()
    if not sequence_id:
        raise ValueError("Configuration de sequence invalide: sequence_id absent.")

    state = load_sequence_state()
    state.update(
        {
            "sequence_id": sequence_id,
            "sequence_name": config.get("sequence_name"),
            "state": "running",
            "status": "running",
            "started_at": state.get("started_at") or _now_iso(),
            "stdout_log_path": str(config.get("stdout_log_path") or ""),
            "stderr_log_path": str(config.get("stderr_log_path") or ""),
            "supervisor_heartbeat": _now_iso(),
            "last_error": None,
        }
    )
    _persist_window_state(state)
    append_training_log(
        f"Sequence V4 distante {sequence_id} demarree ({config.get('sequence_name')}).",
        source="sequence",
    )

    window_index = 0
    smoke_results: dict[tuple[str, str], bool] = {}
    explicit_steps = list(config.get("steps") or [])
    if explicit_steps:
        for step in explicit_steps:
            window_index += 1
            step_kind = str(step.get("kind") or "window").strip().lower()
            if step_kind == "window":
                profile = str(step.get("profile") or "").strip()
                engine = str(step.get("engine") or "").strip().lower()
                mode = str(step.get("mode") or "").strip().lower()
                smoke_key = (profile, engine)
                if engine == "dreamer" and mode in {"proxy_ga", "full"} and smoke_results.get(smoke_key) is False:
                    append_training_log(
                        f"Sequence V4: saut de {engine}/{mode} pour {profile} car le smoke a echoue.",
                        level="WARNING",
                        source="sequence",
                    )
                    continue
                window_result = _execute_window(
                    config=config,
                    sequence_state=state,
                    profile=profile,
                    engine=engine,
                    mode=mode,
                    window_index=window_index,
                )
                if mode == "smoke":
                    smoke_results[smoke_key] = bool(window_result.get("proxy_ready"))
                continue

            if step_kind == "gnn_refresh":
                refresh_result = _execute_gnn_refresh(
                    sequence_state=state,
                    refresh_payload=dict(step.get("refresh_payload") or {}),
                    window_index=window_index,
                )
                if str(refresh_result.get("status") or "").strip().lower() != "completed":
                    raise RuntimeError("Refresh GNN Gold incomplet ou en erreur.")
                continue

            if step_kind == "service_action":
                action = str(step.get("action") or "").strip().lower()
                if action == "restart_vllm":
                    _restart_vllm_service()
                    continue
                raise RuntimeError(f"Action de service inconnue: {action}")

            raise RuntimeError(f"Etape de sequence inconnue: {step_kind}")

        state["state"] = "completed"
        state["status"] = "completed"
        state["finished_at"] = _now_iso()
        state["supervisor_heartbeat"] = _now_iso()
        _persist_window_state(state)
        append_training_log(
            f"Sequence V4 distante {sequence_id} terminee.",
            source="sequence",
        )
        return

    for profile in list(config.get("profiles") or []):
        for window in list(config.get("window_order") or []):
            window_index += 1
            engine = str(window.get("engine") or "").strip().lower()
            mode = str(window.get("mode") or "").strip().lower()
            _execute_window(
                config=config,
                sequence_state=state,
                profile=profile,
                engine=engine,
                mode=mode,
                window_index=window_index,
            )

    state["state"] = "completed"
    state["status"] = "completed"
    state["finished_at"] = _now_iso()
    state["supervisor_heartbeat"] = _now_iso()
    _persist_window_state(state)
    append_training_log(
        f"Sequence V4 distante {sequence_id} terminee.",
        source="sequence",
    )


def main() -> int:
    """Point d'entree CLI du superviseur V4 distant.

    Returns:
        int: Code de retour shell.
    """

    parser = argparse.ArgumentParser(description="Superviseur distant V4.")
    parser.add_argument("--config", required=True, help="Chemin du JSON de configuration distant.")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    _bootstrap_timescale_env()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    config = _load_json(config_path)
    if not config:
        raise RuntimeError(f"Configuration de sequence introuvable ou invalide: {config_path}")

    V4_SEQUENCE_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    V4_SEQUENCE_PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    try:
        _run_sequence(config)
        return 0
    except Exception as exc:
        state = load_sequence_state()
        state["state"] = "paused"
        state["status"] = "paused"
        state["last_error"] = str(exc)
        state["supervisor_heartbeat"] = _now_iso()
        _persist_window_state(state)
        append_training_log(
            f"Sequence V4 mise en pause: {exc}",
            level="ERROR",
            source="sequence",
        )
        print(f"[sequence] Erreur fatale: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if V4_SEQUENCE_PID_PATH.exists():
            try:
                V4_SEQUENCE_PID_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
