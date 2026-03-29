"""Execute la nuit Gold manuelle directement sur le serveur, sans V4.

Le runner consomme une configuration JSON preparee cote local, puis
orchesre un seul run GPU a la fois:

- MuZero proxy repris depuis un checkpoint explicite ;
- un seul fallback proxy MuZero si le premier verdict est faible ;
- un seul full MuZero si un proxy est viable ;
- Dreamer smoke, puis proxy/full si le smoke valide le pipeline ;
- GNN en dernier seulement si non desactive.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_DIR / "src" / "eva-lab"
SHARED_ROOT = PROJECT_DIR / "src" / "shared"
for candidate in (PACKAGE_ROOT, SHARED_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from eva_lab.training_status import (  # type: ignore  # pylint: disable=wrong-import-position
    append_training_log,
    finalize_training_status,
    load_latest_terminal_summary,
    load_sequence_state,
    load_training_status,
    persist_sequence_state,
    set_training_launcher_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gold_manual_remote")

REMOTE_SCRIPT_PATH = PROJECT_DIR / "scripts" / "run_nightly_training_remote.sh"
LOCK_FILE = PROJECT_DIR / "data" / "checkpoints" / "nightly_training.lock"
LOCK_DIR = PROJECT_DIR / "data" / "checkpoints" / "nightly_training.lock.d"


def _now_iso() -> str:
    """Retourne un horodatage ISO 8601 local.

    Returns:
        str: Horodatage courant.
    """

    return datetime.now().isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    """Charge un JSON local si present.

    Args:
        path (Path): Chemin cible.

    Returns:
        dict[str, Any] | None: Charge utile ou ``None``.
    """

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ecrit un JSON local de maniere atomique.

    Args:
        path (Path): Fichier cible.
        payload (dict[str, Any]): Contenu a serialiser.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def _http_json(path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Lit un endpoint JSON local d'EVA Lab.

    Args:
        path (str): Chemin HTTP cible.
        method (str): Methode HTTP.
        payload (dict[str, Any] | None): Corps JSON optionnel.

    Returns:
        dict[str, Any]: Charge utile decodee.

    Raises:
        RuntimeError: Si l'appel echoue.
    """

    url = f"http://127.0.0.1:8600{path}"
    request_data = None
    headers = {}
    if payload is not None:
        request_data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=request_data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Lecture impossible de {url}: {exc}") from exc


def _load_state(state_path: Path) -> dict[str, Any]:
    """Charge l'etat manuel persiste.

    Args:
        state_path (Path): Fichier d'etat cible.

    Returns:
        dict[str, Any]: Etat normalise.
    """

    payload = _read_json(state_path) or {}
    return {
        "run_group_id": payload.get("run_group_id"),
        "current_stage": payload.get("current_stage"),
        "current_engine": payload.get("current_engine"),
        "current_run_id": payload.get("current_run_id"),
        "focus_symbols": list(payload.get("focus_symbols") or []),
        "gate_profile": payload.get("gate_profile"),
        "resume_checkpoint_path": payload.get("resume_checkpoint_path"),
        "resume_step": payload.get("resume_step"),
        "status": payload.get("status") or "idle",
        "last_error": payload.get("last_error"),
        "started_at": payload.get("started_at"),
        "updated_at": payload.get("updated_at"),
        "muzero_results": list(payload.get("muzero_results") or []),
        "dreamer_results": list(payload.get("dreamer_results") or []),
        "gnn_result": payload.get("gnn_result"),
    }


def _persist_state(state_path: Path, **patch: Any) -> dict[str, Any]:
    """Fusionne un patch dans l'etat manuel.

    Args:
        state_path (Path): Fichier d'etat cible.
        **patch (Any): Champs a mettre a jour.

    Returns:
        dict[str, Any]: Etat normalise persiste.
    """

    state = _load_state(state_path)
    state.update(patch)
    state["updated_at"] = _now_iso()
    _write_json(state_path, state)
    return state


def _stop_active_training(reason: str) -> None:
    """Coupe un run actif et nettoie le verrou nightly.

    Args:
        reason (str): Motif humain de l'arret.
    """

    lock_payload = _read_json(LOCK_FILE) or {}
    lock_pid = lock_payload.get("pid")
    if isinstance(lock_pid, int):
        try:
            os.kill(lock_pid, signal.SIGTERM)
            time.sleep(5)
        except OSError:
            pass

    stop_command = (
        "docker ps --format '{{.Names}}' | "
        "grep '^the_hive-eva-trainer-run-' | "
        "xargs -r docker stop"
    )
    subprocess.run(["bash", "-lc", stop_command], cwd=PROJECT_DIR, check=False)

    if LOCK_DIR.exists():
        subprocess.run(["bash", "-lc", f"rm -rf {LOCK_DIR}"], cwd=PROJECT_DIR, check=False)
    if LOCK_FILE.exists():
        LOCK_FILE.unlink(missing_ok=True)

    append_training_log(
        "Run interrompu manuellement pour basculer sur le bypass Gold de la nuit.",
        level="WARNING",
        source="launcher",
    )
    set_training_launcher_state(phase="idle", last_stop_reason=reason)
    finalize_training_status("aborted", reason=reason)


def _pause_v4_supervisor() -> None:
    """Fige le superviseur V4 pour eviter tout redemarrage parasite."""

    subprocess.run(
        ["bash", "-lc", "pkill -f 'scripts/deploy/v4_sequence_runner.py' || true"],
        cwd=PROJECT_DIR,
        check=False,
    )
    sequence_state = load_sequence_state()
    sequence_state.update(
        {
            "state": "paused",
            "status": "paused",
            "next_step": "manual_gold_night",
            "last_error": "manual_gold_bypass_active",
            "retry_reason": "manual_gold_bypass_active",
        }
    )
    persist_sequence_state(sequence_state)


def _resolve_summary(engine: str, run_id: str, explicit_path: str | None = None) -> dict[str, Any] | None:
    """Charge le resume terminal le plus probable du run.

    Args:
        engine (str): Moteur cible.
        run_id (str): Identifiant du run.
        explicit_path (str | None): Chemin de resume explicite si connu.

    Returns:
        dict[str, Any] | None: Resume terminal si disponible.
    """

    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.is_absolute():
            candidate = PROJECT_DIR / candidate
        payload = _read_json(candidate)
        if payload is not None and str(payload.get("run_id") or "").strip() == run_id:
            return payload

    payload = load_latest_terminal_summary(engine=engine)
    if payload is None:
        return None
    return payload if str(payload.get("run_id") or "").strip() == run_id else None


def _wait_for_run_start(expected_trigger: str, previous_run_id: str | None, timeout_seconds: int = 600) -> str:
    """Attend le demarrage visible d'un nouveau run.

    Args:
        expected_trigger (str): Trigger attendu.
        previous_run_id (str | None): Dernier run visible avant lancement.
        timeout_seconds (int): Delai maximal.

    Returns:
        str: Run id observe.

    Raises:
        RuntimeError: Si le run n'apparait pas dans le delai.
    """

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status = _http_json("/training/status")
        run = dict(status.get("run") or {})
        run_id = str(run.get("run_id") or "").strip()
        trigger = str(run.get("trigger") or "").strip()
        if run_id and run_id != str(previous_run_id or "") and trigger == expected_trigger and bool(run.get("active")):
            return run_id
        time.sleep(5)
    raise RuntimeError(f"Le run {expected_trigger} n'a pas demarre dans le delai imparti.")


def _wait_for_run_terminal(
    run_id: str,
    *,
    engine: str,
    state_path: Path | None = None,
    stage_id: str | None = None,
    poll_interval_seconds: int = 30,
) -> dict[str, Any]:
    """Attend la fin observable d'un run et charge son resume terminal.

    Args:
        run_id (str): Run a surveiller.
        engine (str): Moteur associe.
        state_path (Path | None): Etat manuel a heartbeater si disponible.
        stage_id (str | None): Identifiant de stage pour la trace.
        poll_interval_seconds (int): Frequence de poll.

    Returns:
        dict[str, Any]: Charge terminale enrichie.
    """

    last_step_label = ""
    while True:
        status = _http_json("/training/status")
        run = dict(status.get("run") or {})
        current_run_id = str(run.get("run_id") or "").strip()
        current_step = str(run.get("step_label") or run.get("effective_step_label") or "").strip()
        if state_path is not None:
            _persist_state(
                state_path,
                current_stage=stage_id,
                current_engine=engine,
                current_run_id=run_id,
                status="running",
            )
        if current_step and current_step != last_step_label:
            logger.info("[%s] Etape: %s", run_id, current_step)
            last_step_label = current_step
        if current_run_id == run_id and not bool(run.get("active")):
            terminal_summary_path = str(run.get("terminal_summary_path") or "").strip() or None
            return {
                "terminal": {
                    "status": str(run.get("status") or "unknown"),
                    "run_id": run_id,
                    "reason": str(run.get("reason") or ""),
                    "terminal_summary_path": terminal_summary_path,
                    "step_label": current_step or last_step_label,
                },
                "summary": _resolve_summary(engine=engine, run_id=run_id, explicit_path=terminal_summary_path),
            }
        time.sleep(max(10, poll_interval_seconds))


def _score_summary_metrics(summary: dict[str, Any]) -> float:
    """Calcule un score heuristique stable pour un resume terminal.

    Args:
        summary (dict[str, Any]): Resume courant.

    Returns:
        float: Score consolide.
    """

    metrics = dict(summary.get("metrics") or {})
    mechanics = dict(summary.get("metrics_by_position_mechanics") or {})
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
    failure_mode = str(summary.get("failure_mode") or "").strip().lower()
    if failure_mode in {"inactive", "sell_heavy", "buy_heavy", "bad_exit"}:
        score -= 25.0
    return round(score, 4)


def _is_muzero_summary_weak(summary: dict[str, Any] | None) -> tuple[bool, str]:
    """Determine si un proxy MuZero doit etre coupe de la suite.

    Args:
        summary (dict[str, Any] | None): Resume terminal courant.

    Returns:
        tuple[bool, str]: Drapeau de faiblesse et raison associee.
    """

    if not summary:
        return True, "resume_terminal_absent"

    metrics = dict(summary.get("metrics") or {})
    mechanics = dict(summary.get("metrics_by_position_mechanics") or {})
    failure_mode = str(summary.get("failure_mode") or "").strip().lower()
    arena_outcome = str(summary.get("arena_outcome") or "").strip().upper()
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    return_pct = float(metrics.get("return_pct", 0.0) or 0.0)
    net_realized_pct = float(metrics.get("net_realized_pct", 0.0) or 0.0)
    close_quality_score = float(mechanics.get("close_quality_score", 0.0) or 0.0)
    hold_drag_score = float(mechanics.get("hold_drag_score", 0.0) or 0.0)

    if str(summary.get("terminal_status") or "").strip().lower() != "completed":
        return True, f"terminal_status:{summary.get('terminal_status')}"
    if failure_mode in {"inactive", "sell_heavy", "buy_heavy", "bad_exit"}:
        return True, f"failure_mode:{failure_mode}"
    if arena_outcome and arena_outcome != "VICTORY" and profit_factor < 1.0 and return_pct <= 0.0:
        return True, "arena_defeat_net"
    if profit_factor < 1.0 and return_pct <= 0.0 and net_realized_pct <= 0.0:
        return True, "rentabilite_negative"
    if close_quality_score < 0.40 and hold_drag_score > 0.90:
        return True, "sorties_trop_faibles"
    return False, "proxy_viable"


def _append_stage_result(state_path: Path, collection_key: str, payload: dict[str, Any]) -> None:
    """Ajoute un resultat de stage dans l'etat global.

    Args:
        state_path (Path): Fichier d'etat.
        collection_key (str): Cle de collection.
        payload (dict[str, Any]): Verdict a empiler.
    """

    state = _load_state(state_path)
    entries = list(state.get(collection_key) or [])
    entries.append(payload)
    _persist_state(state_path, **{collection_key: entries})


def _build_stage_env(
    *,
    config: dict[str, Any],
    stage_id: str,
    trial_id: str,
    runtime_overrides: dict[str, str],
) -> dict[str, str]:
    """Construit l'environnement final d'un stage manuel.

    Args:
        config (dict[str, Any]): Configuration complete de la nuit.
        stage_id (str): Identifiant technique du stage.
        trial_id (str): Trial logique.
        runtime_overrides (dict[str, str]): Surcharges moteur.

    Returns:
        dict[str, str]: Environnement complet du stage.
    """

    env = os.environ.copy()
    env.update({key: str(value) for key, value in dict(runtime_overrides or {}).items()})
    env["TRAINING_SEQUENCE_ID"] = str(config.get("run_group_id") or "").strip()
    env["TRAINING_SEQUENCE_PROFILE"] = str((config.get("manual_env") or {}).get("sequence_profile") or "gold_manual_night")
    env["TRAINING_SUPERVISOR_STATE"] = str((config.get("manual_env") or {}).get("supervisor_state") or "manual_gold_night")
    env["TRAINING_WINDOW_ID"] = stage_id
    env["TRAINING_TRIAL_ID"] = trial_id
    env["NIGHTLY_KEEP_VLLM"] = "0"
    env["NIGHTLY_DEFER_VLLM_RESTART"] = "1"
    env["PYTHONPATH"] = f"{PACKAGE_ROOT}:{SHARED_ROOT}"
    return env


def _launch_stage_process(
    *,
    config: dict[str, Any],
    state_path: Path,
    stage_id: str,
    engine: str,
    trial_id: str,
    runtime_overrides: dict[str, str],
) -> dict[str, Any]:
    """Lance un stage GPU, attend son verdict et retourne le resume.

    Args:
        config (dict[str, Any]): Configuration globale.
        state_path (Path): Fichier d'etat.
        stage_id (str): Identifiant technique.
        engine (str): Moteur cible.
        trial_id (str): Trial logique.
        runtime_overrides (dict[str, str]): Variables a injecter.

    Returns:
        dict[str, Any]: Verdict terminal enrichi.
    """

    previous_run = dict((_http_json("/training/status").get("run") or {}))
    previous_run_id = str(previous_run.get("run_id") or "").strip() or None
    env = _build_stage_env(
        config=config,
        stage_id=stage_id,
        trial_id=trial_id,
        runtime_overrides=runtime_overrides,
    )
    expected_trigger = str(env.get("TRAINING_RUN_TRIGGER") or "").strip()
    if not expected_trigger:
        raise ValueError("Impossible de lancer un stage sans TRAINING_RUN_TRIGGER.")

    _persist_state(
        state_path,
        current_stage=stage_id,
        current_engine=engine,
        current_run_id=None,
        status="launching",
        last_error=None,
    )
    process = subprocess.Popen(
        ["bash", str(REMOTE_SCRIPT_PATH)],
        cwd=PROJECT_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    run_id = _wait_for_run_start(expected_trigger, previous_run_id)
    _persist_state(
        state_path,
        current_stage=stage_id,
        current_engine=engine,
        current_run_id=run_id,
        status="running",
    )
    verdict = _wait_for_run_terminal(
        run_id,
        engine=engine,
        state_path=state_path,
        stage_id=stage_id,
        poll_interval_seconds=30,
    )
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        logger.warning("Le lanceur local du stage %s est encore actif apres la fin observable du run.", stage_id)
    verdict["engine"] = engine
    verdict["trial_id"] = trial_id
    verdict["stage_id"] = stage_id
    verdict["process_pid"] = process.pid
    return verdict


def _run_muzero_stage(
    *,
    config: dict[str, Any],
    state_path: Path,
    stage_prefix: str,
    trial: dict[str, Any],
    mode: str,
    resume_checkpoint_path: str | None = None,
    resume_step: int | None = None,
) -> dict[str, Any]:
    """Execute un stage MuZero et calcule sa viabilite.

    Args:
        config (dict[str, Any]): Configuration globale.
        state_path (Path): Fichier d'etat.
        stage_prefix (str): Prefixe de stage.
        trial (dict[str, Any]): Definition du trial.
        mode (str): Mode `proxy_ga` ou `full`.
        resume_checkpoint_path (str | None): Checkpoint explicite optionnel.
        resume_step (int | None): Step de reprise explicite.

    Returns:
        dict[str, Any]: Verdict du stage MuZero.
    """

    trial_id = str(trial.get("trial_id") or "").strip() or "baseline"
    runtime_overrides = dict(trial.get("runtime_overrides") or {})
    if resume_checkpoint_path:
        runtime_overrides["MUZERO_RESUME_CHECKPOINT_PATH"] = str(resume_checkpoint_path)
    if resume_step:
        runtime_overrides["MUZERO_RESUME_STEP"] = str(int(resume_step))

    verdict = _launch_stage_process(
        config=config,
        state_path=state_path,
        stage_id=f"{stage_prefix}_{trial_id}",
        engine="muzero",
        trial_id=trial_id,
        runtime_overrides=runtime_overrides,
    )
    summary = dict(verdict.get("summary") or {})
    verdict["score"] = _score_summary_metrics(summary) if summary else None
    weak, weak_reason = _is_muzero_summary_weak(summary)
    verdict["is_weak"] = weak
    verdict["weak_reason"] = weak_reason
    _append_stage_result(
        state_path,
        "muzero_results",
        {
            "stage_id": verdict.get("stage_id"),
            "trial_id": trial_id,
            "mode": mode,
            "run_id": ((verdict.get("terminal") or {}).get("run_id")),
            "status": ((verdict.get("terminal") or {}).get("status")),
            "summary_path": str(((summary or {}).get("path")) or ""),
            "score": verdict.get("score"),
            "is_weak": weak,
            "weak_reason": weak_reason,
        },
    )
    return verdict


def _run_dreamer_stage(
    *,
    config: dict[str, Any],
    state_path: Path,
    stage_prefix: str,
    trial: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Execute un stage Dreamer et retourne son resume terminal.

    Args:
        config (dict[str, Any]): Configuration globale.
        state_path (Path): Fichier d'etat.
        stage_prefix (str): Prefixe d'etape.
        trial (dict[str, Any]): Definition du trial.
        mode (str): Mode `smoke`, `proxy_ga` ou `full`.

    Returns:
        dict[str, Any]: Verdict Dreamer enrichi.
    """

    trial_id = str(trial.get("trial_id") or "").strip() or "baseline"
    runtime_overrides = dict(trial.get("runtime_overrides") or {})
    verdict = _launch_stage_process(
        config=config,
        state_path=state_path,
        stage_id=f"{stage_prefix}_{trial_id}",
        engine="dreamer",
        trial_id=trial_id,
        runtime_overrides=runtime_overrides,
    )
    summary = dict(verdict.get("summary") or {})
    verdict["score"] = _score_summary_metrics(summary) if summary else None
    _append_stage_result(
        state_path,
        "dreamer_results",
        {
            "stage_id": verdict.get("stage_id"),
            "trial_id": trial_id,
            "mode": mode,
            "run_id": ((verdict.get("terminal") or {}).get("run_id")),
            "status": ((verdict.get("terminal") or {}).get("status")),
            "summary_path": str(((summary or {}).get("path")) or ""),
            "score": verdict.get("score"),
            "terminal_status": str(summary.get("terminal_status") or ""),
            "failure_mode": str(summary.get("failure_mode") or ""),
        },
    )
    return verdict


def _select_single_muzero_full_candidate(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Retient un seul finaliste MuZero pour le mode `full`.

    Args:
        results (list[dict[str, Any]]): Resultats proxy deja produits.

    Returns:
        dict[str, Any] | None: Meilleur proxy viable ou ``None``.
    """

    viable = [item for item in results if not bool(item.get("is_weak"))]
    if not viable:
        return None
    return max(viable, key=lambda item: float(item.get("score") or -10**9))


def _run_gnn_refresh(request_payload: dict[str, Any]) -> dict[str, Any]:
    """Declenche puis attend un refresh GNN explicite.

    Args:
        request_payload (dict[str, Any]): Parametres de refresh.

    Returns:
        dict[str, Any]: Etat final du refresh.
    """

    response = _http_json("/gnn/refresh", method="POST", payload=request_payload)
    refresh = dict(response.get("refresh") or {})
    deadline = time.time() + 8 * 3600
    while time.time() < deadline:
        status_payload = _http_json("/gnn/refresh/status")
        refresh = dict(status_payload.get("refresh") or {})
        refresh_status = str(refresh.get("status") or "").strip().lower()
        if refresh_status in {"completed", "error"}:
            return refresh
        time.sleep(20)
    raise RuntimeError("Refresh GNN toujours en cours apres le delai maximal.")


def _run_manual_night(config: dict[str, Any]) -> dict[str, Any]:
    """Execute la nuit Gold critique a partir d'une configuration pre-calculee.

    Args:
        config (dict[str, Any]): Configuration serialisee.

    Returns:
        dict[str, Any]: Etat final de la nuit Gold.
    """

    state_path = Path(str(config.get("state_path") or PROJECT_DIR / "data" / "checkpoints" / "gold_manual_state.json"))
    _persist_state(
        state_path,
        run_group_id=str(config.get("run_group_id") or "").strip(),
        current_stage="bootstrap",
        current_engine=None,
        current_run_id=None,
        focus_symbols=list(config.get("focus_symbols") or []),
        gate_profile=str(config.get("gate_profile") or "").strip() or None,
        resume_checkpoint_path=str(config.get("resume_checkpoint_path") or "").strip() or None,
        resume_step=config.get("resume_step"),
        status="preparing",
        last_error=None,
        started_at=_now_iso(),
        muzero_results=[],
        dreamer_results=[],
        gnn_result=None,
    )

    current_training = load_training_status()
    if bool(current_training.get("active")):
        _stop_active_training("gold_manual_night_cutover")
    _pause_v4_supervisor()

    muzero_config = dict(config.get("muzero") or {})
    dreamer_config = dict(config.get("dreamer") or {})

    muzero_results: list[dict[str, Any]] = []
    primary_proxy = _run_muzero_stage(
        config=config,
        state_path=state_path,
        stage_prefix="muzero_proxy_resume",
        trial=dict(muzero_config.get("primary_proxy") or {}),
        mode="proxy_ga",
        resume_checkpoint_path=str(config.get("resume_checkpoint_path") or "").strip() or None,
        resume_step=int(config.get("resume_step") or 0) or None,
    )
    muzero_results.append(primary_proxy)

    if primary_proxy.get("is_weak"):
        fallback_trial = dict(muzero_config.get("fallback_proxy") or {})
        if fallback_trial:
            fallback_proxy = _run_muzero_stage(
                config=config,
                state_path=state_path,
                stage_prefix="muzero_proxy_fallback",
                trial=fallback_trial,
                mode="proxy_ga",
            )
            muzero_results.append(fallback_proxy)

    selected_proxy = _select_single_muzero_full_candidate(muzero_results)
    muzero_full_result = None
    if selected_proxy is not None:
        selected_trial_id = str(selected_proxy.get("trial_id") or "").strip()
        full_catalog = dict(muzero_config.get("full_catalog") or {})
        full_overrides = dict((full_catalog.get(selected_trial_id) or {}))
        if full_overrides:
            muzero_full_result = _run_muzero_stage(
                config=config,
                state_path=state_path,
                stage_prefix="muzero_full",
                trial={"trial_id": selected_trial_id, "runtime_overrides": full_overrides},
                mode="full",
            )

    smoke_trials = list(dreamer_config.get("smoke_trials") or [])
    dreamer_smoke_result = _run_dreamer_stage(
        config=config,
        state_path=state_path,
        stage_prefix="dreamer_smoke",
        trial=dict(smoke_trials[0] if smoke_trials else {}),
        mode="smoke",
    )
    smoke_summary = dict(dreamer_smoke_result.get("summary") or {})
    if (
        str(smoke_summary.get("terminal_status") or "").strip().lower() == "blocked"
        and str(smoke_summary.get("failure_mode") or "").strip().lower() == "insufficient_sample"
        and len(smoke_trials) > 1
    ):
        dreamer_smoke_result = _run_dreamer_stage(
            config=config,
            state_path=state_path,
            stage_prefix="dreamer_smoke_rescue",
            trial=dict(smoke_trials[1]),
            mode="smoke",
        )
        smoke_summary = dict(dreamer_smoke_result.get("summary") or {})

    dreamer_proxy_result = None
    dreamer_full_result = None
    if (
        str(smoke_summary.get("terminal_status") or "").strip().lower() == "completed"
        and str(smoke_summary.get("failure_mode") or "").strip().lower() in {"", "none"}
    ):
        proxy_trial = dict(dreamer_config.get("proxy_trial") or {})
        if proxy_trial:
            dreamer_proxy_result = _run_dreamer_stage(
                config=config,
                state_path=state_path,
                stage_prefix="dreamer_proxy",
                trial=proxy_trial,
                mode="proxy_ga",
            )
            proxy_summary = dict(dreamer_proxy_result.get("summary") or {})
            if str(proxy_summary.get("terminal_status") or "").strip().lower() == "completed":
                full_trial = dict(dreamer_config.get("full_trial") or {})
                if full_trial:
                    dreamer_full_result = _run_dreamer_stage(
                        config=config,
                        state_path=state_path,
                        stage_prefix="dreamer_full",
                        trial=full_trial,
                        mode="full",
                    )

    gnn_result = None
    if not bool(config.get("skip_gnn")) and config.get("gnn_request"):
        gnn_result = _run_gnn_refresh(dict(config.get("gnn_request") or {}))
        _persist_state(state_path, gnn_result=gnn_result)

    final_state = _persist_state(
        state_path,
        current_stage="completed",
        current_engine=None,
        current_run_id=None,
        status="completed",
        last_error=None,
    )
    final_state["selected_muzero_proxy"] = selected_proxy
    final_state["muzero_full_result"] = muzero_full_result
    final_state["dreamer_smoke_result"] = dreamer_smoke_result
    final_state["dreamer_proxy_result"] = dreamer_proxy_result
    final_state["dreamer_full_result"] = dreamer_full_result
    final_state["gnn_result"] = gnn_result
    _write_json(state_path, final_state)
    return final_state


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI du runner distant.

    Returns:
        argparse.Namespace: Arguments normalises.
    """

    parser = argparse.ArgumentParser(description="Execute la nuit Gold manuelle directement sur le serveur.")
    parser.add_argument("--config", required=True, help="Chemin du JSON de configuration genere cote local.")
    return parser.parse_args()


def main() -> int:
    """Point d'entree du runner distant.

    Returns:
        int: Code de sortie systeme.
    """

    args = parse_args()
    config_path = Path(str(args.config))
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    config = _read_json(config_path)
    if not config:
        raise SystemExit(f"Configuration Gold manuelle introuvable ou invalide: {config_path}")

    try:
        final_state = _run_manual_night(config)
    except Exception as exc:
        logger.exception("La nuit Gold manuelle a echoue: %s", exc)
        state_path = Path(str(config.get("state_path") or PROJECT_DIR / "data" / "checkpoints" / "gold_manual_state.json"))
        _persist_state(state_path, status="error", last_error=str(exc))
        return 1

    logger.info("Nuit Gold manuelle terminee: %s", final_state.get("status"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
