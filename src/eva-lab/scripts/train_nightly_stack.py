"""Orchestre la sequence nocturne complete des entrainements trading."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.gnn_registry import load_market_gnn_registry
from eva_lab.shadow_dataset import summarize_shadow_weighting
from eva_lab.training_notifier import send_nightly_summary, send_training_run_started
from eva_lab.training_status import (
    TERMINAL_SUMMARY_DIR,
    append_training_log,
    build_training_universe_summary,
    finalize_training_status,
    load_cpu_scheduler_state,
    load_latest_terminal_summary,
    load_terminal_summary,
    load_training_status,
    mark_skip_status,
    mark_step_finished,
    mark_step_running,
    persist_cpu_scheduler_state,
    reset_training_status,
    set_continuous_scheduler_state,
    set_training_runtime_state,
    set_training_weighting,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.nightly_training")

WORKDIR = Path(__file__).resolve().parents[1]
SUMMARY_PATH = WORKDIR / "data" / "checkpoints" / "nightly_training_summary.json"
LOCK_PATH = WORKDIR / "data" / "checkpoints" / "nightly_training.lock"
SHADOW_DIR = WORKDIR / "data" / "shadow_learning"
TRADING_REVIEW_PATH = WORKDIR / "data" / "checkpoints" / "trading_reviews" / "latest.json"
CANONICAL_TIMESCALE_SYMBOLS = [
    "EURUSD",
    "XAUUSD",
    "GBPUSD",
    "USDJPY",
    "US30.cash",
    "GER40.cash",
    "US500.cash",
]
SCALP_CORE_SYMBOLS = ["EURUSD", "XAUUSD", "GBPUSD", "USDJPY"]
SCALP_INDEX_SYMBOLS = ["US30.cash", "GER40.cash", "US500.cash"]
CANONICAL_GNN_TIMEFRAMES = ["M5", "H1", "D1"]
CONTINUOUS_TRAINING_TRIGGER = "continuous_auto_improve"
CONTINUOUS_SCHEDULER_MODE = "continuous_auto_improve"
DEFAULT_LIVE_SCALP_CHAMPION_ID = "gen_scalp_20260308_203907"


def _env_flag(name: str, default: bool) -> bool:
    """Interprete une variable d'environnement booleenne.

    Args:
        name (str): Nom de la variable a lire.
        default (bool): Valeur de repli si la variable est absente.

    Returns:
        bool: Valeur booleenne normalisee.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Lit un entier depuis l'environnement avec repli robuste.

    Args:
        name (str): Nom de la variable.
        default (int): Valeur de repli.

    Returns:
        int: Valeur entiere exploitable.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Valeur entiere invalide pour %s=%s. Repli=%s.", name, raw_value, default)
        return default


def _parse_iso_datetime(value: object) -> datetime | None:
    """Convertit une date ISO en objet ``datetime``.

    Args:
        value (object): Valeur brute a convertir.

    Returns:
        datetime | None: Date valide ou ``None`` si la conversion echoue.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _hours_since(timestamp: datetime | None) -> float | None:
    """Calcule l'age en heures d'un evenement.

    Args:
        timestamp (datetime | None): Date de reference.

    Returns:
        float | None: Age en heures ou ``None``.
    """
    if timestamp is None:
        return None
    current_time = datetime.now(timestamp.tzinfo) if timestamp.tzinfo is not None else datetime.now()
    return max((current_time - timestamp).total_seconds() / 3600.0, 0.0)


def _resolve_weighting_value(
    env_overrides: dict[str, object] | None,
    name: str,
    default: str,
) -> float:
    """Lit une ponderation en priorisant les overrides du cycle.

    Args:
        env_overrides (dict[str, object] | None): Overrides runtime du cycle.
        name (str): Nom de la variable cible.
        default (str): Valeur de repli.

    Returns:
        float: Valeur flottante normalisee.
    """
    if env_overrides and name in env_overrides:
        raw_value = str(env_overrides.get(name) or "").strip()
    else:
        raw_value = str(os.getenv(name, default) or default).strip()
    try:
        return max(float(raw_value or default), 0.0)
    except ValueError:
        logger.warning("Valeur de ponderation invalide pour %s=%s. Repli=%s.", name, raw_value, default)
        return max(float(default), 0.0)


def _build_shadow_weighting_profile(
    env_overrides: dict[str, object] | None = None,
) -> dict[str, float]:
    """Construit le profil de ponderation utilise par la nightly.

    Returns:
        dict[str, float]: Profil de pondération shadow exploitable.
    """
    return {
        "base_weight": _resolve_weighting_value(env_overrides, "TRAINING_EPISODE_WEIGHT_BASE", "1.0"),
        "winner_bonus": _resolve_weighting_value(
            env_overrides,
            "TRAINING_EPISODE_WEIGHT_WINNER_BONUS",
            "0.15",
        ),
        "loser_bonus": _resolve_weighting_value(
            env_overrides,
            "TRAINING_EPISODE_WEIGHT_LOSER_BONUS",
            "0.35",
        ),
        "nemesis_bonus": _resolve_weighting_value(
            env_overrides,
            "TRAINING_EPISODE_WEIGHT_NEMESIS_BONUS",
            "0.55",
        ),
        "risk_symbol_bonus": _resolve_weighting_value(
            env_overrides,
            "TRAINING_EPISODE_WEIGHT_RISK_BONUS",
            "0.25",
        ),
        "seed_candidate_bonus": _resolve_weighting_value(
            env_overrides,
            "TRAINING_EPISODE_WEIGHT_SEED_CANDIDATE_BONUS",
            "0.45",
        ),
    }


def _build_training_weighting_summary(
    learning_context: dict[str, object],
) -> dict[str, object]:
    """Resume la ponderation shadow pour la queue nightly.

    Args:
        learning_context (dict[str, object]): Guidance issue de la revue.

    Returns:
        dict[str, object]: Resume compact de ponderation exploitable par l'API.
    """
    allowed_symbols = list(learning_context.get("priority_symbols") or CANONICAL_TIMESCALE_SYMBOLS)
    weighting_profile = _build_shadow_weighting_profile(
        dict(learning_context.get("env_overrides") or {}),
    )
    summary = summarize_shadow_weighting(
        [SHADOW_DIR],
        winner_symbols=list(learning_context.get("winner_symbols") or []),
        risk_symbols=list(learning_context.get("risk_symbols") or []),
        seed_model_versions=list(learning_context.get("seed_model_versions") or []),
        seed_checkpoints=list(learning_context.get("seed_checkpoints") or []),
        allowed_symbols=allowed_symbols,
        max_episodes=_env_int("TRAINING_WEIGHTING_MAX_EPISODES", 250),
        weighting_profile=weighting_profile,
    )
    summary["allowed_symbols"] = list(allowed_symbols)
    summary["shadow_dirs"] = [str(SHADOW_DIR)]
    summary["gnn_focus_symbol"] = learning_context.get("gnn_focus_symbol")
    summary["seed_candidate_id"] = learning_context.get("seed_candidate_id")
    summary["seed_checkpoint"] = learning_context.get("seed_checkpoint")
    return summary


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convertit une valeur arbitraire en flottant robuste.

    Args:
        value (object): Valeur brute a convertir.
        default (float): Valeur de repli si la conversion echoue.

    Returns:
        float: Valeur flottante exploitable.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    """Convertit une valeur arbitraire en entier robuste.

    Args:
        value (object): Valeur brute a convertir.
        default (int): Valeur de repli si la conversion echoue.

    Returns:
        int: Valeur entiere exploitable.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _default_continuous_scheduler_state() -> dict[str, object]:
    """Construit l'etat minimal du scheduler continu.

    Returns:
        dict[str, object]: Etat de base du scheduler continu.
    """

    return {
        "mode": "disabled",
        "cycle_id": None,
        "cycle_index": 0,
        "seed_source": None,
        "seed_checkpoint": None,
        "seed_reason": None,
        "seed_candidate_id": None,
        "auto_promotion_policy": "strict_live_gate",
        "next_jobs": [],
        "current_focus": None,
        "last_completed_cycle": {},
        "degraded_horizons": {},
        "horizon_failures": {},
        "best_scalp_candidate": {},
        "best_for_mutation_candidate": {},
        "best_for_seed_candidate": {},
        "seed_reuse_block_reason": None,
        "improvement_vs_live": {},
        "scheduler_decision": {},
        "mutation_targets": {},
        "last_started_at": None,
        "last_finished_at": None,
        "last_dreamer_completed_at": None,
        "live_policy": "muzero_only",
        "gnn_policy": "weak_veto",
        "dreamer_policy": "offline_locked",
        "ensemble_prod_enabled": False,
    }


def _normalize_continuous_scheduler_state(payload: dict[str, object] | None) -> dict[str, object]:
    """Normalise l'etat persiste du scheduler continu.

    Args:
        payload (dict[str, object] | None): Etat precedemment persiste.

    Returns:
        dict[str, object]: Etat complet et borne.
    """

    normalized = _default_continuous_scheduler_state()
    normalized.update(dict(payload or {}))
    normalized["cycle_index"] = max(_safe_int(normalized.get("cycle_index"), 0), 0)
    normalized["next_jobs"] = list(normalized.get("next_jobs") or [])
    normalized["last_completed_cycle"] = dict(normalized.get("last_completed_cycle") or {})
    normalized["degraded_horizons"] = dict(normalized.get("degraded_horizons") or {})
    normalized["horizon_failures"] = dict(normalized.get("horizon_failures") or {})
    normalized["best_scalp_candidate"] = dict(normalized.get("best_scalp_candidate") or {})
    normalized["best_for_mutation_candidate"] = dict(
        normalized.get("best_for_mutation_candidate") or {}
    )
    normalized["best_for_seed_candidate"] = dict(normalized.get("best_for_seed_candidate") or {})
    normalized["improvement_vs_live"] = dict(normalized.get("improvement_vs_live") or {})
    normalized["scheduler_decision"] = dict(normalized.get("scheduler_decision") or {})
    normalized["mutation_targets"] = dict(normalized.get("mutation_targets") or {})
    return normalized


def _load_continuous_scheduler_state() -> dict[str, object]:
    """Charge et normalise l'etat du scheduler continu.

    Returns:
        dict[str, object]: Etat courant du scheduler continu.
    """

    return _normalize_continuous_scheduler_state(load_cpu_scheduler_state() or {})


def _persist_continuous_scheduler_state(state: dict[str, object]) -> dict[str, object]:
    """Persiste l'etat du scheduler continu dans les deux vues runtime.

    Args:
        state (dict[str, object]): Etat a persister.

    Returns:
        dict[str, object]: Etat normalise et persiste.
    """

    normalized = _normalize_continuous_scheduler_state(state)
    persist_cpu_scheduler_state(normalized)
    set_continuous_scheduler_state(normalized)
    return normalized


def _evaluate_gnn_refresh_policy(requested: bool) -> dict[str, object]:
    """Determine si le refresh nightly du GNN doit vraiment etre lance.

    Args:
        requested (bool): Intention brute issue de l'environnement.

    Returns:
        dict[str, object]: Decision normalisee avec fraicheur et raison.
    """
    threshold_hours = max(_env_int("TRAINING_GNN_REFRESH_AFTER_HOURS", 72), 1)
    registry = load_market_gnn_registry()
    trained_at = _parse_iso_datetime(registry.get("trained_at"))
    freshness_hours = _hours_since(trained_at)
    registry_status = str(registry.get("status") or "").strip().lower() or "unavailable"
    checkpoint_ready = bool(str(registry.get("checkpoint_path") or "").strip())
    refresh_required = requested
    reason = "requested"

    if not requested:
        refresh_required = False
        reason = "disabled_by_env"
    elif not checkpoint_ready:
        refresh_required = True
        reason = "missing_checkpoint"
    elif registry_status in {"stale", "unavailable", "draft"}:
        refresh_required = True
        reason = f"registry_{registry_status}"
    elif freshness_hours is None:
        refresh_required = True
        reason = "missing_freshness"
    elif freshness_hours <= threshold_hours:
        refresh_required = False
        reason = "already_fresh"
    else:
        refresh_required = True
        reason = "freshness_threshold_exceeded"

    return {
        "requested": requested,
        "scheduled": refresh_required,
        "reason": reason,
        "threshold_hours": threshold_hours,
        "freshness_hours": round(freshness_hours, 2) if freshness_hours is not None else None,
        "registry_status": registry_status,
        "checkpoint_ready": checkpoint_ready,
        "trained_at": registry.get("trained_at"),
        "champion_id": registry.get("version"),
    }


def _load_json_file(path: Path) -> dict[str, object] | None:
    """Charge un fichier JSON si present.

    Args:
        path (Path): Chemin cible.

    Returns:
        dict[str, object] | None: Charge utile JSON ou ``None``.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Lecture JSON impossible pour %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _load_previous_summary() -> dict[str, object] | None:
    """Charge le dernier resume nightly si disponible.

    Returns:
        dict[str, object] | None: Resume precedent ou ``None``.
    """
    return _load_json_file(SUMMARY_PATH)


def _load_lock_payload() -> dict[str, object] | None:
    """Charge le verrou actif si present.

    Returns:
        dict[str, object] | None: Charge utile du verrou ou ``None``.
    """
    return _load_json_file(LOCK_PATH)


def _write_lock_payload(payload: dict[str, object]) -> None:
    """Ecrit le verrou de run actif sur disque.

    Args:
        payload (dict[str, object]): Metadonnees du run actif.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_lock_stale(payload: dict[str, object] | None) -> bool:
    """Determine si un verrou parait obsolete.

    Args:
        payload (dict[str, object] | None): Verrou charge.

    Returns:
        bool: ``True`` si le verrou doit etre purge.
    """
    if not payload:
        return True
    started_at = _parse_iso_datetime(payload.get("started_at"))
    age_hours = _hours_since(started_at)
    if age_hours is None:
        return True
    stale_after_hours = _env_int("TRAINING_RUN_LOCK_MAX_AGE_HOURS", 18)
    return age_hours >= stale_after_hours


def _record_skip_event(reason: str, trigger: str, lock_payload: dict[str, object] | None) -> None:
    """Ajoute un evenement de skip dans le resume nightly sans ecraser un run actif.

    Args:
        reason (str): Raison explicite du skip.
        trigger (str): Origine du lancement (`cron`, `manual`, etc.).
        lock_payload (dict[str, object] | None): Verrou qui a provoque le skip.
    """
    summary = _load_previous_summary() or {}
    if not summary:
        summary = {
            "started_at": datetime.now().isoformat(),
            "status": "skipped",
            "strategy": "n/a",
            "reason": reason,
            "steps": [],
        }
    skip_event = {
        "trigger": trigger,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "lock": lock_payload or {},
    }
    skip_events = list(summary.get("skip_events") or [])
    skip_events.append(skip_event)
    summary["skip_events"] = skip_events[-20:]
    summary["last_skip_event"] = skip_event
    if summary.get("status") in {None, "skipped"}:
        summary["status"] = "skipped"
        summary["reason"] = reason
        summary["finished_at"] = datetime.now().isoformat()
    persist_summary(summary)
    mark_skip_status(reason, trigger, lock_payload)


def acquire_run_lock() -> tuple[bool, dict[str, object] | None]:
    """Acquiert le verrou nightly si possible.

    Returns:
        tuple[bool, dict[str, object] | None]: Etat d'acquisition et charge utile.
    """
    if _env_flag("NIGHTLY_RUN_LOCK_ALREADY_HELD", False):
        payload = _load_lock_payload() or {
            "mode": "external",
            "trigger": os.getenv("TRAINING_RUN_TRIGGER", "external"),
            "started_at": datetime.now().isoformat(),
        }
        return True, payload

    existing_lock = _load_lock_payload()
    if existing_lock and not _is_lock_stale(existing_lock):
        return False, existing_lock

    if LOCK_PATH.exists():
        try:
            LOCK_PATH.unlink()
            logger.warning("Verrou nightly obsolete supprime: %s", LOCK_PATH)
        except OSError as exc:
            logger.warning("Suppression du verrou nightly impossible: %s", exc)
            return False, existing_lock

    payload = {
        "pid": os.getpid(),
        "trigger": os.getenv("TRAINING_RUN_TRIGGER", "manual"),
        "started_at": datetime.now().isoformat(),
        "holder": "train_nightly_stack",
        "workdir": str(WORKDIR),
    }
    _write_lock_payload(payload)
    return True, payload


def release_run_lock(lock_payload: dict[str, object] | None) -> None:
    """Libere le verrou nightly si le processus en est proprietaire.

    Args:
        lock_payload (dict[str, object] | None): Charge utile capturee a l'acquisition.
    """
    if not lock_payload:
        return
    if lock_payload.get("mode") == "external":
        return
    current_lock = _load_lock_payload()
    if current_lock and current_lock.get("pid") != lock_payload.get("pid"):
        return
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Liberation du verrou nightly impossible: %s", exc)


def _collect_shadow_learning_stats() -> dict[str, object]:
    """Mesure le volume et la fraicheur des donnees Shadow Learning.

    Returns:
        dict[str, object]: Compteurs de lignes et date de modification.
    """
    latest_modified: datetime | None = None
    total_records = 0
    jsonl_files = sorted(SHADOW_DIR.rglob("*.jsonl")) if SHADOW_DIR.exists() else []

    for file_path in jsonl_files:
        try:
            modified_at = datetime.fromtimestamp(file_path.stat().st_mtime)
            latest_modified = (
                modified_at if latest_modified is None or modified_at > latest_modified else latest_modified
            )
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                total_records += sum(1 for line in handle if line.strip())
        except OSError as exc:
            logger.warning("Lecture Shadow Learning impossible pour %s: %s", file_path, exc)

    return {
        "records": total_records,
        "latest_modified_at": latest_modified.isoformat() if latest_modified else None,
        "files": len(jsonl_files),
    }


def _resolve_horizons() -> list[str]:
    """Retourne les horizons strategiques a traiter.

    Returns:
        list[str]: Horizons ordonnes et nettoyes.
    """
    return [
        item.strip().lower()
        for item in os.getenv("MUZERO_HORIZONS", "scalp,intraday,swing").split(",")
        if item.strip()
    ]


def _canonical_symbols_csv() -> str:
    """Retourne l'univers nightly canonique au format CSV.

    Returns:
        str: Liste CSV stable des symboles nightly.
    """

    return ",".join(CANONICAL_TIMESCALE_SYMBOLS)


def _unique_symbols(symbols: list[str]) -> list[str]:
    """Dedoublonne une liste de symboles sans perdre l'ordre.

    Args:
        symbols (list[str]): Symboles bruts a nettoyer.

    Returns:
        list[str]: Symboles uniques, trims et non vides.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized = str(symbol or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _build_scalp_core_priority_symbols(symbols: list[str] | None = None) -> list[str]:
    """Reordonne les symboles scalp avec le noyau FX/XAU en priorite.

    Args:
        symbols (list[str] | None): Liste source a normaliser.

    Returns:
        list[str]: Univers full-7 avec noyau prioritaire en tete.
    """
    base_symbols = _unique_symbols(list(symbols or CANONICAL_TIMESCALE_SYMBOLS))
    return _unique_symbols(SCALP_CORE_SYMBOLS + base_symbols + SCALP_INDEX_SYMBOLS)


def _candidate_review_paths() -> list[Path]:
    """Retourne les chemins plausibles de revue a charger.

    Returns:
        list[Path]: Liste dedoublonnee des chemins candidats.
    """
    paths: list[Path] = []
    for env_name in ("TRAINING_REVIEW_PATH", "BANKER_TRADING_REVIEW_PATH"):
        raw_value = str(os.getenv(env_name, "")).strip()
        if raw_value:
            paths.append(Path(raw_value).expanduser())
    paths.append(TRADING_REVIEW_PATH)
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def _build_banker_review_url() -> str:
    """Construit l'URL de repli pour lire la revue du banker.

    Returns:
        str: URL HTTP cible pour ``/trading/review/latest``.
    """
    explicit_url = str(os.getenv("BANKER_REVIEW_URL", "")).strip()
    if explicit_url:
        return explicit_url
    host = str(os.getenv("BANKER_API_HOST", "localhost")).strip() or "localhost"
    port = str(os.getenv("BANKER_API_PORT", "8100")).strip() or "8100"
    if host == "0.0.0.0":
        host = "localhost"
    return f"http://{host}:{port}/trading/review/latest"


def _load_latest_trading_review() -> dict[str, object] | None:
    """Charge la derniere revue journaliere disponible.

    La resolution suit trois etapes:
    1. chemin explicite via variables d'environnement ;
    2. chemin local standard dans ``data/checkpoints/trading_reviews`` ;
    3. endpoint HTTP du banker si les fichiers sont absents.

    Returns:
        dict[str, object] | None: Revue chargee ou ``None``.
    """
    for review_path in _candidate_review_paths():
        payload = _load_json_file(review_path)
        if payload:
            payload.setdefault("_review_source", "file")
            payload.setdefault("_review_path", str(review_path))
            return payload

    review_url = _build_banker_review_url()
    timeout_seconds = max(1, _env_int("TRAINING_REVIEW_HTTP_TIMEOUT_SECONDS", 3))
    try:
        with urllib_request.urlopen(review_url, timeout=timeout_seconds) as response:
            if getattr(response, "status", 200) >= 400:
                logger.warning("Lecture HTTP de la revue impossible: %s", review_url)
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, ValueError, OSError) as exc:
        logger.info("Aucune revue banker exploitable via %s: %s", review_url, exc)
        return None

    if isinstance(payload, dict):
        payload.setdefault("_review_source", "http")
        payload.setdefault("_review_path", review_url)
        return payload
    return None


def _collect_review_winner_symbols(review: dict[str, object]) -> list[str]:
    """Classe les meilleurs symboles du jour a partir de la revue.

    Args:
        review (dict[str, object]): Revue journaliere du banker.

    Returns:
        list[str]: Symboles gagnants classes par qualite.
    """
    ranked_rows: list[tuple[str, float, float, int]] = []
    for item in list(review.get("symbols") or []):
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip()
        if not symbol:
            continue
        net_profit = float(item.get("net_profit") or 0.0)
        win_rate = float(item.get("win_rate") or 0.0)
        closed_deals = int(item.get("closed_deals") or 0)
        if net_profit <= 0.0 or closed_deals <= 0:
            continue
        ranked_rows.append((symbol, net_profit, win_rate, closed_deals))
    ranked_rows.sort(key=lambda row: (row[1], row[2], row[3]), reverse=True)
    return [row[0] for row in ranked_rows]


def _collect_review_risk_symbols(review: dict[str, object]) -> list[str]:
    """Classe les symboles a risque a partir de la revue.

    Args:
        review (dict[str, object]): Revue journaliere du banker.

    Returns:
        list[str]: Symboles a surveiller ou penaliser en priorite.
    """
    ranked_rows: list[tuple[int, float, int, str]] = []
    risk_priority = {"quarantaine": 0, "alerte": 1, "surveillance": 2, "normal": 3}
    for item in list(review.get("symbol_risk_map") or []):
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip()
        if not symbol:
            continue
        risk_level = str(item.get("risk_level") or "normal").strip().lower()
        net_profit = float(item.get("net_profit") or 0.0)
        recent_losses = int(item.get("recent_losses_4h") or 0) + int(item.get("recent_events_12h") or 0)
        if risk_level == "normal" and net_profit >= 0.0 and recent_losses <= 0:
            continue
        ranked_rows.append((risk_priority.get(risk_level, 9), net_profit, -recent_losses, symbol))
    ranked_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return [row[3] for row in ranked_rows]


def _translate_mutation_priors_to_env(
    *,
    review: dict[str, object],
    winner_symbols: list[str],
    risk_symbols: list[str],
) -> dict[str, str]:
    """Traduit les priors de review en variables MuZero/GNN concretes.

    Args:
        review (dict[str, object]): Revue journaliere complete.
        winner_symbols (list[str]): Symboles positifs de reference.
        risk_symbols (list[str]): Symboles a risque a traiter en priorite.

    Returns:
        dict[str, str]: Variables d'environnement exploitables par la nightly.
    """
    overrides: dict[str, str] = {
        "TRAINING_REVIEW_AVAILABLE": "1",
        "TRAINING_REVIEW_GENERATED_AT": str(review.get("generated_at") or ""),
        "TRAINING_REVIEW_SOURCE": str(review.get("_review_source") or "unknown"),
        "TRAINING_WINNER_SYMBOLS": ",".join(winner_symbols),
        "TRAINING_RISK_SYMBOLS": ",".join(risk_symbols),
    }
    prioritized_symbols = _unique_symbols(winner_symbols + risk_symbols + CANONICAL_TIMESCALE_SYMBOLS)
    if prioritized_symbols:
        overrides["TRAINING_PRIORITY_SYMBOLS"] = ",".join(prioritized_symbols)

    priors = [item for item in list(review.get("mutation_priors") or []) if isinstance(item, dict)]
    for prior in priors:
        target = str(prior.get("target") or "").strip().lower()
        if target == "muzero_mechanics":
            overrides.update(
                {
                    "MUZERO_HOLD_STALE_PENALTY_AFTER_STEPS": "10",
                    "MUZERO_HOLD_STALE_PENALTY": "1.35",
                    "MUZERO_HOLD_TREND_PENALTY": "0.35",
                    "MUZERO_HOLD_RANGE_PENALTY": "0.20",
                    "MUZERO_SPLIT_MAX_SPLITS": "2",
                    "MUZERO_SPLIT_MIN_TRADE_RETURN": "0.0025",
                    "MUZERO_SPLIT_MIN_REALIZED_PCT": "0.0015",
                    "MUZERO_SPLIT_FAILURE_PENALTY": "0.75",
                    "MUZERO_CLOSE_WINNER_THRESHOLD": "0.0048",
                    "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD": "0.0088",
                    "MUZERO_CLOSE_TP_LIKE_THRESHOLD": "0.0062",
                }
            )
        elif target == "muzero_directional_balance":
            overrides.update(
                {
                    "MUZERO_ACTIVITY_INSUFFICIENT_ENTRIES_PENALTY": "2.50",
                    "MUZERO_DIRECTIONAL_MAX_IMBALANCE": "0.58",
                    "MUZERO_DIRECTIONAL_IMBALANCE_PENALTY": "1.75",
                }
            )
        elif target == "gold_live_filters":
            overrides.setdefault("TRAIN_GNN_FOCUS_SYMBOL", "XAUUSD")
        elif target == "gnn_consultatif":
            symbols = _unique_symbols([str(symbol) for symbol in list(prior.get("symbols") or [])])
            if symbols:
                overrides["TRAIN_GNN_FOCUS_SYMBOL"] = symbols[0]

    if winner_symbols:
        overrides.setdefault("MUZERO_CLOSE_WINNER_THRESHOLD", "0.0052")
        overrides.setdefault("MUZERO_CLOSE_STRONG_WINNER_THRESHOLD", "0.0094")
        overrides.setdefault("MUZERO_CLOSE_TP_LIKE_THRESHOLD", "0.0066")
    if risk_symbols and "TRAIN_GNN_FOCUS_SYMBOL" not in overrides:
        overrides["TRAIN_GNN_FOCUS_SYMBOL"] = risk_symbols[0]

    return overrides


def _build_review_learning_context(review: dict[str, object] | None) -> dict[str, object]:
    """Construit un contexte de guidance a partir de la revue journaliere.

    Args:
        review (dict[str, object] | None): Revue journaliere chargee.

    Returns:
        dict[str, object]: Contexte compact pour la queue nightly.
    """
    if not review:
        return {
            "loaded": False,
            "source": None,
            "path": None,
            "generated_at": None,
            "winner_symbols": [],
            "risk_symbols": [],
            "priority_symbols": list(CANONICAL_TIMESCALE_SYMBOLS),
            "gnn_focus_symbol": None,
            "env_overrides": {"TRAINING_REVIEW_AVAILABLE": "0"},
        }

    winner_symbols = _collect_review_winner_symbols(review)
    risk_symbols = _collect_review_risk_symbols(review)
    env_overrides = _translate_mutation_priors_to_env(
        review=review,
        winner_symbols=winner_symbols,
        risk_symbols=risk_symbols,
    )
    priority_symbols = _unique_symbols(
        winner_symbols
        + risk_symbols
        + list(
            review.get("live_universe", {}).get("symbols") or review.get("runtime", {}).get("cpu_live_symbols") or []
        )
        + CANONICAL_TIMESCALE_SYMBOLS
    )
    if priority_symbols:
        env_overrides["TRAINING_PRIORITY_SYMBOLS"] = ",".join(priority_symbols)
    gnn_focus_symbol = str(env_overrides.get("TRAIN_GNN_FOCUS_SYMBOL") or "").strip() or None
    return {
        "loaded": True,
        "source": review.get("_review_source"),
        "path": review.get("_review_path"),
        "generated_at": review.get("generated_at"),
        "winner_symbols": winner_symbols,
        "risk_symbols": risk_symbols,
        "priority_symbols": priority_symbols,
        "gnn_focus_symbol": gnn_focus_symbol,
        "env_overrides": env_overrides,
    }


def _iter_recent_terminal_summaries(
    *,
    engine: str,
    horizon: str,
    limit: int = 24,
) -> list[dict[str, object]]:
    """Charge les resumes terminaux recents pour un moteur/horizon.

    Args:
        engine (str): Moteur cible.
        horizon (str): Horizon cible.
        limit (int): Nombre maximal de resumes a charger.

    Returns:
        list[dict[str, object]]: Resumes tries du plus recent au plus ancien.
    """

    pattern = f"terminal_{engine}_{horizon}_*.json"
    payloads: list[dict[str, object]] = []
    for candidate in sorted(
        TERMINAL_SUMMARY_DIR.glob(pattern),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        payload = load_terminal_summary(path=candidate)
        if not isinstance(payload, dict):
            continue
        payloads.append(
            {
                **payload,
                "_summary_path": str(candidate),
                "_summary_modified_at": datetime.fromtimestamp(candidate.stat().st_mtime).isoformat(),
            }
        )
        if len(payloads) >= max(limit, 1):
            break
    return payloads


def _candidate_has_positive_metrics(metrics: dict[str, object] | None) -> bool:
    """Retourne vrai si un candidat porte deja un signal economique positif.

    Args:
        metrics (dict[str, object] | None): Metriques challenger.

    Returns:
        bool: ``True`` si les metriques sont toutes positives.
    """

    snapshot = dict(metrics or {})
    return (
        _safe_float(snapshot.get("profit_factor")) > 1.0
        and _safe_float(snapshot.get("return_pct")) > 0.0
        and _safe_float(snapshot.get("net_realized_pct")) > 0.0
    )


def _extract_live_scalp_reference(promoter: ChampionPromoter) -> dict[str, object]:
    """Assemble une reference compacte du live `MuZero scalp`.

    Args:
        promoter (ChampionPromoter): Promoteur central des champions.

    Returns:
        dict[str, object]: Reference live exploitable pour les comparaisons.
    """

    live_status = promoter.build_horizon_status("scalp")
    manifest = dict(live_status.get("manifest") or {})
    arena_report = dict(live_status.get("arena_report") or {})
    live_metrics = dict(
        promoter._extract_live_reference_metrics(
            manifest=manifest,
            arena_report=arena_report,
        )
        or {}
    )
    live_checkpoint_path, _ = promoter.resolve_live_checkpoint("scalp")
    live_mechanics = dict(live_metrics.get("metrics_by_position_mechanics") or {})
    return {
        "live_champion_id": live_status.get("live_champion_id") or DEFAULT_LIVE_SCALP_CHAMPION_ID,
        "live_checkpoint": str(live_checkpoint_path) if live_checkpoint_path else None,
        "metrics": live_metrics,
        "mechanics": live_mechanics,
        "feature_profile": live_status.get("feature_profile"),
        "family": live_status.get("family"),
    }


def _build_scalp_candidate_record(
    terminal_summary: dict[str, object],
    promoter: ChampionPromoter,
) -> dict[str, object]:
    """Transforme un resume terminal `scalp` en seed candidat exploitable.

    Args:
        terminal_summary (dict[str, object]): Resume terminal MuZero scalp.
        promoter (ChampionPromoter): Promoteur des checkpoints.

    Returns:
        dict[str, object]: Vue compacte du candidat.
    """

    metrics = dict(terminal_summary.get("metrics") or {})
    promotion_gate = dict(terminal_summary.get("promotion_gate") or {})
    latest_verdict = dict(terminal_summary.get("latest_verdict") or {})
    metrics_by_symbol = dict(terminal_summary.get("metrics_by_symbol") or metrics.get("metrics_by_symbol") or {})
    metrics["metrics_by_symbol"] = metrics_by_symbol
    candidate_id = str(terminal_summary.get("latest_candidate") or "").strip() or None
    checkpoint_path = None
    if candidate_id:
        candidate_path = promoter.weights_dir / f"{candidate_id}.pkl"
        if candidate_path.exists():
            checkpoint_path = candidate_path

    mechanics = dict(terminal_summary.get("metrics_by_position_mechanics") or {})
    return {
        "run_id": terminal_summary.get("run_id"),
        "summary_path": terminal_summary.get("_summary_path"),
        "summary_modified_at": terminal_summary.get("_summary_modified_at"),
        "candidate_id": candidate_id,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "seed_ready": checkpoint_path is not None,
        "arena_outcome": str(terminal_summary.get("arena_outcome") or "").upper() or None,
        "promotion_allowed": bool(promotion_gate.get("allowed", False)),
        "promotion_reason": latest_verdict.get("reason") or promotion_gate.get("reason"),
        "failure_mode": terminal_summary.get("failure_mode") or promotion_gate.get("failure_mode"),
        "metrics": metrics,
        "mechanics": mechanics,
        "metrics_by_symbol": metrics_by_symbol,
        "battle_report_present": bool(dict(terminal_summary.get("artifact_state") or {}).get("battle_report_present")),
        "positive_metrics": _candidate_has_positive_metrics(metrics),
    }


def _build_improvement_vs_live(
    candidate: dict[str, object],
    live_reference: dict[str, object],
) -> dict[str, object]:
    """Construit un delta compact entre seed candidat et live.

    Args:
        candidate (dict[str, object]): Seed candidat retenu.
        live_reference (dict[str, object]): Reference du champion live.

    Returns:
        dict[str, object]: Delta numerique et drapeaux d'amelioration.
    """

    candidate_metrics = dict(candidate.get("metrics") or {})
    live_metrics = dict(live_reference.get("metrics") or {})
    candidate_mechanics = dict(candidate.get("mechanics") or {})
    live_mechanics = dict(live_reference.get("mechanics") or {})
    return {
        "candidate_id": candidate.get("candidate_id"),
        "live_champion_id": live_reference.get("live_champion_id"),
        "profit_factor_delta": round(
            _safe_float(candidate_metrics.get("profit_factor"))
            - _safe_float(live_metrics.get("profit_factor")),
            6,
        ),
        "return_pct_delta": round(
            _safe_float(candidate_metrics.get("return_pct"))
            - _safe_float(live_metrics.get("return_pct")),
            6,
        ),
        "net_realized_pct_delta": round(
            _safe_float(candidate_metrics.get("net_realized_pct"))
            - _safe_float(live_metrics.get("net_realized_pct")),
            6,
        ),
        "close_quality_delta": round(
            _safe_float(candidate_mechanics.get("close_quality_score"))
            - _safe_float(live_mechanics.get("close_quality_score")),
            6,
        ),
        "hold_drag_delta": round(
            _safe_float(candidate_mechanics.get("hold_drag_score"))
            - _safe_float(live_mechanics.get("hold_drag_score")),
            6,
        ),
    }


def _build_scalp_mutation_targets(seed_candidate: dict[str, object]) -> dict[str, object]:
    """Traduit le meilleur seed `scalp` en mutations runtime ciblees.

    Args:
        seed_candidate (dict[str, object]): Seed candidat retenu.

    Returns:
        dict[str, object]: Cibles de mutation et surcharges d'environnement.
    """

    metrics = dict(seed_candidate.get("metrics") or {})
    mechanics = dict(seed_candidate.get("mechanics") or {})
    env_overrides: dict[str, str] = {}
    targets: dict[str, object] = {}
    directional_bias = str(metrics.get("directional_bias") or "").strip().lower()
    directional_imbalance = _safe_float(metrics.get("directional_imbalance"))
    long_entry_share = _safe_float(metrics.get("long_entry_share"))
    short_entry_share = _safe_float(metrics.get("short_entry_share"))
    close_quality_score = _safe_float(mechanics.get("close_quality_score"), _safe_float(metrics.get("close_quality_score")))
    hold_drag_score = _safe_float(mechanics.get("hold_drag_score"), _safe_float(metrics.get("hold_drag_score")))
    split_efficiency = _safe_float(mechanics.get("split_efficiency"), _safe_float(metrics.get("split_efficiency")))
    pyramid_efficiency = _safe_float(
        mechanics.get("pyramid_efficiency"),
        _safe_float(metrics.get("pyramid_efficiency")),
    )

    if directional_bias in {"sell_heavy", "buy_heavy"} or directional_imbalance > 0.60:
        env_overrides.update(
            {
                "MUZERO_ACTIVITY_MIN_ENTRIES": "4",
                "MUZERO_ACTIVITY_INSUFFICIENT_ENTRIES_PENALTY": "3.40",
                "MUZERO_DIRECTIONAL_MIN_ENTRY_SHARE": "0.28",
                "MUZERO_DIRECTIONAL_MAX_IMBALANCE": "0.46",
                "MUZERO_DIRECTIONAL_IMBALANCE_PENALTY": "3.20",
            }
        )
        targets["directional_balance"] = {
            "reason": directional_bias or "directional_imbalance",
            "objective": "reequilibrer_long_short",
            "long_entry_share": long_entry_share,
            "short_entry_share": short_entry_share,
            "directional_imbalance": directional_imbalance,
            "env_overrides": {
                key: env_overrides[key]
                for key in (
                    "MUZERO_ACTIVITY_MIN_ENTRIES",
                    "MUZERO_ACTIVITY_INSUFFICIENT_ENTRIES_PENALTY",
                    "MUZERO_DIRECTIONAL_MIN_ENTRY_SHARE",
                    "MUZERO_DIRECTIONAL_MAX_IMBALANCE",
                    "MUZERO_DIRECTIONAL_IMBALANCE_PENALTY",
                )
            },
        }

    if close_quality_score < 0.45:
        env_overrides.update(
            {
                "MUZERO_CLOSE_WINNER_THRESHOLD": "0.0032",
                "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD": "0.0064",
                "MUZERO_CLOSE_TP_LIKE_THRESHOLD": "0.0038",
                "MUZERO_REWARD_REALIZED_PNL_MULTIPLIER": "1.45",
                "MUZERO_REWARD_CLOSE_REALIZED_MULTIPLIER": "2.85",
                "MUZERO_SLBE_EXIT_BONUS": "1.90",
            }
        )
        targets["close_quality"] = {
            "reason": "close_quality_low",
            "value": close_quality_score,
            "env_overrides": {
                key: env_overrides[key]
                for key in (
                    "MUZERO_CLOSE_WINNER_THRESHOLD",
                    "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD",
                    "MUZERO_CLOSE_TP_LIKE_THRESHOLD",
                    "MUZERO_REWARD_REALIZED_PNL_MULTIPLIER",
                    "MUZERO_REWARD_CLOSE_REALIZED_MULTIPLIER",
                    "MUZERO_SLBE_EXIT_BONUS",
                )
            },
        }

    if hold_drag_score > 0.40:
        env_overrides.update(
            {
                "MUZERO_HOLD_STALE_PENALTY_AFTER_STEPS": "6",
                "MUZERO_HOLD_STALE_PENALTY": "2.10",
                "MUZERO_HOLD_TREND_PENALTY": "0.68",
                "MUZERO_HOLD_RANGE_PENALTY": "0.32",
                "MUZERO_REWARD_HOLD_DRAG_MULTIPLIER": "0.95",
            }
        )
        targets["hold_drag"] = {
            "reason": "hold_drag_high",
            "value": hold_drag_score,
            "env_overrides": {
                key: env_overrides[key]
                for key in (
                    "MUZERO_HOLD_STALE_PENALTY_AFTER_STEPS",
                    "MUZERO_HOLD_STALE_PENALTY",
                    "MUZERO_HOLD_TREND_PENALTY",
                    "MUZERO_HOLD_RANGE_PENALTY",
                    "MUZERO_REWARD_HOLD_DRAG_MULTIPLIER",
                )
            },
        }

    if split_efficiency <= 0.0 or pyramid_efficiency <= 0.0:
        env_overrides.update(
            {
                "MUZERO_SPLIT_MAX_SPLITS": "0",
                "MUZERO_SPLIT_MIN_TRADE_RETURN": "0.0060",
                "MUZERO_SPLIT_MIN_REALIZED_PCT": "0.0045",
                "MUZERO_SPLIT_FAILURE_PENALTY": "1.35",
                "MUZERO_PYRAMID_MAX_ADDITIONS": "0",
                "MUZERO_REWARD_PYRAMID_FAILURE_PENALTY": "0.55",
                "MUZERO_REWARD_PYRAMID_NEGATIVE_EXIT_PENALTY": "0.85",
            }
        )
        targets["split_pyramid_efficiency"] = {
            "reason": "efficiency_non_positive",
            "split_efficiency": split_efficiency,
            "pyramid_efficiency": pyramid_efficiency,
            "env_overrides": {
                key: env_overrides[key]
                for key in (
                    "MUZERO_SPLIT_MAX_SPLITS",
                    "MUZERO_SPLIT_MIN_TRADE_RETURN",
                    "MUZERO_SPLIT_MIN_REALIZED_PCT",
                    "MUZERO_SPLIT_FAILURE_PENALTY",
                    "MUZERO_PYRAMID_MAX_ADDITIONS",
                    "MUZERO_REWARD_PYRAMID_FAILURE_PENALTY",
                    "MUZERO_REWARD_PYRAMID_NEGATIVE_EXIT_PENALTY",
                )
            },
        }

    return {
        "seed_candidate_id": seed_candidate.get("candidate_id"),
        "env_overrides": env_overrides,
        "targets": targets,
    }


def _summarize_promotion_decision(terminal_summary: dict[str, object] | None) -> dict[str, object]:
    """Resout une decision de promotion compacte a partir d'un resume terminal.

    Args:
        terminal_summary (dict[str, object] | None): Resume terminal cible.

    Returns:
        dict[str, object]: Verdict compact de promotion.
    """

    snapshot = dict(terminal_summary or {})
    promotion_gate = dict(snapshot.get("promotion_gate") or {})
    latest_verdict = dict(snapshot.get("latest_verdict") or {})
    return {
        "candidate_id": snapshot.get("latest_candidate"),
        "arena_outcome": snapshot.get("arena_outcome"),
        "allowed": bool(promotion_gate.get("allowed", False)),
        "status": latest_verdict.get("status") or ("promoted" if promotion_gate.get("allowed") else "blocked"),
        "reason": latest_verdict.get("reason") or promotion_gate.get("reason"),
        "failure_mode": latest_verdict.get("failure_mode") or snapshot.get("failure_mode"),
    }


def _merge_learning_context_with_scheduler(
    learning_context: dict[str, object],
    *,
    seed_candidate: dict[str, object],
    mutation_targets: dict[str, object],
    seed_reuse_block_reason: str | None = None,
) -> dict[str, object]:
    """Fusionne le seed et ses mutations dans le contexte nightly.

    Args:
        learning_context (dict[str, object]): Contexte de review existant.
        seed_candidate (dict[str, object]): Seed retenu pour le scalp.
        mutation_targets (dict[str, object]): Mutations ciblees derivees du seed.
        seed_reuse_block_reason (str | None): Raison de blocage d'un reseed direct.

    Returns:
        dict[str, object]: Nouveau contexte nightly enrichi.
    """

    merged = dict(learning_context or {})
    env_overrides = dict(merged.get("env_overrides") or {})
    env_overrides.update(dict(mutation_targets.get("env_overrides") or {}))
    priority_symbols = list(merged.get("priority_symbols") or CANONICAL_TIMESCALE_SYMBOLS)
    priority_symbols = _unique_symbols(priority_symbols) or list(CANONICAL_TIMESCALE_SYMBOLS)
    if seed_reuse_block_reason == "core_symbol_balance":
        priority_symbols = _build_scalp_core_priority_symbols(priority_symbols)
        env_overrides["TRAINING_EPISODE_WEIGHT_SEED_CANDIDATE_BONUS"] = "0.20"
    merged["priority_symbols"] = priority_symbols
    env_overrides["TRAINING_PRIORITY_SYMBOLS"] = ",".join(priority_symbols)
    env_overrides["TRAINING_FOCUS_SYMBOLS"] = _canonical_symbols_csv()
    seed_candidate_id = str(seed_candidate.get("candidate_id") or "").strip()
    seed_checkpoint = str(seed_candidate.get("checkpoint_path") or "").strip()
    if seed_candidate_id:
        env_overrides["TRAINING_SEED_CANDIDATE_ID"] = seed_candidate_id
        merged["seed_candidate_id"] = seed_candidate_id
        merged["seed_model_versions"] = [seed_candidate_id]
    if seed_checkpoint:
        env_overrides["TRAINING_SEED_CHECKPOINTS"] = ",".join(
            _unique_symbols([seed_checkpoint, Path(seed_checkpoint).name])
        )
        merged["seed_checkpoint"] = seed_checkpoint
        merged["seed_checkpoints"] = [seed_checkpoint, Path(seed_checkpoint).name]
    merged["seed_source"] = seed_candidate.get("source")
    merged["seed_reason"] = seed_candidate.get("reason")
    if merged.get("seed_source"):
        env_overrides["TRAINING_SEED_SOURCE"] = str(merged.get("seed_source"))
    if merged.get("seed_reason"):
        env_overrides["TRAINING_SEED_REASON"] = str(merged.get("seed_reason"))
    merged["seed_reuse_block_reason"] = seed_reuse_block_reason
    if seed_reuse_block_reason:
        env_overrides["TRAINING_SEED_REUSE_BLOCK_REASON"] = str(seed_reuse_block_reason)
    merged["env_overrides"] = env_overrides
    return merged


def _select_best_scalp_seed(
    promoter: ChampionPromoter,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    str | None,
]:
    """Selectionne separement le meilleur candidat de mutation et le seed reel.

    Args:
        promoter (ChampionPromoter): Promoteur des champions MuZero.

    Returns:
        tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], str | None]:
            Seed reel, meilleur candidat pour mutation, meilleur candidat seedable,
            reference live et raison de blocage du reseed direct.
    """

    def _decorate_candidate(
        candidate: dict[str, object],
        *,
        source: str,
        reason: str,
    ) -> dict[str, object]:
        decorated = dict(candidate)
        decorated["source"] = source
        decorated["reason"] = reason
        seed_gate = promoter.evaluate_scalp_seed_gate(decorated.get("metrics") or {})
        decorated["seed_gate"] = seed_gate
        decorated["seed_gate_allowed"] = bool(seed_gate.get("allowed", False))
        decorated["seed_gate_reason"] = seed_gate.get("primary_reason")
        return decorated

    live_reference = _extract_live_scalp_reference(promoter)
    summaries = _iter_recent_terminal_summaries(engine="muzero", horizon="scalp")
    best_for_mutation: dict[str, object] | None = None
    best_for_seed: dict[str, object] | None = None
    fallback_positive_mutation: dict[str, object] | None = None
    fallback_positive_seed: dict[str, object] | None = None

    for summary in summaries:
        candidate = _build_scalp_candidate_record(summary, promoter)
        if not candidate.get("candidate_id") or not candidate.get("seed_ready"):
            continue
        if candidate.get("arena_outcome") == "VICTORY" and candidate.get("positive_metrics"):
            decorated = _decorate_candidate(
                candidate,
                source="scalp_victory_candidate",
                reason="dernier_scalp_victorieux_positif",
            )
            if best_for_mutation is None:
                best_for_mutation = decorated
            if decorated.get("seed_gate_allowed") and best_for_seed is None:
                best_for_seed = dict(decorated)
            if best_for_mutation is not None and best_for_seed is not None:
                break
            continue
        if candidate.get("positive_metrics"):
            decorated = _decorate_candidate(
                candidate,
                source="scalp_positive_candidate",
                reason="dernier_scalp_positif_avec_resume_terminal",
            )
            if fallback_positive_mutation is None:
                fallback_positive_mutation = decorated
            if decorated.get("seed_gate_allowed") and fallback_positive_seed is None:
                fallback_positive_seed = dict(decorated)

    if best_for_mutation is None:
        best_for_mutation = fallback_positive_mutation
    if best_for_seed is None:
        best_for_seed = fallback_positive_seed

    seed_reuse_block_reason: str | None = None
    if best_for_mutation is not None and not best_for_seed:
        seed_reuse_block_reason = str(best_for_mutation.get("seed_gate_reason") or "").strip() or None
    elif (
        best_for_mutation is not None
        and best_for_seed is not None
        and best_for_mutation.get("candidate_id") != best_for_seed.get("candidate_id")
    ):
        seed_reuse_block_reason = str(best_for_mutation.get("seed_gate_reason") or "").strip() or None

    if best_for_seed is not None:
        seed = dict(best_for_seed)
    else:
        fallback_reason = (
            f"seed_gate_blocked_{seed_reuse_block_reason}"
            if seed_reuse_block_reason
            else "aucun_challenger_scalp_seedable"
        )
        seed = {
            "source": "live_champion_fallback",
            "reason": fallback_reason,
            "candidate_id": live_reference.get("live_champion_id") or DEFAULT_LIVE_SCALP_CHAMPION_ID,
            "checkpoint_path": live_reference.get("live_checkpoint"),
            "seed_ready": bool(live_reference.get("live_checkpoint")),
            "metrics": dict(live_reference.get("metrics") or {}),
            "mechanics": dict(live_reference.get("mechanics") or {}),
            "metrics_by_symbol": dict((live_reference.get("metrics") or {}).get("metrics_by_symbol") or {}),
        }

    return (
        seed,
        dict(best_for_mutation or {}),
        dict(best_for_seed or {}),
        live_reference,
        seed_reuse_block_reason,
    )


def _plan_continuous_scheduler(
    *,
    scheduler_state: dict[str, object],
    promoter: ChampionPromoter,
    run_gnn_requested: bool,
    run_muzero_requested: bool,
    run_dreamer_requested: bool,
    gnn_refresh_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    """Construit la decision de cycle du scheduler continu.

    Args:
        scheduler_state (dict[str, object]): Etat precedemment persiste.
        promoter (ChampionPromoter): Promoteur des champions.
        run_gnn_requested (bool): Demande brute de refresh GNN.
        run_muzero_requested (bool): Demande brute de file MuZero.
        run_dreamer_requested (bool): Demande brute de Dreamer offline.
        gnn_refresh_policy (dict[str, object] | None): Politique GNN deja resolue.

    Returns:
        dict[str, object]: Decision complete du cycle courant.
    """

    continuous_mode = _env_flag("TRAINING_CONTINUOUS_MODE", False)
    previous_state = _normalize_continuous_scheduler_state(scheduler_state)
    if not continuous_mode:
        return {
            "mode": "nightly_once",
            "cycle_id": None,
            "cycle_index": previous_state.get("cycle_index", 0),
            "current_focus": "default",
            "degraded_horizons": {},
            "next_horizons": _resolve_horizons(),
            "run_gnn": run_gnn_requested,
            "run_muzero": run_muzero_requested,
            "run_dreamer": run_dreamer_requested,
            "seed_candidate": {},
            "best_scalp_candidate": {},
            "best_for_mutation_candidate": {},
            "best_for_seed_candidate": {},
            "seed_reuse_block_reason": None,
            "improvement_vs_live": {},
            "scheduler_decision": {
                "mode": "nightly_once",
                "reason": "continuous_mode_disabled",
            },
            "mutation_targets": {},
        }

    current_cycle_index = max(_safe_int(previous_state.get("cycle_index"), 0), 0) + 1
    cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_cycle_index:04d}"
    cleaned_degraded: dict[str, object] = {}
    for horizon, payload in dict(previous_state.get("degraded_horizons") or {}).items():
        degraded_until_cycle = _safe_int(dict(payload).get("degraded_until_cycle"), 0)
        if degraded_until_cycle >= current_cycle_index:
            cleaned_degraded[horizon] = dict(payload)

    (
        seed_candidate,
        best_for_mutation_candidate,
        best_for_seed_candidate,
        live_reference,
        seed_reuse_block_reason,
    ) = _select_best_scalp_seed(promoter)
    comparison_candidate = dict(best_for_mutation_candidate or seed_candidate)
    improvement_vs_live = _build_improvement_vs_live(comparison_candidate, live_reference)
    mutation_targets = _build_scalp_mutation_targets(comparison_candidate)
    last_completed_cycle = dict(previous_state.get("last_completed_cycle") or {})
    last_scalp_decision = dict(last_completed_cycle.get("scalp_result") or {})
    scalp_gate_blocked = (
        str(last_scalp_decision.get("arena_outcome") or "").upper() == "VICTORY"
        and not bool(last_scalp_decision.get("allowed", False))
    )
    cycle_slot = ((current_cycle_index - 1) % 3) + 1
    current_focus = "scalp_only" if scalp_gate_blocked or cycle_slot in {1, 2} else "full_muzero"
    next_horizons = ["scalp"] if current_focus == "scalp_only" else ["scalp", "intraday", "swing"]
    next_horizons = [
        horizon
        for horizon in next_horizons
        if horizon not in cleaned_degraded
    ]
    last_dreamer_completed_at = _parse_iso_datetime(previous_state.get("last_dreamer_completed_at"))
    dreamer_interval_hours = max(
        _env_int("TRAINING_CONTINUOUS_DREAMER_MIN_INTERVAL_HOURS", 24),
        1,
    )
    dreamer_elapsed_hours = _hours_since(last_dreamer_completed_at)
    allow_dreamer = (
        run_dreamer_requested
        and current_focus == "full_muzero"
        and (
            dreamer_elapsed_hours is None
            or dreamer_elapsed_hours >= float(dreamer_interval_hours)
        )
    )
    scheduler_decision = {
        "mode": CONTINUOUS_SCHEDULER_MODE,
        "cycle_slot": cycle_slot,
        "current_focus": current_focus,
        "seed_source": seed_candidate.get("source"),
        "seed_reason": seed_candidate.get("reason"),
        "seed_checkpoint": seed_candidate.get("checkpoint_path"),
        "seed_candidate_id": seed_candidate.get("candidate_id"),
        "best_for_mutation_candidate_id": comparison_candidate.get("candidate_id"),
        "best_for_seed_candidate_id": best_for_seed_candidate.get("candidate_id"),
        "seed_reuse_block_reason": seed_reuse_block_reason,
        "degraded_horizons": cleaned_degraded,
        "gnn_requested": run_gnn_requested,
        "gnn_scheduled": bool((gnn_refresh_policy or {}).get("scheduled", run_gnn_requested)),
        "dreamer_requested": run_dreamer_requested,
        "dreamer_scheduled": allow_dreamer,
        "reason": "scalp_gate_blocked" if scalp_gate_blocked else f"cycle_slot_{cycle_slot}",
    }
    next_jobs = []
    if bool((gnn_refresh_policy or {}).get("scheduled", run_gnn_requested)):
        next_jobs.append("gnn")
    next_jobs.extend(f"muzero_{horizon}" for horizon in next_horizons)
    if allow_dreamer:
        next_jobs.append("dreamer_offline")

    return {
        "mode": CONTINUOUS_SCHEDULER_MODE,
        "cycle_id": cycle_id,
        "cycle_index": current_cycle_index,
        "current_focus": current_focus,
        "degraded_horizons": cleaned_degraded,
        "next_horizons": next_horizons,
        "next_jobs": next_jobs,
        "run_gnn": bool((gnn_refresh_policy or {}).get("scheduled", run_gnn_requested)),
        "run_muzero": run_muzero_requested and bool(next_horizons),
        "run_dreamer": allow_dreamer,
        "seed_candidate": seed_candidate,
        "best_scalp_candidate": comparison_candidate,
        "best_for_mutation_candidate": comparison_candidate,
        "best_for_seed_candidate": best_for_seed_candidate,
        "seed_reuse_block_reason": seed_reuse_block_reason,
        "live_reference": live_reference,
        "improvement_vs_live": improvement_vs_live,
        "scheduler_decision": scheduler_decision,
        "mutation_targets": mutation_targets,
    }


def _update_scheduler_after_cycle_success(
    *,
    scheduler_state: dict[str, object],
    scheduler_plan: dict[str, object],
    step_results: list[dict[str, object]],
) -> dict[str, object]:
    """Met a jour le scheduler apres un cycle termine sans erreur infra.

    Args:
        scheduler_state (dict[str, object]): Etat precedent.
        scheduler_plan (dict[str, object]): Decision du cycle courant.
        step_results (list[dict[str, object]]): Resultats de toutes les etapes.

    Returns:
        dict[str, object]: Nouvel etat du scheduler.
    """

    updated = _normalize_continuous_scheduler_state(scheduler_state)
    updated["mode"] = CONTINUOUS_SCHEDULER_MODE
    updated["cycle_id"] = scheduler_plan.get("cycle_id")
    updated["cycle_index"] = scheduler_plan.get("cycle_index")
    updated["current_focus"] = scheduler_plan.get("current_focus")
    updated["seed_source"] = scheduler_plan.get("seed_candidate", {}).get("source")
    updated["seed_checkpoint"] = scheduler_plan.get("seed_candidate", {}).get("checkpoint_path")
    updated["seed_reason"] = scheduler_plan.get("seed_candidate", {}).get("reason")
    updated["seed_candidate_id"] = scheduler_plan.get("seed_candidate", {}).get("candidate_id")
    updated["best_scalp_candidate"] = dict(scheduler_plan.get("best_scalp_candidate") or {})
    updated["best_for_mutation_candidate"] = dict(
        scheduler_plan.get("best_for_mutation_candidate") or {}
    )
    updated["best_for_seed_candidate"] = dict(
        scheduler_plan.get("best_for_seed_candidate") or {}
    )
    updated["seed_reuse_block_reason"] = scheduler_plan.get("seed_reuse_block_reason")
    updated["improvement_vs_live"] = dict(scheduler_plan.get("improvement_vs_live") or {})
    updated["mutation_targets"] = dict((scheduler_plan.get("mutation_targets") or {}).get("targets") or {})
    updated["scheduler_decision"] = dict(scheduler_plan.get("scheduler_decision") or {})
    updated["next_jobs"] = []
    updated["last_started_at"] = updated.get("last_started_at") or datetime.now().isoformat()
    updated["last_finished_at"] = datetime.now().isoformat()
    step_map = {str(result.get("step_name") or ""): dict(result) for result in step_results}
    scalp_result = dict((step_map.get("muzero_scalp") or {}).get("promotion_decision") or {})
    updated["last_completed_cycle"] = {
        "cycle_id": scheduler_plan.get("cycle_id"),
        "cycle_index": scheduler_plan.get("cycle_index"),
        "current_focus": scheduler_plan.get("current_focus"),
        "completed_at": updated["last_finished_at"],
        "steps": [dict(result) for result in step_results],
        "scalp_result": scalp_result,
    }
    horizon_failures = dict(updated.get("horizon_failures") or {})
    for horizon in ("scalp", "intraday", "swing"):
        if f"muzero_{horizon}" in step_map:
            horizon_failures.pop(horizon, None)
    updated["horizon_failures"] = horizon_failures
    updated["degraded_horizons"] = {
        horizon: payload
        for horizon, payload in dict(updated.get("degraded_horizons") or {}).items()
        if _safe_int(dict(payload).get("degraded_until_cycle"), 0) > _safe_int(updated.get("cycle_index"), 0)
    }
    if "dreamer_offline" in step_map:
        updated["last_dreamer_completed_at"] = updated["last_finished_at"]
    return updated


def _update_scheduler_after_cycle_failure(
    *,
    scheduler_state: dict[str, object],
    scheduler_plan: dict[str, object],
    failed_job: dict[str, object] | None,
    failed_phase: str | None,
    exception_message: str,
) -> dict[str, object]:
    """Met a jour le scheduler apres un echec de cycle.

    Args:
        scheduler_state (dict[str, object]): Etat precedent.
        scheduler_plan (dict[str, object]): Decision du cycle courant.
        failed_job (dict[str, object] | None): Job ayant echoue.
        failed_phase (str | None): Sous-phase runtime ayant echoue.
        exception_message (str): Message d'erreur compact.

    Returns:
        dict[str, object]: Nouvel etat du scheduler.
    """

    updated = _normalize_continuous_scheduler_state(scheduler_state)
    updated["mode"] = CONTINUOUS_SCHEDULER_MODE
    updated["cycle_id"] = scheduler_plan.get("cycle_id")
    updated["cycle_index"] = scheduler_plan.get("cycle_index")
    updated["current_focus"] = scheduler_plan.get("current_focus")
    updated["seed_source"] = scheduler_plan.get("seed_candidate", {}).get("source")
    updated["seed_checkpoint"] = scheduler_plan.get("seed_candidate", {}).get("checkpoint_path")
    updated["seed_reason"] = scheduler_plan.get("seed_candidate", {}).get("reason")
    updated["seed_candidate_id"] = scheduler_plan.get("seed_candidate", {}).get("candidate_id")
    updated["best_scalp_candidate"] = dict(scheduler_plan.get("best_scalp_candidate") or {})
    updated["best_for_mutation_candidate"] = dict(
        scheduler_plan.get("best_for_mutation_candidate") or {}
    )
    updated["best_for_seed_candidate"] = dict(
        scheduler_plan.get("best_for_seed_candidate") or {}
    )
    updated["seed_reuse_block_reason"] = scheduler_plan.get("seed_reuse_block_reason")
    updated["improvement_vs_live"] = dict(scheduler_plan.get("improvement_vs_live") or {})
    updated["mutation_targets"] = dict((scheduler_plan.get("mutation_targets") or {}).get("targets") or {})
    updated["scheduler_decision"] = dict(scheduler_plan.get("scheduler_decision") or {})
    updated["next_jobs"] = []
    updated["last_started_at"] = updated.get("last_started_at") or datetime.now().isoformat()
    updated["last_finished_at"] = datetime.now().isoformat()
    updated["last_completed_cycle"] = {
        "cycle_id": scheduler_plan.get("cycle_id"),
        "cycle_index": scheduler_plan.get("cycle_index"),
        "current_focus": scheduler_plan.get("current_focus"),
        "completed_at": updated["last_finished_at"],
        "failed_job": dict(failed_job or {}),
        "failed_phase": failed_phase,
        "exception_message": exception_message,
    }
    engine = str((failed_job or {}).get("engine") or "").strip().lower()
    horizon = str((failed_job or {}).get("horizon") or "").strip().lower()
    if engine == "muzero" and horizon:
        horizon_failures = dict(updated.get("horizon_failures") or {})
        previous_failure = dict(horizon_failures.get(horizon) or {})
        repeat_count = (
            _safe_int(previous_failure.get("repeat_count"), 0) + 1
            if str(previous_failure.get("failed_phase") or "") == str(failed_phase or "")
            else 1
        )
        horizon_failures[horizon] = {
            "failed_phase": failed_phase,
            "repeat_count": repeat_count,
            "updated_at": updated["last_finished_at"],
            "last_exception_message": exception_message,
        }
        updated["horizon_failures"] = horizon_failures
        degraded_horizons = dict(updated.get("degraded_horizons") or {})
        if repeat_count >= 2:
            degraded_horizons[horizon] = {
                "failed_phase": failed_phase,
                "repeat_count": repeat_count,
                "reason": "same_failed_phase_repeated",
                "degraded_until_cycle": _safe_int(updated.get("cycle_index"), 0) + 1,
                "updated_at": updated["last_finished_at"],
            }
        updated["degraded_horizons"] = degraded_horizons
    return updated


def _filter_env_overrides(env_overrides: dict[str, str], prefixes: tuple[str, ...]) -> dict[str, str]:
    """Filtre un dictionnaire d'environnement par prefixes.

    Args:
        env_overrides (dict[str, str]): Variables candidates.
        prefixes (tuple[str, ...]): Prefixes autorises.

    Returns:
        dict[str, str]: Sous-ensemble filtre.
    """
    return {
        key: value
        for key, value in env_overrides.items()
        if any(key.startswith(prefix) for prefix in prefixes)
    }


def _build_champion_snapshot(promoter: ChampionPromoter, horizons: list[str]) -> dict[str, object]:
    """Assemble l'etat des champions par horizon.

    Args:
        promoter (ChampionPromoter): Promoteur central des champions.
        horizons (list[str]): Horizons a inspecter.

    Returns:
        dict[str, object]: Snapshot d'etat par horizon.
    """
    snapshot: dict[str, object] = {}
    for horizon in horizons:
        status = promoter.build_horizon_status(horizon)
        promotion_gate = status.get("promotion_gate", {}) or {}
        live_checkpoint = status.get("live_checkpoint", {}) or {}
        snapshot[horizon] = {
            "allowed": bool(promotion_gate.get("allowed", False)),
            "reason": promotion_gate.get("reason", "unknown"),
            "selection": status.get("selection"),
            "champion_id": status.get("champion_id"),
            "live_checkpoint_exists": bool(live_checkpoint.get("exists", False)),
            "live_checkpoint_modified_at": live_checkpoint.get("modified_at"),
            "live_universe_count": ((status.get("live_universe") or {}).get("count")) or 0,
        }
    return snapshot


def decide_training_strategy() -> dict[str, object]:
    """Choisit automatiquement la strategie nightly.

    La decision suit trois modes:
    - ``skip``: aucun retrain utile ce soir.
    - ``refresh``: retrain de maintien sur echantillon reduit.
    - ``research``: run massif pour chercher un nouveau champion.

    Returns:
        dict[str, object]: Strategie retenue, raison et contexte de decision.
    """
    automation_mode = os.getenv("TRAINING_AUTOMATION_MODE", "smart").strip().lower()
    previous_summary = _load_previous_summary() or {}
    previous_finished_at = _parse_iso_datetime(previous_summary.get("finished_at"))
    shadow_stats = _collect_shadow_learning_stats()
    shadow_latest = _parse_iso_datetime(shadow_stats.get("latest_modified_at"))
    shadow_age_hours = _hours_since(shadow_latest)
    new_shadow_since_last = bool(
        previous_finished_at is None
        or (shadow_latest is not None and shadow_latest > previous_finished_at)
    )
    min_shadow_records = _env_int("TRAINING_MIN_SHADOW_RECORDS", 25)
    refresh_after_hours = _env_int("TRAINING_REFRESH_AFTER_HOURS", 24)
    max_champion_age_hours = _env_int("TRAINING_MAX_CHAMPION_AGE_HOURS", 72)
    promoter = ChampionPromoter(
        weights_dir=str(WORKDIR / "data" / "muzero" / "weights"),
        results_dir=str(WORKDIR / "data" / "muzero" / "results"),
    )
    horizons = _resolve_horizons()
    champion_snapshot = _build_champion_snapshot(promoter, horizons)
    deployable_horizons = [
        horizon
        for horizon, status in champion_snapshot.items()
        if isinstance(status, dict)
        and status.get("allowed")
        and status.get("live_checkpoint_exists")
    ]
    checkpoint_dates = [
        _parse_iso_datetime((status or {}).get("live_checkpoint_modified_at"))
        for status in champion_snapshot.values()
        if isinstance(status, dict)
    ]
    checkpoint_dates = [value for value in checkpoint_dates if value is not None]
    oldest_live_hours = max((_hours_since(value) or 0.0) for value in checkpoint_dates) if checkpoint_dates else None

    decision: dict[str, object] = {
        "mode": automation_mode,
        "strategy": "research",
        "reason": "default_research",
        "shadow_stats": shadow_stats,
        "champion_snapshot": champion_snapshot,
        "deployable_horizons": deployable_horizons,
        "previous_finished_at": previous_summary.get("finished_at"),
        "oldest_live_hours": oldest_live_hours,
    }

    if automation_mode in {"always", "force_research"}:
        decision["reason"] = "forced_research"
        return decision

    if automation_mode == "disabled":
        decision["strategy"] = "skip"
        decision["reason"] = "automation_disabled"
        return decision

    if not deployable_horizons:
        decision["strategy"] = "research"
        decision["reason"] = "no_deployable_champion"
        return decision

    if oldest_live_hours is not None and oldest_live_hours >= max_champion_age_hours:
        decision["strategy"] = "research"
        decision["reason"] = "champion_stale"
        return decision

    if (
        new_shadow_since_last
        and int(shadow_stats.get("records", 0)) >= min_shadow_records
        and (shadow_age_hours is None or shadow_age_hours <= refresh_after_hours + 12)
    ):
        decision["strategy"] = "refresh"
        decision["reason"] = "new_shadow_data"
        return decision

    if previous_finished_at is None:
        decision["strategy"] = "refresh"
        decision["reason"] = "no_previous_nightly"
        return decision

    since_last_hours = _hours_since(previous_finished_at)
    if since_last_hours is not None and since_last_hours >= refresh_after_hours:
        decision["strategy"] = "refresh"
        decision["reason"] = "refresh_window_elapsed"
        return decision

    decision["strategy"] = "skip"
    decision["reason"] = "champion_recent_and_no_new_data"
    return decision


def _set_env_default(name: str, value: str) -> None:
    """Definit une variable d'environnement sans ecraser un choix explicite.

    Args:
        name (str): Nom de la variable.
        value (str): Valeur par defaut a appliquer.
    """
    if not os.getenv(name):
        os.environ[name] = value


def apply_training_strategy(decision: dict[str, object]) -> None:
    """Applique les parametres adaptes a la strategie nightly choisie.

    Args:
        decision (dict[str, object]): Strategie issue de `decide_training_strategy`.
    """
    strategy = str(decision.get("strategy", "research")).lower()
    canonical_symbols_csv = _canonical_symbols_csv()
    if strategy == "research":
        _set_env_default("TRAINING_PROFILE", "research")
        _set_env_default("RUN_TRAIN_GNN", "1")
        _set_env_default("RUN_TRAIN_MUZERO", "1")
        _set_env_default("RUN_TRAIN_DREAMER", "0")
        _set_env_default("MUZERO_TRAINING_STEPS", "32000")
        _set_env_default("MUZERO_GAMES_PER_SYMBOL", "20")
        _set_env_default("ARENA_GAMES_PER_SYMBOL", "8")
        _set_env_default("ARENA_MIN_GAMES", "24")
        _set_env_default("ARENA_MIN_SYMBOLS", "6")
        _set_env_default("MUZERO_MAX_SYMBOLS", str(len(CANONICAL_TIMESCALE_SYMBOLS)))
        _set_env_default("ARENA_MAX_SYMBOLS", str(len(CANONICAL_TIMESCALE_SYMBOLS)))
        _set_env_default("MUZERO_DATASET_SOURCE", "timescaledb")
        _set_env_default("TRAINING_TIMESCALE_ENABLED", "1")
        _set_env_default("TRAINING_FOCUS_SYMBOLS", canonical_symbols_csv)
        _set_env_default("MUZERO_SYMBOLS", canonical_symbols_csv)
        _set_env_default("ARENA_SYMBOLS", canonical_symbols_csv)
        _set_env_default("MUZERO_SYMBOLS_SCALP", canonical_symbols_csv)
        _set_env_default("MUZERO_SYMBOLS_INTRADAY", canonical_symbols_csv)
        _set_env_default("MUZERO_SYMBOLS_SWING", canonical_symbols_csv)
        _set_env_default("ARENA_SYMBOLS_SCALP", canonical_symbols_csv)
        _set_env_default("ARENA_SYMBOLS_INTRADAY", canonical_symbols_csv)
        _set_env_default("ARENA_SYMBOLS_SWING", canonical_symbols_csv)
        _set_env_default("TRAIN_GNN_SYMBOLS", canonical_symbols_csv)
        _set_env_default("TRAIN_GNN_CONTEXT_SYMBOLS", canonical_symbols_csv)
        _set_env_default("TRAIN_GNN_MAX_SYMBOLS", str(len(CANONICAL_TIMESCALE_SYMBOLS)))
        return

    if strategy == "refresh":
        _set_env_default("TRAINING_PROFILE", "refresh")
        _set_env_default("RUN_TRAIN_GNN", "1")
        _set_env_default("RUN_TRAIN_MUZERO", "1")
        _set_env_default("RUN_TRAIN_DREAMER", "0")
        _set_env_default("MUZERO_TRAINING_STEPS", "8000")
        _set_env_default("MUZERO_GAMES_PER_SYMBOL", "8")
        _set_env_default("ARENA_GAMES_PER_SYMBOL", "4")
        _set_env_default("ARENA_MIN_GAMES", "12")
        _set_env_default("ARENA_MIN_SYMBOLS", "3")
        _set_env_default("MUZERO_MAX_SYMBOLS", str(len(CANONICAL_TIMESCALE_SYMBOLS)))
        _set_env_default("ARENA_MAX_SYMBOLS", str(len(CANONICAL_TIMESCALE_SYMBOLS)))
        _set_env_default("MUZERO_DATASET_SOURCE", "timescaledb")
        _set_env_default("TRAINING_TIMESCALE_ENABLED", "1")
        _set_env_default("TRAINING_FOCUS_SYMBOLS", canonical_symbols_csv)
        _set_env_default("MUZERO_SYMBOLS", canonical_symbols_csv)
        _set_env_default("ARENA_SYMBOLS", canonical_symbols_csv)
        _set_env_default("MUZERO_SYMBOLS_SCALP", canonical_symbols_csv)
        _set_env_default("MUZERO_SYMBOLS_INTRADAY", canonical_symbols_csv)
        _set_env_default("MUZERO_SYMBOLS_SWING", canonical_symbols_csv)
        _set_env_default("ARENA_SYMBOLS_SCALP", canonical_symbols_csv)
        _set_env_default("ARENA_SYMBOLS_INTRADAY", canonical_symbols_csv)
        _set_env_default("ARENA_SYMBOLS_SWING", canonical_symbols_csv)
        _set_env_default("TRAIN_GNN_SYMBOLS", canonical_symbols_csv)
        _set_env_default("TRAIN_GNN_CONTEXT_SYMBOLS", canonical_symbols_csv)
        _set_env_default("TRAIN_GNN_MAX_SYMBOLS", str(len(CANONICAL_TIMESCALE_SYMBOLS)))


def _build_gnn_job(learning_context: dict[str, object] | None = None) -> dict[str, object]:
    """Construit l'etape explicite de refresh GNN.

    Args:
        learning_context (dict[str, object] | None): Contexte derive de la review.

    Returns:
        dict[str, object]: Definition complete du job GNN.
    """

    review_overrides = dict((learning_context or {}).get("env_overrides") or {})
    focus_symbols = list((learning_context or {}).get("priority_symbols") or CANONICAL_TIMESCALE_SYMBOLS)
    focus_symbols = _unique_symbols(focus_symbols) or list(CANONICAL_TIMESCALE_SYMBOLS)
    canonical_symbols_csv = ",".join(focus_symbols)
    extra_env = {
        "TRAIN_GNN_SYMBOLS": canonical_symbols_csv,
        "TRAIN_GNN_CONTEXT_SYMBOLS": canonical_symbols_csv,
        "TRAIN_GNN_MAX_SYMBOLS": str(len(CANONICAL_TIMESCALE_SYMBOLS)),
        "TRAIN_GNN_DEPLOYMENT_CLASS": "consultative",
    }
    extra_env.update(_filter_env_overrides(review_overrides, ("TRAINING_", "TRAIN_GNN_")))
    return {
        "name": "gnn",
        "engine": "gnn",
        "horizon": None,
        "dataset_source": "timescaledb",
        "focus_symbols": focus_symbols,
        "command": [sys.executable, "scripts/train_gnn.py"],
        "extra_env": extra_env,
    }


def _build_muzero_job(
    horizon: str,
    learning_context: dict[str, object] | None = None,
    *,
    seed_candidate: dict[str, object] | None = None,
) -> dict[str, object]:
    """Construit un job nightly MuZero borne a TimeScaleDB.

    Args:
        horizon (str): Horizon cible.
        learning_context (dict[str, object] | None): Contexte derive de la review.
        seed_candidate (dict[str, object] | None): Seed scalp a reutiliser.

    Returns:
        dict[str, object]: Definition complete du job MuZero.
    """

    normalized_horizon = str(horizon).strip().lower()
    review_overrides = dict((learning_context or {}).get("env_overrides") or {})
    focus_symbols = list((learning_context or {}).get("priority_symbols") or CANONICAL_TIMESCALE_SYMBOLS)
    focus_symbols = _unique_symbols(focus_symbols) or list(CANONICAL_TIMESCALE_SYMBOLS)
    canonical_symbols_csv = ",".join(focus_symbols)
    extra_env = {
        "MUZERO_HORIZON": normalized_horizon,
        "MUZERO_MODEL_FAMILY": "",
        "MUZERO_DATASET_SOURCE": "timescaledb",
        "TRAINING_TIMESCALE_ENABLED": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85"),
        "TRAINING_FOCUS_SYMBOLS": canonical_symbols_csv,
        "MUZERO_SYMBOLS": canonical_symbols_csv,
        "ARENA_SYMBOLS": canonical_symbols_csv,
        f"MUZERO_SYMBOLS_{normalized_horizon.upper()}": canonical_symbols_csv,
        f"ARENA_SYMBOLS_{normalized_horizon.upper()}": canonical_symbols_csv,
        "MUZERO_MAX_SYMBOLS": str(len(CANONICAL_TIMESCALE_SYMBOLS)),
        "ARENA_MAX_SYMBOLS": str(len(CANONICAL_TIMESCALE_SYMBOLS)),
        "MUZERO_COLLECTOR_MODE": os.getenv("MUZERO_COLLECTOR_MODE", "batched_symbol_workers"),
        "MUZERO_COLLECTOR_WORKERS": os.getenv("MUZERO_COLLECTOR_WORKERS", "7"),
        "MUZERO_COLLECTOR_QUEUE_DEPTH": os.getenv("MUZERO_COLLECTOR_QUEUE_DEPTH", "128"),
        "MUZERO_INFERENCE_BATCH_MAX": os.getenv("MUZERO_INFERENCE_BATCH_MAX", "64"),
        "MUZERO_INFERENCE_BATCH_TIMEOUT_MS": os.getenv("MUZERO_INFERENCE_BATCH_TIMEOUT_MS", "2"),
        "MUZERO_BATCH_AUTOTUNE": os.getenv("MUZERO_BATCH_AUTOTUNE", "1"),
        "MUZERO_BATCH_CANDIDATES": os.getenv("MUZERO_BATCH_CANDIDATES", "32,64,96,128"),
    }
    seed_payload = dict(seed_candidate or {})
    if normalized_horizon == "scalp":
        seed_checkpoint = str(seed_payload.get("checkpoint_path") or "").strip()
        seed_candidate_id = str(seed_payload.get("candidate_id") or "").strip()
        seed_source = str(seed_payload.get("source") or "").strip()
        seed_reason = str(seed_payload.get("reason") or "").strip()
        if seed_checkpoint:
            extra_env["MUZERO_RESUME_CHECKPOINT_PATH"] = seed_checkpoint
        if seed_candidate_id:
            extra_env["TRAINING_SEED_CANDIDATE_ID"] = seed_candidate_id
        if seed_source:
            extra_env["TRAINING_SEED_SOURCE"] = seed_source
        if seed_reason:
            extra_env["TRAINING_SEED_REASON"] = seed_reason
    extra_env.update(_filter_env_overrides(review_overrides, ("TRAINING_", "MUZERO_", "ARENA_")))
    return {
        "name": f"muzero_{normalized_horizon}",
        "engine": "muzero",
        "horizon": normalized_horizon,
        "dataset_source": "timescaledb",
        "focus_symbols": focus_symbols,
        "command": [sys.executable, "scripts/train_global_models.py"],
        "extra_env": extra_env,
    }


def _build_dreamer_job() -> dict[str, object]:
    """Construit un job Dreamer offline explicitement optionnel.

    Returns:
        dict[str, object]: Definition complete du job Dreamer.
    """

    return {
        "name": "dreamer_offline",
        "engine": "dreamer",
        "horizon": None,
        "dataset_source": "timescaledb",
        "focus_symbols": list(CANONICAL_TIMESCALE_SYMBOLS),
        "command": [sys.executable, "-m", "eva_lab.muzero.offline_trainer"],
        "extra_env": {
            "DREAMER_EPOCHS": os.getenv("DREAMER_EPOCHS", "1500"),
            "MUZERO_DATASET_SOURCE": "timescaledb",
            "TRAINING_TIMESCALE_ENABLED": "1",
            "TRAINING_FOCUS_SYMBOLS": _canonical_symbols_csv(),
        },
    }


def build_nightly_job_queue(
    *,
    run_gnn: bool,
    run_muzero: bool,
    run_dreamer: bool,
    learning_context: dict[str, object] | None = None,
    scheduler_plan: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Construit la file explicite des jobs nocturnes.

    Args:
        run_gnn (bool): Active le refresh GNN.
        run_muzero (bool): Active les jobs MuZero multi-horizon.
        run_dreamer (bool): Active le job Dreamer offline.
        learning_context (dict[str, object] | None): Guidance derivee de la review.
        scheduler_plan (dict[str, object] | None): Decision du scheduler continu.

    Returns:
        list[dict[str, object]]: File ordonnee des jobs a executer.
    """

    jobs: list[dict[str, object]] = []
    if run_gnn:
        jobs.append(_build_gnn_job(learning_context))
    if run_muzero:
        horizons = list((scheduler_plan or {}).get("next_horizons") or _resolve_horizons())
        seed_candidate = dict((scheduler_plan or {}).get("seed_candidate") or {})
        for horizon in horizons:
            jobs.append(
                _build_muzero_job(
                    horizon,
                    learning_context,
                    seed_candidate=seed_candidate if str(horizon).strip().lower() == "scalp" else None,
                )
            )
    if run_dreamer:
        jobs.append(_build_dreamer_job())
    return jobs


def _summarize_job(job: dict[str, object]) -> dict[str, object]:
    """Reduit un job nightly a une vue legere pour le resume JSON.

    Args:
        job (dict[str, object]): Definition complete du job.

    Returns:
        dict[str, object]: Vue compacte du job.
    """

    return {
        "name": job.get("name"),
        "engine": job.get("engine"),
        "horizon": job.get("horizon"),
        "dataset_source": job.get("dataset_source"),
        "focus_symbols": list(job.get("focus_symbols") or []),
    }


def persist_summary(summary: dict[str, object]) -> None:
    """Ecrit le resume courant sur disque pour garder une trace meme en cas d'echec.

    Args:
        summary (dict[str, object]): Resume courant de la sequence nightly.
    """
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def append_step(
    summary: dict[str, object],
    name: str,
    status: str,
    error: str | None = None,
    details: dict[str, object] | None = None,
) -> None:
    """Ajoute le resultat d'une etape dans le resume JSON.

    Args:
        summary (dict[str, object]): Resume global en construction.
        name (str): Nom de l'etape.
        status (str): Statut final de l'etape.
        error (str | None): Erreur eventuelle.
        details (dict[str, object] | None): Metadonnees compactes optionnelles.
    """
    step: dict[str, object] = {"name": name, "status": status}
    if error:
        step["error"] = error
    if details:
        step["details"] = dict(details)
    summary.setdefault("steps", []).append(step)
    persist_summary(summary)
    mark_step_finished(name, status, error)


def run_step(name: str, command: list[str], extra_env: dict[str, str] | None = None) -> dict[str, object]:
    """Execute une etape d'entrainement dans un processus isole.

    Args:
        name (str): Nom de l'etape.
        command (list[str]): Commande a lancer.
        extra_env (dict[str, str] | None): Variables d'environnement additionnelles.

    Returns:
        dict[str, object]: Resume court de l'execution.

    Raises:
        RuntimeError: Si le sous-processus se termine en erreur.
    """
    env = os.environ.copy()
    pythonpath_entries = [str(WORKDIR), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([entry for entry in pythonpath_entries if entry])
    if extra_env:
        env.update(extra_env)
    gpu_owner = "idle"
    if name.startswith("muzero_"):
        gpu_owner = "muzero"
    elif name.startswith("dreamer"):
        gpu_owner = "dreamer"
    elif name.startswith("gnn"):
        gpu_owner = "gnn"

    logger.info("Debut etape %s: %s", name, command)
    mark_step_running(name, phase="demarrage")
    set_training_runtime_state(gpu_owner=gpu_owner)
    append_training_log(
        f"Debut de l'etape {name}.",
        source="nightly",
    )
    result = subprocess.run(
        command,
        cwd=WORKDIR,
        env=env,
        check=False,
        text=True,
        stderr=subprocess.PIPE,
        errors="replace",
    )
    stderr_tail = [
        line.strip()
        for line in str(result.stderr or "").splitlines()
        if str(line or "").strip()
    ][-12:]
    runtime_status = dict(load_training_status())
    failed_phase = runtime_status.get("failed_phase")
    run_id = runtime_status.get("run_id")
    if result.returncode != 0:
        set_training_runtime_state(gpu_owner="idle")
        error_summary = {
            "step_name": name,
            "return_code": int(result.returncode),
            "stderr_tail": stderr_tail,
            "failed_phase": failed_phase,
            "run_id": run_id,
            "gpu_owner": gpu_owner,
        }
        append_training_log(
            (
                f"Echec de l'etape {name} "
                f"(code={result.returncode}, phase={failed_phase or 'inconnue'}, run_id={run_id or 'n/a'})."
            ),
            level="ERROR",
            source="nightly",
        )
        if stderr_tail:
            append_training_log(
                "stderr nightly: " + " || ".join(stderr_tail[-3:]),
                level="ERROR",
                source="nightly",
            )
        raise RuntimeError(
            "Echec de l'etape "
            f"{name} | code={result.returncode} | phase={failed_phase or 'inconnue'} | "
            f"run_id={run_id or 'n/a'} | stderr_tail={stderr_tail!r}"
        )
    logger.info("Etape %s terminee avec succes.", name)
    set_training_runtime_state(gpu_owner="idle")
    result_payload = {
        "step_name": name,
        "return_code": int(result.returncode),
        "stderr_tail": stderr_tail,
        "failed_phase": failed_phase,
        "run_id": run_id,
        "gpu_owner": gpu_owner,
        "batch_autotune_result": runtime_status.get("jax_batch_profile"),
        "collector_mode": runtime_status.get("collector_mode"),
        "collector_workers": runtime_status.get("collector_workers"),
        "collector_queue_depth": runtime_status.get("collector_queue_depth"),
        "inference_batch_profile": runtime_status.get("inference_batch_profile"),
    }
    if name.startswith("muzero_"):
        horizon = str(name.split("_", 1)[1] if "_" in name else "").strip().lower()
        terminal_summary = load_latest_terminal_summary(engine="muzero", horizon=horizon) or {}
        result_payload["terminal_summary"] = terminal_summary
        result_payload["promotion_decision"] = _summarize_promotion_decision(terminal_summary)
    return result_payload


def main() -> dict[str, object]:
    """Lance la file explicite des jobs nocturnes.

    Returns:
        dict[str, object]: Resume complet de la sequence nightly.
    """
    decision = decide_training_strategy()
    trigger = os.getenv("TRAINING_RUN_TRIGGER", "manual")
    promoter = ChampionPromoter(
        weights_dir=str(WORKDIR / "data" / "muzero" / "weights"),
        results_dir=str(WORKDIR / "data" / "muzero" / "results"),
    )
    scheduler_state_before = _load_continuous_scheduler_state()
    summary: dict[str, object] = {
        "started_at": datetime.now().isoformat(),
        "workdir": str(WORKDIR),
        "steps": [],
        "status": "running",
        "strategy": decision.get("strategy"),
        "reason": decision.get("reason"),
        "decision": decision,
        "trigger": trigger,
        "lock_file": str(LOCK_PATH),
        "continuous_scheduler": dict(scheduler_state_before),
    }
    run_id = f"nightly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    lock_payload: dict[str, object] | None = None
    lock_acquired, lock_payload = acquire_run_lock()
    if not lock_acquired:
        logger.warning("Sequence nightly ignoree: verrou deja actif (%s).", lock_payload)
        _record_skip_event("run_already_active", trigger, lock_payload)
        return {
            "status": "skipped",
            "reason": "run_already_active",
            "trigger": trigger,
            "active_lock": lock_payload,
        }

    summary["lock"] = lock_payload
    persist_summary(summary)
    logger.info(
        "Strategie nightly retenue: %s (%s).",
        summary["strategy"],
        summary["reason"],
    )
    append_training_log(
        f"Strategie nightly retenue: {summary['strategy']} ({summary['reason']}).",
        source="nightly",
    )

    review_payload = _load_latest_trading_review()
    review_learning = _build_review_learning_context(review_payload)
    raw_review_summary = {
        "loaded": review_learning.get("loaded", False),
        "source": review_learning.get("source"),
        "path": review_learning.get("path"),
        "generated_at": review_learning.get("generated_at"),
        "winner_symbols": list(review_learning.get("winner_symbols") or []),
        "risk_symbols": list(review_learning.get("risk_symbols") or []),
        "priority_symbols": list(review_learning.get("priority_symbols") or []),
        "gnn_focus_symbol": review_learning.get("gnn_focus_symbol"),
    }
    summary["review_learning"] = dict(raw_review_summary)
    decision["review_learning"] = dict(raw_review_summary)

    if summary["strategy"] == "skip":
        summary["status"] = "skipped"
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        mark_skip_status(str(summary["reason"]), trigger, lock_payload)
        send_nightly_summary(summary)
        logger.info("Sequence nightly ignoree: %s", summary["reason"])
        release_run_lock(lock_payload)
        return summary

    apply_training_strategy(decision)
    requested_run_gnn = _env_flag("RUN_TRAIN_GNN", True)
    requested_run_muzero = _env_flag("RUN_TRAIN_MUZERO", True)
    requested_run_dreamer = _env_flag("RUN_TRAIN_DREAMER", False)
    gnn_refresh_policy = _evaluate_gnn_refresh_policy(requested_run_gnn)
    scheduler_plan = _plan_continuous_scheduler(
        scheduler_state=scheduler_state_before,
        promoter=promoter,
        run_gnn_requested=requested_run_gnn,
        run_muzero_requested=requested_run_muzero,
        run_dreamer_requested=requested_run_dreamer,
        gnn_refresh_policy=gnn_refresh_policy,
    )
    review_learning = _merge_learning_context_with_scheduler(
        review_learning,
        seed_candidate=dict(scheduler_plan.get("seed_candidate") or {}),
        mutation_targets=dict(scheduler_plan.get("mutation_targets") or {}),
        seed_reuse_block_reason=str(scheduler_plan.get("seed_reuse_block_reason") or "").strip() or None,
    )
    summary["review_learning"] = {
        **raw_review_summary,
        "priority_symbols": list(review_learning.get("priority_symbols") or []),
        "gnn_focus_symbol": review_learning.get("gnn_focus_symbol"),
        "seed_candidate_id": review_learning.get("seed_candidate_id"),
        "seed_checkpoint": review_learning.get("seed_checkpoint"),
        "seed_source": review_learning.get("seed_source"),
        "seed_reason": review_learning.get("seed_reason"),
        "seed_reuse_block_reason": review_learning.get("seed_reuse_block_reason"),
    }
    decision["review_learning"] = dict(summary["review_learning"])
    training_weighting = _build_training_weighting_summary(review_learning)
    summary["training_weighting"] = training_weighting
    decision["training_weighting"] = training_weighting
    summary["best_scalp_candidate"] = dict(scheduler_plan.get("best_scalp_candidate") or {})
    summary["best_for_mutation_candidate"] = dict(
        scheduler_plan.get("best_for_mutation_candidate") or {}
    )
    summary["best_for_seed_candidate"] = dict(scheduler_plan.get("best_for_seed_candidate") or {})
    summary["seed_reuse_block_reason"] = scheduler_plan.get("seed_reuse_block_reason")
    summary["improvement_vs_live"] = dict(scheduler_plan.get("improvement_vs_live") or {})
    summary["scheduler_decision"] = dict(scheduler_plan.get("scheduler_decision") or {})
    summary["mutation_targets"] = dict((scheduler_plan.get("mutation_targets") or {}).get("targets") or {})
    summary["gnn_refresh_policy"] = gnn_refresh_policy
    decision["gnn_refresh_policy"] = gnn_refresh_policy
    summary["gnn_policy"] = {
        "mode": "refresh_if_stale",
        "refresh_after_hours": gnn_refresh_policy.get("threshold_hours"),
        "scheduled": bool(scheduler_plan.get("run_gnn")),
        "reason": gnn_refresh_policy.get("reason"),
        "concurrent_with_muzero": False,
    }
    summary["dreamer_policy"] = {
        "live_policy": "offline_locked",
        "queue_position": "after_muzero",
        "battle_report_required": True,
        "scheduled": bool(scheduler_plan.get("run_dreamer")),
        "min_interval_hours": _env_int("TRAINING_CONTINUOUS_DREAMER_MIN_INTERVAL_HOURS", 24),
    }
    summary["muzero_scheduler_policy"] = {
        "gpu_priority": "muzero_first",
        "collector_mode": os.getenv("MUZERO_COLLECTOR_MODE", "batched_symbol_workers"),
        "collector_workers": _env_int("MUZERO_COLLECTOR_WORKERS", 7),
        "collector_queue_depth": _env_int("MUZERO_COLLECTOR_QUEUE_DEPTH", 128),
        "inference_batch_max": _env_int("MUZERO_INFERENCE_BATCH_MAX", 64),
        "inference_batch_timeout_ms": _env_int("MUZERO_INFERENCE_BATCH_TIMEOUT_MS", 2),
        "batch_autotune": _env_flag("MUZERO_BATCH_AUTOTUNE", True),
        "batch_candidates": [
            int(item)
            for item in str(os.getenv("MUZERO_BATCH_CANDIDATES", "32,64,96,128")).split(",")
            if str(item).strip()
        ],
        "continuous_focus": scheduler_plan.get("current_focus"),
    }
    summary["promotion_decision"] = {}
    summary["batch_autotune_result"] = None

    scheduler_runtime_state = _normalize_continuous_scheduler_state(scheduler_state_before)
    if scheduler_plan.get("mode") == CONTINUOUS_SCHEDULER_MODE:
        scheduler_runtime_state.update(
            {
                "mode": CONTINUOUS_SCHEDULER_MODE,
                "cycle_id": scheduler_plan.get("cycle_id"),
                "cycle_index": scheduler_plan.get("cycle_index"),
                "current_focus": scheduler_plan.get("current_focus"),
                "seed_source": scheduler_plan.get("seed_candidate", {}).get("source"),
                "seed_checkpoint": scheduler_plan.get("seed_candidate", {}).get("checkpoint_path"),
                "seed_reason": scheduler_plan.get("seed_candidate", {}).get("reason"),
                "seed_candidate_id": scheduler_plan.get("seed_candidate", {}).get("candidate_id"),
                "auto_promotion_policy": "strict_live_gate",
                "next_jobs": list(scheduler_plan.get("next_jobs") or []),
                "best_scalp_candidate": dict(scheduler_plan.get("best_scalp_candidate") or {}),
                "best_for_mutation_candidate": dict(
                    scheduler_plan.get("best_for_mutation_candidate") or {}
                ),
                "best_for_seed_candidate": dict(
                    scheduler_plan.get("best_for_seed_candidate") or {}
                ),
                "seed_reuse_block_reason": scheduler_plan.get("seed_reuse_block_reason"),
                "improvement_vs_live": dict(scheduler_plan.get("improvement_vs_live") or {}),
                "scheduler_decision": dict(scheduler_plan.get("scheduler_decision") or {}),
                "mutation_targets": dict((scheduler_plan.get("mutation_targets") or {}).get("targets") or {}),
                "last_started_at": datetime.now().isoformat(),
                "live_policy": "muzero_only",
                "gnn_policy": "weak_veto",
                "dreamer_policy": "offline_locked",
                "ensemble_prod_enabled": False,
            }
        )
        _persist_continuous_scheduler_state(scheduler_runtime_state)
    summary["continuous_scheduler"] = dict(scheduler_runtime_state)
    persist_summary(summary)

    if review_learning.get("loaded"):
        append_training_log(
            "Review nightly chargee: "
            f"winners={','.join(summary['review_learning']['winner_symbols']) or 'aucun'} | "
            f"risks={','.join(summary['review_learning']['risk_symbols']) or 'aucun'}",
            source="nightly",
        )
    else:
        append_training_log(
            "Aucune review nightly disponible; file lancee sur priorites canoniques.",
            source="nightly",
        )
    append_training_log(
        (
            "Ponderation nightly: "
            f"episodes={training_weighting.get('episodes_loaded', 0)} | "
            f"tags={training_weighting.get('weighted_episode_counts', {})}"
        ),
        source="nightly",
    )

    universe_summary = build_training_universe_summary()
    reset_training_status(
        run_id=run_id,
        trigger=trigger,
        strategy=str(summary.get("strategy") or "research"),
        reason=str(summary.get("reason") or "manual"),
        universe=universe_summary,
    )
    set_training_weighting(training_weighting)
    set_continuous_scheduler_state(scheduler_runtime_state)
    send_training_run_started(
        run_id=run_id,
        strategy=str(summary.get("strategy") or "research"),
        reason=str(summary.get("reason") or "manual"),
        trigger=trigger,
        universe=universe_summary,
    )

    run_gnn = bool(scheduler_plan.get("run_gnn"))
    run_muzero = bool(scheduler_plan.get("run_muzero"))
    run_dreamer = bool(scheduler_plan.get("run_dreamer"))
    if requested_run_gnn and not run_gnn:
        append_training_log(
            (
                "Refresh GNN saute: "
                f"{gnn_refresh_policy.get('reason')} "
                f"(fraicheur={gnn_refresh_policy.get('freshness_hours')}h)."
            ),
            source="nightly",
        )
    job_queue = build_nightly_job_queue(
        run_gnn=run_gnn,
        run_muzero=run_muzero,
        run_dreamer=run_dreamer,
        learning_context=review_learning,
        scheduler_plan=scheduler_plan,
    )
    summary["job_queue"] = [_summarize_job(job) for job in job_queue]
    summary["continuous_scheduler"] = dict(
        _persist_continuous_scheduler_state(
            {
                **scheduler_runtime_state,
                "next_jobs": [str(job.get("name") or "") for job in job_queue],
            }
        )
    )
    persist_summary(summary)
    logger.info("File nightly preparee: %s", [job.get("name") for job in job_queue])
    append_training_log(
        "File nightly preparee: " + ", ".join(str(job.get("name")) for job in job_queue),
        source="nightly",
    )

    if not job_queue:
        summary["status"] = "skipped"
        summary["reason"] = "continuous_queue_empty"
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        finalize_training_status("blocked", reason="continuous_queue_empty")
        send_nightly_summary(summary)
        logger.warning("Aucun job a lancer pour ce cycle continu.")
        release_run_lock(lock_payload)
        return summary

    step_results: list[dict[str, object]] = []
    current_job: dict[str, object] | None = None
    try:
        for job in job_queue:
            current_job = dict(job)
            step_result = run_step(
                str(job.get("name") or "unknown"),
                list(job.get("command") or []),
                extra_env=dict(job.get("extra_env") or {}),
            )
            step_results.append(dict(step_result))
            if (
                str(job.get("engine") or "").strip().lower() == "muzero"
                and isinstance(step_result.get("batch_autotune_result"), dict)
                and step_result.get("batch_autotune_result")
            ):
                summary["batch_autotune_result"] = dict(step_result.get("batch_autotune_result") or {})
            if str(job.get("name") or "").strip().lower() == "muzero_scalp":
                terminal_summary = dict(step_result.get("terminal_summary") or {})
                summary["promotion_decision"] = dict(step_result.get("promotion_decision") or {})
                if terminal_summary:
                    candidate_record = _build_scalp_candidate_record(terminal_summary, promoter)
                    seed_gate = promoter.evaluate_scalp_seed_gate(candidate_record.get("metrics") or {})
                    candidate_record["seed_gate"] = seed_gate
                    candidate_record["seed_gate_allowed"] = bool(seed_gate.get("allowed", False))
                    candidate_record["seed_gate_reason"] = seed_gate.get("primary_reason")
                    summary["best_scalp_candidate"] = candidate_record
                    summary["best_for_mutation_candidate"] = dict(candidate_record)
                    if candidate_record["seed_gate_allowed"]:
                        summary["best_for_seed_candidate"] = dict(candidate_record)
                        summary["seed_reuse_block_reason"] = None
                    else:
                        summary["best_for_seed_candidate"] = {}
                        summary["seed_reuse_block_reason"] = candidate_record["seed_gate_reason"]
                    summary["improvement_vs_live"] = _build_improvement_vs_live(
                        summary["best_scalp_candidate"],
                        dict(scheduler_plan.get("live_reference") or {}),
                    )
                    summary["mutation_targets"] = dict(
                        (_build_scalp_mutation_targets(summary["best_scalp_candidate"]) or {}).get("targets") or {}
                    )
                    scheduler_plan["best_scalp_candidate"] = dict(summary["best_scalp_candidate"])
                    scheduler_plan["best_for_mutation_candidate"] = dict(
                        summary.get("best_for_mutation_candidate") or {}
                    )
                    scheduler_plan["best_for_seed_candidate"] = dict(
                        summary.get("best_for_seed_candidate") or {}
                    )
                    scheduler_plan["seed_reuse_block_reason"] = summary.get("seed_reuse_block_reason")
                    scheduler_plan["improvement_vs_live"] = dict(summary.get("improvement_vs_live") or {})
                    scheduler_plan["mutation_targets"] = {
                        **dict(scheduler_plan.get("mutation_targets") or {}),
                        "targets": dict(summary.get("mutation_targets") or {}),
                    }
            append_step(
                summary,
                str(job.get("name") or "unknown"),
                "ok",
                details=step_result,
            )

        if scheduler_plan.get("mode") == CONTINUOUS_SCHEDULER_MODE:
            summary["continuous_scheduler"] = dict(
                _persist_continuous_scheduler_state(
                    _update_scheduler_after_cycle_success(
                        scheduler_state=scheduler_runtime_state,
                        scheduler_plan=scheduler_plan,
                        step_results=step_results,
                    )
                )
            )
        summary["status"] = "ok"
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        finalize_training_status("ok", reason=str(summary.get("reason") or "complete"))
        send_nightly_summary(summary)
        logger.info("Resume nocturne ecrit dans %s", SUMMARY_PATH)
        return summary
    except Exception as exc:
        logger.exception("Sequence nocturne en echec: %s", exc)
        summary["status"] = "error"
        summary["error"] = str(exc)
        current_status = dict(load_training_status())
        summary["failure_context"] = {
            "failed_phase": current_status.get("failed_phase"),
            "run_id": current_status.get("run_id"),
            "exception_type": current_status.get("exception_type"),
            "exception_message": current_status.get("exception_message"),
            "traceback_tail": list(current_status.get("traceback_tail") or []),
            "last_nonzero_exit": current_status.get("last_nonzero_exit"),
        }
        if scheduler_plan.get("mode") == CONTINUOUS_SCHEDULER_MODE:
            summary["continuous_scheduler"] = dict(
                _persist_continuous_scheduler_state(
                    _update_scheduler_after_cycle_failure(
                        scheduler_state=scheduler_runtime_state,
                        scheduler_plan=scheduler_plan,
                        failed_job=current_job,
                        failed_phase=str(current_status.get("failed_phase") or "") or None,
                        exception_message=str(exc),
                    )
                )
            )
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        finalize_training_status("error", reason=str(exc))
        send_nightly_summary(summary)
        raise
    finally:
        release_run_lock(lock_payload)


if __name__ == "__main__":
    report = main()
    logger.info("Sequence nocturne terminee: %s", report)
