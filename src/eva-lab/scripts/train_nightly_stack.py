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
    append_training_log,
    build_training_universe_summary,
    finalize_training_status,
    mark_skip_status,
    mark_step_finished,
    mark_step_running,
    reset_training_status,
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
CANONICAL_GNN_TIMEFRAMES = ["M5", "H1", "D1"]


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


def _build_shadow_weighting_profile() -> dict[str, float]:
    """Construit le profil de ponderation utilise par la nightly.

    Returns:
        dict[str, float]: Profil de pondération shadow exploitable.
    """
    return {
        "base_weight": max(float(os.getenv("TRAINING_EPISODE_WEIGHT_BASE", "1.0") or 1.0), 0.0),
        "winner_bonus": max(
            float(os.getenv("TRAINING_EPISODE_WEIGHT_WINNER_BONUS", "0.15") or 0.15),
            0.0,
        ),
        "loser_bonus": max(
            float(os.getenv("TRAINING_EPISODE_WEIGHT_LOSER_BONUS", "0.35") or 0.35),
            0.0,
        ),
        "nemesis_bonus": max(
            float(os.getenv("TRAINING_EPISODE_WEIGHT_NEMESIS_BONUS", "0.55") or 0.55),
            0.0,
        ),
        "risk_symbol_bonus": max(
            float(os.getenv("TRAINING_EPISODE_WEIGHT_RISK_BONUS", "0.25") or 0.25),
            0.0,
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
    weighting_profile = _build_shadow_weighting_profile()
    summary = summarize_shadow_weighting(
        [SHADOW_DIR],
        winner_symbols=list(learning_context.get("winner_symbols") or []),
        risk_symbols=list(learning_context.get("risk_symbols") or []),
        allowed_symbols=allowed_symbols,
        max_episodes=_env_int("TRAINING_WEIGHTING_MAX_EPISODES", 250),
        weighting_profile=weighting_profile,
    )
    summary["allowed_symbols"] = list(allowed_symbols)
    summary["shadow_dirs"] = [str(SHADOW_DIR)]
    summary["gnn_focus_symbol"] = learning_context.get("gnn_focus_symbol")
    return summary


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


def _build_muzero_job(horizon: str, learning_context: dict[str, object] | None = None) -> dict[str, object]:
    """Construit un job nightly MuZero borne a TimeScaleDB.

    Args:
        horizon (str): Horizon cible.
        learning_context (dict[str, object] | None): Contexte derive de la review.

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
        "TRAINING_FOCUS_SYMBOLS": canonical_symbols_csv,
        "MUZERO_SYMBOLS": canonical_symbols_csv,
        "ARENA_SYMBOLS": canonical_symbols_csv,
        f"MUZERO_SYMBOLS_{normalized_horizon.upper()}": canonical_symbols_csv,
        f"ARENA_SYMBOLS_{normalized_horizon.upper()}": canonical_symbols_csv,
        "MUZERO_MAX_SYMBOLS": str(len(CANONICAL_TIMESCALE_SYMBOLS)),
        "ARENA_MAX_SYMBOLS": str(len(CANONICAL_TIMESCALE_SYMBOLS)),
    }
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
) -> list[dict[str, object]]:
    """Construit la file explicite des jobs nocturnes.

    Args:
        run_gnn (bool): Active le refresh GNN.
        run_muzero (bool): Active les jobs MuZero multi-horizon.
        run_dreamer (bool): Active le job Dreamer offline.
        learning_context (dict[str, object] | None): Guidance derivee de la review.

    Returns:
        list[dict[str, object]]: File ordonnee des jobs a executer.
    """

    jobs: list[dict[str, object]] = []
    if run_gnn:
        jobs.append(_build_gnn_job(learning_context))
    if run_muzero:
        for horizon in _resolve_horizons():
            jobs.append(_build_muzero_job(horizon, learning_context))
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
) -> None:
    """Ajoute le resultat d'une etape dans le resume JSON.

    Args:
        summary (dict[str, object]): Resume global en construction.
        name (str): Nom de l'etape.
        status (str): Statut final de l'etape.
        error (str | None): Erreur eventuelle.
    """
    step: dict[str, object] = {"name": name, "status": status}
    if error:
        step["error"] = error
    summary.setdefault("steps", []).append(step)
    persist_summary(summary)
    mark_step_finished(name, status, error)


def run_step(name: str, command: list[str], extra_env: dict[str, str] | None = None) -> None:
    """Execute une etape d'entrainement dans un processus isole.

    Args:
        name (str): Nom de l'etape.
        command (list[str]): Commande a lancer.
        extra_env (dict[str, str] | None): Variables d'environnement additionnelles.

    Raises:
        RuntimeError: Si le sous-processus se termine en erreur.
    """
    env = os.environ.copy()
    pythonpath_entries = [str(WORKDIR), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([entry for entry in pythonpath_entries if entry])
    if extra_env:
        env.update(extra_env)

    logger.info("Debut etape %s: %s", name, command)
    mark_step_running(name, phase="demarrage")
    append_training_log(
        f"Debut de l'etape {name}.",
        source="nightly",
    )
    result = subprocess.run(command, cwd=WORKDIR, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Echec de l'etape {name} (code {result.returncode}).")
    logger.info("Etape %s terminee avec succes.", name)


def main() -> dict[str, object]:
    """Lance la file explicite des jobs nocturnes.

    Returns:
        dict[str, object]: Resume complet de la sequence nightly.
    """
    decision = decide_training_strategy()
    trigger = os.getenv("TRAINING_RUN_TRIGGER", "manual")
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
    summary["review_learning"] = {
        "loaded": review_learning.get("loaded", False),
        "source": review_learning.get("source"),
        "path": review_learning.get("path"),
        "generated_at": review_learning.get("generated_at"),
        "winner_symbols": list(review_learning.get("winner_symbols") or []),
        "risk_symbols": list(review_learning.get("risk_symbols") or []),
        "priority_symbols": list(review_learning.get("priority_symbols") or []),
        "gnn_focus_symbol": review_learning.get("gnn_focus_symbol"),
    }
    decision["review_learning"] = summary["review_learning"]
    training_weighting = _build_training_weighting_summary(review_learning)
    summary["training_weighting"] = training_weighting
    decision["training_weighting"] = training_weighting
    set_training_weighting(training_weighting)
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
    universe_summary = build_training_universe_summary()
    reset_training_status(
        run_id=run_id,
        trigger=trigger,
        strategy=str(summary.get("strategy") or "research"),
        reason=str(summary.get("reason") or "manual"),
        universe=universe_summary,
    )
    set_training_weighting(training_weighting)
    send_training_run_started(
        run_id=run_id,
        strategy=str(summary.get("strategy") or "research"),
        reason=str(summary.get("reason") or "manual"),
        trigger=trigger,
        universe=universe_summary,
    )

    requested_run_gnn = _env_flag("RUN_TRAIN_GNN", True)
    gnn_refresh_policy = _evaluate_gnn_refresh_policy(requested_run_gnn)
    summary["gnn_refresh_policy"] = gnn_refresh_policy
    decision["gnn_refresh_policy"] = gnn_refresh_policy
    persist_summary(summary)
    run_gnn = bool(gnn_refresh_policy.get("scheduled"))
    run_muzero = _env_flag("RUN_TRAIN_MUZERO", True)
    run_dreamer = _env_flag("RUN_TRAIN_DREAMER", False)
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
    )
    summary["job_queue"] = [_summarize_job(job) for job in job_queue]
    persist_summary(summary)
    logger.info("File nightly preparee: %s", [job.get("name") for job in job_queue])
    append_training_log(
        "File nightly preparee: " + ", ".join(str(job.get("name")) for job in job_queue),
        source="nightly",
    )

    try:
        for job in job_queue:
            run_step(
                str(job.get("name") or "unknown"),
                list(job.get("command") or []),
                extra_env=dict(job.get("extra_env") or {}),
            )
            append_step(summary, str(job.get("name") or "unknown"), "ok")

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
