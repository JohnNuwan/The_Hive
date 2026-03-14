"""Utilitaires partages pour suivre un run d'entrainement EVA Lab."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

STATUS_DIR = Path(os.getenv("TRAINING_CHECKPOINT_DIR", "data/checkpoints"))
STATUS_PATH = STATUS_DIR / "training_status.json"
RUN_LOG_PATH = STATUS_DIR / "training_run.log"
NIGHTLY_SUMMARY_PATH = STATUS_DIR / "nightly_training_summary.json"
MAX_LOG_LINES = int(os.getenv("TRAINING_STATUS_MAX_LOG_LINES", "400"))

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


def _now_iso() -> str:
    """Retourne l'horodatage courant au format ISO."""

    return datetime.now().isoformat()


def _default_status() -> dict[str, Any]:
    """Construit la structure de base du statut training."""

    return {
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
    }


def _ensure_status_dir() -> None:
    """Cree le dossier de statut si necessaire."""

    STATUS_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ecrit un JSON de facon atomique."""

    _ensure_status_dir()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


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


def tail_training_log(limit: int = 30) -> list[str]:
    """Retourne les dernieres lignes du journal partage."""

    if not RUN_LOG_PATH.exists():
        return []
    try:
        lines = RUN_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    safe_limit = max(limit, 1)
    return lines[-safe_limit:]


def reset_training_status(
    *,
    run_id: str,
    trigger: str,
    strategy: str,
    reason: str,
    universe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reinitialise le statut d'un nouveau run en preservant le lanceur."""

    previous = load_training_status()
    status = _default_status()
    status["run_id"] = run_id
    status["active"] = True
    status["status"] = "running"
    status["trigger"] = trigger
    status["strategy"] = strategy
    status["reason"] = reason
    status["skip_reason"] = None
    status["started_at"] = _now_iso()
    status["launcher"] = dict(previous.get("launcher") or {})
    status["dependencies"] = dict(previous.get("dependencies") or {})
    status["universe"] = universe or build_training_universe_summary()
    persisted = persist_training_status(status)
    append_training_log(
        f"Run {run_id} demarre | strategie={strategy} | trigger={trigger} | raison={reason}",
        source="nightly",
    )
    return persisted


def mark_step_running(
    step_name: str,
    *,
    phase: str | None = None,
    horizon: str | None = None,
    symbol: str | None = None,
    symbol_index: int | None = None,
    symbol_total: int | None = None,
    part_index: int | None = None,
    part_total: int | None = None,
    epoch_current: int | None = None,
    epoch_total: int | None = None,
    training_step_current: int | None = None,
    training_step_total: int | None = None,
) -> dict[str, Any]:
    """Met a jour l'etape courante d'un run."""

    step = {
        "name": step_name,
        "status": "running",
        "phase": phase,
        "horizon": horizon,
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
    status["active"] = True
    status["status"] = "running"
    status["current_step"] = {key: value for key, value in step.items() if value is not None}
    status["failed_step"] = None
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
    """Met a jour le statut pour un skip propre du cron."""

    status = load_training_status()
    status["active"] = False
    status["status"] = "skipped"
    status["trigger"] = trigger
    status["skip_reason"] = reason
    status["finished_at"] = _now_iso()
    launcher = dict(status.get("launcher") or {})
    if lock_payload:
        launcher["skip_lock"] = lock_payload
    status["launcher"] = launcher
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
