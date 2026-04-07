"""Registre et projection lecture seule du Market GNN.

Ce module fournit une couche de verite pour le GNN de marche sans modifier
la chaine d'entrainement active. Il consolide les artefacts existants
(`gnn_master.pth` et `gnn_master_metrics.json`) dans un registre stable et
peut exposer un graphe reel derive des historiques disponibles.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eva_lab.training_utils import (
    classify_training_symbol,
    get_scalp_multi_universe_symbols,
    load_history_frame,
    normalize_training_symbols,
)

logger = logging.getLogger(__name__)

DEFAULT_GNN_TIMEFRAMES = ["M5", "H1", "D1"]
DEFAULT_GRAPH_TIMEFRAME = "H1"
DEFAULT_GRAPH_TIMEFRAME_FALLBACKS = ["H1", "M5", "D1"]
DEFAULT_STALE_HOURS = 168
DEFAULT_GRAPH_MAX_NODES = 18
DEFAULT_GRAPH_MIN_POINTS = 120
DEFAULT_GRAPH_HISTORY_BARS = 600
DEFAULT_GRAPH_MIN_ABS_CORRELATION = 0.25
DEFAULT_GRAPH_MAX_LINKS = 32
GNN_REGISTRY_NAME = "gnn_registry.json"
GNN_REFRESH_STATE_NAME = "gnn_refresh_state.json"


def _now_iso() -> str:
    """Retourne l'horodatage UTC courant au format ISO 8601."""

    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    """Charge un fichier JSON si disponible.

    Args:
        path (Path): Chemin du fichier a lire.

    Returns:
        dict[str, Any] | None: Contenu JSON charge ou ``None``.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Lecture JSON impossible pour %s: %s", path, exc)
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ecrit un JSON de maniere atomique.

    Args:
        path (Path): Fichier cible.
        payload (dict[str, Any]): Contenu a serialiser.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _default_refresh_state() -> dict[str, Any]:
    """Retourne l'etat de refresh GNN par defaut."""

    return {
        "status": "idle",
        "queued": False,
        "requested_at": None,
        "started_at": None,
        "finished_at": None,
        "run_id": None,
        "failure_reason": None,
        "source_run_id": None,
        "requested_by": None,
    }


def _file_description(path: Path) -> dict[str, Any]:
    """Retourne des metadonnees simples sur un fichier.

    Args:
        path (Path): Fichier a decrire.

    Returns:
        dict[str, Any]: Presence, taille et date de modification.
    """
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "modified_at": None,
        }
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "modified_at": datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def _build_family_counts(symbols: list[str]) -> dict[str, int]:
    """Compte les symboles par famille d'actifs.

    Args:
        symbols (list[str]): Univers de symboles.

    Returns:
        dict[str, int]: Nombre de symboles par famille.
    """
    counts: dict[str, int] = {}
    for symbol in symbols:
        family = classify_training_symbol(symbol)
        counts[family] = counts.get(family, 0) + 1
    return counts


def _normalize_metrics(raw_metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Normalise les metriques publiques du GNN.

    Args:
        raw_metrics (dict[str, Any] | None): Rapport brut issu du trainer.

    Returns:
        dict[str, Any]: Vue stable et sure des metriques.
    """
    metrics = raw_metrics or {}
    return {
        "loss": float(metrics.get("loss", 0.0) or 0.0),
        "scalp_accuracy": float(metrics.get("scalp_accuracy", 0.0) or 0.0),
        "intraday_accuracy": float(metrics.get("intraday_accuracy", 0.0) or 0.0),
        "swing_accuracy": float(metrics.get("swing_accuracy", 0.0) or 0.0),
        "epochs": int(metrics.get("epochs", 0) or 0),
        "batch_size": int(metrics.get("batch_size", 0) or 0),
        "samples": int(metrics.get("samples", 0) or 0),
    }


def _is_stale(trained_at: str | None) -> bool:
    """Indique si un checkpoint doit etre considere comme ancien.

    Args:
        trained_at (str | None): Horodatage ISO du dernier entrainement.

    Returns:
        bool: ``True`` si le checkpoint est plus vieux que le seuil.
    """
    if not trained_at:
        return False
    try:
        trained_dt = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    stale_after_hours = int(os.getenv("GNN_STALE_HOURS", str(DEFAULT_STALE_HOURS)))
    age_seconds = (datetime.now(tz=timezone.utc) - trained_dt.astimezone(timezone.utc)).total_seconds()
    return age_seconds > stale_after_hours * 3600


def _derive_status(
    previous_registry: dict[str, Any] | None,
    checkpoint: dict[str, Any],
    metrics: dict[str, Any],
    trained_at: str | None,
) -> str:
    """Derive le statut public du GNN.

    Args:
        previous_registry (dict[str, Any] | None): Registre precedent.
        checkpoint (dict[str, Any]): Description du checkpoint principal.
        metrics (dict[str, Any]): Metriques normalisees.
        trained_at (str | None): Date du dernier entrainement.

    Returns:
        str: Statut public (`draft`, `validated`, `live`, `stale`, `unavailable`).
    """
    previous_status = str((previous_registry or {}).get("status") or "").strip().lower()
    if not checkpoint.get("exists") and metrics["epochs"] <= 0:
        return "unavailable"
    if previous_status == "live" and checkpoint.get("exists"):
        return "live" if not _is_stale(trained_at) else "stale"
    if checkpoint.get("exists") and metrics["epochs"] > 0:
        return "validated" if not _is_stale(trained_at) else "stale"
    if checkpoint.get("exists"):
        return "draft"
    return "unavailable"


def _resolve_graph_timeframe(timeframes: list[str]) -> str:
    """Choisit le timeframe utilise pour projeter le graphe.

    Args:
        timeframes (list[str]): Timeframes declares par le registre.

    Returns:
        str: Timeframe retenu pour la projection.
    """
    normalized = [str(item).upper() for item in timeframes]
    if DEFAULT_GRAPH_TIMEFRAME in normalized:
        return DEFAULT_GRAPH_TIMEFRAME
    return normalized[0] if normalized else DEFAULT_GRAPH_TIMEFRAME


def _candidate_graph_timeframes(timeframes: list[str]) -> list[str]:
    """Construit l'ordre de fallback des timeframes du graphe.

    Args:
        timeframes (list[str]): Timeframes disponibles dans le registre.

    Returns:
        list[str]: Timeframes tries selon la politique de fallback.
    """

    normalized = [str(item).upper() for item in timeframes if str(item).strip()]
    ordered: list[str] = []
    for timeframe in DEFAULT_GRAPH_TIMEFRAME_FALLBACKS:
        if timeframe in normalized and timeframe not in ordered:
            ordered.append(timeframe)
    for timeframe in normalized:
        if timeframe not in ordered:
            ordered.append(timeframe)
    return ordered or [DEFAULT_GRAPH_TIMEFRAME]


def load_market_gnn_refresh_state(
    models_dir: str | os.PathLike[str] = "data/models",
) -> dict[str, Any]:
    """Charge l'etat courant de refresh du GNN.

    Args:
        models_dir (str | os.PathLike[str]): Dossier des artefacts GNN.

    Returns:
        dict[str, Any]: Etat persistant du refresh.
    """

    refresh_path = Path(models_dir) / GNN_REFRESH_STATE_NAME
    payload = _read_json(refresh_path) or {}
    state = _default_refresh_state()
    state.update(payload)
    return state


def persist_market_gnn_refresh_state(
    payload: dict[str, Any],
    models_dir: str | os.PathLike[str] = "data/models",
) -> dict[str, Any]:
    """Persiste l'etat courant du refresh GNN.

    Args:
        payload (dict[str, Any]): Etat a serialiser.
        models_dir (str | os.PathLike[str]): Dossier cible.

    Returns:
        dict[str, Any]: Etat normalise ecrit sur disque.
    """

    refresh_path = Path(models_dir) / GNN_REFRESH_STATE_NAME
    state = _default_refresh_state()
    state.update(payload or {})
    _write_json(refresh_path, state)
    return state


def update_market_gnn_registry(
    patch: dict[str, Any],
    models_dir: str | os.PathLike[str] = "data/models",
) -> dict[str, Any]:
    """Met a jour le registre GNN avec un patch simple.

    Args:
        patch (dict[str, Any]): Valeurs a fusionner dans le registre.
        models_dir (str | os.PathLike[str]): Dossier du registre.

    Returns:
        dict[str, Any]: Registre mis a jour.
    """

    registry_path = Path(models_dir) / GNN_REGISTRY_NAME
    current = _read_json(registry_path) or {}
    merged = dict(current)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**dict(merged.get(key) or {}), **value}
        else:
            merged[key] = value
    _write_json(registry_path, merged)
    return merged


def load_market_gnn_registry(
    models_dir: str | os.PathLike[str] = "data/models",
) -> dict[str, Any]:
    """Construit le registre stable du Market GNN.

    Args:
        models_dir (str | os.PathLike[str]): Dossier contenant les artefacts GNN.

    Returns:
        dict[str, Any]: Registre consolide du GNN de marche.
    """
    models_path = Path(models_dir)
    registry_path = models_path / GNN_REGISTRY_NAME
    checkpoint_path = models_path / "gnn_master.pth"
    metrics_path = models_path / "gnn_master_metrics.json"

    previous_registry = _read_json(registry_path) or {}
    refresh_state = load_market_gnn_refresh_state(models_dir=models_dir)
    metrics_payload = _read_json(metrics_path) or {}
    metrics = _normalize_metrics(metrics_payload)
    checkpoint = _file_description(checkpoint_path)
    metrics_file = _file_description(metrics_path)

    trained_at = checkpoint.get("modified_at") or metrics_file.get("modified_at")
    symbols = normalize_training_symbols(
        metrics_payload.get("symbols")
        or previous_registry.get("universe", {}).get("symbols")
        or []
    )
    focus_symbols = normalize_training_symbols(
        metrics_payload.get("focus_symbols")
        or previous_registry.get("focus_symbols")
        or symbols
    )
    focus_symbol = str(
        metrics_payload.get("focus_symbol")
        or previous_registry.get("focus_symbol")
        or (focus_symbols[0] if focus_symbols else "")
    ).strip() or None
    context_symbols = normalize_training_symbols(
        metrics_payload.get("context_symbols")
        or previous_registry.get("context_symbols")
        or [symbol for symbol in symbols if symbol != focus_symbol]
    )
    deployment_class = str(
        metrics_payload.get("deployment_class")
        or previous_registry.get("deployment_class")
        or "consultative"
    ).strip() or "consultative"
    timeframes = previous_registry.get("timeframes") or list(DEFAULT_GNN_TIMEFRAMES)
    status = _derive_status(previous_registry, checkpoint, metrics, trained_at)
    status_reason = str(previous_registry.get("status_reason") or "").strip()
    if not status_reason:
        if status == "unavailable":
            status_reason = "Aucun checkpoint GNN exploitable n'est disponible."
        elif status == "stale":
            status_reason = "Le dernier checkpoint GNN est plus ancien que le seuil de fraicheur."
        elif status == "draft":
            status_reason = "Le checkpoint existe mais le rapport d'entrainement reste incomplet."
        else:
            status_reason = "Le registre GNN est coherent."
    coverage_summary = dict(previous_registry.get("coverage_summary") or {})
    refresh_view = {
        "queued": bool(refresh_state.get("queued")),
        "status": str(refresh_state.get("status") or "idle"),
        "requested_at": refresh_state.get("requested_at"),
        "started_at": refresh_state.get("started_at"),
        "finished_at": refresh_state.get("finished_at"),
        "failure_reason": refresh_state.get("failure_reason"),
    }
    consultative_universe = normalize_training_symbols(get_scalp_multi_universe_symbols())
    universe_ready = bool(symbols) and all(symbol in symbols for symbol in consultative_universe)
    metrics_ready = bool(metrics.get("epochs", 0) > 0 and metrics.get("samples", 0) > 0)
    champion_kind = "consultative"
    champion_id = str(previous_registry.get("champion_id") or "").strip() or None
    if champion_id is None and checkpoint.get("exists") and trained_at:
        safe_timestamp = trained_at.replace(":", "").replace("-", "").replace("+", "_").replace(".", "_")
        champion_id = f"gnn_consultative_{safe_timestamp}"
    champion_ready = bool(
        status in {"validated", "live"}
        and not _is_stale(trained_at)
        and bool(previous_registry.get("source_run_id") or refresh_state.get("source_run_id"))
        and metrics_ready
        and universe_ready
    )
    registry = {
        "name": "market_gnn",
        "version": str(previous_registry.get("version") or checkpoint_path.stem),
        "status": status,
        "status_reason": status_reason,
        "trained_at": trained_at,
        "champion_id": champion_id,
        "champion_ready": champion_ready,
        "champion_kind": champion_kind,
        "checkpoint_path": checkpoint["path"] if checkpoint["exists"] else None,
        "source_run_id": previous_registry.get("source_run_id") or refresh_state.get("source_run_id"),
        "focus_symbols": focus_symbols,
        "focus_symbol": focus_symbol,
        "context_symbols": context_symbols,
        "deployment_class": deployment_class,
        "timeframes": [str(item).upper() for item in timeframes],
        "universe": {
            "symbols": symbols,
            "count": len(symbols),
            "family_counts": _build_family_counts(symbols),
            "canonical_scalp_multi_universe": consultative_universe,
            "universe_ready": universe_ready,
        },
        "metrics": metrics,
        "last_refresh_requested_at": previous_registry.get("last_refresh_requested_at") or refresh_state.get("requested_at"),
        "last_refresh_started_at": previous_registry.get("last_refresh_started_at") or refresh_state.get("started_at"),
        "last_refresh_finished_at": previous_registry.get("last_refresh_finished_at") or refresh_state.get("finished_at"),
        "last_refresh_status": previous_registry.get("last_refresh_status") or refresh_state.get("status") or "idle",
        "coverage_summary": coverage_summary,
        "refresh_state": refresh_view,
        "artifacts": {
            "registry": _file_description(registry_path),
            "checkpoint": checkpoint,
            "metrics": metrics_file,
        },
    }

    if registry != previous_registry:
        _write_json(registry_path, registry)
    return registry


def build_market_gnn_graph_snapshot(
    registry: dict[str, Any] | None = None,
    data_dir: str | os.PathLike[str] = "data/history",
) -> dict[str, Any]:
    """Construit un graphe reel a partir des historiques utilises par le GNN.

    Args:
        registry (dict[str, Any] | None): Registre GNN deja charge.
        data_dir (str | os.PathLike[str]): Dossier des historiques.

    Returns:
        dict[str, Any]: Snapshot graphique derive des correlations historiques.
    """
    current_registry = registry or load_market_gnn_registry()
    symbols = list(current_registry.get("universe", {}).get("symbols") or [])
    candidate_timeframes = _candidate_graph_timeframes(list(current_registry.get("timeframes") or []))
    if not symbols:
        return {
            "status": "unavailable",
            "reason": "Aucun univers GNN versionne n'est disponible.",
            "nodes": [],
            "links": [],
            "graph_timeframe": None,
            "selected_timeframe": None,
            "candidate_timeframes": candidate_timeframes,
            "overlap_points": 0,
            "missing_symbols": [],
            "displayed_symbol_count": 0,
            "universe_symbol_count": 0,
        }

    history_bars = int(os.getenv("GNN_GRAPH_HISTORY_BARS", str(DEFAULT_GRAPH_HISTORY_BARS)))
    min_points = int(os.getenv("GNN_GRAPH_MIN_POINTS", str(DEFAULT_GRAPH_MIN_POINTS)))
    max_nodes = int(os.getenv("GNN_GRAPH_MAX_NODES", str(DEFAULT_GRAPH_MAX_NODES)))
    max_links = int(os.getenv("GNN_GRAPH_MAX_LINKS", str(DEFAULT_GRAPH_MAX_LINKS)))
    min_abs_correlation = float(
        os.getenv("GNN_GRAPH_MIN_ABS_CORRELATION", str(DEFAULT_GRAPH_MIN_ABS_CORRELATION))
    )

    selected_symbols = symbols[:max_nodes] if max_nodes > 0 else symbols
    best_attempt: dict[str, Any] | None = None

    for graph_timeframe in candidate_timeframes:
        series_by_symbol: dict[str, pd.Series] = {}
        missing_symbols: list[str] = []
        short_symbols: list[str] = []
        for symbol in selected_symbols:
            frame = load_history_frame(symbol, graph_timeframe, data_dir)
            if frame is None or "close" not in frame.columns:
                missing_symbols.append(symbol)
                continue
            returns = (
                frame["close"]
                .astype(float)
                .pct_change()
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .tail(history_bars)
            )
            if returns.shape[0] < min_points:
                short_symbols.append(symbol)
                continue
            series_by_symbol[symbol] = returns.rename(symbol)

        attempt = {
            "timeframe": graph_timeframe,
            "series_by_symbol": series_by_symbol,
            "missing_symbols": missing_symbols,
            "short_symbols": short_symbols,
            "overlap_points": 0,
        }
        if len(series_by_symbol) >= 2:
            matrix = pd.concat(series_by_symbol.values(), axis=1, join="inner").dropna()
            attempt["overlap_points"] = int(matrix.shape[0])
            if matrix.shape[0] >= min_points:
                correlation_matrix = matrix.corr().fillna(0.0)
                nodes: list[dict[str, Any]] = []
                core_links: list[dict[str, Any]] = []
                candidate_links: list[dict[str, Any]] = []

                symbols_in_graph = list(correlation_matrix.columns)
                for symbol in symbols_in_graph:
                    row = correlation_matrix.loc[symbol].drop(symbol, errors="ignore")
                    centrality = float(row.abs().mean()) if not row.empty else 0.0
                    nodes.append(
                        {
                            "id": symbol,
                            "label": symbol,
                            "role": "asset",
                            "family": classify_training_symbol(symbol),
                            "centrality": round(centrality, 4),
                            "timestamp": current_registry.get("trained_at"),
                        }
                    )
                    core_links.append(
                        {
                            "source": "market_core",
                            "target": symbol,
                            "value": round(max(centrality, 0.05), 4),
                            "kind": "core",
                        }
                    )

                for source_index, source_symbol in enumerate(symbols_in_graph):
                    for target_symbol in symbols_in_graph[source_index + 1 :]:
                        correlation = float(correlation_matrix.loc[source_symbol, target_symbol])
                        if abs(correlation) < min_abs_correlation:
                            continue
                        candidate_links.append(
                            {
                                "source": source_symbol,
                                "target": target_symbol,
                                "value": round(abs(correlation), 4),
                                "correlation": round(correlation, 4),
                                "kind": "correlation",
                            }
                        )

                candidate_links.sort(key=lambda item: item["value"], reverse=True)
                return {
                    "status": "ok",
                    "reason": "Projection reelle basee sur les correlations historiques du GNN.",
                    "nodes": [
                        {
                            "id": "market_core",
                            "label": "Market GNN Core",
                            "role": "core",
                            "family": "core",
                            "centrality": 1.0,
                            "timestamp": current_registry.get("trained_at"),
                        },
                        *nodes,
                    ],
                    "links": core_links + candidate_links[:max_links],
                    "graph_timeframe": graph_timeframe,
                    "selected_timeframe": graph_timeframe,
                    "candidate_timeframes": candidate_timeframes,
                    "overlap_points": int(matrix.shape[0]),
                    "missing_symbols": missing_symbols + short_symbols,
                    "displayed_symbol_count": len(symbols_in_graph),
                    "universe_symbol_count": len(symbols),
                    "correlation_points": int(matrix.shape[0]),
                }

        if best_attempt is None or int(attempt["overlap_points"]) > int(best_attempt.get("overlap_points", 0)):
            best_attempt = attempt

    best_attempt = best_attempt or {
        "timeframe": _resolve_graph_timeframe(candidate_timeframes),
        "series_by_symbol": {},
        "missing_symbols": selected_symbols,
        "short_symbols": [],
        "overlap_points": 0,
    }
    missing_symbols = list(best_attempt.get("missing_symbols") or []) + list(best_attempt.get("short_symbols") or [])
    overlap_points = int(best_attempt.get("overlap_points", 0) or 0)
    reason = "Historique insuffisant pour construire le graphe GNN."
    if overlap_points > 0:
        reason = "Les historiques ne se recoupent pas assez pour un graphe GNN fiable."
    return {
        "status": "unavailable",
        "reason": reason,
        "nodes": [],
        "links": [],
        "graph_timeframe": best_attempt.get("timeframe"),
        "selected_timeframe": None,
        "candidate_timeframes": candidate_timeframes,
        "overlap_points": overlap_points,
        "missing_symbols": missing_symbols,
        "displayed_symbol_count": len(best_attempt.get("series_by_symbol") or {}),
        "universe_symbol_count": len(symbols),
        "correlation_points": overlap_points,
    }
