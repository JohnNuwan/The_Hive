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

from eva_lab.training_utils import classify_training_symbol, load_history_frame

logger = logging.getLogger(__name__)

DEFAULT_GNN_TIMEFRAMES = ["M5", "H1", "D1"]
DEFAULT_GRAPH_TIMEFRAME = "H1"
DEFAULT_STALE_HOURS = 168
DEFAULT_GRAPH_MAX_NODES = 18
DEFAULT_GRAPH_MIN_POINTS = 120
DEFAULT_GRAPH_HISTORY_BARS = 600
DEFAULT_GRAPH_MIN_ABS_CORRELATION = 0.25
DEFAULT_GRAPH_MAX_LINKS = 32


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
    registry_path = models_path / "gnn_registry.json"
    checkpoint_path = models_path / "gnn_master.pth"
    metrics_path = models_path / "gnn_master_metrics.json"

    previous_registry = _read_json(registry_path) or {}
    metrics_payload = _read_json(metrics_path) or {}
    metrics = _normalize_metrics(metrics_payload)
    checkpoint = _file_description(checkpoint_path)
    metrics_file = _file_description(metrics_path)

    trained_at = checkpoint.get("modified_at") or metrics_file.get("modified_at")
    symbols = list(dict.fromkeys(metrics_payload.get("symbols") or previous_registry.get("universe", {}).get("symbols") or []))
    timeframes = previous_registry.get("timeframes") or list(DEFAULT_GNN_TIMEFRAMES)
    status = _derive_status(previous_registry, checkpoint, metrics, trained_at)
    registry = {
        "name": "market_gnn",
        "version": str(previous_registry.get("version") or checkpoint_path.stem),
        "status": status,
        "trained_at": trained_at,
        "checkpoint_path": checkpoint["path"] if checkpoint["exists"] else None,
        "source_run_id": previous_registry.get("source_run_id"),
        "timeframes": [str(item).upper() for item in timeframes],
        "universe": {
            "symbols": symbols,
            "count": len(symbols),
            "family_counts": _build_family_counts(symbols),
        },
        "metrics": metrics,
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
    if not symbols:
        return {
            "status": "unavailable",
            "reason": "Aucun univers GNN versionne n'est disponible.",
            "nodes": [],
            "links": [],
            "graph_timeframe": None,
            "displayed_symbol_count": 0,
            "universe_symbol_count": 0,
        }

    graph_timeframe = _resolve_graph_timeframe(list(current_registry.get("timeframes") or []))
    history_bars = int(os.getenv("GNN_GRAPH_HISTORY_BARS", str(DEFAULT_GRAPH_HISTORY_BARS)))
    min_points = int(os.getenv("GNN_GRAPH_MIN_POINTS", str(DEFAULT_GRAPH_MIN_POINTS)))
    max_nodes = int(os.getenv("GNN_GRAPH_MAX_NODES", str(DEFAULT_GRAPH_MAX_NODES)))
    max_links = int(os.getenv("GNN_GRAPH_MAX_LINKS", str(DEFAULT_GRAPH_MAX_LINKS)))
    min_abs_correlation = float(
        os.getenv("GNN_GRAPH_MIN_ABS_CORRELATION", str(DEFAULT_GRAPH_MIN_ABS_CORRELATION))
    )

    selected_symbols = symbols[:max_nodes] if max_nodes > 0 else symbols
    series_by_symbol: dict[str, pd.Series] = {}
    for symbol in selected_symbols:
        frame = load_history_frame(symbol, graph_timeframe, data_dir)
        if frame is None or "close" not in frame.columns:
            continue
        returns = frame["close"].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(history_bars)
        if returns.shape[0] >= min_points:
            series_by_symbol[symbol] = returns.rename(symbol)

    if len(series_by_symbol) < 2:
        return {
            "status": "unavailable",
            "reason": "Historique insuffisant pour construire le graphe GNN.",
            "nodes": [],
            "links": [],
            "graph_timeframe": graph_timeframe,
            "displayed_symbol_count": len(series_by_symbol),
            "universe_symbol_count": len(symbols),
        }

    matrix = pd.concat(series_by_symbol.values(), axis=1, join="inner").dropna()
    if matrix.shape[0] < min_points:
        return {
            "status": "unavailable",
            "reason": "Les historiques ne se recoupent pas assez pour un graphe fiable.",
            "nodes": [],
            "links": [],
            "graph_timeframe": graph_timeframe,
            "displayed_symbol_count": len(series_by_symbol),
            "universe_symbol_count": len(symbols),
        }

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
    links = core_links + candidate_links[:max_links]
    graph_nodes = [
        {
            "id": "market_core",
            "label": "Market GNN Core",
            "role": "core",
            "family": "core",
            "centrality": 1.0,
            "timestamp": current_registry.get("trained_at"),
        },
        *nodes,
    ]

    return {
        "status": "ok",
        "reason": "Projection reelle basee sur les correlations historiques du GNN.",
        "nodes": graph_nodes,
        "links": links,
        "graph_timeframe": graph_timeframe,
        "displayed_symbol_count": len(symbols_in_graph),
        "universe_symbol_count": len(symbols),
        "correlation_points": int(matrix.shape[0]),
    }

