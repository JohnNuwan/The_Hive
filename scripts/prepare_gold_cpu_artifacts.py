#!/usr/bin/env python3
"""Prepare les artefacts CPU Monday Gold pour Dreamer et GNN."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]


def _ensure_local_import_paths() -> None:
    """Ajoute les chemins source necessaires quel que soit le layout."""

    candidate_roots = [ROOT_DIR]
    candidate_roots.extend(ROOT_DIR.parents)
    seen: set[str] = set()

    for root in candidate_roots:
        root_str = str(root)
        if root_str in seen:
            continue
        seen.add(root_str)

        repo_shared = root / "src" / "shared"
        repo_lab = root / "src" / "eva-lab"
        flat_lab = root

        if repo_shared.exists() and repo_lab.exists():
            for path in (repo_shared, repo_lab):
                normalized = str(path)
                if normalized not in sys.path:
                    sys.path.insert(0, normalized)
            return

        if (flat_lab / "eva_lab").exists():
            normalized = str(flat_lab)
            if normalized not in sys.path:
                sys.path.insert(0, normalized)
            return


_ensure_local_import_paths()

from eva_lab.gold_cpu_prep import (
    build_default_gnn_cache_payload,
    build_dreamer_replay_cache_payload,
    load_dreamer_replay_cache,
    load_gnn_dataset_cache,
    resolve_cpu_prep_dir,
    save_dreamer_replay_cache,
)
from eva_lab.muzero.config import MuZeroConfigV3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_gold_cpu_artifacts")

DEFAULT_FOCUS_SYMBOL = "XAUUSD"
DEFAULT_CONTEXT_SYMBOLS = ["XAGUSD", "DXY.cash", "US500.cash", "EURUSD"]
DEFAULT_DREAMER_PROFILES = [
    {
        "name": "smoke_primary",
        "sequence_length": 24,
        "sequence_stride": 6,
    },
    {
        "name": "smoke_fallback",
        "sequence_length": 16,
        "sequence_stride": 4,
    },
    {
        "name": "gold_balanced_short_seq",
        "sequence_length": 32,
        "sequence_stride": 8,
    },
    {
        "name": "gold_fast_close",
        "sequence_length": 24,
        "sequence_stride": 6,
    },
    {
        "name": "gold_memory_mid",
        "sequence_length": 40,
        "sequence_stride": 8,
    },
]


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI du prepareur CPU.

    Returns:
        argparse.Namespace: Arguments resolves.
    """
    parser = argparse.ArgumentParser(description="Prepare les caches CPU Monday Gold.")
    parser.add_argument(
        "--mode",
        choices=["all", "dreamer", "gnn"],
        default="all",
        help="Sous-ensemble d'artefacts a preparer.",
    )
    parser.add_argument(
        "--focus-symbol",
        default=DEFAULT_FOCUS_SYMBOL,
        help="Actif principal cible.",
    )
    parser.add_argument(
        "--context-symbols",
        default=",".join(DEFAULT_CONTEXT_SYMBOLS),
        help="Actifs de contexte separes par des virgules.",
    )
    parser.add_argument(
        "--family",
        default="metals",
        help="Famille d'entrainement Dreamer.",
    )
    parser.add_argument(
        "--horizon",
        default="scalp",
        help="Horizon d'entrainement Dreamer.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("TRAINING_DATA_DIR", str(ROOT_DIR / "data" / "history")),
        help="Dossier historique de secours.",
    )
    parser.add_argument(
        "--shadow-dirs",
        default=os.getenv("DREAMER_SHADOW_DATA_DIRS", str(ROOT_DIR / "data" / "shadow_learning")),
        help="Dossiers shadow separes par os.pathsep.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("GOLD_CPU_PREP_DIR", str(ROOT_DIR / "data" / "checkpoints" / "gold_cpu_prep")),
        help="Dossier de cache cible.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstruit les artefacts meme si un cache existe deja.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Chemin de sortie du rapport JSON. Par defaut dans le dossier de cache.",
    )
    return parser.parse_args()


def _resolve_context_symbols(raw_value: str) -> list[str]:
    """Normalise la liste des symboles de contexte.

    Args:
        raw_value (str): Liste brute separee par des virgules.

    Returns:
        list[str]: Symboles distincts et ordonnes.
    """
    symbols: list[str] = []
    for item in raw_value.split(","):
        symbol = str(item).strip()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _prepare_gnn_cache(
    *,
    symbols: list[str],
    cache_dir: str,
    force: bool,
) -> dict[str, Any]:
    """Prepare le cache CPU GNN.

    Args:
        symbols (list[str]): Univers GNN cible.
        cache_dir (str): Dossier de cache.
        force (bool): Force la reconstruction.

    Returns:
        dict[str, Any]: Resume du cache genere ou reutilise.
    """
    cached = None if force else load_gnn_dataset_cache(symbols, cache_dir=cache_dir)
    if cached:
        logger.info("Cache CPU GNN reutilise: %s", cached["cache_path"])
        return {
            "status": "cached",
            "cache_path": cached["cache_path"],
            "valid_symbols": list(cached.get("valid_symbols") or []),
            "samples": min(
                len(cached["dataset"][symbol][timeframe]["labels"])
                for symbol in list(cached.get("valid_symbols") or [])
                for timeframe in cached["dataset"][symbol].keys()
            )
            if list(cached.get("valid_symbols") or [])
            else 0,
        }

    payload, cache_path = build_default_gnn_cache_payload(symbols)
    logger.info("Cache CPU GNN cree: %s", cache_path)
    return {
        "status": "prepared",
        "cache_path": str(cache_path),
        "valid_symbols": list(payload.get("valid_symbols") or []),
        "samples": min(
            len(payload["dataset"][symbol][timeframe]["labels"])
            for symbol in list(payload.get("valid_symbols") or [])
            for timeframe in payload["dataset"][symbol].keys()
        )
        if list(payload.get("valid_symbols") or [])
        else 0,
    }


def _prepare_dreamer_caches(
    *,
    symbols: list[str],
    horizon: str,
    family: str,
    data_dir: str,
    shadow_dirs: list[str],
    cache_dir: str,
    force: bool,
) -> list[dict[str, Any]]:
    """Prepare les caches Dreamer pour les profils Monday Gold.

    Args:
        symbols (list[str]): Univers Dreamer cible.
        horizon (str): Horizon cible.
        family (str): Famille cible.
        data_dir (str): Dossier historique de secours.
        shadow_dirs (list[str]): Dossiers shadow.
        cache_dir (str): Dossier de cache.
        force (bool): Force la reconstruction.

    Returns:
        list[dict[str, Any]]: Resumes par profil.
    """
    config = MuZeroConfigV3(horizon=horizon, model_family=family)
    results: list[dict[str, Any]] = []

    for profile in DEFAULT_DREAMER_PROFILES:
        existing = None
        if not force:
            existing = load_dreamer_replay_cache(
                horizon=horizon,
                family=family,
                symbols=symbols,
                sequence_length=int(profile["sequence_length"]),
                sequence_stride=int(profile["sequence_stride"]),
                cache_dir=cache_dir,
            )
        if existing:
            logger.info(
                "Cache Dreamer reutilise pour %s: %s",
                profile["name"],
                existing["cache_path"],
            )
            results.append(
                {
                    "profile": profile["name"],
                    "status": "cached",
                    "cache_path": existing["cache_path"],
                    "history_games": int(existing.get("history_games") or 0),
                    "shadow_games": int(existing.get("shadow_games") or 0),
                    "total_steps": int(existing.get("total_steps") or 0),
                }
            )
            continue

        payload = build_dreamer_replay_cache_payload(
            symbols=symbols,
            horizon=horizon,
            family=family,
            data_dir=data_dir,
            shadow_data_dirs=shadow_dirs,
            observation_size=int(config.observation_shape[0]),
            action_space_size=int(config.action_space_size),
            sequence_length=int(profile["sequence_length"]),
            sequence_stride=int(profile["sequence_stride"]),
        )
        cache_path = save_dreamer_replay_cache(payload, cache_dir=cache_dir)
        logger.info("Cache Dreamer cree pour %s: %s", profile["name"], cache_path)
        results.append(
            {
                "profile": profile["name"],
                "status": "prepared",
                "cache_path": str(cache_path),
                "history_games": int(payload.get("history_games") or 0),
                "shadow_games": int(payload.get("shadow_games") or 0),
                "total_steps": int(payload.get("total_steps") or 0),
            }
        )

    return results


def main() -> int:
    """Execute la preparation CPU Monday Gold.

    Returns:
        int: Code de retour POSIX.
    """
    args = parse_args()
    cache_dir = str(resolve_cpu_prep_dir(args.cache_dir))
    focus_symbol = str(args.focus_symbol).strip()
    context_symbols = _resolve_context_symbols(args.context_symbols)
    gnn_symbols = [focus_symbol, *[symbol for symbol in context_symbols if symbol != focus_symbol]]
    dreamer_symbols = [focus_symbol]
    shadow_dirs = [
        item.strip()
        for item in str(args.shadow_dirs).split(os.pathsep)
        if item.strip()
    ]

    report: dict[str, Any] = {
        "focus_symbol": focus_symbol,
        "context_symbols": context_symbols,
        "cache_dir": cache_dir,
        "dreamer": [],
        "gnn": None,
    }

    if args.mode in {"all", "dreamer"}:
        report["dreamer"] = _prepare_dreamer_caches(
            symbols=dreamer_symbols,
            horizon=str(args.horizon).strip().lower(),
            family=str(args.family).strip().lower(),
            data_dir=str(args.data_dir),
            shadow_dirs=shadow_dirs,
            cache_dir=cache_dir,
            force=bool(args.force),
        )

    if args.mode in {"all", "gnn"}:
        report["gnn"] = _prepare_gnn_cache(
            symbols=gnn_symbols,
            cache_dir=cache_dir,
            force=bool(args.force),
        )

    report_path = Path(args.report_path) if args.report_path else Path(cache_dir) / "gold_cpu_prep_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    logger.info("Rapport CPU Gold ecrit dans %s", report_path)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
