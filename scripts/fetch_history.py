"""Recupere un univers MT5 et exporte les historiques d'entrainement."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import MetaTrader5 as mt5
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_history")

OUTPUT_DIR = Path("data") / "history"
INVENTORY_PATH = OUTPUT_DIR / "inventory.json"
FOREX_QUOTES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"}
CRYPTO_TOKENS = {
    "BTC", "ETH", "ADA", "BNB", "DOGE", "DOT", "LTC", "SOL", "XRP", "UNI", "LINK", "AVAX", "MATIC",
    "ATOM", "NEAR", "ETC", "FIL", "APT", "ARB", "OP", "TRX", "BCH", "XLM", "SUI", "PEPE",
}
INDEX_HINTS = {"US30", "US500", "USTEC", "NAS100", "GER40", "UK100", "JP225", "FRA40", "SPX500", "AUS200"}
TIMEFRAMES: dict[str, tuple[int, int]] = {
    "M5": (mt5.TIMEFRAME_M5, int(os.getenv("HISTORY_M5_BARS", "50000"))),
    "H1": (mt5.TIMEFRAME_H1, int(os.getenv("HISTORY_H1_BARS", "10000"))),
}


@dataclass
class SymbolCandidate:
    """Decrit un symbole retenu pour l'export historique.

    Args:
        name (str): Nom MT5 du symbole.
        category (str): Categorie deduite pour le symbole.
        reason (str): Raison textuelle du classement.
    """

    name: str
    category: str
    reason: str


def parse_args() -> argparse.Namespace:
    """Construit les arguments CLI du collecteur historique.

    Returns:
        argparse.Namespace: Arguments resolves.
    """
    parser = argparse.ArgumentParser(description="Exporte un univers MT5 et ses historiques CSV.")
    parser.add_argument(
        "--classes",
        default=os.getenv("HISTORY_UNIVERSE_CLASSES", "forex,cfd,crypto,metals"),
        help="Classes d'actifs a exporter, separees par des virgules.",
    )
    parser.add_argument(
        "--symbols",
        default=os.getenv("TRAINING_SYMBOLS", ""),
        help="Liste explicite de symboles a exporter. Prioritaire sur la decouverte.",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=int(os.getenv("HISTORY_FETCH_SLEEP_MS", "250")),
        help="Pause entre deux requetes MT5 en millisecondes.",
    )
    parser.add_argument(
        "--max-forex",
        type=int,
        default=int(os.getenv("HISTORY_MAX_FOREX", "28")),
        help="Nombre max de symboles Forex.",
    )
    parser.add_argument(
        "--max-cfd",
        type=int,
        default=int(os.getenv("HISTORY_MAX_CFD", "20")),
        help="Nombre max de symboles CFD/indices.",
    )
    parser.add_argument(
        "--max-crypto",
        type=int,
        default=int(os.getenv("HISTORY_MAX_CRYPTO", "20")),
        help="Nombre max de symboles crypto.",
    )
    parser.add_argument(
        "--max-metals",
        type=int,
        default=int(os.getenv("HISTORY_MAX_METALS", "6")),
        help="Nombre max de symboles metaux.",
    )
    return parser.parse_args()


def _normalize_symbol_name(name: str) -> str:
    """Nettoie un nom de symbole pour les heuristiques.

    Args:
        name (str): Nom brut MT5.

    Returns:
        str: Nom simplifie en majuscules.
    """
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def _is_forex_symbol(name: str, path: str, description: str) -> bool:
    """Determine si un symbole ressemble a une paire Forex.

    Args:
        name (str): Nom du symbole.
        path (str): Chemin MT5 du symbole.
        description (str): Description broker.

    Returns:
        bool: ``True`` si le symbole est classe Forex.
    """
    normalized = _normalize_symbol_name(name)
    if "FOREX" in path or "FOREX" in description:
        return True
    if len(normalized) < 6:
        return False
    prefix = normalized[:3]
    suffix = normalized[3:6]
    return prefix in FOREX_QUOTES and suffix in FOREX_QUOTES


def _is_metal_symbol(name: str, path: str, description: str) -> bool:
    """Determine si un symbole correspond aux metaux.

    Args:
        name (str): Nom du symbole.
        path (str): Chemin MT5 du symbole.
        description (str): Description broker.

    Returns:
        bool: ``True`` si le symbole est un metal.
    """
    haystack = f"{name} {path} {description}".upper()
    return any(token in haystack for token in {"XAU", "XAG", "GOLD", "SILVER"})


def _is_crypto_symbol(name: str, path: str, description: str) -> bool:
    """Determine si un symbole correspond a la crypto.

    Args:
        name (str): Nom du symbole.
        path (str): Chemin MT5 du symbole.
        description (str): Description broker.

    Returns:
        bool: ``True`` si le symbole est une crypto.
    """
    haystack = f"{name} {path} {description}".upper()
    if "CRYPTO" in haystack:
        return True
    normalized = _normalize_symbol_name(name)
    for token in CRYPTO_TOKENS:
        if normalized.startswith(token) and any(quote in normalized for quote in {"USD", "USDT", "EUR"}):
            return True
    return False


def _is_cfd_symbol(name: str, path: str, description: str) -> bool:
    """Determine si un symbole correspond a un CFD ou indice.

    Args:
        name (str): Nom du symbole.
        path (str): Chemin MT5 du symbole.
        description (str): Description broker.

    Returns:
        bool: ``True`` si le symbole est classe CFD.
    """
    haystack = f"{name} {path} {description}".upper()
    normalized = _normalize_symbol_name(name)
    if ".CASH" in haystack or "INDEX" in haystack or "INDICES" in haystack or "CFD" in haystack:
        return True
    return any(token in normalized for token in INDEX_HINTS)


def classify_symbol(symbol_info: object) -> SymbolCandidate | None:
    """Classe un symbole MT5 dans un univers d'entrainement.

    Args:
        symbol_info (object): Objet renvoye par ``mt5.symbols_get``.

    Returns:
        SymbolCandidate | None: Symbole classe ou ``None`` si ignore.
    """
    name = str(getattr(symbol_info, "name", "") or "")
    path = str(getattr(symbol_info, "path", "") or "")
    description = str(getattr(symbol_info, "description", "") or "")
    if not name:
        return None
    if _is_metal_symbol(name, path, description):
        return SymbolCandidate(name=name, category="metals", reason="heuristique_metaux")
    if _is_crypto_symbol(name, path, description):
        return SymbolCandidate(name=name, category="crypto", reason="heuristique_crypto")
    if _is_forex_symbol(name, path, description):
        return SymbolCandidate(name=name, category="forex", reason="heuristique_forex")
    if _is_cfd_symbol(name, path, description):
        return SymbolCandidate(name=name, category="cfd", reason="heuristique_cfd")
    return None


def _preferred_rank(category: str, symbol: str) -> tuple[int, str]:
    """Retourne un ordre de priorite stable pour les symboles.

    Args:
        category (str): Categorie de l'actif.
        symbol (str): Nom du symbole.

    Returns:
        tuple[int, str]: Cle de tri stable.
    """
    priority_map = {
        "BTCUSD": 0,
        "ETHUSD": 1,
        "XAUUSD": 2,
        "EURUSD": 3,
        "GBPUSD": 4,
        "USDJPY": 5,
        "US30.cash": 6,
        "US500.cash": 7,
        "GER40.cash": 8,
    }
    base_rank = priority_map.get(symbol, 999)
    if category == "metals" and symbol.startswith("XAU"):
        base_rank = min(base_rank, 1)
    return (base_rank, symbol)


def select_target_symbols(args: argparse.Namespace) -> list[SymbolCandidate]:
    """Construit la liste finale des symboles a exporter.

    Args:
        args (argparse.Namespace): Arguments CLI.

    Returns:
        list[SymbolCandidate]: Symboles retenus pour l'export.
    """
    manual_symbols = [item.strip() for item in str(args.symbols or "").split(",") if item.strip()]
    if manual_symbols:
        return [SymbolCandidate(name=symbol, category="manual", reason="selection_manuelle") for symbol in manual_symbols]

    requested_classes = {item.strip().lower() for item in str(args.classes).split(",") if item.strip()}
    limits = {
        "forex": args.max_forex,
        "cfd": args.max_cfd,
        "crypto": args.max_crypto,
        "metals": args.max_metals,
    }
    raw_symbols = mt5.symbols_get()
    if raw_symbols is None:
        raise RuntimeError(f"Lecture des symboles MT5 impossible: {mt5.last_error()}")

    grouped: dict[str, list[SymbolCandidate]] = defaultdict(list)
    for symbol_info in raw_symbols:
        candidate = classify_symbol(symbol_info)
        if candidate is None or candidate.category not in requested_classes:
            continue
        grouped[candidate.category].append(candidate)

    selected: list[SymbolCandidate] = []
    for category in ["forex", "cfd", "crypto", "metals"]:
        if category not in requested_classes:
            continue
        ranked = sorted(grouped.get(category, []), key=lambda item: _preferred_rank(category, item.name))
        limit = limits.get(category, 0)
        if limit > 0:
            ranked = ranked[:limit]
        selected.extend(ranked)

    deduped: list[SymbolCandidate] = []
    seen: set[str] = set()
    for candidate in selected:
        if candidate.name in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate.name)
    return deduped


def fetch_data(symbol: str, timeframe_name: str, timeframe_value: int, count: int) -> Path | None:
    """Recupere les bougies MT5 et ecrit le CSV local.

    Args:
        symbol (str): Symbole a exporter.
        timeframe_name (str): Nom logique du timeframe.
        timeframe_value (int): Constante MT5 du timeframe.
        count (int): Nombre de bougies a demander.

    Returns:
        Path | None: Fichier CSV ecrit ou ``None`` si echec.
    """
    logger.info("Recuperation %s [%s] - %s bougies", symbol, timeframe_name, count)
    if not mt5.symbol_select(symbol, True):
        logger.warning("Selection MT5 impossible pour %s: %s", symbol, mt5.last_error())
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe_value, 0, count)
    if rates is None:
        logger.warning("Historique indisponible pour %s [%s]: %s", symbol, timeframe_name, mt5.last_error())
        return None

    frame = pd.DataFrame(rates)
    if frame.empty:
        logger.warning("Historique vide pour %s [%s].", symbol, timeframe_name)
        return None
    frame["time"] = pd.to_datetime(frame["time"], unit="s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{symbol}_{timeframe_name}.csv"
    frame.to_csv(output_path, index=False)
    logger.info("CSV ecrit: %s (%s lignes)", output_path, len(frame))
    return output_path


def write_inventory(selected: Iterable[SymbolCandidate], generated_files: list[Path]) -> None:
    """Ecrit un inventaire JSON des symboles et fichiers produits.

    Args:
        selected (Iterable[SymbolCandidate]): Symboles retenus.
        generated_files (list[Path]): Fichiers CSV ecrits.
    """
    inventory = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "symbols": [
            {"name": item.name, "category": item.category, "reason": item.reason}
            for item in selected
        ],
        "files": [str(path) for path in generated_files],
        "counts": {
            "symbols": len(list(selected)),
            "files": len(generated_files),
        },
    }
    INVENTORY_PATH.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    logger.info("Inventaire historique mis a jour: %s", INVENTORY_PATH)


def main() -> int:
    """Initialise MT5, decouvre l'univers et exporte les historiques.

    Returns:
        int: Code de retour processus.
    """
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not mt5.initialize():
        logger.error("Initialisation MT5 impossible: %s", mt5.last_error())
        return 1

    logger.info("Version MT5: %s", mt5.version())
    try:
        selected = select_target_symbols(args)
        if not selected:
            logger.error("Aucun symbole retenu pour l'univers d'entrainement.")
            return 2

        logger.info("Univers historique retenu: %s symboles", len(selected))
        by_category: dict[str, int] = defaultdict(int)
        for item in selected:
            by_category[item.category] += 1
        for category, count in sorted(by_category.items()):
            logger.info(" - %s: %s", category, count)

        generated_files: list[Path] = []
        for candidate in selected:
            for timeframe_name, (timeframe_value, count) in TIMEFRAMES.items():
                output_path = fetch_data(candidate.name, timeframe_name, timeframe_value, count)
                if output_path is not None:
                    generated_files.append(output_path)
                time.sleep(max(args.sleep_ms, 0) / 1000.0)

        write_inventory(selected, generated_files)
        logger.info("Export historique termine: %s fichiers.", len(generated_files))
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
