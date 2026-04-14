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
from typing import Any, Iterable

import MetaTrader5 as mt5
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependance optionnelle en CLI.
    load_dotenv = None

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:  # pragma: no cover - dependance optionnelle pour la persistence TimescaleDB.
    psycopg2 = None
    execute_values = None

LOCAL_ROOT = Path(__file__).resolve().parents[1]
EVA_LAB_SRC = LOCAL_ROOT / "src" / "eva-lab"
if EVA_LAB_SRC.is_dir():
    sys.path.insert(0, str(EVA_LAB_SRC))

try:
    from eva_lab.timescale_store import (
        evaluate_ohlc_write_request,
        get_timescale_runtime_status,
        get_timescale_settings,
    )
except ImportError:  # pragma: no cover - utile si le script est isole hors depot.
    evaluate_ohlc_write_request = None
    get_timescale_runtime_status = None
    get_timescale_settings = None

if load_dotenv is not None:
    load_dotenv()

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
    "M1": (mt5.TIMEFRAME_M1, int(os.getenv("HISTORY_M1_BARS", "100000"))),
    "M5": (mt5.TIMEFRAME_M5, int(os.getenv("HISTORY_M5_BARS", "120000"))),
    "M15": (mt5.TIMEFRAME_M15, int(os.getenv("HISTORY_M15_BARS", "40000"))),
    "H1": (mt5.TIMEFRAME_H1, int(os.getenv("HISTORY_H1_BARS", "30000"))),
    "D1": (mt5.TIMEFRAME_D1, int(os.getenv("HISTORY_D1_BARS", "2500"))),
    "W1": (mt5.TIMEFRAME_W1, int(os.getenv("HISTORY_W1_BARS", "1040"))),
}
TIMESCALE_BATCH_SIZE = int(os.getenv("HISTORY_TIMESCALE_BATCH_SIZE", "5000"))


def _sql_identifier(identifier: str) -> str:
    """Quote un identifiant SQL simple ou schema.table.

    Args:
        identifier (str): Identifiant brut.

    Returns:
        str: Identifiant quote pour PostgreSQL.

    Raises:
        ValueError: Si l'identifiant est vide.
    """

    parts = [part.strip() for part in str(identifier or "").split(".") if part.strip()]
    if not parts:
        raise ValueError("Identifiant SQL vide pour TimeScaleDB.")
    return ".".join(f'"{part.replace(chr(34), chr(34) * 2)}"' for part in parts)


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


class TimescaleWriter:
    """Persiste les bougies OHLC dans TimescaleDB en mode best-effort.

    Cette ecriture est volontairement optionnelle pour conserver un collecteur
    robuste: si TimescaleDB est absent ou mal configure, l'export CSV continue.
    """

    def __init__(self, enabled: bool) -> None:
        """Initialise l'ecrivain TimescaleDB.

        Args:
            enabled (bool): Active l'ecriture en base si ``True``.
        """
        self.enabled = bool(enabled)
        self._disabled_reason: str | None = None
        self._storage_profile = "balanced"
        self._bars_table = "market.market_bars"
        if not self.enabled:
            return
        if psycopg2 is None or execute_values is None:
            self.enabled = False
            self._disabled_reason = "psycopg2 indisponible"
            logger.warning(
                "Persistence TimescaleDB desactivee: %s. Les CSV restent la source de secours.",
                self._disabled_reason,
            )
            return
        if evaluate_ohlc_write_request is None or get_timescale_runtime_status is None:
            self.enabled = False
            self._disabled_reason = "socle eva_lab.timescale_store indisponible"
            logger.warning(
                "Persistence TimescaleDB desactivee: %s. Les CSV restent la source de secours.",
                self._disabled_reason,
            )
            return

        # Le flag CLI doit pouvoir activer explicitement l'ecriture OHLC.
        os.environ["TRAINING_TIMESCALE_ENABLED"] = "1"
        settings = get_timescale_settings() if get_timescale_settings is not None else {}
        self._storage_profile = str(settings.get("storage_profile") or "balanced")
        self._bars_table = str(settings.get("bars_table") or "market.market_bars")
        runtime = get_timescale_runtime_status(repair=True)
        if not bool(runtime.get("database_exists", False)):
            self._disable_writer("base applicative TimeScaleDB absente")
            return
        if not bool(runtime.get("extension_ready", False)):
            self._disable_writer("extension TimescaleDB absente")
            return
        if not bool(runtime.get("schema_ready", False)):
            self._disable_writer("schema TimeScaleDB incomplet")
            return
        guard_status = dict(runtime.get("write_guard_status") or {})
        if str(guard_status.get("status") or "") == "blocked":
            self._disable_writer("limite disque TimeScaleDB atteinte")
            return
        if str(guard_status.get("status") or "") == "degraded":
            logger.warning(
                "TimeScaleDB en mode degrade: taille=%s octets, profil=%s.",
                guard_status.get("db_size_bytes"),
                self._storage_profile,
            )

    def write_ohlc(self, symbol: str, timeframe_name: str, frame: pd.DataFrame) -> None:
        """Persiste un DataFrame OHLC dans TimescaleDB.

        Args:
            symbol (str): Symbole exporte.
            timeframe_name (str): Timeframe logique.
            frame (pd.DataFrame): Bougies OHLCV a inserer.
        """
        if not self.enabled:
            return
        if frame.empty:
            return

        guard_before = self._validate_write_guard(timeframe_name, repair=True)
        if guard_before is None:
            return

        rows = self._build_rows(symbol, timeframe_name, frame)
        if not rows:
            return

        try:
            inserted_rows = 0
            with psycopg2.connect(self._build_dsn()) as connection:
                with connection.cursor() as cursor:
                    for chunk in self._chunk_rows(rows):
                        execute_values(
                            cursor,
                            f"""
                            INSERT INTO {_sql_identifier(self._bars_table)} (
                                timestamp,
                                symbol,
                                timeframe,
                                open,
                                high,
                                low,
                                close,
                                tick_volume,
                                real_volume,
                                spread,
                                source,
                                ingested_at
                            )
                            VALUES %s
                            ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                tick_volume = EXCLUDED.tick_volume,
                                real_volume = EXCLUDED.real_volume,
                                spread = EXCLUDED.spread,
                                source = EXCLUDED.source,
                                ingested_at = EXCLUDED.ingested_at
                            """,
                            chunk,
                            page_size=min(len(chunk), TIMESCALE_BATCH_SIZE),
                        )
                        connection.commit()
                        inserted_rows += len(chunk)
                        guard_after_batch = self._validate_write_guard(timeframe_name, repair=False)
                        if guard_after_batch is None:
                            logger.warning(
                                "Arret de l'ecriture OHLC apres lot sur %s [%s].",
                                symbol,
                                timeframe_name,
                            )
                            break
            logger.info(
                "TimescaleDB mis a jour: %s [%s] (%s lignes, profil=%s).",
                symbol,
                timeframe_name,
                inserted_rows,
                self._storage_profile,
            )
        except Exception as exc:  # pragma: no cover - depend du service externe.
            self._disable_writer(
                f"echec d'ecriture sur {symbol} [{timeframe_name}]",
                details=str(exc),
            )

    @staticmethod
    def _build_rows(symbol: str, timeframe_name: str, frame: pd.DataFrame) -> list[tuple[Any, ...]]:
        """Construit la liste de tuples a inserer.

        Args:
            symbol (str): Symbole source.
            timeframe_name (str): Timeframe associe.
            frame (pd.DataFrame): DataFrame normalise.

        Returns:
            list[tuple[Any, ...]]: Lignes SQL prêtes a inserer.
        """
        rows: list[tuple[Any, ...]] = []
        for row in frame.itertuples(index=False):
            rows.append(
                (
                    row.time.to_pydatetime(),
                    symbol,
                    timeframe_name,
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                    int(getattr(row, "tick_volume", 0) or 0),
                    int(getattr(row, "real_volume", 0) or 0),
                    int(getattr(row, "spread", 0) or 0),
                    "mt5",
                    pd.Timestamp.utcnow().to_pydatetime(),
                )
            )
        return rows

    @staticmethod
    def _build_dsn() -> str:
        """Construit la chaine de connexion PostgreSQL.

        Returns:
            str: DSN de connexion TimescaleDB.
        """
        settings = get_timescale_settings() if get_timescale_settings is not None else {}
        host = str(
            settings.get("host")
            or os.getenv("TRAINING_TIMESCALE_HOST")
            or os.getenv("TIMESCALE_HOST")
            or "localhost"
        )
        port = str(
            settings.get("port")
            or os.getenv("TRAINING_TIMESCALE_PORT")
            or os.getenv("TIMESCALE_PORT")
            or "5432"
        )
        database = str(
            settings.get("database")
            or os.getenv("TRAINING_TIMESCALE_DB")
            or os.getenv("TIMESCALE_DB")
            or "thehive"
        )
        user = str(
            settings.get("user")
            or os.getenv("TRAINING_TIMESCALE_USER")
            or os.getenv("TIMESCALE_USER")
            or "eva"
        )
        password = str(
            settings.get("password")
            or os.getenv("TRAINING_TIMESCALE_PASSWORD")
            or os.getenv("TIMESCALE_PASSWORD")
            or ""
        )
        return (
            f"host={host} port={port} dbname={database} "
            f"user={user} password={password}"
        )

    @staticmethod
    def _chunk_rows(rows: list[tuple[Any, ...]]) -> Iterable[list[tuple[Any, ...]]]:
        """Decoupe un lot de lignes pour verifier le garde-fou regulierement.

        Args:
            rows (list[tuple[Any, ...]]): Lignes a inserer.

        Yields:
            Iterable[list[tuple[Any, ...]]]: Paquets de taille controlee.
        """

        if not rows:
            return
        batch_size = max(TIMESCALE_BATCH_SIZE, 1)
        for index in range(0, len(rows), batch_size):
            yield rows[index:index + batch_size]

    def _disable_writer(self, reason: str, *, details: str | None = None) -> None:
        """Desactive l'ecriture TimescaleDB apres un echec structurel.

        Args:
            reason (str): Raison stable de desactivation.
            details (str | None): Detail technique optionnel.
        """

        self.enabled = False
        self._disabled_reason = str(reason or "").strip() or "raison_inconnue"
        if details:
            logger.warning(
                "Ecriture TimescaleDB desactivee: %s (%s).",
                self._disabled_reason,
                details,
            )
        else:
            logger.warning(
                "Ecriture TimescaleDB desactivee: %s.",
                self._disabled_reason,
            )

    def _validate_write_guard(self, timeframe_name: str, *, repair: bool) -> dict[str, Any] | None:
        """Valide le timeframe et la volumetrie avant ou apres un lot.

        Args:
            timeframe_name (str): Timeframe a valider.
            repair (bool): Tente un bootstrap idempotent si ``True``.

        Returns:
            dict[str, Any] | None: Diagnostic d'autorisation ou ``None`` si ecriture refusee.
        """

        if evaluate_ohlc_write_request is None:
            self._disable_writer("diagnostic TimeScaleDB indisponible")
            return None
        guard = evaluate_ohlc_write_request(timeframe_name, repair=repair)
        status = str(guard.get("status") or "")
        reason = str(guard.get("reason") or "")
        if status == "timeframe_blocked":
            logger.warning(
                "Ecriture OHLC refusee pour [%s]: timeframe non autorise. Autorises=%s.",
                timeframe_name,
                ",".join(guard.get("allowed_timeframes", [])),
            )
            return None
        if not bool(guard.get("allowed", False)):
            message = f"garde-fou TimeScaleDB bloque ({status or 'unknown'} / {reason or 'unknown'})"
            if status in {"blocked", "unavailable", "invalid_request"}:
                self._disable_writer(message, details=str(guard.get("last_bootstrap_error") or ""))
            else:
                logger.warning("Ecriture OHLC refusee: %s.", message)
            return None
        if status == "degraded":
            logger.warning(
                "Ecriture OHLC en mode degrade pour [%s]: taille=%s octets, seuil soft=%s.",
                timeframe_name,
                guard.get("db_size_bytes"),
                guard.get("soft_limit_bytes"),
            )
        return guard


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
        "--timeframes",
        default=os.getenv("HISTORY_TIMEFRAMES", ",".join(TIMEFRAMES.keys())),
        help="Timeframes a exporter, separes par des virgules (ex: M5,H1,D1).",
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
    parser.add_argument(
        "--write-timescale",
        action="store_true",
        default=os.getenv("HISTORY_WRITE_TIMESCALE", "0").strip().lower() in {"1", "true", "yes", "on"},
        help="Ecrit aussi les bougies dans TimescaleDB en plus des CSV.",
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


def fetch_data(
    symbol: str,
    timeframe_name: str,
    timeframe_value: int,
    count: int,
    timescale_writer: TimescaleWriter | None = None,
) -> Path | None:
    """Recupere les bougies MT5 et ecrit le CSV local.

    Args:
        symbol (str): Symbole a exporter.
        timeframe_name (str): Nom logique du timeframe.
        timeframe_value (int): Constante MT5 du timeframe.
        count (int): Nombre de bougies a demander.
        timescale_writer (TimescaleWriter | None): Ecrivain TimescaleDB optionnel.

    Returns:
        Path | None: Fichier CSV ecrit ou ``None`` si echec.
    """
    logger.info("Recuperation %s [%s] - %s bougies", symbol, timeframe_name, count)
    if not mt5.symbol_select(symbol, True):
        logger.warning("Selection MT5 impossible pour %s: %s", symbol, mt5.last_error())
        return None

    rates = _copy_rates_chunked(symbol, timeframe_value, count)
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
    if timescale_writer is not None:
        timescale_writer.write_ohlc(symbol, timeframe_name, frame)
    return output_path


def _copy_rates_chunked(symbol: str, timeframe_value: int, count: int) -> object | None:
    """Recupere un historique MT5 par paquets pour eviter les limites broker.

    Args:
        symbol (str): Symbole a recuperer.
        timeframe_value (int): Constante MT5 du timeframe.
        count (int): Nombre total de bougies souhaite.

    Returns:
        object | None: Tableau ``rates`` concatene ou ``None`` si aucun lot n'est disponible.
    """
    chunk_size = max(1000, int(os.getenv("HISTORY_FETCH_CHUNK_SIZE", "40000")))
    frames: list[pd.DataFrame] = []
    offset = 0

    while offset < count:
        request_size = min(chunk_size, count - offset)
        chunk = mt5.copy_rates_from_pos(symbol, timeframe_value, offset, request_size)
        if chunk is None:
            if offset == 0:
                return None
            logger.warning(
                "Historique partiel pour %s [%s]: arret a %s bougies (%s).",
                symbol,
                timeframe_value,
                offset,
                mt5.last_error(),
            )
            break
        chunk_frame = pd.DataFrame(chunk)
        if chunk_frame.empty:
            break
        frames.append(chunk_frame)
        fetched = len(chunk_frame)
        offset += fetched
        if fetched < request_size:
            break

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["time"]).sort_values("time")
    return merged.to_records(index=False)


def write_inventory(selected: Iterable[SymbolCandidate], generated_files: list[Path]) -> None:
    """Ecrit un inventaire JSON des symboles et fichiers produits.

    Args:
        selected (Iterable[SymbolCandidate]): Symboles retenus.
        generated_files (list[Path]): Fichiers CSV ecrits.
    """
    selected_list = list(selected)
    existing_files = sorted(OUTPUT_DIR.glob("*.csv"))
    inventory = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "symbols": [
            {"name": item.name, "category": item.category, "reason": item.reason}
            for item in selected_list
        ],
        "files": [str(path) for path in existing_files],
        "files_generated_this_run": [str(path) for path in generated_files],
        "counts": {
            "symbols": len(selected_list),
            "files": len(existing_files),
            "files_generated_this_run": len(generated_files),
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
    timescale_writer = TimescaleWriter(enabled=args.write_timescale)

    if not mt5.initialize():
        logger.error("Initialisation MT5 impossible: %s", mt5.last_error())
        return 1

    logger.info("Version MT5: %s", mt5.version())
    try:
        selected = select_target_symbols(args)
        if not selected:
            logger.error("Aucun symbole retenu pour l'univers d'entrainement.")
            return 2

        requested_timeframes = [
            item.strip().upper()
            for item in str(args.timeframes).split(",")
            if item.strip()
        ]
        invalid_timeframes = [item for item in requested_timeframes if item not in TIMEFRAMES]
        if invalid_timeframes:
            logger.error("Timeframes inconnus: %s", ", ".join(invalid_timeframes))
            return 3

        logger.info("Univers historique retenu: %s symboles", len(selected))
        by_category: dict[str, int] = defaultdict(int)
        for item in selected:
            by_category[item.category] += 1
        for category, count in sorted(by_category.items()):
            logger.info(" - %s: %s", category, count)

        generated_files: list[Path] = []
        for candidate in selected:
            for timeframe_name in requested_timeframes:
                timeframe_value, count = TIMEFRAMES[timeframe_name]
                output_path = fetch_data(
                    candidate.name,
                    timeframe_name,
                    timeframe_value,
                    count,
                    timescale_writer=timescale_writer,
                )
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
