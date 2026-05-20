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

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - autorise l'injection Timescale sans terminal MT5.
    mt5 = None

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

try:
    from eva_lab.training_utils import resolve_feature_profile
    from shared.indicators import IndicatorFactory
except ImportError:  # pragma: no cover - utile si le script est lance hors pile complete.
    resolve_feature_profile = None
    IndicatorFactory = None

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
    "M1": ((mt5.TIMEFRAME_M1 if mt5 is not None else 1), int(os.getenv("HISTORY_M1_BARS", "100000"))),
    "M5": ((mt5.TIMEFRAME_M5 if mt5 is not None else 5), int(os.getenv("HISTORY_M5_BARS", "120000"))),
    "M15": ((mt5.TIMEFRAME_M15 if mt5 is not None else 15), int(os.getenv("HISTORY_M15_BARS", "40000"))),
    "H1": ((mt5.TIMEFRAME_H1 if mt5 is not None else 60), int(os.getenv("HISTORY_H1_BARS", "30000"))),
    "D1": ((mt5.TIMEFRAME_D1 if mt5 is not None else 1440), int(os.getenv("HISTORY_D1_BARS", "2500"))),
    "W1": ((mt5.TIMEFRAME_W1 if mt5 is not None else 10080), int(os.getenv("HISTORY_W1_BARS", "1040"))),
}
TIMESCALE_BATCH_SIZE = int(os.getenv("HISTORY_TIMESCALE_BATCH_SIZE", "5000"))
METAL_PREFIXES = ("XAU", "XAG", "XPT", "XPD")


def _as_bool(value: str | None) -> bool:
    """Convertit une valeur texte en booleen.

    Args:
        value (str | None): Valeur brute a interpreter.

    Returns:
        bool: True si la valeur represente un booleen actif.
    """

    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


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


def _infer_training_family(symbol: str) -> str:
    """Deduit la famille logique d'un symbole pour les features.

    Args:
        symbol (str): Symbole brut issu de MT5.

    Returns:
        str: Famille canonique (`forex`, `index_cfd`, `crypto` ou `metal`).
    """

    normalized = str(symbol or "").strip().upper()
    if any(normalized.startswith(prefix) for prefix in METAL_PREFIXES):
        return "metal"
    if any(token in normalized for token in INDEX_HINTS) or normalized.endswith(".CASH"):
        return "index_cfd"
    if any(token in normalized for token in CRYPTO_TOKENS) or normalized.endswith("USD.E"):
        return "crypto"
    if len(normalized) >= 6 and normalized[:3] in FOREX_QUOTES and normalized[3:6] in FOREX_QUOTES:
        return "forex"
    return "index_cfd"


def _resolve_session_phase(timestamp: pd.Timestamp) -> str:
    """Mappe une horodate vers une phase de session simplifiee.

    Args:
        timestamp (pd.Timestamp): Horodatage de la bougie.

    Returns:
        str: Session simplifiee exploitable par les features.
    """

    hour = int(pd.Timestamp(timestamp).hour)
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "europe_open"
    if 12 <= hour < 17:
        return "london_ny_overlap"
    if 17 <= hour < 22:
        return "ny"
    return "late"


def _classify_adx_regime(adx_value: float) -> str:
    """Traduit une valeur ADX en regime textuel stable.

    Args:
        adx_value (float): Valeur ADX courante.

    Returns:
        str: Regime synthétique.
    """

    value = float(adx_value or 0.0)
    if value >= 30.0:
        return "strong_trend"
    if value >= 20.0:
        return "trend"
    return "range"


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
        self._features_table = "market.market_features"
        self._allowed_feature_timeframes = {"M5", "H1", "D1"}
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
        self._features_table = str(settings.get("features_table") or "market.market_features")
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
                    cursor.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
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
            self.write_features(symbol, timeframe_name, frame)
        except Exception as exc:  # pragma: no cover - depend du service externe.
            self._disable_writer(
                f"echec d'ecriture sur {symbol} [{timeframe_name}]",
                details=str(exc),
            )

    def write_features(self, symbol: str, timeframe_name: str, frame: pd.DataFrame) -> None:
        """Persiste les features de marche canoniques dans TimescaleDB.

        Args:
            symbol (str): Symbole exporte.
            timeframe_name (str): Timeframe logique.
            frame (pd.DataFrame): Bougies OHLCV source.
        """
        if not self.enabled:
            return
        if IndicatorFactory is None or resolve_feature_profile is None:
            return
        if timeframe_name.upper() not in self._allowed_feature_timeframes:
            return
        if frame.empty:
            return

        family = _infer_training_family(symbol)
        feature_profile = resolve_feature_profile("scalp", family)
        profile_name = str(feature_profile.get("profile_name") or "scalp_unknown_v1").strip()
        feature_rows = self._build_feature_rows(symbol, timeframe_name, frame, family, profile_name, feature_profile)
        if not feature_rows:
            return

        try:
            with psycopg2.connect(self._build_dsn()) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
                    for chunk in self._chunk_rows(feature_rows):
                        execute_values(
                            cursor,
                            f"""
                            INSERT INTO {_sql_identifier(self._features_table)} (
                                timestamp,
                                symbol,
                                timeframe,
                                feature_profile,
                                ema_fast,
                                ema_slow,
                                ema200,
                                vwap,
                                obv,
                                rsi,
                                adx,
                                atr,
                                bb_width,
                                relative_volume,
                                session_phase,
                                payload,
                                computed_at
                            )
                            VALUES %s
                            ON CONFLICT (symbol, timeframe, feature_profile, timestamp) DO UPDATE SET
                                ema_fast = EXCLUDED.ema_fast,
                                ema_slow = EXCLUDED.ema_slow,
                                ema200 = EXCLUDED.ema200,
                                vwap = EXCLUDED.vwap,
                                obv = EXCLUDED.obv,
                                rsi = EXCLUDED.rsi,
                                adx = EXCLUDED.adx,
                                atr = EXCLUDED.atr,
                                bb_width = EXCLUDED.bb_width,
                                relative_volume = EXCLUDED.relative_volume,
                                session_phase = EXCLUDED.session_phase,
                                payload = EXCLUDED.payload,
                                computed_at = EXCLUDED.computed_at
                            """,
                            chunk,
                            template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)",
                            page_size=min(len(chunk), TIMESCALE_BATCH_SIZE),
                        )
                    connection.commit()
            logger.info(
                "TimescaleDB features mis a jour: %s [%s] (%s lignes, profil=%s).",
                symbol,
                timeframe_name,
                len(feature_rows),
                profile_name,
            )
        except Exception as exc:  # pragma: no cover - depend du service externe.
            logger.warning(
                "Ecriture TimeScaleDB des features impossible pour %s [%s]: %s",
                symbol,
                timeframe_name,
                exc,
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
    def _build_feature_rows(
        symbol: str,
        timeframe_name: str,
        frame: pd.DataFrame,
        family: str,
        profile_name: str,
        feature_profile: dict[str, Any],
    ) -> list[tuple[Any, ...]]:
        """Construit les lignes de features canoniques a inserer.

        Args:
            symbol (str): Symbole source.
            timeframe_name (str): Timeframe logique.
            frame (pd.DataFrame): Bougies OHLCV normalisees.
            family (str): Famille logique du symbole.
            profile_name (str): Nom de profil de features.
            feature_profile (dict[str, Any]): Profil de features resolu.

        Returns:
            list[tuple[Any, ...]]: Lignes pretes pour l'UPSERT SQL.
        """
        enriched = frame.copy()
        enriched["time"] = pd.to_datetime(enriched["time"], utc=False)
        enriched = enriched.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

        for column in ["tick_volume", "real_volume", "spread"]:
            if column not in enriched.columns:
                enriched[column] = 0.0

        close = enriched["close"].astype(float)
        high = enriched["high"].astype(float)
        low = enriched["low"].astype(float)
        volume = enriched["tick_volume"].astype(float)

        enriched["ema_fast"] = close.ewm(span=20, adjust=False).mean()
        enriched["ema_slow"] = close.ewm(span=50, adjust=False).mean()
        enriched["ema200"] = close.ewm(span=200, adjust=False).mean()
        enriched["vwap"] = IndicatorFactory.vwap(high, low, close, volume)
        enriched["obv"] = IndicatorFactory.obv(close, volume)
        enriched["rsi"] = IndicatorFactory.rsi(close, 14)
        enriched["adx"] = IndicatorFactory.adx(high, low, close)["adx"]
        enriched["atr"] = IndicatorFactory.atr(high, low, close, 14)
        bollinger = IndicatorFactory.bollinger_bands(close)
        enriched["bb_width"] = (bollinger["upper"] - bollinger["lower"]) / close.replace(0, pd.NA)
        volume_baseline = volume.replace(0.0, pd.NA).rolling(20, min_periods=1).mean()
        enriched["relative_volume"] = volume / volume_baseline
        enriched["price_vs_vwap"] = (close - enriched["vwap"]) / close.replace(0, pd.NA)
        enriched["obv_slope"] = enriched["obv"].diff().fillna(0.0)
        enriched["atr_pct"] = enriched["atr"] / close.replace(0, pd.NA)
        enriched["momentum"] = IndicatorFactory.momentum(close)
        enriched["adx_regime"] = enriched["adx"].apply(_classify_adx_regime)
        enriched["session_phase"] = enriched["time"].apply(_resolve_session_phase)
        enriched = enriched.ffill().bfill().fillna(0.0)

        computed_at = pd.Timestamp.utcnow().to_pydatetime()
        rows: list[tuple[Any, ...]] = []
        for row in enriched.itertuples(index=False):
            payload = {
                "family": family,
                "feature_version": str(feature_profile.get("feature_version") or "v1"),
                "profile_version": str(feature_profile.get("profile_version") or "v1"),
                "entry_features": list(feature_profile.get("entry_features") or []),
                "audit_features": list(feature_profile.get("audit_features") or []),
                "price_vs_vwap": float(getattr(row, "price_vs_vwap", 0.0) or 0.0),
                "obv_slope": float(getattr(row, "obv_slope", 0.0) or 0.0),
                "atr_pct": float(getattr(row, "atr_pct", 0.0) or 0.0),
                "momentum": float(getattr(row, "momentum", 0.0) or 0.0),
                "adx_regime": str(getattr(row, "adx_regime", "flat") or "flat"),
            }
            rows.append(
                (
                    row.time.to_pydatetime() if hasattr(row.time, "to_pydatetime") else row.time,
                    symbol,
                    timeframe_name,
                    profile_name,
                    float(getattr(row, "ema_fast", 0.0) or 0.0),
                    float(getattr(row, "ema_slow", 0.0) or 0.0),
                    float(getattr(row, "ema200", 0.0) or 0.0),
                    float(getattr(row, "vwap", 0.0) or 0.0),
                    float(getattr(row, "obv", 0.0) or 0.0),
                    float(getattr(row, "rsi", 0.0) or 0.0),
                    float(getattr(row, "adx", 0.0) or 0.0),
                    float(getattr(row, "atr", 0.0) or 0.0),
                    float(getattr(row, "bb_width", 0.0) or 0.0),
                    float(getattr(row, "relative_volume", 0.0) or 0.0),
                    str(getattr(row, "session_phase", "unknown") or "unknown"),
                    json.dumps(payload, ensure_ascii=False, default=str),
                    computed_at,
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
    parser.add_argument(
        "--ingest-existing-only",
        action="store_true",
        help="Ignore MT5 et pousse uniquement les CSV deja presents vers TimescaleDB.",
    )
    parser.add_argument(
        "--terminal-path",
        default=os.getenv("HISTORY_MT5_TERMINAL_PATH") or os.getenv("MT5_TERMINAL_PATH") or "",
        help="Chemin du terminal MT5 source utilise pour l'export historique.",
    )
    parser.add_argument(
        "--terminal-portable",
        action="store_true",
        default=_as_bool(os.getenv("HISTORY_MT5_TERMINAL_PORTABLE") or os.getenv("MT5_TERMINAL_PORTABLE")),
        help="Initialise le terminal MT5 source en mode portable.",
    )
    parser.add_argument(
        "--login",
        type=int,
        default=int(
            os.getenv("HISTORY_MT5_LOGIN")
            or os.getenv("MT5_LOGIN")
            or "0"
        ),
        help="Login MT5 attendu pour la source historique.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("HISTORY_MT5_PASSWORD") or os.getenv("MT5_PASSWORD") or "",
        help="Mot de passe MT5 source pour l'export historique.",
    )
    parser.add_argument(
        "--server",
        default=os.getenv("HISTORY_MT5_SERVER") or os.getenv("MT5_SERVER") or "",
        help="Serveur MT5 attendu pour la source historique.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=int(os.getenv("HISTORY_MT5_TIMEOUT_MS") or "120000"),
        help="Delai maximal d'initialisation MT5 en millisecondes.",
    )
    return parser.parse_args()


def initialize_history_terminal(args: argparse.Namespace) -> None:
    """Initialise explicitement la source MT5 de l'export historique.

    Args:
        args (argparse.Namespace): Arguments CLI resolves.

    Raises:
        RuntimeError: Si le terminal MT5 ne peut pas etre initialise.
        ValueError: Si le compte connecte ne correspond pas au compte attendu.
    """
    if mt5 is None:
        raise RuntimeError("MetaTrader5 indisponible dans cet environnement Python.")

    init_kwargs: dict[str, Any] = {
        "timeout": max(int(args.timeout_ms), 1000),
    }
    if str(args.terminal_path or "").strip():
        init_kwargs["path"] = str(args.terminal_path).strip()
        init_kwargs["portable"] = bool(args.terminal_portable)
    if int(args.login or 0) > 0:
        init_kwargs["login"] = int(args.login)
    if str(args.password or "").strip():
        init_kwargs["password"] = str(args.password)
    if str(args.server or "").strip():
        init_kwargs["server"] = str(args.server).strip()

    if not mt5.initialize(**init_kwargs):
        raise RuntimeError(f"Initialisation MT5 impossible: {mt5.last_error()}")

    terminal_info = mt5.terminal_info()
    account_info = mt5.account_info()
    logger.info(
        "Source MT5 historique connectee: compte=%s serveur=%s terminal=%s portable=%s.",
        getattr(account_info, "login", None),
        getattr(account_info, "server", None),
        getattr(terminal_info, "path", None),
        init_kwargs.get("portable", False),
    )

    expected_login = int(args.login or 0)
    expected_server = str(args.server or "").strip()
    current_login = int(getattr(account_info, "login", 0) or 0)
    current_server = str(getattr(account_info, "server", "") or "").strip()
    if expected_login and current_login != expected_login:
        raise ValueError(
            f"Compte MT5 source inattendu: attendu={expected_login}, obtenu={current_login}."
        )
    if expected_server and current_server.lower() != expected_server.lower():
        raise ValueError(
            f"Serveur MT5 source inattendu: attendu={expected_server}, obtenu={current_server}."
        )


def validate_symbol_access(selected: Iterable[SymbolCandidate]) -> None:
    """Valide l'acces aux symboles critiques avant l'export complet.

    Args:
        selected (Iterable[SymbolCandidate]): Symboles retenus pour l'export.

    Raises:
        RuntimeError: Si aucun symbole teste n'est exploitable sur le terminal source.
    """

    selected_list = list(selected)
    sample = selected_list[: min(3, len(selected_list))]
    failed: list[str] = []
    for candidate in sample:
        if mt5.symbol_select(candidate.name, True):
            continue
        failed.append(candidate.name)
        logger.warning(
            "Selection MT5 impossible pour %s sur la source historique: %s.",
            candidate.name,
            mt5.last_error(),
        )
    if len(failed) == len(sample) and failed:
        raise RuntimeError(
            "Aucun symbole test n'est accessible sur le terminal MT5 source. "
            "Verifier le compte, le serveur et la disponibilite du Market Watch."
        )


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


def ingest_existing_csv_to_timescale(
    symbol: str,
    timeframe_name: str,
    *,
    data_dir: Path,
    timescale_writer: TimescaleWriter,
) -> Path | None:
    """Injecte un CSV deja present dans TimescaleDB.

    Args:
        symbol (str): Symbole a injecter.
        timeframe_name (str): Timeframe logique du CSV.
        data_dir (Path): Dossier racine des historiques.
        timescale_writer (TimescaleWriter): Ecrivain TimescaleDB actif.

    Returns:
        Path | None: Chemin du CSV injecte, ou ``None`` si introuvable.
    """

    csv_path = data_dir / f"{symbol}_{timeframe_name}.csv"
    if not csv_path.exists():
        logger.warning("Injection TimeScale ignoree: CSV absent pour %s [%s].", symbol, timeframe_name)
        return None
    frame = pd.read_csv(csv_path)
    if frame.empty:
        logger.warning("Injection TimeScale ignoree: CSV vide pour %s [%s].", symbol, timeframe_name)
        return None
    frame["time"] = pd.to_datetime(frame["time"], utc=False)
    timescale_writer.write_ohlc(symbol, timeframe_name, frame)
    logger.info("CSV injecte dans TimeScaleDB: %s [%s].", symbol, timeframe_name)
    return csv_path


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

    if not args.ingest_existing_only:
        try:
            initialize_history_terminal(args)
        except Exception as exc:
            logger.error("Initialisation MT5 impossible pour la source historique: %s", exc)
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
        if args.ingest_existing_only:
            if not args.write_timescale:
                logger.error("Le mode --ingest-existing-only exige --write-timescale.")
                return 4
            for candidate in selected:
                for timeframe_name in requested_timeframes:
                    output_path = ingest_existing_csv_to_timescale(
                        candidate.name,
                        timeframe_name,
                        data_dir=OUTPUT_DIR,
                        timescale_writer=timescale_writer,
                    )
                    if output_path is not None:
                        generated_files.append(output_path)
            write_inventory(selected, generated_files)
            logger.info("Injection TimeScale terminee: %s fichiers utilises.", len(generated_files))
            return 0

        validate_symbol_access(selected)

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
        if mt5 is not None:
            mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
