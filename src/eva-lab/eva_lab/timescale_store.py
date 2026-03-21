"""Adaptateur optionnel TimescaleDB pour les historiques et analyses de training."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    """Interprete une variable booleenne d'environnement.

    Args:
        name (str): Nom de la variable.
        default (bool): Valeur de repli.

    Returns:
        bool: Etat booleen normalise.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_timescale_settings() -> dict[str, Any]:
    """Retourne la configuration courante de la source TimescaleDB.

    Returns:
        dict[str, Any]: Parametres de connexion et de tables.
    """
    return {
        "enabled": _env_flag("TRAINING_TIMESCALE_ENABLED", False),
        "host": os.getenv("TRAINING_TIMESCALE_HOST", "timescaledb"),
        "port": int(os.getenv("TRAINING_TIMESCALE_PORT", "5432")),
        "database": os.getenv("TRAINING_TIMESCALE_DB", "hive"),
        "user": os.getenv("TRAINING_TIMESCALE_USER", "hive"),
        "password": os.getenv("TRAINING_TIMESCALE_PASSWORD", "hive"),
        "sslmode": os.getenv("TRAINING_TIMESCALE_SSLMODE", "prefer"),
        "bars_table": os.getenv("TRAINING_TIMESCALE_BARS_TABLE", "market_bars"),
        "features_table": os.getenv("TRAINING_TIMESCALE_FEATURES_TABLE", "market_features"),
        "datasets_table": os.getenv("TRAINING_TIMESCALE_DATASETS_TABLE", "training_datasets"),
        "arena_table": os.getenv("TRAINING_TIMESCALE_ARENA_TABLE", "arena_results"),
    }


def describe_timescale_source() -> dict[str, Any]:
    """Expose la source TimescaleDB pour les endpoints de supervision.

    Returns:
        dict[str, Any]: Description stable de la source et des tables.
    """
    settings = get_timescale_settings()
    return {
        "enabled": bool(settings["enabled"]),
        "kind": "timescaledb",
        "source": "timescaledb" if bool(settings["enabled"]) else "csv",
        "host": settings["host"],
        "port": settings["port"],
        "database": settings["database"],
        "bars_table": settings["bars_table"],
        "features_table": settings["features_table"],
        "datasets_table": settings["datasets_table"],
        "arena_table": settings["arena_table"],
    }


def _load_driver():
    """Charge le pilote PostgreSQL de facon optionnelle.

    Returns:
        Any | None: Module `psycopg2` si disponible, sinon `None`.
    """
    try:
        import psycopg2

        return psycopg2
    except Exception:
        return None


@contextmanager
def _connect() -> Iterator[Any]:
    """Ouvre une connexion TimescaleDB si la source est active.

    Yields:
        Any: Connexion PostgreSQL active.
    """
    settings = get_timescale_settings()
    if not settings["enabled"]:
        yield None
        return

    driver = _load_driver()
    if driver is None:
        logger.debug("Pilote psycopg2 indisponible pour TimescaleDB.")
        yield None
        return

    connection = None
    try:
        connection = driver.connect(
            host=settings["host"],
            port=settings["port"],
            dbname=settings["database"],
            user=settings["user"],
            password=settings["password"],
            sslmode=settings["sslmode"],
        )
        yield connection
    except Exception as exc:
        logger.warning("Connexion TimeDB impossible: %s", exc)
        yield None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def discover_timescale_inventory() -> dict[str, set[str]]:
    """Construit un inventaire minimal des symboles presents dans TimeDB.

    Returns:
        dict[str, set[str]]: Mapping `symbole -> timeframes disponibles`.
    """
    settings = get_timescale_settings()
    if not settings["enabled"]:
        return {}

    query = (
        f"SELECT symbol, timeframe "
        f"FROM {settings['bars_table']} "
        "GROUP BY symbol, timeframe"
    )
    inventory: dict[str, set[str]] = {}
    with _connect() as connection:
        if connection is None:
            return inventory
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                for symbol, timeframe in cursor.fetchall():
                    inventory.setdefault(str(symbol), set()).add(str(timeframe).upper())
        except Exception as exc:
            logger.warning("Inventaire TimeDB indisponible: %s", exc)
            return {}
    return inventory


def load_history_frame_from_timescale(
    symbol: str,
    timeframe: str,
    limit: int | None = None,
) -> pd.DataFrame | None:
    """Charge un historique OHLCV depuis TimescaleDB.

    Args:
        symbol (str): Symbole de marche.
        timeframe (str): Timeframe cible.
        limit (int | None): Nombre maximal de lignes a lire.

    Returns:
        pd.DataFrame | None: Historique OHLCV indexe par `time`, ou `None`.
    """
    settings = get_timescale_settings()
    if not settings["enabled"]:
        return None

    query = (
        "SELECT timestamp AS time, open, high, low, close, "
        "COALESCE(tick_volume, 0) AS tick_volume, "
        "COALESCE(spread, 0) AS spread, "
        "COALESCE(real_volume, 0) AS real_volume "
        f"FROM {settings['bars_table']} "
        "WHERE symbol = %s AND timeframe = %s "
        "ORDER BY timestamp DESC"
    )
    params: list[Any] = [symbol, str(timeframe).upper()]
    if limit and int(limit) > 0:
        query += " LIMIT %s"
        params.append(int(limit))

    with _connect() as connection:
        if connection is None:
            return None
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                if not rows:
                    return None
                frame = pd.DataFrame.from_records(
                    rows,
                    columns=[
                        "time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "tick_volume",
                        "spread",
                        "real_volume",
                    ],
                )
        except Exception as exc:
            logger.warning("Lecture TimeDB impossible pour %s %s: %s", symbol, timeframe, exc)
            return None

    frame["time"] = pd.to_datetime(frame["time"], utc=False)
    frame = frame.sort_values("time").drop_duplicates(subset=["time"]).set_index("time")
    frame.attrs["dataset_source"] = "timescaledb"
    frame.attrs["dataset_timeframe"] = str(timeframe).upper()
    frame.attrs["dataset_symbol"] = str(symbol)
    return frame


def _ensure_metadata_tables(connection: Any, settings: dict[str, Any]) -> None:
    """Cree les tables metadata si elles n'existent pas encore.

    Args:
        connection (Any): Connexion PostgreSQL active.
        settings (dict[str, Any]): Parametres de table TimescaleDB.
    """
    dataset_table = settings["datasets_table"]
    arena_table = settings["arena_table"]
    statements = [
        f"""
        CREATE TABLE IF NOT EXISTS {dataset_table} (
            dataset_id TEXT PRIMARY KEY,
            horizon TEXT,
            family TEXT,
            timeframe TEXT,
            source TEXT,
            feature_profile TEXT,
            mechanics_profile TEXT,
            payload JSONB NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {arena_table} (
            id BIGSERIAL PRIMARY KEY,
            dataset_id TEXT,
            horizon TEXT,
            family TEXT,
            feature_profile TEXT,
            challenger_id TEXT,
            champion_id TEXT,
            outcome TEXT,
            payload JSONB NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    ]
    with connection.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)
    connection.commit()


def record_training_dataset(
    descriptor: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Enregistre le manifeste de dataset si TimeDB est disponible.

    Args:
        descriptor (dict[str, Any]): Descripteur immuable du dataset.
        metadata (dict[str, Any] | None): Metadonnees de contexte du run.

    Returns:
        bool: ``True`` si l'ecriture a reussi, sinon ``False``.
    """
    settings = get_timescale_settings()
    if not settings["enabled"]:
        return False

    dataset_id = str(descriptor.get("dataset_id") or "").strip()
    if not dataset_id:
        logger.warning("Manifeste dataset ignore: dataset_id absent.")
        return False

    payload_json = json.dumps(descriptor, ensure_ascii=False, default=str)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False, default=str)
    query = f"""
        INSERT INTO {settings['datasets_table']} (
            dataset_id,
            horizon,
            family,
            timeframe,
            source,
            feature_profile,
            mechanics_profile,
            payload,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
        ON CONFLICT (dataset_id) DO UPDATE SET
            horizon = EXCLUDED.horizon,
            family = EXCLUDED.family,
            timeframe = EXCLUDED.timeframe,
            source = EXCLUDED.source,
            feature_profile = EXCLUDED.feature_profile,
            mechanics_profile = EXCLUDED.mechanics_profile,
            payload = EXCLUDED.payload,
            metadata = EXCLUDED.metadata,
            created_at = NOW()
    """

    with _connect() as connection:
        if connection is None:
            return False
        try:
            _ensure_metadata_tables(connection, settings)
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        dataset_id,
                        str(descriptor.get("horizon") or "").strip() or None,
                        str(descriptor.get("family") or "").strip() or None,
                        str(descriptor.get("timeframe") or "").strip() or None,
                        str(descriptor.get("source") or "").strip() or None,
                        str(descriptor.get("feature_profile") or "").strip() or None,
                        str(descriptor.get("mechanics_profile") or "").strip() or None,
                        payload_json,
                        metadata_json,
                    ),
                )
            connection.commit()
            return True
        except Exception as exc:
            logger.warning("Ecriture du manifeste dataset impossible: %s", exc)
            return False


def record_arena_result(
    report_payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Enregistre un rapport d'Arena ou de promotion si TimeDB est disponible.

    Args:
        report_payload (dict[str, Any]): Rapport complet du duel ou de promotion.
        metadata (dict[str, Any] | None): Metadonnees de contexte du run.

    Returns:
        bool: ``True`` si l'ecriture a reussi, sinon ``False``.
    """
    settings = get_timescale_settings()
    if not settings["enabled"]:
        return False

    battle_report = dict(report_payload.get("battle_report") or {})
    challenger = dict(battle_report.get("challenger") or {})
    champion = dict(battle_report.get("champion") or {})
    payload_json = json.dumps(report_payload, ensure_ascii=False, default=str)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False, default=str)
    query = f"""
        INSERT INTO {settings['arena_table']} (
            dataset_id,
            horizon,
            family,
            feature_profile,
            challenger_id,
            champion_id,
            outcome,
            payload,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
    """

    with _connect() as connection:
        if connection is None:
            return False
        try:
            _ensure_metadata_tables(connection, settings)
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        str(report_payload.get("dataset_id") or "").strip() or None,
                        str(report_payload.get("horizon") or "").strip() or None,
                        str(report_payload.get("family") or "").strip() or None,
                        str(report_payload.get("feature_profile") or "").strip() or None,
                        str(challenger.get("id") or report_payload.get("challenger_id") or "").strip() or None,
                        str(champion.get("id") or report_payload.get("live_champion_id") or "").strip() or None,
                        str(battle_report.get("outcome") or report_payload.get("outcome") or "").strip() or None,
                        payload_json,
                        metadata_json,
                    ),
                )
            connection.commit()
            return True
        except Exception as exc:
            logger.warning("Ecriture du rapport Arena impossible: %s", exc)
            return False
