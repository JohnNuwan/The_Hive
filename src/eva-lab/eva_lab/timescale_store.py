"""Socle canonique TimescaleDB pour les historiques, runs et contextes."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

try:
    import pandas as pd
except ImportError:  # pragma: no cover - repli utile pour le superviseur hors conteneur.
    pd = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TimescaleTableMap:
    """Regroupe les tables canoniques exploitees par l'usine V5.1."""

    bars: str
    features: str
    datasets: str
    arena: str
    ga_trials: str
    replay_metadata: str
    market_context: str
    investment_theses: str
    gpu_metrics: str
    cpu_jobs: str
    run_windows: str
    metrics_history: str


DEFAULT_TABLES = TimescaleTableMap(
    bars="market.market_bars",
    features="market.market_features",
    datasets="training.training_datasets",
    arena="training.arena_results",
    ga_trials="training.ga_trials",
    replay_metadata="training.replay_metadata",
    market_context="research.market_context_snapshots",
    investment_theses="research.investment_theses",
    gpu_metrics="ops.gpu_metrics",
    cpu_jobs="ops.cpu_jobs_history",
    run_windows="training.run_windows",
    metrics_history="training.metrics_history",
)

CANONICAL_TIMESCALE_DATABASE = "thehive"
CANONICAL_TIMESCALE_USER = "eva"
DEFAULT_STORAGE_PROFILE = "balanced"
DEFAULT_ALLOWED_WRITE_TIMEFRAMES = ("M5", "H1", "D1")
DEFAULT_SOFT_LIMIT_GB = 120.0
DEFAULT_HARD_LIMIT_GB = 150.0
TIMESCALE_EXTENSION_NAME = "timescaledb"
_LAST_BOOTSTRAP_ERROR: str | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    """Interprete une variable booleenne d'environnement."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    """Lit un flottant simple depuis l'environnement."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def _json_dumps(payload: Any) -> str:
    """Serialise un objet Python de maniere robuste pour JSONB."""

    return json.dumps(payload if payload is not None else {}, ensure_ascii=False, default=str)


def _normalize_database_name(raw_value: str | None) -> str:
    """Normalise les alias historiques de base PostgreSQL."""

    value = str(raw_value or "").strip()
    if not value:
        return CANONICAL_TIMESCALE_DATABASE
    if value.lower() in {"the_hive", "the-hive"}:
        return CANONICAL_TIMESCALE_DATABASE
    return value


def _normalize_user_name(raw_value: str | None) -> str:
    """Normalise les alias historiques d'utilisateur PostgreSQL."""

    value = str(raw_value or "").strip()
    if not value:
        return CANONICAL_TIMESCALE_USER
    if value.lower() in {"the_hive", "thehive"}:
        return CANONICAL_TIMESCALE_USER
    return value


def _normalize_table_name(raw_value: str | None, default_value: str) -> str:
    """Normalise un identifiant de table vers le schema canonique."""

    value = str(raw_value or "").strip()
    if not value:
        return default_value
    if "." in value:
        return value
    default_schema, _default_table = _split_identifier(default_value)
    return f"{default_schema}.{value}"


def _parse_allowed_timeframes(raw_value: str | None) -> tuple[str, ...]:
    """Normalise la liste des timeframes autorises pour l'ecriture OHLC."""

    raw_items = str(raw_value or "").split(",")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        label = str(item or "").strip().upper()
        if not label or label in seen:
            continue
        normalized.append(label)
        seen.add(label)
    if not normalized:
        return DEFAULT_ALLOWED_WRITE_TIMEFRAMES
    return tuple(normalized)


def _bytes_from_gb(value_gb: float) -> int:
    """Convertit une taille en Go vers des octets entiers."""

    return max(int(float(value_gb) * 1024 * 1024 * 1024), 0)


def _split_identifier(identifier: str) -> tuple[str, str]:
    """Separe un identifiant SQL `schema.table` en deux parties."""

    normalized = str(identifier or "").strip()
    if not normalized:
        raise ValueError("Identifiant SQL vide.")
    if "." in normalized:
        schema_name, table_name = normalized.split(".", 1)
        return schema_name.strip(), table_name.strip()
    return "public", normalized


def _sql_identifier(identifier: str) -> str:
    """Construit un identifiant SQL quote pour schema ou table."""

    parts = [part.strip() for part in str(identifier or "").split(".") if part.strip()]
    if not parts:
        raise ValueError("Identifiant SQL vide.")
    return ".".join(f'"{part.replace(chr(34), chr(34) * 2)}"' for part in parts)


def get_timescale_settings() -> dict[str, Any]:
    """Retourne la configuration courante de la source TimescaleDB."""

    table_map = TimescaleTableMap(
        bars=_normalize_table_name(os.getenv("TRAINING_TIMESCALE_BARS_TABLE"), DEFAULT_TABLES.bars),
        features=_normalize_table_name(os.getenv("TRAINING_TIMESCALE_FEATURES_TABLE"), DEFAULT_TABLES.features),
        datasets=_normalize_table_name(os.getenv("TRAINING_TIMESCALE_DATASETS_TABLE"), DEFAULT_TABLES.datasets),
        arena=_normalize_table_name(os.getenv("TRAINING_TIMESCALE_ARENA_TABLE"), DEFAULT_TABLES.arena),
        ga_trials=_normalize_table_name(os.getenv("TRAINING_TIMESCALE_GA_TABLE"), DEFAULT_TABLES.ga_trials),
        replay_metadata=_normalize_table_name(
            os.getenv("TRAINING_TIMESCALE_REPLAY_TABLE"),
            DEFAULT_TABLES.replay_metadata,
        ),
        market_context=_normalize_table_name(
            os.getenv("TRAINING_TIMESCALE_MARKET_CONTEXT_TABLE"),
            DEFAULT_TABLES.market_context,
        ),
        investment_theses=_normalize_table_name(
            os.getenv("TRAINING_TIMESCALE_INVEST_TABLE"),
            DEFAULT_TABLES.investment_theses,
        ),
        gpu_metrics=_normalize_table_name(os.getenv("TRAINING_TIMESCALE_GPU_TABLE"), DEFAULT_TABLES.gpu_metrics),
        cpu_jobs=_normalize_table_name(os.getenv("TRAINING_TIMESCALE_CPU_JOBS_TABLE"), DEFAULT_TABLES.cpu_jobs),
        run_windows=_normalize_table_name(
            os.getenv("TRAINING_TIMESCALE_RUN_WINDOWS_TABLE"),
            DEFAULT_TABLES.run_windows,
        ),
        metrics_history=_normalize_table_name(
            os.getenv("TRAINING_TIMESCALE_METRICS_HISTORY_TABLE"),
            DEFAULT_TABLES.metrics_history,
        ),
    )
    storage_profile = str(os.getenv("TIMESCALE_STORAGE_PROFILE", DEFAULT_STORAGE_PROFILE) or "").strip().lower()
    if not storage_profile:
        storage_profile = DEFAULT_STORAGE_PROFILE
    soft_limit_gb = _env_float("TIMESCALE_DB_SOFT_LIMIT_GB", DEFAULT_SOFT_LIMIT_GB)
    hard_limit_gb = _env_float("TIMESCALE_DB_HARD_LIMIT_GB", DEFAULT_HARD_LIMIT_GB)
    if hard_limit_gb < soft_limit_gb:
        hard_limit_gb = soft_limit_gb
    return {
        "enabled": _env_flag("TRAINING_TIMESCALE_ENABLED", False),
        "host": os.getenv("TRAINING_TIMESCALE_HOST", os.getenv("TIMESCALE_HOST", "timescaledb")),
        "port": int(os.getenv("TRAINING_TIMESCALE_PORT", os.getenv("TIMESCALE_PORT", "5432"))),
        "database": _normalize_database_name(
            os.getenv("TRAINING_TIMESCALE_DB", os.getenv("TIMESCALE_DB", CANONICAL_TIMESCALE_DATABASE))
        ),
        "user": _normalize_user_name(
            os.getenv("TRAINING_TIMESCALE_USER", os.getenv("TIMESCALE_USER", CANONICAL_TIMESCALE_USER))
        ),
        "password": os.getenv("TRAINING_TIMESCALE_PASSWORD", os.getenv("TIMESCALE_PASSWORD", "")),
        "sslmode": os.getenv("TRAINING_TIMESCALE_SSLMODE", "prefer"),
        "storage_profile": storage_profile,
        "allowed_write_timeframes": _parse_allowed_timeframes(
            os.getenv(
                "HISTORY_TIMESCALE_ALLOWED_TIMEFRAMES",
                ",".join(DEFAULT_ALLOWED_WRITE_TIMEFRAMES),
            )
        ),
        "soft_limit_gb": soft_limit_gb,
        "hard_limit_gb": hard_limit_gb,
        "soft_limit_bytes": _bytes_from_gb(soft_limit_gb),
        "hard_limit_bytes": _bytes_from_gb(hard_limit_gb),
        "tables": table_map,
        "bars_table": table_map.bars,
        "features_table": table_map.features,
        "datasets_table": table_map.datasets,
        "arena_table": table_map.arena,
    }


def describe_timescale_source() -> dict[str, Any]:
    """Expose la source TimeDB pour les endpoints de supervision."""
    return get_timescale_runtime_status(repair=False)


def _load_driver():
    """Charge le pilote PostgreSQL de facon optionnelle."""

    try:
        import psycopg2

        return psycopg2
    except Exception:
        return None


@contextmanager
def _connect(
    database: str | None = None,
    *,
    log_errors: bool = True,
    autocommit: bool = False,
) -> Iterator[Any]:
    """Ouvre une connexion TimescaleDB si la source est active."""

    settings = get_timescale_settings()
    if not settings["enabled"]:
        yield None
        return

    driver = _load_driver()
    if driver is None:
        logger.debug("Pilote psycopg2 indisponible pour TimeDB.")
        yield None
        return

    connection = None
    try:
        connection = driver.connect(
            host=settings["host"],
            port=settings["port"],
            dbname=_normalize_database_name(database or settings["database"]),
            user=settings["user"],
            password=settings["password"],
            sslmode=settings["sslmode"],
        )
        if autocommit:
            connection.autocommit = True
        yield connection
    except Exception as exc:
        if log_errors:
            logger.warning("Connexion TimeDB impossible: %s", exc)
        yield None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _ensure_schema(cursor: Any, schema_name: str) -> None:
    """Cree un schema si necessaire."""

    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {_sql_identifier(schema_name)}")


def _ensure_column(
    cursor: Any,
    table_name: str,
    column_name: str,
    column_sql: str,
) -> None:
    """Ajoute une colonne manquante sans casser les schemas existants.

    Args:
        cursor (Any): Curseur SQL actif.
        table_name (str): Table cible au format ``schema.table``.
        column_name (str): Nom de la colonne a garantir.
        column_sql (str): Definition SQL complete de la colonne.
    """

    cursor.execute(
        f"ALTER TABLE IF EXISTS {_sql_identifier(table_name)} "
        f"ADD COLUMN IF NOT EXISTS {_sql_identifier(column_name)} {column_sql}"
    )


def _drop_not_null_if_present(cursor: Any, table_name: str, column_name: str) -> None:
    """Retire une contrainte `NOT NULL` legacy si la colonne existe encore.

    Args:
        cursor (Any): Curseur SQL actif.
        table_name (str): Table cible au format ``schema.table``.
        column_name (str): Colonne legacy a assouplir.
    """

    schema_name, raw_table_name = _split_identifier(table_name)
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
          AND column_name = %s
          AND is_nullable = 'NO'
        """,
        (schema_name, raw_table_name, column_name),
    )
    if cursor.fetchone():
        cursor.execute(
            f"ALTER TABLE IF EXISTS {_sql_identifier(table_name)} "
            f"ALTER COLUMN {_sql_identifier(column_name)} DROP NOT NULL"
        )


def _table_exists(cursor: Any, table_name: str) -> bool:
    """Indique si une table SQL existe deja."""

    schema_name, raw_table_name = _split_identifier(table_name)
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_name = %s
        """,
        (schema_name, raw_table_name),
    )
    return cursor.fetchone() is not None


def _ensure_schema_objects(connection: Any, settings: dict[str, Any]) -> None:
    """Cree les schemas et tables canoniques si necessaire."""

    tables: TimescaleTableMap = settings["tables"]
    schema_names = {
        _split_identifier(tables.bars)[0],
        _split_identifier(tables.features)[0],
        _split_identifier(tables.datasets)[0],
        _split_identifier(tables.arena)[0],
        _split_identifier(tables.ga_trials)[0],
        _split_identifier(tables.replay_metadata)[0],
        _split_identifier(tables.market_context)[0],
        _split_identifier(tables.investment_theses)[0],
        _split_identifier(tables.gpu_metrics)[0],
        _split_identifier(tables.cpu_jobs)[0],
        _split_identifier(tables.run_windows)[0],
        _split_identifier(tables.metrics_history)[0],
    }
    with connection.cursor() as cursor:
        for schema_name in sorted(schema_names):
            _ensure_schema(cursor, schema_name)

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.bars)} (
                "timestamp" TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                tick_volume BIGINT NOT NULL DEFAULT 0,
                real_volume BIGINT NOT NULL DEFAULT 0,
                spread INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'mt5',
                ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (symbol, timeframe, "timestamp")
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.features)} (
                "timestamp" TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                feature_profile TEXT NOT NULL,
                ema_fast DOUBLE PRECISION NULL,
                ema_slow DOUBLE PRECISION NULL,
                ema200 DOUBLE PRECISION NULL,
                vwap DOUBLE PRECISION NULL,
                obv DOUBLE PRECISION NULL,
                rsi DOUBLE PRECISION NULL,
                adx DOUBLE PRECISION NULL,
                atr DOUBLE PRECISION NULL,
                bb_width DOUBLE PRECISION NULL,
                relative_volume DOUBLE PRECISION NULL,
                session_phase TEXT NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (symbol, timeframe, feature_profile, "timestamp")
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.datasets)} (
                dataset_id TEXT PRIMARY KEY,
                engine TEXT NULL,
                horizon TEXT NULL,
                family TEXT NULL,
                timeframe TEXT NULL,
                feature_profile TEXT NULL,
                mechanics_profile_version TEXT NULL,
                source TEXT NULL,
                bars_table TEXT NULL,
                features_table TEXT NULL,
                symbols JSONB NOT NULL DEFAULT '[]'::jsonb,
                start_at TIMESTAMPTZ NULL,
                end_at TIMESTAMPTZ NULL,
                coverage_ratio DOUBLE PRECISION NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.arena)} (
                id BIGSERIAL PRIMARY KEY,
                dataset_id TEXT NULL,
                engine TEXT NULL,
                horizon TEXT NULL,
                family TEXT NULL,
                feature_profile TEXT NULL,
                challenger_id TEXT NULL,
                champion_id TEXT NULL,
                outcome TEXT NULL,
                failure_mode TEXT NULL,
                metrics JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                metrics_by_symbol JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                metrics_by_position_mechanics JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.ga_trials)} (
                trial_id TEXT PRIMARY KEY,
                engine TEXT NULL,
                sequence_id TEXT NULL,
                profile TEXT NULL,
                horizon TEXT NULL,
                family TEXT NULL,
                feature_profile TEXT NULL,
                mechanics_profile_version TEXT NULL,
                ga_generation INTEGER NULL,
                ga_trial TEXT NULL,
                trial_mode TEXT NULL,
                trial_cost_profile TEXT NULL,
                params JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                fitness_score DOUBLE PRECISION NULL,
                failure_mode TEXT NULL,
                run_id TEXT NULL,
                dataset_id TEXT NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                finished_at TIMESTAMPTZ NULL
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.replay_metadata)} (
                cache_key TEXT PRIMARY KEY,
                engine TEXT NULL,
                horizon TEXT NULL,
                family TEXT NULL,
                feature_profile TEXT NULL,
                mechanics_profile_version TEXT NULL,
                entries INTEGER NOT NULL DEFAULT 0,
                source TEXT NULL,
                reuse_ratio DOUBLE PRECISION NULL,
                last_run_id TEXT NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.market_context)} (
                snapshot_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                family TEXT NULL,
                mode TEXT NOT NULL DEFAULT 'prop',
                macro_bias TEXT NULL,
                event_risk TEXT NULL,
                geo_risk TEXT NULL,
                blocked BOOLEAN NOT NULL DEFAULT FALSE,
                confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                generated_at TIMESTAMPTZ NOT NULL,
                expires_at TIMESTAMPTZ NULL
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.investment_theses)} (
                thesis_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                issuer TEXT NULL,
                mode TEXT NOT NULL DEFAULT 'invest',
                conviction_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                fundamental_risk TEXT NULL,
                governance_risk TEXT NULL,
                horizon_months INTEGER NOT NULL DEFAULT 12,
                thesis TEXT NOT NULL DEFAULT '',
                sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                generated_at TIMESTAMPTZ NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'draft'
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.gpu_metrics)} (
                "timestamp" TIMESTAMPTZ NOT NULL,
                host TEXT NOT NULL,
                gpu_name TEXT NULL,
                utilization_percent DOUBLE PRECISION NULL,
                memory_used_mb DOUBLE PRECISION NULL,
                memory_total_mb DOUBLE PRECISION NULL,
                temperature_c DOUBLE PRECISION NULL,
                power_draw_w DOUBLE PRECISION NULL,
                source TEXT NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                PRIMARY KEY (host, "timestamp")
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.cpu_jobs)} (
                job_id TEXT PRIMARY KEY,
                lane TEXT NOT NULL,
                job_name TEXT NOT NULL,
                status TEXT NOT NULL,
                host TEXT NULL,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                started_at TIMESTAMPTZ NULL,
                finished_at TIMESTAMPTZ NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.run_windows)} (
                window_id TEXT PRIMARY KEY,
                sequence_id TEXT NOT NULL,
                profile TEXT NULL,
                engine TEXT NULL,
                mode TEXT NULL,
                trial_id TEXT NULL,
                window_index INTEGER NULL,
                status TEXT NOT NULL,
                last_run_id TEXT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                started_at TIMESTAMPTZ NULL,
                finished_at TIMESTAMPTZ NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_sql_identifier(tables.metrics_history)} (
                "timestamp" TIMESTAMPTZ NOT NULL,
                run_id TEXT NOT NULL,
                engine TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL,
                step INTEGER NOT NULL,
                epoch INTEGER NULL,
                PRIMARY KEY (run_id, engine, metric_name, step, "timestamp")
            )
            """
        )
        _ensure_column(cursor, tables.ga_trials, "sequence_id", "TEXT NULL")
        _ensure_column(cursor, tables.ga_trials, "profile", "TEXT NULL")
        _ensure_column(cursor, tables.run_windows, "sequence_id", "TEXT NULL")
        _ensure_column(cursor, tables.run_windows, "profile", "TEXT NULL")
        _ensure_column(cursor, tables.run_windows, "mode", "TEXT NULL")
        _ensure_column(cursor, tables.run_windows, "trial_id", "TEXT NULL")
        _ensure_column(cursor, tables.run_windows, "window_index", "INTEGER NULL")
        _ensure_column(cursor, tables.run_windows, "status", "TEXT NULL")
        _ensure_column(cursor, tables.run_windows, "last_run_id", "TEXT NULL")
        _ensure_column(cursor, tables.run_windows, "retry_count", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(cursor, tables.run_windows, "updated_at", "TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        _drop_not_null_if_present(cursor, tables.run_windows, "run_id")
        cursor.execute(f'DROP VIEW IF EXISTS {_sql_identifier("public.market_ohlc")}')
        cursor.execute(
            f"""
            CREATE VIEW {_sql_identifier('public.market_ohlc')} AS
            SELECT
                "timestamp" AS time,
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
            FROM {_sql_identifier(tables.bars)}
            """
        )
        cursor.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_market_bars_timeframe_symbol" ON {_sql_identifier(tables.bars)} (timeframe, symbol, "timestamp" DESC)'
        )
        cursor.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_market_features_profile_symbol" ON {_sql_identifier(tables.features)} (feature_profile, symbol, timeframe, "timestamp" DESC)'
        )
        connection.commit()


def _set_last_bootstrap_error(message: str | None) -> None:
    """Memorise la derniere erreur de bootstrap TimeDB."""

    global _LAST_BOOTSTRAP_ERROR
    _LAST_BOOTSTRAP_ERROR = str(message or "").strip() or None


def _database_exists(connection: Any, database_name: str) -> bool:
    """Indique si une base existe deja sur l'instance PostgreSQL."""

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (_normalize_database_name(database_name),),
        )
        return cursor.fetchone() is not None


def _create_database(connection: Any, database_name: str, owner: str) -> None:
    """Cree la base cible si elle n'existe pas encore."""

    normalized_database = _normalize_database_name(database_name)
    normalized_owner = _normalize_user_name(owner)
    with connection.cursor() as cursor:
        cursor.execute(
            f"CREATE DATABASE {_sql_identifier(normalized_database)} OWNER {_sql_identifier(normalized_owner)}"
        )


def _extension_exists(connection: Any, extension_name: str = TIMESCALE_EXTENSION_NAME) -> bool:
    """Retourne ``True`` si l'extension cible est installee."""

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM pg_extension WHERE extname = %s",
            (str(extension_name or "").strip(),),
        )
        return cursor.fetchone() is not None


def _ensure_extension(connection: Any, extension_name: str = TIMESCALE_EXTENSION_NAME) -> None:
    """Installe l'extension cible si necessaire."""

    with connection.cursor() as cursor:
        cursor.execute(f"CREATE EXTENSION IF NOT EXISTS {_sql_identifier(extension_name)} CASCADE")
    connection.commit()


def _database_size_bytes(connection: Any) -> int:
    """Retourne la taille de la base courante en octets."""

    with connection.cursor() as cursor:
        cursor.execute("SELECT COALESCE(pg_database_size(current_database()), 0)")
        row = cursor.fetchone()
    return int((row or [0])[0] or 0)


def _schema_objects_ready(connection: Any, settings: dict[str, Any]) -> bool:
    """Verifie la presence des objets canoniques indispensables."""

    tables: TimescaleTableMap = settings["tables"]
    required_tables = [
        tables.bars,
        tables.features,
        tables.datasets,
        tables.arena,
        tables.ga_trials,
        tables.replay_metadata,
        tables.market_context,
        tables.investment_theses,
        tables.gpu_metrics,
        tables.cpu_jobs,
        tables.run_windows,
    ]
    with connection.cursor() as cursor:
        for table_name in required_tables:
            if not _table_exists(cursor, table_name):
                return False
    return True


def _table_is_hypertable(connection: Any, table_name: str) -> bool:
    """Indique si une table est deja declaree comme hypertable."""

    schema_name, raw_table_name = _split_identifier(table_name)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM timescaledb_information.hypertables
                WHERE hypertable_schema = %s
                  AND hypertable_name = %s
                """,
                (schema_name, raw_table_name),
            )
            return cursor.fetchone() is not None
    except Exception:
        return False


def _ensure_hypertable(connection: Any, table_name: str, time_column: str) -> None:
    """Convertit une table en hypertable de maniere idempotente."""

    if _table_is_hypertable(connection, table_name):
        return
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT create_hypertable(%s, %s, if_not_exists => TRUE, migrate_data => TRUE)",
            (table_name, str(time_column or "").strip()),
        )
    connection.commit()


def _set_table_compression(connection: Any, table_name: str) -> None:
    """Active la compression TimescaleDB pour une hypertable."""

    with connection.cursor() as cursor:
        cursor.execute(f"ALTER TABLE {_sql_identifier(table_name)} SET (timescaledb.compress = true)")
    connection.commit()


def _ensure_compression_policy(connection: Any, table_name: str, older_than: str) -> None:
    """Ajoute une politique de compression idempotente."""

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT add_compression_policy(%s, %s::interval, if_not_exists => TRUE)",
            (table_name, older_than),
        )
    connection.commit()


def _ensure_retention_policy(connection: Any, table_name: str, older_than: str) -> None:
    """Ajoute une politique de retention idempotente."""

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT add_retention_policy(%s, %s::interval, if_not_exists => TRUE)",
            (table_name, older_than),
        )
    connection.commit()


def _purge_cpu_jobs_history(connection: Any, table_name: str, older_than: str) -> None:
    """Purge les anciens jobs CPU sur une table classique."""

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            DELETE FROM {_sql_identifier(table_name)}
            WHERE COALESCE(finished_at, started_at, created_at, NOW())
                < NOW() - %s::interval
            """,
            (older_than,),
        )
    connection.commit()


def _apply_balanced_storage_profile(connection: Any, settings: dict[str, Any]) -> None:
    """Applique le profil `balanced` sans exploser le disque."""

    tables: TimescaleTableMap = settings["tables"]
    _ensure_hypertable(connection, tables.bars, "timestamp")
    _set_table_compression(connection, tables.bars)
    _ensure_compression_policy(connection, tables.bars, "7 days")
    _ensure_retention_policy(connection, tables.bars, "400 days")

    _ensure_hypertable(connection, tables.features, "timestamp")
    _set_table_compression(connection, tables.features)
    _ensure_compression_policy(connection, tables.features, "3 days")
    _ensure_retention_policy(connection, tables.features, "90 days")

    _ensure_hypertable(connection, tables.gpu_metrics, "timestamp")
    _ensure_retention_policy(connection, tables.gpu_metrics, "30 days")
    _purge_cpu_jobs_history(connection, tables.cpu_jobs, "30 days")

    _ensure_hypertable(connection, tables.metrics_history, "timestamp")
    _ensure_retention_policy(connection, tables.metrics_history, "90 days")


def _apply_storage_profile(connection: Any, settings: dict[str, Any]) -> None:
    """Applique les garde-fous de stockage associes au profil courant."""

    profile = str(settings.get("storage_profile") or DEFAULT_STORAGE_PROFILE).strip().lower()
    if profile == "balanced":
        _apply_balanced_storage_profile(connection, settings)


def _build_write_guard_status(
    settings: dict[str, Any],
    *,
    database_ready: bool,
    db_size_bytes: int | None,
) -> dict[str, Any]:
    """Construit un diagnostic stable de garde-fou disque."""

    allowed_timeframes = list(settings.get("allowed_write_timeframes") or DEFAULT_ALLOWED_WRITE_TIMEFRAMES)
    soft_limit_bytes = int(settings.get("soft_limit_bytes") or _bytes_from_gb(DEFAULT_SOFT_LIMIT_GB))
    hard_limit_bytes = int(settings.get("hard_limit_bytes") or _bytes_from_gb(DEFAULT_HARD_LIMIT_GB))
    if not settings.get("enabled", False):
        return {
            "allowed": False,
            "status": "disabled",
            "reason": "timescaledb_desactive",
            "db_size_bytes": db_size_bytes,
            "soft_limit_bytes": soft_limit_bytes,
            "hard_limit_bytes": hard_limit_bytes,
            "allowed_timeframes": allowed_timeframes,
        }
    if not database_ready:
        return {
            "allowed": False,
            "status": "unavailable",
            "reason": "bootstrap_incomplet",
            "db_size_bytes": db_size_bytes,
            "soft_limit_bytes": soft_limit_bytes,
            "hard_limit_bytes": hard_limit_bytes,
            "allowed_timeframes": allowed_timeframes,
        }

    current_size = int(db_size_bytes or 0)
    if current_size >= hard_limit_bytes:
        return {
            "allowed": False,
            "status": "blocked",
            "reason": "hard_limit_reached",
            "db_size_bytes": current_size,
            "soft_limit_bytes": soft_limit_bytes,
            "hard_limit_bytes": hard_limit_bytes,
            "allowed_timeframes": allowed_timeframes,
        }
    if current_size >= soft_limit_bytes:
        return {
            "allowed": True,
            "status": "degraded",
            "reason": "soft_limit_reached",
            "db_size_bytes": current_size,
            "soft_limit_bytes": soft_limit_bytes,
            "hard_limit_bytes": hard_limit_bytes,
            "allowed_timeframes": allowed_timeframes,
        }
    return {
        "allowed": True,
        "status": "allowed",
        "reason": "within_limits",
        "db_size_bytes": current_size,
        "soft_limit_bytes": soft_limit_bytes,
        "hard_limit_bytes": hard_limit_bytes,
        "allowed_timeframes": allowed_timeframes,
    }


def evaluate_ohlc_write_request(
    timeframe: str,
    *,
    repair: bool = False,
) -> dict[str, Any]:
    """Valide une demande d'ecriture OHLC contre les garde-fous TimeScaleDB.

    Args:
        timeframe (str): Timeframe demande pour l'ecriture.
        repair (bool): Tente un bootstrap idempotent si ``True``.

    Returns:
        dict[str, Any]: Diagnostic stable exploitable par les scripts CLI.
    """

    normalized_timeframe = str(timeframe or "").strip().upper()
    runtime = get_timescale_runtime_status(repair=repair)
    guard = dict(runtime.get("write_guard_status") or {})
    allowed_timeframes = [
        str(item or "").strip().upper()
        for item in guard.get("allowed_timeframes", runtime.get("allowed_write_timeframes", []))
        if str(item or "").strip()
    ]
    diagnostic = {
        "allowed": bool(guard.get("allowed", False)),
        "status": str(guard.get("status") or "unknown"),
        "reason": str(guard.get("reason") or "unknown"),
        "timeframe": normalized_timeframe,
        "allowed_timeframes": allowed_timeframes,
        "db_size_bytes": runtime.get("db_size_bytes"),
        "soft_limit_bytes": guard.get("soft_limit_bytes"),
        "hard_limit_bytes": guard.get("hard_limit_bytes"),
        "storage_profile": runtime.get("storage_profile"),
        "state": runtime.get("state"),
        "database_exists": bool(runtime.get("database_exists", False)),
        "extension_ready": bool(runtime.get("extension_ready", False)),
        "schema_ready": bool(runtime.get("schema_ready", False)),
        "last_bootstrap_error": runtime.get("last_bootstrap_error"),
    }
    if not normalized_timeframe:
        diagnostic.update(
            {
                "allowed": False,
                "status": "invalid_request",
                "reason": "timeframe_absent",
            }
        )
        return diagnostic
    if normalized_timeframe not in allowed_timeframes:
        diagnostic.update(
            {
                "allowed": False,
                "status": "timeframe_blocked",
                "reason": "timeframe_non_autorise",
            }
        )
        return diagnostic
    return diagnostic


def get_timescale_runtime_status(*, repair: bool = False) -> dict[str, Any]:
    """Retourne l'etat reel de TimeScaleDB et repare si demande."""

    settings = get_timescale_settings()
    tables: TimescaleTableMap = settings["tables"]
    runtime = {
        "enabled": bool(settings["enabled"]),
        "kind": "timescaledb",
        "source": "timescaledb" if bool(settings["enabled"]) else "csv",
        "state": "disabled" if not bool(settings["enabled"]) else "offline",
        "host": settings["host"],
        "port": settings["port"],
        "database": settings["database"],
        "user": settings["user"],
        "bars_table": tables.bars,
        "features_table": tables.features,
        "datasets_table": tables.datasets,
        "arena_table": tables.arena,
        "ga_table": tables.ga_trials,
        "replay_table": tables.replay_metadata,
        "market_context_table": tables.market_context,
        "investment_table": tables.investment_theses,
        "ops_gpu_table": tables.gpu_metrics,
        "ops_cpu_jobs_table": tables.cpu_jobs,
        "run_windows_table": tables.run_windows,
        "storage_profile": settings["storage_profile"],
        "allowed_write_timeframes": list(settings["allowed_write_timeframes"]),
        "database_exists": False,
        "extension_ready": False,
        "schema_ready": False,
        "db_size_bytes": None,
        "reachable": False,
        "ok": False,
        "last_bootstrap_error": _LAST_BOOTSTRAP_ERROR,
    }
    runtime["write_guard_status"] = _build_write_guard_status(
        settings,
        database_ready=False,
        db_size_bytes=None,
    )
    if not settings["enabled"]:
        return runtime

    try:
        with _connect("postgres", log_errors=False, autocommit=True) as admin_connection:
            if admin_connection is None:
                _set_last_bootstrap_error("Connexion a la base postgres impossible.")
                runtime["last_bootstrap_error"] = _LAST_BOOTSTRAP_ERROR
                return runtime

            runtime["reachable"] = True
            runtime["database_exists"] = _database_exists(admin_connection, settings["database"])
            if repair and not runtime["database_exists"]:
                _create_database(admin_connection, settings["database"], settings["user"])
                runtime["database_exists"] = True

        if not runtime["database_exists"]:
            runtime["state"] = "missing_database"
            _set_last_bootstrap_error(f"Base {settings['database']} absente.")
            runtime["last_bootstrap_error"] = _LAST_BOOTSTRAP_ERROR
            return runtime

        with _connect(settings["database"], log_errors=False) as connection:
            if connection is None:
                _set_last_bootstrap_error(f"Connexion a la base {settings['database']} impossible.")
                runtime["last_bootstrap_error"] = _LAST_BOOTSTRAP_ERROR
                return runtime

            runtime["reachable"] = True
            if repair:
                _ensure_extension(connection)
            runtime["extension_ready"] = _extension_exists(connection)
            if repair and runtime["extension_ready"]:
                _ensure_schema_objects(connection, settings)
                _apply_storage_profile(connection, settings)
            runtime["schema_ready"] = _schema_objects_ready(connection, settings)
            runtime["db_size_bytes"] = _database_size_bytes(connection)

    except Exception as exc:
        _set_last_bootstrap_error(str(exc))
        runtime["last_bootstrap_error"] = _LAST_BOOTSTRAP_ERROR
        runtime["state"] = "offline"
        return runtime

    database_ready = bool(
        runtime["database_exists"]
        and runtime["extension_ready"]
        and runtime["schema_ready"]
    )
    runtime["write_guard_status"] = _build_write_guard_status(
        settings,
        database_ready=database_ready,
        db_size_bytes=runtime["db_size_bytes"],
    )
    if not runtime["extension_ready"]:
        runtime["state"] = "missing_extension"
        _set_last_bootstrap_error("Extension TimescaleDB absente.")
    elif not runtime["schema_ready"]:
        runtime["state"] = "schema_incomplete"
        _set_last_bootstrap_error("Schema TimeScaleDB incomplet.")
    elif runtime["write_guard_status"]["status"] == "blocked":
        runtime["state"] = "write_guard_blocked"
        _set_last_bootstrap_error("Ecriture OHLC bloquee par la limite disque.")
    elif runtime["write_guard_status"]["status"] == "degraded":
        runtime["state"] = "degraded"
        _set_last_bootstrap_error(None)
    else:
        runtime["state"] = "ready"
        _set_last_bootstrap_error(None)

    runtime["last_bootstrap_error"] = _LAST_BOOTSTRAP_ERROR
    runtime["ok"] = bool(database_ready)
    return runtime


def ensure_timescale_ready() -> bool:
    """Garantit l'existence des schemas et tables canoniques."""

    runtime = get_timescale_runtime_status(repair=True)
    if runtime.get("ok"):
        return True
    last_error = str(runtime.get("last_bootstrap_error") or "").strip()
    if last_error:
        logger.warning("Initialisation TimeDB impossible: %s", last_error)
    return False


_timescale_inventory_cache = None
_timescale_inventory_cache_time = 0.0


def discover_timescale_inventory() -> dict[str, set[str]]:
    """Construit un inventaire minimal des symboles presents dans TimeDB."""
    global _timescale_inventory_cache, _timescale_inventory_cache_time
    import time
    now = time.time()
    if _timescale_inventory_cache is not None and (now - _timescale_inventory_cache_time) < 300.0:
        return _timescale_inventory_cache

    settings = get_timescale_settings()
    if not settings["enabled"]:
        return {}
    if not ensure_timescale_ready():
        return {}

    query = f"SELECT symbol, timeframe FROM {_sql_identifier(settings['bars_table'])} GROUP BY symbol, timeframe"
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
            
    _timescale_inventory_cache = inventory
    _timescale_inventory_cache_time = now
    return inventory


def load_history_frame_from_timescale(
    symbol: str,
    timeframe: str,
    limit: int | None = None,
) -> pd.DataFrame | None:
    """Charge un historique OHLCV depuis TimeDB."""

    if pd is None:
        logger.warning(
            "Lecture TimeDB indisponible pour %s %s: pandas n'est pas installe sur ce runtime.",
            symbol,
            timeframe,
        )
        return None

    settings = get_timescale_settings()
    if not settings["enabled"]:
        return None
    if not ensure_timescale_ready():
        return None

    query = (
        'SELECT "timestamp" AS time, open, high, low, close, '
        "COALESCE(tick_volume, 0) AS tick_volume, "
        "COALESCE(spread, 0) AS spread, "
        "COALESCE(real_volume, 0) AS real_volume "
        f"FROM {_sql_identifier(settings['bars_table'])} "
        "WHERE symbol = %s AND timeframe = %s "
        'ORDER BY "timestamp" DESC'
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


def _execute_upsert(query: str, params: Sequence[Any]) -> bool:
    """Execute un UPSERT simple si la base est disponible."""

    settings = get_timescale_settings()
    if not settings["enabled"]:
        return False
    if not ensure_timescale_ready():
        return False

    with _connect() as connection:
        if connection is None:
            return False
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
            connection.commit()
            return True
        except Exception as exc:
            logger.warning("Ecriture TimeDB impossible: %s", exc)
            return False


def record_training_dataset(
    descriptor: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Enregistre le manifeste de dataset si TimeDB est disponible."""

    settings = get_timescale_settings()
    dataset_id = str(descriptor.get("dataset_id") or "").strip()
    if not dataset_id:
        logger.warning("Manifeste dataset ignore: dataset_id absent.")
        return False

    coverage = dict(descriptor.get("dataset_coverage") or {})
    query = f"""
        INSERT INTO {_sql_identifier(settings['datasets_table'])} (
            dataset_id,
            engine,
            horizon,
            family,
            timeframe,
            feature_profile,
            mechanics_profile_version,
            source,
            bars_table,
            features_table,
            symbols,
            start_at,
            end_at,
            coverage_ratio,
            payload,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb)
        ON CONFLICT (dataset_id) DO UPDATE SET
            engine = EXCLUDED.engine,
            horizon = EXCLUDED.horizon,
            family = EXCLUDED.family,
            timeframe = EXCLUDED.timeframe,
            feature_profile = EXCLUDED.feature_profile,
            mechanics_profile_version = EXCLUDED.mechanics_profile_version,
            source = EXCLUDED.source,
            bars_table = EXCLUDED.bars_table,
            features_table = EXCLUDED.features_table,
            symbols = EXCLUDED.symbols,
            start_at = EXCLUDED.start_at,
            end_at = EXCLUDED.end_at,
            coverage_ratio = EXCLUDED.coverage_ratio,
            payload = EXCLUDED.payload,
            metadata = EXCLUDED.metadata,
            created_at = NOW()
    """
    return _execute_upsert(
        query,
        (
            dataset_id,
            str(descriptor.get("engine") or "").strip() or None,
            str(descriptor.get("horizon") or "").strip() or None,
            str(descriptor.get("family") or "").strip() or None,
            str(descriptor.get("timeframe") or "").strip() or None,
            str(descriptor.get("feature_profile") or "").strip() or None,
            str(descriptor.get("mechanics_profile_version") or "").strip() or None,
            str(descriptor.get("source") or "").strip() or None,
            settings["bars_table"],
            settings["features_table"],
            _json_dumps(list(descriptor.get("symbols") or [])),
            descriptor.get("start_at"),
            descriptor.get("end_at"),
            coverage.get("coverage_ratio"),
            _json_dumps(descriptor),
            _json_dumps(metadata or {}),
        ),
    )


def record_arena_result(
    report_payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Enregistre un rapport d'Arena ou de promotion si TimeDB est disponible."""

    settings = get_timescale_settings()
    battle_report = dict(report_payload.get("battle_report") or {})
    challenger = dict(battle_report.get("challenger") or {})
    champion = dict(battle_report.get("champion") or {})
    challenger_metrics = dict(challenger.get("metrics") or report_payload.get("metrics") or {})
    mechanics = dict(
        report_payload.get("metrics_by_position_mechanics")
        or challenger_metrics.get("metrics_by_position_mechanics")
        or {}
    )
    query = f"""
        INSERT INTO {_sql_identifier(settings['arena_table'])} (
            dataset_id,
            engine,
            horizon,
            family,
            feature_profile,
            challenger_id,
            champion_id,
            outcome,
            failure_mode,
            metrics,
            metrics_by_symbol,
            metrics_by_position_mechanics,
            payload,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
    """
    return _execute_upsert(
        query,
        (
            str(report_payload.get("dataset_id") or "").strip() or None,
            str(report_payload.get("engine") or "muzero"),
            str(report_payload.get("horizon") or "").strip() or None,
            str(report_payload.get("family") or "").strip() or None,
            str(report_payload.get("feature_profile") or "").strip() or None,
            str(challenger.get("id") or report_payload.get("challenger_id") or "").strip() or None,
            str(champion.get("id") or report_payload.get("live_champion_id") or "").strip() or None,
            str(battle_report.get("outcome") or report_payload.get("outcome") or "").strip() or None,
            str(report_payload.get("failure_mode") or "").strip() or None,
            _json_dumps(challenger_metrics),
            _json_dumps(report_payload.get("metrics_by_symbol") or {}),
            _json_dumps(mechanics),
            _json_dumps(report_payload),
            _json_dumps(metadata or {}),
        ),
    )


def record_ga_trial(trial_payload: dict[str, Any]) -> bool:
    """Enregistre un essai GA et son score dans TimeDB."""

    settings = get_timescale_settings()
    trial_id = str(trial_payload.get("trial_id") or "").strip()
    if not trial_id:
        logger.warning("Essai GA ignore: trial_id absent.")
        return False
    query = f"""
        INSERT INTO {_sql_identifier(settings['tables'].ga_trials)} (
            trial_id,
            engine,
            sequence_id,
            profile,
            horizon,
            family,
            feature_profile,
            mechanics_profile_version,
            ga_generation,
            ga_trial,
            trial_mode,
            trial_cost_profile,
            params,
            fitness_score,
            failure_mode,
            run_id,
            dataset_id,
            payload,
            finished_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (trial_id) DO UPDATE SET
            engine = EXCLUDED.engine,
            sequence_id = EXCLUDED.sequence_id,
            profile = EXCLUDED.profile,
            horizon = EXCLUDED.horizon,
            family = EXCLUDED.family,
            feature_profile = EXCLUDED.feature_profile,
            mechanics_profile_version = EXCLUDED.mechanics_profile_version,
            ga_generation = EXCLUDED.ga_generation,
            ga_trial = EXCLUDED.ga_trial,
            trial_mode = EXCLUDED.trial_mode,
            trial_cost_profile = EXCLUDED.trial_cost_profile,
            params = EXCLUDED.params,
            fitness_score = EXCLUDED.fitness_score,
            failure_mode = EXCLUDED.failure_mode,
            run_id = EXCLUDED.run_id,
            dataset_id = EXCLUDED.dataset_id,
            payload = EXCLUDED.payload,
            finished_at = EXCLUDED.finished_at
    """
    return _execute_upsert(
        query,
        (
            trial_id,
            str(trial_payload.get("engine") or "").strip() or None,
            str(trial_payload.get("sequence_id") or "").strip() or None,
            str(trial_payload.get("profile") or "").strip() or None,
            str(trial_payload.get("horizon") or "").strip() or None,
            str(trial_payload.get("family") or "").strip() or None,
            str(trial_payload.get("feature_profile") or "").strip() or None,
            str(trial_payload.get("mechanics_profile_version") or "").strip() or None,
            trial_payload.get("ga_generation"),
            str(trial_payload.get("ga_trial") or "").strip() or None,
            str(trial_payload.get("trial_mode") or "").strip() or None,
            str(trial_payload.get("trial_cost_profile") or "").strip() or None,
            _json_dumps(trial_payload.get("params") or {}),
            trial_payload.get("fitness_score"),
            str(trial_payload.get("failure_mode") or "").strip() or None,
            str(trial_payload.get("run_id") or "").strip() or None,
            str(trial_payload.get("dataset_id") or "").strip() or None,
            _json_dumps(trial_payload),
            trial_payload.get("finished_at"),
        ),
    )


def load_ga_trials(
    *,
    campaign_id: str | None = None,
    generation: int | None = None,
    limit: int = 128,
) -> list[dict[str, Any]]:
    """Charge les essais GA recents depuis TimeDB.

    Args:
        campaign_id (str | None): Identifiant optionnel de campagne seedee.
        generation (int | None): Generation optionnelle a filtrer.
        limit (int): Nombre maximal d'essais a retourner.

    Returns:
        list[dict[str, Any]]: Liste normalisee des essais charges.
    """

    settings = get_timescale_settings()
    if not settings["enabled"]:
        return []

    safe_limit = max(1, min(int(limit or 128), 512))
    where_clauses: list[str] = []
    params: list[Any] = []

    normalized_campaign_id = str(campaign_id or "").strip()
    if normalized_campaign_id:
        where_clauses.append("(payload->>'campaign_id') = %s")
        params.append(normalized_campaign_id)

    if generation is not None:
        where_clauses.append("ga_generation = %s")
        params.append(int(generation))

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    query = f"""
        SELECT
            trial_id,
            engine,
            sequence_id,
            profile,
            horizon,
            family,
            feature_profile,
            mechanics_profile_version,
            ga_generation,
            ga_trial,
            trial_mode,
            trial_cost_profile,
            fitness_score,
            failure_mode,
            run_id,
            dataset_id,
            payload,
            created_at,
            finished_at
        FROM {_sql_identifier(settings['tables'].ga_trials)}
        {where_sql}
        ORDER BY ga_generation ASC NULLS LAST, COALESCE(finished_at, created_at) DESC
        LIMIT %s
    """
    params.append(safe_limit)

    with _connect() as connection:
        if connection is None:
            return []
        try:
            _ensure_schema_objects(connection, settings)
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        except Exception as exc:
            logger.warning("Lecture des essais GA impossible: %s", exc)
            return []

    records: list[dict[str, Any]] = []
    for (
        trial_id,
        engine,
        sequence_id,
        profile,
        horizon,
        family,
        feature_profile,
        mechanics_profile_version,
        ga_generation,
        ga_trial,
        trial_mode,
        trial_cost_profile,
        fitness_score,
        failure_mode,
        run_id,
        dataset_id,
        payload,
        created_at,
        finished_at,
    ) in rows:
        parsed_payload = payload
        if isinstance(parsed_payload, str):
            try:
                parsed_payload = json.loads(parsed_payload)
            except Exception:
                parsed_payload = {}
        if not isinstance(parsed_payload, dict):
            parsed_payload = {}

        records.append(
            {
                "trial_id": str(trial_id or "").strip() or None,
                "engine": str(engine or "").strip() or None,
                "sequence_id": str(sequence_id or "").strip() or None,
                "profile": str(profile or "").strip() or None,
                "horizon": str(horizon or "").strip() or None,
                "family": str(family or "").strip() or None,
                "feature_profile": str(feature_profile or "").strip() or None,
                "mechanics_profile_version": str(mechanics_profile_version or "").strip() or None,
                "ga_generation": int(ga_generation) if ga_generation is not None else None,
                "ga_trial": str(ga_trial or "").strip() or None,
                "trial_mode": str(trial_mode or "").strip() or None,
                "trial_cost_profile": str(trial_cost_profile or "").strip() or None,
                "fitness_score": float(fitness_score) if fitness_score is not None else None,
                "failure_mode": str(failure_mode or "").strip() or None,
                "run_id": str(run_id or "").strip() or None,
                "dataset_id": str(dataset_id or "").strip() or None,
                "payload": parsed_payload,
                "created_at": created_at.isoformat() if created_at is not None else None,
                "finished_at": finished_at.isoformat() if finished_at is not None else None,
            }
        )
    return records


def record_replay_metadata(replay_payload: dict[str, Any]) -> bool:
    """Enregistre l'etat d'un cache replay partage."""

    settings = get_timescale_settings()
    cache_key = str(replay_payload.get("cache_key") or "").strip()
    if not cache_key:
        logger.warning("Replay metadata ignore: cache_key absent.")
        return False
    query = f"""
        INSERT INTO {_sql_identifier(settings['tables'].replay_metadata)} (
            cache_key,
            engine,
            horizon,
            family,
            feature_profile,
            mechanics_profile_version,
            entries,
            source,
            reuse_ratio,
            last_run_id,
            payload,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
        ON CONFLICT (cache_key) DO UPDATE SET
            engine = EXCLUDED.engine,
            horizon = EXCLUDED.horizon,
            family = EXCLUDED.family,
            feature_profile = EXCLUDED.feature_profile,
            mechanics_profile_version = EXCLUDED.mechanics_profile_version,
            entries = EXCLUDED.entries,
            source = EXCLUDED.source,
            reuse_ratio = EXCLUDED.reuse_ratio,
            last_run_id = EXCLUDED.last_run_id,
            payload = EXCLUDED.payload,
            updated_at = NOW()
    """
    return _execute_upsert(
        query,
        (
            cache_key,
            str(replay_payload.get("engine") or "").strip() or None,
            str(replay_payload.get("horizon") or "").strip() or None,
            str(replay_payload.get("family") or "").strip() or None,
            str(replay_payload.get("feature_profile") or "").strip() or None,
            str(replay_payload.get("mechanics_profile_version") or "").strip() or None,
            int(replay_payload.get("entries") or 0),
            str(replay_payload.get("source") or "").strip() or None,
            replay_payload.get("reuse_ratio"),
            str(replay_payload.get("last_run_id") or "").strip() or None,
            _json_dumps(replay_payload),
        ),
    )


def record_market_context_snapshot(context_payload: dict[str, Any]) -> bool:
    """Enregistre un snapshot de contexte prop consultatif."""

    settings = get_timescale_settings()
    snapshot_id = str(context_payload.get("snapshot_id") or "").strip()
    if not snapshot_id:
        logger.warning("Contexte marche ignore: snapshot_id absent.")
        return False
    query = f"""
        INSERT INTO {_sql_identifier(settings['tables'].market_context)} (
            snapshot_id,
            symbol,
            family,
            mode,
            macro_bias,
            event_risk,
            geo_risk,
            blocked,
            confidence,
            sources,
            payload,
            generated_at,
            expires_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
        ON CONFLICT (snapshot_id) DO UPDATE SET
            symbol = EXCLUDED.symbol,
            family = EXCLUDED.family,
            mode = EXCLUDED.mode,
            macro_bias = EXCLUDED.macro_bias,
            event_risk = EXCLUDED.event_risk,
            geo_risk = EXCLUDED.geo_risk,
            blocked = EXCLUDED.blocked,
            confidence = EXCLUDED.confidence,
            sources = EXCLUDED.sources,
            payload = EXCLUDED.payload,
            generated_at = EXCLUDED.generated_at,
            expires_at = EXCLUDED.expires_at
    """
    return _execute_upsert(
        query,
        (
            snapshot_id,
            str(context_payload.get("symbol") or "").strip(),
            str(context_payload.get("family") or "").strip() or None,
            str(context_payload.get("mode") or "prop"),
            str(context_payload.get("macro_bias") or "").strip() or None,
            str(context_payload.get("event_risk") or "").strip() or None,
            str(context_payload.get("geo_risk") or "").strip() or None,
            bool(context_payload.get("blocked", False)),
            float(context_payload.get("confidence") or 0.0),
            _json_dumps(list(context_payload.get("sources") or [])),
            _json_dumps(context_payload),
            context_payload.get("generated_at"),
            context_payload.get("expires_at"),
        ),
    )


def record_investment_thesis(thesis_payload: dict[str, Any]) -> bool:
    """Enregistre une these long terme consultative."""

    settings = get_timescale_settings()
    thesis_id = str(thesis_payload.get("thesis_id") or "").strip()
    if not thesis_id:
        logger.warning("These investissement ignoree: thesis_id absent.")
        return False
    query = f"""
        INSERT INTO {_sql_identifier(settings['tables'].investment_theses)} (
            thesis_id,
            symbol,
            issuer,
            mode,
            conviction_score,
            fundamental_risk,
            governance_risk,
            horizon_months,
            thesis,
            sources,
            payload,
            generated_at,
            review_status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
        ON CONFLICT (thesis_id) DO UPDATE SET
            symbol = EXCLUDED.symbol,
            issuer = EXCLUDED.issuer,
            mode = EXCLUDED.mode,
            conviction_score = EXCLUDED.conviction_score,
            fundamental_risk = EXCLUDED.fundamental_risk,
            governance_risk = EXCLUDED.governance_risk,
            horizon_months = EXCLUDED.horizon_months,
            thesis = EXCLUDED.thesis,
            sources = EXCLUDED.sources,
            payload = EXCLUDED.payload,
            generated_at = EXCLUDED.generated_at,
            review_status = EXCLUDED.review_status
    """
    return _execute_upsert(
        query,
        (
            thesis_id,
            str(thesis_payload.get("symbol") or "").strip(),
            str(thesis_payload.get("issuer") or "").strip() or None,
            str(thesis_payload.get("mode") or "invest"),
            float(thesis_payload.get("conviction_score") or 0.0),
            str(thesis_payload.get("fundamental_risk") or "").strip() or None,
            str(thesis_payload.get("governance_risk") or "").strip() or None,
            int(thesis_payload.get("horizon_months") or 12),
            str(thesis_payload.get("thesis") or ""),
            _json_dumps(list(thesis_payload.get("sources") or [])),
            _json_dumps(thesis_payload),
            thesis_payload.get("generated_at"),
            str(thesis_payload.get("review_status") or "draft"),
        ),
    )


def record_gpu_metric(metric_payload: dict[str, Any]) -> bool:
    """Enregistre un snapshot GPU structure."""

    settings = get_timescale_settings()
    query = f"""
        INSERT INTO {_sql_identifier(settings['tables'].gpu_metrics)} (
            "timestamp",
            host,
            gpu_name,
            utilization_percent,
            memory_used_mb,
            memory_total_mb,
            temperature_c,
            power_draw_w,
            source,
            payload
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (host, "timestamp") DO UPDATE SET
            gpu_name = EXCLUDED.gpu_name,
            utilization_percent = EXCLUDED.utilization_percent,
            memory_used_mb = EXCLUDED.memory_used_mb,
            memory_total_mb = EXCLUDED.memory_total_mb,
            temperature_c = EXCLUDED.temperature_c,
            power_draw_w = EXCLUDED.power_draw_w,
            source = EXCLUDED.source,
            payload = EXCLUDED.payload
    """
    return _execute_upsert(
        query,
        (
            metric_payload.get("timestamp"),
            str(metric_payload.get("host") or "").strip() or None,
            str(metric_payload.get("gpu_name") or "").strip() or None,
            metric_payload.get("utilization_percent"),
            metric_payload.get("memory_used_mb"),
            metric_payload.get("memory_total_mb"),
            metric_payload.get("temperature_c"),
            metric_payload.get("power_draw_w"),
            str(metric_payload.get("source") or "").strip() or None,
            _json_dumps(metric_payload),
        ),
    )


def record_cpu_job_history(job_payload: dict[str, Any]) -> bool:
    """Enregistre l'execution d'un job CPU ordonnance."""

    settings = get_timescale_settings()
    job_id = str(job_payload.get("job_id") or "").strip()
    if not job_id:
        logger.warning("Historique CPU ignore: job_id absent.")
        return False
    query = f"""
        INSERT INTO {_sql_identifier(settings['tables'].cpu_jobs)} (
            job_id,
            lane,
            job_name,
            status,
            host,
            payload,
            started_at,
            finished_at
        )
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
        ON CONFLICT (job_id) DO UPDATE SET
            lane = EXCLUDED.lane,
            job_name = EXCLUDED.job_name,
            status = EXCLUDED.status,
            host = EXCLUDED.host,
            payload = EXCLUDED.payload,
            started_at = EXCLUDED.started_at,
            finished_at = EXCLUDED.finished_at
    """
    return _execute_upsert(
        query,
        (
            job_id,
            str(job_payload.get("lane") or "").strip(),
            str(job_payload.get("job_name") or "").strip(),
            str(job_payload.get("status") or "").strip(),
            str(job_payload.get("host") or "").strip() or None,
            _json_dumps(job_payload),
            job_payload.get("started_at"),
            job_payload.get("finished_at"),
        ),
    )


def record_run_window(window_payload: dict[str, Any]) -> bool:
    """Enregistre l'etat historise d'une fenetre de sequence V4."""

    settings = get_timescale_settings()
    window_id = str(window_payload.get("window_id") or "").strip()
    sequence_id = str(window_payload.get("sequence_id") or "").strip()
    if not window_id or not sequence_id:
        logger.warning("Fenetre de run ignoree: window_id ou sequence_id absent.")
        return False
    query = f"""
        INSERT INTO {_sql_identifier(settings['tables'].run_windows)} (
            window_id,
            sequence_id,
            profile,
            engine,
            mode,
            trial_id,
            window_index,
            status,
            last_run_id,
            retry_count,
            payload,
            started_at,
            finished_at,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, NOW())
        ON CONFLICT (window_id) DO UPDATE SET
            sequence_id = EXCLUDED.sequence_id,
            profile = EXCLUDED.profile,
            engine = EXCLUDED.engine,
            mode = EXCLUDED.mode,
            trial_id = EXCLUDED.trial_id,
            window_index = EXCLUDED.window_index,
            status = EXCLUDED.status,
            last_run_id = EXCLUDED.last_run_id,
            retry_count = EXCLUDED.retry_count,
            payload = EXCLUDED.payload,
            started_at = EXCLUDED.started_at,
            finished_at = EXCLUDED.finished_at,
            updated_at = NOW()
    """
    return _execute_upsert(
        query,
        (
            window_id,
            sequence_id,
            str(window_payload.get("profile") or "").strip() or None,
            str(window_payload.get("engine") or "").strip() or None,
            str(window_payload.get("mode") or "").strip() or None,
            str(window_payload.get("trial_id") or "").strip() or None,
            window_payload.get("window_index"),
            str(window_payload.get("status") or "unknown"),
            str(window_payload.get("last_run_id") or "").strip() or None,
            int(window_payload.get("retry_count") or 0),
            _json_dumps(window_payload),
            window_payload.get("started_at"),
            window_payload.get("finished_at"),
        ),
    )


def insert_training_metric(
    engine: str,
    metric_name: str,
    metric_value: float,
    step: int,
    epoch: int | None = None,
    run_id: str | None = None,
) -> None:
    """Insere une metrique d'entrainement en temps reel dans la hypertable TimescaleDB."""
    settings = get_timescale_settings()
    if not settings["enabled"]:
        return

    # Si run_id n'est pas fourni, on tente de le charger depuis le training status
    if not run_id:
        try:
            from eva_lab.training_status import load_training_status
            run_status = load_training_status()
            run_id = run_status.get("run_id") or "nightly_fallback"
        except Exception:
            run_id = "nightly_fallback"

    tables = settings["tables"]
    with _connect() as connection:
        if connection is None:
            return
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {_sql_identifier(tables.metrics_history)}
                    ("timestamp", run_id, engine, metric_name, metric_value, step, epoch)
                    VALUES (NOW(), %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (run_id, engine, metric_name, float(metric_value), int(step), epoch),
                )
                connection.commit()
        except Exception as exc:
            logger.warning("Impossible d'inserer la metrique dans TimescaleDB: %s", exc)

