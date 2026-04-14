-- Migration canonique V5.1 pour TimescaleDB / PostgreSQL.
-- Cette migration unifie les schemas market, training, research et ops.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS training;
CREATE SCHEMA IF NOT EXISTS research;
CREATE SCHEMA IF NOT EXISTS ops;

CREATE TABLE IF NOT EXISTS market.market_bars (
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
);

CREATE INDEX IF NOT EXISTS idx_market_bars_timestamp
    ON market.market_bars ("timestamp" DESC);

CREATE TABLE IF NOT EXISTS market.market_features (
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
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, feature_profile, "timestamp")
);

CREATE INDEX IF NOT EXISTS idx_market_features_timestamp
    ON market.market_features ("timestamp" DESC);

CREATE TABLE IF NOT EXISTS market.macro_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    region TEXT NULL,
    symbol TEXT NULL,
    severity TEXT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    starts_at TIMESTAMPTZ NULL,
    ends_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS market.asset_sessions (
    session_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    session_phase TEXT NOT NULL,
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS training.training_datasets (
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
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training.arena_results (
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
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics_by_symbol JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics_by_position_mechanics JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training.ga_trials (
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
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    fitness_score DOUBLE PRECISION NULL,
    failure_mode TEXT NULL,
    run_id TEXT NULL,
    dataset_id TEXT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ NULL
);

CREATE TABLE IF NOT EXISTS training.replay_metadata (
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
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training.run_windows (
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
    started_at TIMESTAMPTZ NULL,
    finished_at TIMESTAMPTZ NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE IF EXISTS training.ga_trials
    ADD COLUMN IF NOT EXISTS sequence_id TEXT NULL,
    ADD COLUMN IF NOT EXISTS profile TEXT NULL;

ALTER TABLE IF EXISTS training.run_windows
    ADD COLUMN IF NOT EXISTS sequence_id TEXT NULL,
    ADD COLUMN IF NOT EXISTS profile TEXT NULL,
    ADD COLUMN IF NOT EXISTS mode TEXT NULL,
    ADD COLUMN IF NOT EXISTS trial_id TEXT NULL,
    ADD COLUMN IF NOT EXISTS window_index INTEGER NULL,
    ADD COLUMN IF NOT EXISTS status TEXT NULL,
    ADD COLUMN IF NOT EXISTS last_run_id TEXT NULL,
    ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'training'
          AND table_name = 'run_windows'
          AND column_name = 'run_id'
          AND is_nullable = 'NO'
    ) THEN
        ALTER TABLE training.run_windows ALTER COLUMN run_id DROP NOT NULL;
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS research.market_context_snapshots (
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
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    generated_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NULL
);

CREATE TABLE IF NOT EXISTS research.investment_theses (
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
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    generated_at TIMESTAMPTZ NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'draft'
);

CREATE TABLE IF NOT EXISTS research.osint_signals (
    signal_id TEXT PRIMARY KEY,
    symbol TEXT NULL,
    issuer TEXT NULL,
    category TEXT NOT NULL,
    signal_strength DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS research.source_registry (
    source_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    last_success_at TIMESTAMPTZ NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS ops.service_health_history (
    health_id TEXT PRIMARY KEY,
    service_name TEXT NOT NULL,
    status TEXT NOT NULL,
    host TEXT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ops.capacity_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    host TEXT NOT NULL,
    cpu_percent DOUBLE PRECISION NULL,
    ram_percent DOUBLE PRECISION NULL,
    disk_percent DOUBLE PRECISION NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ops.gpu_metrics (
    "timestamp" TIMESTAMPTZ NOT NULL,
    host TEXT NOT NULL,
    gpu_name TEXT NULL,
    utilization_percent DOUBLE PRECISION NULL,
    memory_used_mb DOUBLE PRECISION NULL,
    memory_total_mb DOUBLE PRECISION NULL,
    temperature_c DOUBLE PRECISION NULL,
    power_draw_w DOUBLE PRECISION NULL,
    source TEXT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (host, "timestamp")
);

CREATE TABLE IF NOT EXISTS ops.cpu_jobs_history (
    job_id TEXT PRIMARY KEY,
    lane TEXT NOT NULL,
    job_name TEXT NOT NULL,
    status TEXT NOT NULL,
    host TEXT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NULL,
    finished_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE IF EXISTS ops.cpu_jobs_history
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

DROP VIEW IF EXISTS public.market_ohlc;

CREATE VIEW public.market_ohlc AS
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
FROM market.market_bars;

SELECT create_hypertable(
    'market.market_bars',
    'timestamp',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

ALTER TABLE market.market_bars
    SET (timescaledb.compress = true);

SELECT add_compression_policy(
    'market.market_bars',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'market.market_bars',
    INTERVAL '400 days',
    if_not_exists => TRUE
);

SELECT create_hypertable(
    'market.market_features',
    'timestamp',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

ALTER TABLE market.market_features
    SET (timescaledb.compress = true);

SELECT add_compression_policy(
    'market.market_features',
    INTERVAL '3 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'market.market_features',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

SELECT create_hypertable(
    'ops.gpu_metrics',
    'timestamp',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

SELECT add_retention_policy(
    'ops.gpu_metrics',
    INTERVAL '30 days',
    if_not_exists => TRUE
);

DELETE FROM ops.cpu_jobs_history
WHERE COALESCE(finished_at, started_at, created_at, NOW()) < NOW() - INTERVAL '30 days';
