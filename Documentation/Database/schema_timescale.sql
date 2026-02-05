-- ═══════════════════════════════════════════════════════════════════════════════
-- THE HIVE - TimescaleDB Schema Definition
-- ═══════════════════════════════════════════════════════════════════════════════
-- Database: PostgreSQL 15+ with TimescaleDB extension
-- Purpose: Stockage des données de marché, trades, logs système et audit
-- ═══════════════════════════════════════════════════════════════════════════════

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ═══════════════════════════════════════════════════════════════════════════════
-- MARKET DATA TABLES (The Banker)
-- ═══════════════════════════════════════════════════════════════════════════════

-- Tick Data (Données brutes temps réel)
CREATE TABLE market_ticks (
    time        TIMESTAMPTZ         NOT NULL,
    symbol      VARCHAR(20)         NOT NULL,
    bid         DECIMAL(18, 8)      NOT NULL,
    ask         DECIMAL(18, 8)      NOT NULL,
    spread      DECIMAL(10, 2)      GENERATED ALWAYS AS ((ask - bid) * 10000) STORED,
    volume      BIGINT              DEFAULT 0
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('market_ticks', 'time');

-- Index for fast symbol lookup
CREATE INDEX idx_ticks_symbol ON market_ticks (symbol, time DESC);

-- OHLC Bars (Aggregated candles)
CREATE TABLE market_ohlc (
    time        TIMESTAMPTZ         NOT NULL,
    symbol      VARCHAR(20)         NOT NULL,
    timeframe   VARCHAR(5)          NOT NULL CHECK (timeframe IN ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN')),
    open        DECIMAL(18, 8)      NOT NULL,
    high        DECIMAL(18, 8)      NOT NULL,
    low         DECIMAL(18, 8)      NOT NULL,
    close       DECIMAL(18, 8)      NOT NULL,
    tick_volume BIGINT              DEFAULT 0,
    real_volume BIGINT              DEFAULT 0,
    spread      INTEGER             DEFAULT 0
);

SELECT create_hypertable('market_ohlc', 'time');

-- Compound index for quick OHLC queries
CREATE UNIQUE INDEX idx_ohlc_unique ON market_ohlc (symbol, timeframe, time);


-- ═══════════════════════════════════════════════════════════════════════════════
-- TRADING TABLES (The Banker)
-- ═══════════════════════════════════════════════════════════════════════════════

-- Enum for trade actions
CREATE TYPE trade_action AS ENUM ('BUY', 'SELL');
CREATE TYPE order_status AS ENUM ('pending', 'filled', 'partial', 'cancelled', 'rejected');
CREATE TYPE close_reason AS ENUM ('manual', 'sl_hit', 'tp_hit', 'margin_call', 'algo', 'timeout');

-- Trading Accounts (Hydra system - multiple Prop Firm accounts)
CREATE TABLE trading_accounts (
    id              UUID                PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100)        NOT NULL,
    broker          VARCHAR(50)         NOT NULL,  -- FTMO, FundedNext, The5ers, etc.
    login           INTEGER             NOT NULL,
    server          VARCHAR(100)        NOT NULL,
    phase           VARCHAR(20)         CHECK (phase IN ('challenge', 'verification', 'funded')),
    initial_balance DECIMAL(15, 2)      NOT NULL,
    current_balance DECIMAL(15, 2)      NOT NULL,
    currency        VARCHAR(3)          DEFAULT 'USD',
    is_master       BOOLEAN             DEFAULT FALSE,  -- Master account for copy trading
    copy_enabled    BOOLEAN             DEFAULT TRUE,
    status          VARCHAR(20)         DEFAULT 'active',
    created_at      TIMESTAMPTZ         DEFAULT NOW(),
    updated_at      TIMESTAMPTZ         DEFAULT NOW()
);

-- Trade Orders (Historique complet)
CREATE TABLE trade_orders (
    id              UUID                PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id      UUID                REFERENCES trading_accounts(id),
    ticket          BIGINT              NOT NULL,       -- MT5 ticket number
    symbol          VARCHAR(20)         NOT NULL,
    action          trade_action        NOT NULL,
    volume          DECIMAL(10, 4)      NOT NULL,
    entry_price     DECIMAL(18, 8)      NOT NULL,
    close_price     DECIMAL(18, 8),
    stop_loss       DECIMAL(18, 8)      NOT NULL,       -- Obligatoire (ROE)
    take_profit     DECIMAL(18, 8),
    profit          DECIMAL(15, 2),
    swap            DECIMAL(10, 2)      DEFAULT 0,
    commission      DECIMAL(10, 2)      DEFAULT 0,
    magic_number    INTEGER             DEFAULT 12345,
    comment         VARCHAR(100),
    source          VARCHAR(20)         DEFAULT 'manual' CHECK (source IN ('manual', 'algo', 'copier')),
    status          order_status        NOT NULL DEFAULT 'pending',
    close_reason    close_reason,
    open_time       TIMESTAMPTZ         NOT NULL,
    close_time      TIMESTAMPTZ,
    execution_ms    INTEGER,            -- Execution latency
    slippage_pts    INTEGER,            -- Slippage in points
    created_at      TIMESTAMPTZ         DEFAULT NOW()
);

SELECT create_hypertable('trade_orders', 'open_time');

CREATE INDEX idx_orders_account ON trade_orders (account_id, open_time DESC);
CREATE INDEX idx_orders_symbol ON trade_orders (symbol, open_time DESC);

-- Daily Risk Snapshots (Pour tracking drawdown - Loi 2)
CREATE TABLE daily_risk_snapshots (
    date            DATE                NOT NULL,
    account_id      UUID                REFERENCES trading_accounts(id),
    opening_balance DECIMAL(15, 2)      NOT NULL,
    closing_balance DECIMAL(15, 2),
    high_water_mark DECIMAL(15, 2)      NOT NULL,
    daily_pnl       DECIMAL(15, 2),
    daily_dd_pct    DECIMAL(5, 2),      -- Daily drawdown %
    total_dd_pct    DECIMAL(5, 2),      -- Total drawdown %
    trades_count    INTEGER             DEFAULT 0,
    wins            INTEGER             DEFAULT 0,
    losses          INTEGER             DEFAULT 0,
    anti_tilt_triggered BOOLEAN         DEFAULT FALSE,
    PRIMARY KEY (date, account_id)
);


-- ═══════════════════════════════════════════════════════════════════════════════
-- AGENT ACTIVITY TABLES (EVA Core)
-- ═══════════════════════════════════════════════════════════════════════════════

-- Agent types enum
CREATE TYPE agent_type AS ENUM (
    'core', 'banker', 'shadow', 'builder', 'sentinel', 
    'muse', 'sage', 'wraith', 'researcher', 'advocate', 'sovereign'
);

-- Sessions de conversation
CREATE TABLE chat_sessions (
    id              UUID                PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         VARCHAR(100)        DEFAULT 'admin',
    started_at      TIMESTAMPTZ         DEFAULT NOW(),
    last_activity   TIMESTAMPTZ         DEFAULT NOW(),
    context         JSONB,              -- User context (location, mood, etc.)
    summary         TEXT                -- AI-generated session summary
);

-- Messages de conversation
CREATE TABLE chat_messages (
    id              UUID                PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID                REFERENCES chat_sessions(id),
    role            VARCHAR(20)         NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT                NOT NULL,
    intent_type     VARCHAR(50),        -- Classified intent
    routed_to       agent_type,         -- Which expert handled this
    tokens_used     INTEGER,
    processing_ms   INTEGER,
    timestamp       TIMESTAMPTZ         DEFAULT NOW(),
    metadata        JSONB
);

SELECT create_hypertable('chat_messages', 'timestamp');

CREATE INDEX idx_messages_session ON chat_messages (session_id, timestamp DESC);

-- Agent Activity Log
CREATE TABLE agent_activity_log (
    time            TIMESTAMPTZ         NOT NULL,
    agent           agent_type          NOT NULL,
    action          VARCHAR(100)        NOT NULL,
    status          VARCHAR(20)         CHECK (status IN ('started', 'completed', 'failed', 'timeout')),
    duration_ms     INTEGER,
    input_summary   TEXT,
    output_summary  TEXT,
    error           TEXT,
    metadata        JSONB
);

SELECT create_hypertable('agent_activity_log', 'time');


-- ═══════════════════════════════════════════════════════════════════════════════
-- SECURITY & AUDIT TABLES (The Sentinel / Black Box)
-- ═══════════════════════════════════════════════════════════════════════════════

-- Security Events (Sentinel)
CREATE TABLE security_events (
    time            TIMESTAMPTZ         NOT NULL,
    event_type      VARCHAR(50)         NOT NULL,  -- intrusion_attempt, auth_failure, scan_detected, etc.
    severity        VARCHAR(10)         CHECK (severity IN ('info', 'low', 'medium', 'high', 'critical')),
    source_ip       INET,
    target_service  VARCHAR(50),
    description     TEXT,
    action_taken    VARCHAR(100),       -- blocked, logged, alerted
    raw_data        JSONB,
    handled_by      VARCHAR(50)         DEFAULT 'sentinel'
);

SELECT create_hypertable('security_events', 'time');

CREATE INDEX idx_security_severity ON security_events (severity, time DESC);

-- Audit Trail (Black Box - Immutable Log)
-- Note: Cette table simule une blockchain-like immutable log
CREATE TABLE audit_trail (
    id              BIGSERIAL           PRIMARY KEY,
    time            TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    event_type      VARCHAR(50)         NOT NULL,
    actor           VARCHAR(50)         NOT NULL,  -- admin, eva_core, banker, kernel
    action          TEXT                NOT NULL,
    target          TEXT,
    old_value       JSONB,
    new_value       JSONB,
    context         JSONB,              -- System state at time of action
    prev_hash       VARCHAR(128),       -- Hash of previous record (blockchain-like)
    record_hash     VARCHAR(128)        NOT NULL   -- SHA-512 of this record
);

-- Trigger to compute hash on insert
CREATE OR REPLACE FUNCTION compute_audit_hash()
RETURNS TRIGGER AS $$
DECLARE
    prev_hash_val VARCHAR(128);
    hash_input TEXT;
BEGIN
    -- Get previous record hash
    SELECT record_hash INTO prev_hash_val 
    FROM audit_trail 
    ORDER BY id DESC 
    LIMIT 1;
    
    NEW.prev_hash := COALESCE(prev_hash_val, 'GENESIS');
    
    -- Compute hash of current record
    hash_input := NEW.time::TEXT || NEW.event_type || NEW.actor || NEW.action || 
                  COALESCE(NEW.target, '') || COALESCE(NEW.prev_hash, '');
    NEW.record_hash := encode(sha512(hash_input::bytea), 'hex');
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_hash_trigger
    BEFORE INSERT ON audit_trail
    FOR EACH ROW
    EXECUTE FUNCTION compute_audit_hash();

-- Prevent updates and deletes on audit_trail
CREATE OR REPLACE FUNCTION prevent_audit_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Audit trail is immutable - modifications are forbidden';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_immutable_trigger
    BEFORE UPDATE OR DELETE ON audit_trail
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_modification();


-- ═══════════════════════════════════════════════════════════════════════════════
-- SYSTEM MONITORING TABLES (The Keeper)
-- ═══════════════════════════════════════════════════════════════════════════════

-- Hardware Metrics
CREATE TABLE system_metrics (
    time            TIMESTAMPTZ         NOT NULL,
    metric_name     VARCHAR(50)         NOT NULL,
    value           DECIMAL(15, 4)      NOT NULL,
    unit            VARCHAR(20),
    host            VARCHAR(50)         DEFAULT 'the-hive',
    tags            JSONB
);

SELECT create_hypertable('system_metrics', 'time');

-- Common metrics: gpu_temp, gpu_power, gpu_vram_used, cpu_load, ram_used, disk_io, etc.
CREATE INDEX idx_metrics_name ON system_metrics (metric_name, time DESC);

-- Retention Policies for automatic data cleanup
SELECT add_retention_policy('market_ticks', INTERVAL '7 days');       -- Ticks: 7 jours
SELECT add_retention_policy('system_metrics', INTERVAL '30 days');    -- Metrics: 30 jours
SELECT add_retention_policy('security_events', INTERVAL '365 days');  -- Security: 1 an
-- audit_trail: No retention - kept forever (Black Box requirement)


-- ═══════════════════════════════════════════════════════════════════════════════
-- VIEWS & FUNCTIONS
-- ═══════════════════════════════════════════════════════════════════════════════

-- View: Current day P&L per account
CREATE VIEW v_daily_pnl AS
SELECT 
    ta.id AS account_id,
    ta.name AS account_name,
    ta.broker,
    ta.current_balance,
    COALESCE(SUM(to_pnl.profit), 0) AS today_pnl,
    COUNT(to_pnl.id) AS trades_today
FROM trading_accounts ta
LEFT JOIN trade_orders to_pnl ON ta.id = to_pnl.account_id 
    AND to_pnl.close_time >= CURRENT_DATE
    AND to_pnl.status = 'filled'
GROUP BY ta.id, ta.name, ta.broker, ta.current_balance;

-- View: Active positions across all accounts
CREATE VIEW v_active_positions AS
SELECT 
    ta.name AS account_name,
    ta.broker,
    tr.ticket,
    tr.symbol,
    tr.action,
    tr.volume,
    tr.entry_price,
    tr.stop_loss,
    tr.take_profit,
    tr.open_time
FROM trade_orders tr
JOIN trading_accounts ta ON tr.account_id = ta.id
WHERE tr.status = 'filled' AND tr.close_time IS NULL;

-- Function: Check if trading is allowed (Risk Management)
CREATE OR REPLACE FUNCTION check_trading_allowed(p_account_id UUID)
RETURNS TABLE (
    allowed BOOLEAN,
    reason TEXT
) AS $$
DECLARE
    v_daily_dd DECIMAL(5,2);
    v_total_dd DECIMAL(5,2);
    v_recent_losses INTEGER;
BEGIN
    -- Get current drawdown
    SELECT daily_dd_pct, total_dd_pct 
    INTO v_daily_dd, v_total_dd
    FROM daily_risk_snapshots 
    WHERE account_id = p_account_id AND date = CURRENT_DATE;
    
    -- Check daily drawdown (Loi 2: max 4%)
    IF COALESCE(v_daily_dd, 0) >= 3.95 THEN
        RETURN QUERY SELECT FALSE, 'Daily drawdown limit reached (4%)';
        RETURN;
    END IF;
    
    -- Check total drawdown (Loi 2: max 8%)
    IF COALESCE(v_total_dd, 0) >= 7.95 THEN
        RETURN QUERY SELECT FALSE, 'Total drawdown limit reached (8%)';
        RETURN;
    END IF;
    
    -- Anti-tilt check (2 consecutive losses)
    SELECT COUNT(*) INTO v_recent_losses
    FROM (
        SELECT profit FROM trade_orders 
        WHERE account_id = p_account_id 
        AND status = 'filled' 
        AND close_time >= NOW() - INTERVAL '24 hours'
        ORDER BY close_time DESC
        LIMIT 2
    ) recent
    WHERE profit < 0;
    
    IF v_recent_losses >= 2 THEN
        RETURN QUERY SELECT FALSE, 'Anti-tilt active (2 consecutive losses)';
        RETURN;
    END IF;
    
    RETURN QUERY SELECT TRUE, 'Trading allowed';
END;
$$ LANGUAGE plpgsql;


-- ═══════════════════════════════════════════════════════════════════════════════
-- INITIAL DATA
-- ═══════════════════════════════════════════════════════════════════════════════

-- Insert genesis audit record
INSERT INTO audit_trail (event_type, actor, action, context)
VALUES ('GENESIS', 'system', 'Database schema initialized', 
        '{"version": "1.0.0", "constitution_hash": "pending"}'::jsonb);
