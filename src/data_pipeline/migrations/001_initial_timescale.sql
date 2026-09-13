-- ============================================================
-- Migration 001: Initial TimescaleDB Schema
-- ============================================================

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 1. OHLCV Bar Data (Hypertable)
CREATE TABLE IF NOT EXISTS ohlcv_data (
    timestamp       TIMESTAMPTZ      NOT NULL,
    ticker          VARCHAR(16)      NOT NULL,
    open            NUMERIC(14, 4)   NOT NULL,
    high            NUMERIC(14, 4)   NOT NULL,
    low             NUMERIC(14, 4)   NOT NULL,
    close           NUMERIC(14, 4)   NOT NULL,
    adj_close       NUMERIC(14, 4),
    volume          BIGINT           NOT NULL,
    source          VARCHAR(32)      DEFAULT 'yfinance',
    created_at      TIMESTAMPTZ      DEFAULT NOW(),
    PRIMARY KEY (ticker, timestamp)
);

SELECT create_hypertable(
    'ohlcv_data',
    'timestamp',
    chunk_time_interval => INTERVAL '1 year',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_time ON ohlcv_data (ticker, timestamp DESC);

-- 2. Corporate Actions Table
CREATE TABLE IF NOT EXISTS corporate_actions (
    id                  BIGSERIAL        PRIMARY KEY,
    ticker              VARCHAR(16)      NOT NULL,
    action_date         DATE             NOT NULL,
    action_type         VARCHAR(32)      NOT NULL,
    ratio               NUMERIC(10, 4),
    adjustment_factor   NUMERIC(10, 6)   NOT NULL DEFAULT 1.0,
    estimated_dividend  NUMERIC(10, 4),
    source              VARCHAR(32)      DEFAULT 'yfinance',
    created_at          TIMESTAMPTZ      DEFAULT NOW(),
    CONSTRAINT uq_corp_action UNIQUE (ticker, action_date, action_type)
);

CREATE INDEX IF NOT EXISTS idx_corp_actions_ticker ON corporate_actions (ticker, action_date);

-- 3. Point-in-Time Universe Membership Log
CREATE TABLE IF NOT EXISTS universe_membership (
    id                  BIGSERIAL        PRIMARY KEY,
    as_of_date          DATE             NOT NULL,
    ticker              VARCHAR(16)      NOT NULL,
    in_universe         BOOLEAN          NOT NULL DEFAULT TRUE,
    is_delisted         BOOLEAN          NOT NULL DEFAULT FALSE,
    delisting_date      DATE,
    delisting_reason    VARCHAR(64),
    created_at          TIMESTAMPTZ      DEFAULT NOW(),
    CONSTRAINT uq_universe_member UNIQUE (as_of_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_universe_date ON universe_membership (as_of_date);

-- 4. Data Quality Audit Log
CREATE TABLE IF NOT EXISTS data_quality_log (
    id                  BIGSERIAL        PRIMARY KEY,
    ticker              VARCHAR(16)      NOT NULL,
    timestamp           TIMESTAMPTZ      NOT NULL,
    issue_type          VARCHAR(64)      NOT NULL,
    raw_value           TEXT,
    corrected_value     TEXT,
    strategy_used       VARCHAR(32)      DEFAULT 'flag_only',
    created_at          TIMESTAMPTZ      DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dq_ticker ON data_quality_log (ticker, timestamp);
