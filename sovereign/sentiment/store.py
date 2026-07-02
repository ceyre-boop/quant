"""sovereign/sentiment/store.py — DuckDB storage + secrets for the sentiment pipeline.

Owns the DB path, the connection factory, the schema (4 tables), and a self-contained .env key reader
(deliberately replicated, not imported from layer1, to keep sovereign/sentiment/ decoupled and
isolation-clean). All four feeders + the board builder go through here.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb

from config.loader import params

ROOT = Path(__file__).resolve().parents[2]
# DB path is config-driven (config/parameters.yml :: sentiment.db_path), relative to repo root.
DB_PATH = ROOT / params["sentiment"].get("db_path", "data/sentiment.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sentiment_news_daily (
    date        DATE,
    pair        VARCHAR,
    n_articles  INTEGER,
    n_pos       INTEGER,
    n_neg       INTEGER,
    news_score  DOUBLE,
    fetched_at  TIMESTAMP,
    PRIMARY KEY (date, pair)
);
CREATE TABLE IF NOT EXISTS sentiment_macro_daily (
    date        DATE,
    series      VARCHAR,
    value       DOUBLE,
    delta_1d    DOUBLE,
    delta_5d    DOUBLE,
    fetched_at  TIMESTAMP,
    PRIMARY KEY (date, series)
);
CREATE TABLE IF NOT EXISTS sentiment_vix_daily (
    date         DATE,
    vix_close    DOUBLE,
    vix_5d_ago   DOUBLE,
    vix_momentum DOUBLE,
    vix_regime   VARCHAR,
    fetched_at   TIMESTAMP,
    PRIMARY KEY (date)
);
CREATE TABLE IF NOT EXISTS sentiment_gdelt_daily (
    date        DATE,
    pair        VARCHAR,
    tone_raw    DOUBLE,           -- GDELT avg article tone, [-100, 100]
    tone_score  DOUBLE,           -- tone_raw / 100, [-1, 1]
    tone_5d     DOUBLE,           -- tone_score - tone_score 5 trading days ago
    volume      DOUBLE,           -- log1p(article count) — attention/intensity
    fetched_at  TIMESTAMP,
    PRIMARY KEY (date, pair)
);
-- Economic "release innovation" — first-print actual vs naive baseline, z-scored. NOT a consensus surprise.
CREATE TABLE IF NOT EXISTS sentiment_surprise_release (
    publish_date DATE,            -- ALFRED realtime_start of the FIRST print (when the market saw it)
    series       VARCHAR,
    ref_date     DATE,            -- reference period of the data point
    first_print  DOUBLE,
    baseline     DOUBLE,
    surprise     DOUBLE,          -- first_print - baseline (release innovation)
    surprise_z   DOUBLE,          -- standardized over trailing releases (RAW per-series innovation)
    usd_sign     DOUBLE,          -- economic USD direction of a +surprise (config; UNRATE=-1, rest +1)
    fetched_at   TIMESTAMP,
    PRIMARY KEY (publish_date, series)
);
CREATE TABLE IF NOT EXISTS sentiment_surprise_daily (
    date            DATE,
    econ_surprise_z DOUBLE,       -- EWMA-decayed sum of standardized release innovations (US, broadcast)
    fetched_at      TIMESTAMP,
    PRIMARY KEY (date)
);
-- CFTC Commitment of Traders — large-spec (non-commercial) net positioning. publish_date = Tuesday
-- measurement + lag (Friday); the board may ONLY see a row on/after publish_date (no look-ahead).
CREATE TABLE IF NOT EXISTS sentiment_cot_weekly (
    measurement_date DATE,        -- CFTC "as of" Tuesday
    publish_date     DATE,        -- Tuesday + publish_lag (~Friday) — first date the board may use it
    pair             VARCHAR,
    noncomm_long     BIGINT,
    noncomm_short    BIGINT,
    net_spec         BIGINT,       -- long - short (large speculators)
    net_oi           DOUBLE,       -- net_spec / open_interest (scale-invariant level)
    net_pct          DOUBLE,       -- trailing-3yr percentile of net_spec [0,1] — the contrarian-extreme feature
    net_pct_1y       DOUBLE,       -- trailing-1yr (52w) percentile of net_spec [0,1]
    net_z_1y         DOUBLE,       -- trailing-1yr z-score of net_spec
    net_z_3y         DOUBLE,       -- trailing-3yr z-score of net_spec
    flush_1w         DOUBLE,       -- WoW Δnet_spec / trailing-1yr std of Δnet_spec (the "flush" feature)
    release_ts       TIMESTAMP,    -- provenance: CFTC release ~Friday 15:30 ET (naive ET; publish_date + 15:30)
    open_interest    BIGINT,
    fetched_at       TIMESTAMP,
    PRIMARY KEY (measurement_date, pair)
);
-- CFTC Traders in Financial Futures (TFF, futures-only, gpe5-46if) — leveraged-funds positioning,
-- the professional-speculator analog of legacy non-commercial. Same Tuesday-measured/Friday-published
-- keying and look-ahead guard as sentiment_cot_weekly. Crosses (AUDNZD) are leg differences.
CREATE TABLE IF NOT EXISTS sentiment_cot_tff_weekly (
    measurement_date DATE,
    publish_date     DATE,
    pair             VARCHAR,
    lev_long         BIGINT,
    lev_short        BIGINT,
    lev_net          BIGINT,       -- leveraged-funds long − short
    lev_net_oi       DOUBLE,       -- lev_net / open_interest
    lev_net_pct      DOUBLE,       -- trailing-3yr percentile of lev_net [0,1]
    lev_net_pct_1y   DOUBLE,       -- trailing-1yr percentile of lev_net [0,1]
    release_ts       TIMESTAMP,
    open_interest    BIGINT,
    fetched_at       TIMESTAMP,
    PRIMARY KEY (measurement_date, pair)
);
-- VRP-001 diversifier FEATURE — iv_atm (currency-ETF option implied vol) − rv_trailing (pair-spot
-- realized vol). Dated to the OBSERVABLE EOD close: iv_obs_date/rv_last_date both == date (the
-- forward-look test asserts both <= date). vrp_pct = trailing percentile of vrp_signal (rich vs cheap).
CREATE TABLE IF NOT EXISTS sentiment_vrp_daily (
    date         DATE,
    pair         VARCHAR,
    symbol       VARCHAR,       -- currency-ETF option underlying used as the IV proxy (FXE/FXB/FXY/FXA)
    expiry       DATE,          -- option expiration used for iv_atm
    dte          INTEGER,
    atm_strike   DOUBLE,
    iv_atm       DOUBLE,        -- at-the-money implied vol, annualized (native tier IV or BS-inverted)
    rv_trailing  DOUBLE,        -- trailing realized vol of the PAIR spot, annualized (causal)
    vrp_signal   DOUBLE,        -- iv_atm − rv_trailing
    vrp_pct      DOUBLE,        -- trailing percentile of vrp_signal over its own history [0,1]
    iv_source    VARCHAR,       -- 'native' | 'bs_invert'
    iv_obs_date  DATE,          -- provenance: last data date feeding IV (== date; forward-look guard)
    rv_last_date DATE,          -- provenance: last data date feeding RV (== date; forward-look guard)
    fetched_at   TIMESTAMP,
    PRIMARY KEY (date, pair)
);
CREATE TABLE IF NOT EXISTS sentiment_board_state (
    date            DATE,
    pair            VARCHAR,
    news_score      DOUBLE,
    gdelt_tone      DOUBLE,
    gdelt_tone_5d   DOUBLE,
    gdelt_volume    DOUBLE,
    econ_surprise_z DOUBLE,
    cot_net_pct     DOUBLE,
    cot_net_oi      DOUBLE,
    vrp_signal      DOUBLE,
    vrp_pct         DOUBLE,
    vrp_iv_atm      DOUBLE,
    vrp_rv_trailing DOUBLE,
    macro_curve     DOUBLE,
    macro_spread    DOUBLE,
    macro_inflation DOUBLE,
    vix_level       DOUBLE,
    vix_momentum    DOUBLE,
    vix_regime      VARCHAR,
    built_at        TIMESTAMP,
    PRIMARY KEY (date, pair)
);
-- migrate pre-Step-1 board tables (gitignored runtime DBs) without a rebuild-from-scratch
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS gdelt_tone DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS gdelt_tone_5d DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS gdelt_volume DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS econ_surprise_z DOUBLE;
ALTER TABLE sentiment_surprise_release ADD COLUMN IF NOT EXISTS usd_sign DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS cot_net_pct DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS cot_net_oi DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS vrp_signal DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS vrp_pct DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS vrp_iv_atm DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS vrp_rv_trailing DOUBLE;
ALTER TABLE sentiment_cot_weekly ADD COLUMN IF NOT EXISTS net_pct_1y DOUBLE;
ALTER TABLE sentiment_cot_weekly ADD COLUMN IF NOT EXISTS net_z_1y DOUBLE;
ALTER TABLE sentiment_cot_weekly ADD COLUMN IF NOT EXISTS net_z_3y DOUBLE;
ALTER TABLE sentiment_cot_weekly ADD COLUMN IF NOT EXISTS flush_1w DOUBLE;
ALTER TABLE sentiment_cot_weekly ADD COLUMN IF NOT EXISTS release_ts TIMESTAMP;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS cot_net_pct_1y DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS cot_net_z DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS cot_flush_1w DOUBLE;
ALTER TABLE sentiment_board_state ADD COLUMN IF NOT EXISTS tff_lev_net_pct DOUBLE;
"""


def connect(read_only: bool = False, path: Path | str | None = None) -> "duckdb.DuckDBPyConnection":
    """Open the sentiment DuckDB (creating the dir on write). Pass path=':memory:' for tests."""
    if path is None:
        path = DB_PATH
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        init_schema(con)
    return con


def init_schema(con: "duckdb.DuckDBPyConnection") -> None:
    """Create the four sentiment tables if absent (idempotent)."""
    con.execute(SCHEMA)


def upsert(con: "duckdb.DuckDBPyConnection", table: str, df, key_cols: list[str]) -> int:
    """Idempotent write: delete any rows whose keys collide with df, then insert df.

    Keeps the feeders safe to re-run (NON-NEGOTIABLE: scheduled jobs are idempotent). Column
    order is taken from df; the DataFrame must carry exactly the table's columns it intends to set.
    Returns the number of rows written.
    """
    if df is None or len(df) == 0:
        return 0
    con.register("_incoming", df)
    try:
        cond = " AND ".join(f"t.{k} = i.{k}" for k in key_cols)
        con.execute(f"DELETE FROM {table} t WHERE EXISTS (SELECT 1 FROM _incoming i WHERE {cond})")
        cols = ", ".join(df.columns)
        con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _incoming")
    finally:
        con.unregister("_incoming")
    return len(df)


def env_key(name: str) -> str:
    """Read an API key from the environment, falling back to a manual parse of ROOT/.env.

    Replicates layer1/data_loader.fred_api_key's pattern (NOT imported, to keep sentiment/ decoupled).
    """
    val = os.environ.get(name)
    if val:
        return val
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not found in environment or {env}")
