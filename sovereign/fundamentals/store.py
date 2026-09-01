"""sovereign/fundamentals/store.py — DuckDB storage for the fundamentals pipeline.

Owns the DB path, the connection factory, the schema (11 tables), and typed
upsert helpers for every dataclass in ``sovereign.fundamentals.types``.

Deliberately a SEPARATE database file from data/sentiment.db (sovereign/sentiment/store.py),
not a shared connection or a new schema in that file. Fundamentals harvesting touches SEC
EDGAR full-text/XBRL, FINRA short-volume, and 13F bulk parses — much higher blast radius
per bad run than the sentiment feeders. Keeping the DB file boundary hard means a corrupt
or half-written fundamentals harvest can never poison the sentiment tables the live carry
board reads every session. The pattern (module-level DB_PATH from config.loader params, a
SCHEMA string, connect()/init(), INSERT OR REPLACE upserts) is copied from
sovereign/sentiment/store.py, but nothing is imported from it — that module's own docstring
states it is deliberately decoupled, and fundamentals/ follows the same discipline.

Never imports from ict/ (NN#1 isolation) or from the execution path.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb

from config.loader import params

from sovereign.fundamentals.types import (
    EarningsEvent,
    EstimateSnapshot,
    InsiderTxn,
    InstitutionalPosition,
    PriceReaction,
    ShortInterestPoint,
    ShortVolumePoint,
    BorrowPoint,
)

ROOT = Path(__file__).resolve().parents[2]
# DB path is config-driven (config/parameters.yml :: fundamentals.db_path), relative to repo
# root. `params` may not have a "fundamentals" key on an older checkout (this module ships the
# key, but a stale config file on disk should not hard-crash an import) — .get() with the same
# default keeps this importable before the key lands.
DB_PATH = ROOT / params.get("fundamentals", {}).get("db_path", "data/fundamentals.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fund_symbol (
    ticker        VARCHAR,
    cik           INTEGER,
    cusip         VARCHAR,
    name          VARCHAR,
    in_watchlist  BOOLEAN,
    updated_at    TIMESTAMP,
    PRIMARY KEY (ticker)
);
CREATE TABLE IF NOT EXISTS fund_earnings_event (
    ticker            VARCHAR,
    fiscal_end        DATE,
    report_date       DATE,
    report_time       VARCHAR,     -- bmo | amc | unknown
    eps_estimate      DOUBLE,
    eps_actual        DOUBLE,
    eps_surprise      DOUBLE,
    eps_surprise_pct  DOUBLE,
    rev_estimate      DOUBLE,
    rev_actual        DOUBLE,
    guide_eps_low     DOUBLE,
    guide_eps_high    DOUBLE,
    eps_actual_gaap   DOUBLE,      -- SEC XBRL, independent cross-check of the vendor "actual"
    source            VARCHAR,
    published_ts      TIMESTAMP,
    fetched_at        TIMESTAMP,
    PRIMARY KEY (ticker, fiscal_end)
);
CREATE TABLE IF NOT EXISTS fund_estimate_snapshot (
    snapshot_date  DATE,
    ticker         VARCHAR,
    period         VARCHAR,        -- 0q | +1q | 0y | +1y
    period_end     DATE,
    eps_avg        DOUBLE,
    eps_low        DOUBLE,
    eps_high       DOUBLE,
    n_analysts     INTEGER,
    up_30d         INTEGER,
    down_30d       INTEGER,
    source         VARCHAR,
    fetched_at     TIMESTAMP,
    PRIMARY KEY (snapshot_date, ticker, period)
);
CREATE TABLE IF NOT EXISTS fund_price_reaction (
    ticker           VARCHAR,
    report_date      DATE,
    react_date       DATE,         -- the session that actually absorbed the print
    gap_pct          DOUBLE,
    d0_pct           DOUBLE,
    d1_pct           DOUBLE,
    d5_pct           DOUBLE,
    d0_excess_spy    DOUBLE,
    atr20_pre        DOUBLE,
    gap_over_atr     DOUBLE,
    bars_source      VARCHAR,
    computed_at      TIMESTAMP,
    PRIMARY KEY (ticker, report_date)
);
CREATE TABLE IF NOT EXISTS fund_insider_txn (
    accession      VARCHAR,
    line_no        INTEGER,
    ticker         VARCHAR,
    issuer_cik     INTEGER,
    owner_name     VARCHAR,
    owner_title    VARCHAR,
    is_director    BOOLEAN,
    is_officer     BOOLEAN,
    is_ten_pct     BOOLEAN,
    txn_date       DATE,
    filing_date    DATE,           -- the only date that gates knowability
    code           VARCHAR,        -- P purchase, S sale, A grant, M option, F tax
    shares         DOUBLE,
    price          DOUBLE,
    value_usd      DOUBLE,
    shares_after   DOUBLE,
    is_open_market BOOLEAN,        -- persisted copy of InsiderTxn.is_open_market (code in P,S)
    source         VARCHAR,
    fetched_at     TIMESTAMP,
    PRIMARY KEY (accession, line_no)
);
CREATE TABLE IF NOT EXISTS fund_institution_holding (
    period_end    DATE,
    filer_cik     INTEGER,
    cusip         VARCHAR,
    ticker        VARCHAR,
    filer_name    VARCHAR,
    filing_date   DATE,
    shares        DOUBLE,
    value_usd     DOUBLE,
    is_amendment  BOOLEAN,
    source        VARCHAR,
    fetched_at    TIMESTAMP,
    PRIMARY KEY (period_end, filer_cik, cusip)
);
CREATE TABLE IF NOT EXISTS fund_institution_agg (
    period_end        DATE,
    ticker             VARCHAR,
    n_holders          INTEGER,
    total_shares       DOUBLE,
    total_value_usd    DOUBLE,
    d_shares_qoq       DOUBLE,
    d_holders_qoq      INTEGER,
    new_positions      INTEGER,
    closed_positions   INTEGER,
    top_buyers         JSON,
    top_sellers        JSON,
    computed_at        TIMESTAMP,
    PRIMARY KEY (period_end, ticker)
);
CREATE TABLE IF NOT EXISTS fund_short_interest (
    settlement_date    DATE,
    ticker             VARCHAR,
    shares_short       DOUBLE,
    avg_daily_volume   DOUBLE,
    days_to_cover      DOUBLE,
    pct_float          DOUBLE,
    source             VARCHAR,
    published_ts       TIMESTAMP,
    fetched_at         TIMESTAMP,
    PRIMARY KEY (settlement_date, ticker, source)
);
CREATE TABLE IF NOT EXISTS fund_short_volume_daily (
    date                 DATE,
    ticker               VARCHAR,
    short_volume         DOUBLE,
    short_exempt_volume  DOUBLE,
    total_volume         DOUBLE,
    short_pct            DOUBLE,
    source               VARCHAR,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS fund_borrow (
    date               DATE,
    ticker             VARCHAR,
    tier               VARCHAR,
    available_shares   DOUBLE,
    fee_rate           DOUBLE,
    source             VARCHAR,
    PRIMARY KEY (date, ticker)
);
-- Append-only: every fetch attempt (success or failure) is a row, never upserted, so the
-- harvest history can never be silently overwritten by a later run.
CREATE TABLE IF NOT EXISTS fund_fetch_log (
    run_id        VARCHAR,
    source        VARCHAR,
    section       VARCHAR,
    ticker        VARCHAR,
    endpoint      VARCHAR,
    http_status   INTEGER,
    ok            BOOLEAN,
    rows          INTEGER,
    latency_ms    INTEGER,
    error         VARCHAR,
    fetched_at    TIMESTAMP
);
"""


def connect(read_only: bool = False, path: Path | str | None = None) -> "duckdb.DuckDBPyConnection":
    """Open the fundamentals DuckDB (creating the dir on write). Pass path=':memory:' for tests."""
    if path is None:
        path = DB_PATH
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        init_schema(con)
    return con


def init_schema(con: "duckdb.DuckDBPyConnection") -> None:
    """Create the eleven fundamentals tables if absent (idempotent)."""
    con.execute(SCHEMA)


def init() -> None:
    """Convenience entry point: open (creating the DB file + schema) and close."""
    connect().close()


# ── generic upsert plumbing ─────────────────────────────────────────────────────────────

def _upsert_rows(con: "duckdb.DuckDBPyConnection", table: str, columns: Sequence[str],
                  rows: Iterable[Sequence[Any]]) -> int:
    """INSERT OR REPLACE a batch of already-ordered row tuples. Empty input is a no-op —
    callers pass a possibly-empty section result straight through without a guard of their own."""
    rows = list(rows)
    if not rows:
        return 0
    placeholders = ", ".join(["?"] * len(columns))
    cols = ", ".join(columns)
    con.executemany(
        f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})", rows
    )
    return len(rows)


def _now() -> datetime:
    return datetime.utcnow()


# ── typed upsert helpers, one per dataclass in types.py ─────────────────────────────────

def upsert_symbols(rows: Iterable[dict]) -> int:
    """rows: dicts with ticker, cik, cusip, name, in_watchlist."""
    cols = ["ticker", "cik", "cusip", "name", "in_watchlist", "updated_at"]
    now = _now()
    data = [
        (r["ticker"], r.get("cik"), r.get("cusip"), r.get("name"),
         r.get("in_watchlist"), now)
        for r in rows
    ]
    with connect() as con:
        return _upsert_rows(con, "fund_symbol", cols, data)


def upsert_earnings(rows: Iterable[EarningsEvent]) -> int:
    cols = ["ticker", "fiscal_end", "report_date", "report_time", "eps_estimate",
             "eps_actual", "eps_surprise", "eps_surprise_pct", "rev_estimate",
             "rev_actual", "guide_eps_low", "guide_eps_high", "eps_actual_gaap",
             "source", "published_ts", "fetched_at"]
    now = _now()
    data = [
        (r.ticker, r.fiscal_end, r.report_date, r.report_time, r.eps_estimate,
         r.eps_actual, r.eps_surprise, r.eps_surprise_pct, r.rev_estimate,
         r.rev_actual, r.guide_eps_low, r.guide_eps_high, r.eps_actual_gaap,
         r.source, r.published_ts, now)
        for r in rows
    ]
    with connect() as con:
        return _upsert_rows(con, "fund_earnings_event", cols, data)


def upsert_estimates(rows: Iterable[EstimateSnapshot]) -> int:
    cols = ["snapshot_date", "ticker", "period", "period_end", "eps_avg", "eps_low",
             "eps_high", "n_analysts", "up_30d", "down_30d", "source", "fetched_at"]
    now = _now()
    data = [
        (r.snapshot_date, r.ticker, r.period, r.period_end, r.eps_avg, r.eps_low,
         r.eps_high, r.n_analysts, r.up_30d, r.down_30d, r.source, now)
        for r in rows
    ]
    with connect() as con:
        return _upsert_rows(con, "fund_estimate_snapshot", cols, data)


def upsert_reactions(rows: Iterable[PriceReaction]) -> int:
    cols = ["ticker", "report_date", "react_date", "gap_pct", "d0_pct", "d1_pct",
             "d5_pct", "d0_excess_spy", "atr20_pre", "gap_over_atr", "bars_source",
             "computed_at"]
    now = _now()
    data = [
        (r.ticker, r.report_date, r.react_date, r.gap_pct, r.d0_pct, r.d1_pct,
         r.d5_pct, r.d0_excess_spy, r.atr20_pre, r.gap_over_atr, r.source, now)
        for r in rows
    ]
    with connect() as con:
        return _upsert_rows(con, "fund_price_reaction", cols, data)


def upsert_insider(rows: Iterable[InsiderTxn]) -> int:
    cols = ["accession", "line_no", "ticker", "issuer_cik", "owner_name", "owner_title",
             "is_director", "is_officer", "is_ten_pct", "txn_date", "filing_date", "code",
             "shares", "price", "value_usd", "shares_after", "is_open_market", "source",
             "fetched_at"]
    now = _now()
    data = [
        (r.accession, r.line_no, r.ticker, None, r.owner_name, r.owner_title,
         r.is_director, r.is_officer, r.is_ten_pct, r.txn_date, r.filing_date, r.code,
         r.shares, r.price, r.value_usd, r.shares_after, r.is_open_market, r.source, now)
        for r in rows
    ]
    with connect() as con:
        return _upsert_rows(con, "fund_insider_txn", cols, data)


def upsert_holdings(rows: Iterable[InstitutionalPosition]) -> int:
    cols = ["period_end", "filer_cik", "cusip", "ticker", "filer_name", "filing_date",
             "shares", "value_usd", "is_amendment", "source", "fetched_at"]
    now = _now()
    data = [
        (r.period_end, r.filer_cik, r.cusip, r.ticker, r.filer_name, r.filing_date,
         r.shares, r.value_usd, r.is_amendment, r.source, now)
        for r in rows
    ]
    with connect() as con:
        return _upsert_rows(con, "fund_institution_holding", cols, data)


def upsert_short_interest(rows: Iterable[ShortInterestPoint]) -> int:
    cols = ["settlement_date", "ticker", "shares_short", "avg_daily_volume",
             "days_to_cover", "pct_float", "source", "published_ts", "fetched_at"]
    now = _now()
    data = [
        (r.settlement_date, r.ticker, r.shares_short, r.avg_daily_volume,
         r.days_to_cover, r.pct_float, r.source, r.published_ts, now)
        for r in rows
    ]
    with connect() as con:
        return _upsert_rows(con, "fund_short_interest", cols, data)


def upsert_short_volume(rows: Iterable[ShortVolumePoint]) -> int:
    cols = ["date", "ticker", "short_volume", "short_exempt_volume", "total_volume",
             "short_pct", "source"]
    data = [
        (r.date, r.ticker, r.short_volume, r.short_exempt_volume, r.total_volume,
         r.short_pct, r.source)
        for r in rows
    ]
    with connect() as con:
        return _upsert_rows(con, "fund_short_volume_daily", cols, data)


def upsert_borrow(rows: Iterable[BorrowPoint]) -> int:
    cols = ["date", "ticker", "tier", "available_shares", "fee_rate", "source"]
    data = [
        (r.date, r.ticker, r.tier, r.available_shares, r.fee_rate, r.source)
        for r in rows
    ]
    with connect() as con:
        return _upsert_rows(con, "fund_borrow", cols, data)


def log_fetch(run_id: str, source: str, section: str, ticker: str | None, endpoint: str,
              http_status: int | None, ok: bool, rows: int, latency_ms: int | None,
              error: str | None = None) -> None:
    """Append one fetch-attempt record. Never upserted — fund_fetch_log has no PK by design,
    so a re-run never erases the history of a prior run's failures."""
    with connect() as con:
        con.execute(
            "INSERT INTO fund_fetch_log (run_id, source, section, ticker, endpoint, "
            "http_status, ok, rows, latency_ms, error, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [run_id, source, section, ticker, endpoint, http_status, ok, rows,
             latency_ms, error, _now()],
        )
