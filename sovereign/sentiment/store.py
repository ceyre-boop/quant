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
CREATE TABLE IF NOT EXISTS sentiment_board_state (
    date            DATE,
    pair            VARCHAR,
    news_score      DOUBLE,
    macro_curve     DOUBLE,
    macro_spread    DOUBLE,
    macro_inflation DOUBLE,
    vix_level       DOUBLE,
    vix_momentum    DOUBLE,
    vix_regime      VARCHAR,
    built_at        TIMESTAMP,
    PRIMARY KEY (date, pair)
);
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
