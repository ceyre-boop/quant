"""Holdout fences — miners get data ONLY through these loaders.

Locked at M0, before any mining ran (Plans/immutable-wondering-alpaca.md §Holdouts):
  equities: mining 2025-07-01..2026-06-30 (on disk); holdout 2024-07-01..2025-06-30
            is PHYSICALLY ABSENT from disk until G1 fetches it post-prereg-lock.
  NQ:       mining rows ts <= 2024-06-30; holdout = everything after.
  options:  mining chain files with quote date <= 2023-09-30; holdout = later files.

Miners MUST NOT read the underlying parquets/dirs directly — the fence tests
assert the loaders cannot return holdout rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

EQUITIES_MINING = ("2025-07-01", "2026-06-30")
EQUITIES_HOLDOUT = ("2024-07-01", "2025-06-30")   # G1-only; never fetched in M
NQ_MINING_END = pd.Timestamp("2024-06-30 23:59:59", tz="UTC")
OPTIONS_QUOTE_CUTOFF = "2023-09-30"

NQ_DIR = REPO / "data/es_nq"
CHAIN_DIR = REPO / "data/research/vrp_data_cache/SPY"
GAPPER_DIR = REPO / "data/research/gapper"


def load_nq(kind: str = "1min") -> pd.DataFrame:
    """NQ frames hard-truncated to the mining window. kind: 1min|5min|daily|aux."""
    fname = {"1min": "nq_globex_1min.parquet", "5min": "nq_historical_5min.parquet",
             "daily": "nq_daily.parquet", "aux": "aux_daily.parquet"}[kind]
    df = pd.read_parquet(NQ_DIR / fname)
    tcol = next(c for c in ("ts_event", "session", "date", "ts")
                if c in df.columns or c == df.index.name)
    if tcol == df.index.name:
        df = df.reset_index()
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    if ts.isna().all():  # date-like column without tz
        ts = pd.to_datetime(df[tcol]).dt.tz_localize("UTC")
    return df[ts <= NQ_MINING_END].copy()


def chain_files() -> list[Path]:
    """SPY chain parquets with quote date <= cutoff (filename: {quote}_{expiry}.parquet)."""
    out = []
    for fp in sorted(CHAIN_DIR.glob("*.parquet")):
        quote = fp.stem.split("_")[0]
        if quote <= OPTIONS_QUOTE_CUTOFF:
            out.append(fp)
    return out


def gapper_grouped_files() -> list[Path]:
    """Grouped-daily cache files, asserted inside the mining year."""
    files = sorted((GAPPER_DIR / "cache/grouped").glob("*.json.gz"))
    for fp in files:
        d = fp.name.split(".")[0]   # NOT .stem — "x.json.gz".stem == "x.json"
        assert EQUITIES_MINING[0] <= d <= EQUITIES_MINING[1], \
            f"holdout-dated file in mining cache: {fp.name}"
    return files


def assert_no_equities_holdout_on_disk() -> None:
    """The strongest fence: the equities holdout year must not exist locally
    during M-phase (G1 fetches it into data/research/yield_frontier/holdout_equities/)."""
    hd = REPO / "data/research/yield_frontier/holdout_equities"
    assert not any(hd.glob("**/*.json.gz")) if hd.exists() else True, \
        "equities holdout present on disk during mining phase"
    for fp in (GAPPER_DIR / "cache/grouped").glob("*.json.gz"):
        assert not (EQUITIES_HOLDOUT[0] <= fp.name.split(".")[0] <= EQUITIES_HOLDOUT[1]), \
            f"holdout-year file leaked into gapper cache: {fp.name}"
