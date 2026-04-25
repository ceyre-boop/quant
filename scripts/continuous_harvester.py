"""
Continuous Backtest Harvester — 24/7 Data Generation Engine
============================================================
Walk-forward across 7 years of history × 64 assets × 4 strategies × 25 param combos.
Each trade record is enriched with market context so XGBoost can learn:
  "in regime R, with Hurst H, strategy S fails/works"

Writes to DuckDB at data/harvest.db — append-only, columnar, SQL-queryable.
Run alongside live trading: zero interference (separate process, read-only market data).

Usage:
    python scripts/continuous_harvester.py
    python scripts/continuous_harvester.py --passes 1  # single pass then exit
    python scripts/continuous_harvester.py --symbols AAPL NVDA  # subset
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import multiprocessing as mp
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.fast_engine import FastBacktestEngine, SweepParams
from config.universe import UNIVERSE

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH    = ROOT / "data" / "harvest.db"
CACHE_DIR  = ROOT / "data" / "_price_cache"
LOG_PATH   = ROOT / "logs" / "harvester.log"

HISTORY_START = "2017-01-01"
HISTORY_END   = "2024-12-31"
WINDOW_DAYS   = 90          # walk-forward window size
STEP_DAYS     = 5           # slide step — smaller = more overlap = more records
MIN_BARS      = 60          # skip windows with fewer bars (thin data)
N_CORES       = min(mp.cpu_count(), 12)

STRATEGIES = ["momentum_sma", "donchian_breakout", "bb_reversion", "atr_channel"]

PARAM_GRID = [
    SweepParams(stop_atr_mult=s, tp_rr=t)
    for s in [1.0, 1.5, 2.0, 2.5, 3.0]
    for t in [1.5, 2.0, 2.5, 3.0, 4.0]
]  # 25 combos

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HARVEST] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── DB schema ─────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id        VARCHAR PRIMARY KEY,
    symbol          VARCHAR,
    strategy        VARCHAR,
    window_start    DATE,
    window_end      DATE,
    -- Params
    stop_atr_mult   FLOAT,
    tp_rr           FLOAT,
    atr_period      INT,
    -- Trade record
    entry_bar       INT,
    exit_bar        INT,
    direction       INT,
    pnl             FLOAT,
    pnl_r           FLOAT,
    is_stop         INT,
    is_profitable   INT,
    -- Market context at entry bar
    regime          INT,        -- 0=trending-low-vol  1=high-vol-ranging  -1=unknown
    hurst           FLOAT,      -- 0.5=random  <0.5=mean-rev  >0.5=trending
    atr_norm        FLOAT,      -- ATR / close price  (volatility proxy)
    vol_pct         FLOAT,      -- rolling vol percentile 0-1
    month           INT,
    day_of_week     INT,
    -- Metadata
    harvested_at    TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_symbol   ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_regime   ON trades(regime);
CREATE INDEX IF NOT EXISTS idx_harvested ON trades(harvested_at);
"""

PROGRESS_SCHEMA = """
CREATE TABLE IF NOT EXISTS harvest_progress (
    key     VARCHAR PRIMARY KEY,
    value   VARCHAR
);
"""


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA)
    con.execute(PROGRESS_SCHEMA)


# ── Price data (cached to disk) ───────────────────────────────────────────────

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    cache_file = CACHE_DIR / f"{symbol.replace('=','_')}.parquet"

    # Refresh cache if older than 12 hours or missing
    needs_refresh = (
        not cache_file.exists()
        or (time.time() - cache_file.stat().st_mtime) > 43200
    )

    if needs_refresh:
        try:
            raw = yf.download(
                symbol,
                start=HISTORY_START,
                end=HISTORY_END,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty or len(raw) < 100:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw = raw[["open", "high", "low", "close", "volume"]].dropna()
            raw.to_parquet(cache_file)
        except Exception as e:
            log.warning(f"Download failed {symbol}: {e}")
            return None

    try:
        return pd.read_parquet(cache_file)
    except Exception:
        return None


# ── Signal generators ─────────────────────────────────────────────────────────

def _sma_signals(df: pd.DataFrame, fast=9, slow=21):
    c = df["close"].to_numpy(np.float32)
    def sma(a, w):
        out = np.empty_like(a); cs = np.cumsum(a)
        out[:w] = a[:w].mean(); out[w:] = (cs[w:] - cs[:-w]) / w
        return out
    fm, sm = sma(c, fast), sma(c, slow)
    d = fm - sm; pd_ = np.roll(d, 1); pd_[0] = d[0]
    sig = np.zeros(len(c), np.int8)
    sig[(d > 0) & (pd_ <= 0)] = 1
    sig[(d < 0) & (pd_ >= 0)] = -1
    slope = np.abs(d - pd_) / (sm + 1e-9)
    return sig, np.clip(slope * 1000, 0, 1).astype(np.float32)


def _donchian_signals(df: pd.DataFrame, period=20):
    h = df["high"].to_numpy(np.float32)
    l = df["low"].to_numpy(np.float32)
    c = df["close"].to_numpy(np.float32)
    n = len(c)
    sig = np.zeros(n, np.int8)
    conf = np.zeros(n, np.float32)
    for i in range(period, n):
        highest = h[i-period:i].max()
        lowest  = l[i-period:i].min()
        rng = highest - lowest + 1e-9
        if c[i] > highest:
            sig[i] = 1;  conf[i] = min((c[i] - highest) / rng * 10, 1.0)
        elif c[i] < lowest:
            sig[i] = -1; conf[i] = min((lowest - c[i]) / rng * 10, 1.0)
    return sig, conf


def _bb_reversion_signals(df: pd.DataFrame, period=20, n_std=2.0):
    c = df["close"].to_numpy(np.float32)
    n = len(c)
    sig = np.zeros(n, np.int8)
    conf = np.zeros(n, np.float32)
    for i in range(period, n):
        w = c[i-period:i]
        mu = w.mean(); std = w.std() + 1e-9
        z = (c[i] - mu) / std
        if z < -n_std:
            sig[i] = 1;  conf[i] = min((-z - n_std) / 2, 1.0)
        elif z > n_std:
            sig[i] = -1; conf[i] = min((z - n_std) / 2, 1.0)
    return sig, conf


def _atr_channel_signals(df: pd.DataFrame, period=14, mult=2.0):
    h = df["high"].to_numpy(np.float32)
    l = df["low"].to_numpy(np.float32)
    c = df["close"].to_numpy(np.float32)
    n = len(c)
    sig = np.zeros(n, np.int8)
    conf = np.zeros(n, np.float32)
    atr = np.zeros(n, np.float32)
    atr[0] = h[0] - l[0]
    for i in range(1, n):
        tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        atr[i] = atr[i-1] * (1 - 1/period) + tr * (1/period)
    for i in range(period, n - 1):
        mid = c[i-period:i].mean()
        band = mult * atr[i]
        if c[i] > mid + band:
            sig[i] = 1;  conf[i] = min((c[i] - mid - band) / band, 1.0)
        elif c[i] < mid - band:
            sig[i] = -1; conf[i] = min((mid - band - c[i]) / band, 1.0)
    return sig, conf


SIGNAL_FNS = {
    "momentum_sma":      _sma_signals,
    "donchian_breakout": _donchian_signals,
    "bb_reversion":      _bb_reversion_signals,
    "atr_channel":       _atr_channel_signals,
}


# ── Context features (lightweight, no hmmlearn dependency) ────────────────────

def _compute_context(df: pd.DataFrame) -> pd.DataFrame:
    """Returns per-bar context features aligned with df index."""
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    n = len(c)

    # ATR normalized
    atr = np.zeros(n)
    atr[0] = h[0] - l[0]
    for i in range(1, n):
        tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        atr[i] = atr[i-1] * 0.93 + tr * 0.07
    atr_norm = atr / (c + 1e-9)

    # Rolling 20-bar return volatility
    rets = np.zeros(n)
    rets[1:] = np.diff(np.log(c + 1e-9))
    vol = np.zeros(n)
    for i in range(20, n):
        vol[i] = rets[i-20:i].std()

    # Vol percentile (rolling 252-bar rank of current vol)
    vol_pct = np.full(n, 0.5)
    for i in range(252, n):
        w = vol[i-252:i]
        vol_pct[i] = (w < vol[i]).sum() / 252

    # Simple regime: 0=trending (low vol, positive trend) 1=ranging (high vol)
    trend = np.zeros(n)
    for i in range(50, n):
        slope = (c[i] - c[i-50]) / (c[i-50] + 1e-9)
        trend[i] = slope
    regime = np.where(vol_pct > 0.6, 1, 0)

    # Hurst approximation: variance ratio (fast proxy)
    hurst = np.full(n, 0.5)
    for i in range(40, n):
        w = c[i-40:i]
        r1 = np.var(np.diff(w))
        r2 = np.var(w[::2][1:] - w[::2][:-1]) / 2 if len(w[::2]) > 2 else 0.5
        h_val = 0.5 * (1 + np.log(r1 / (r2 + 1e-12) + 1e-12) / np.log(2)) if r2 > 0 else 0.5
        hurst[i] = float(np.clip(h_val, 0.1, 0.9))

    ctx = pd.DataFrame({
        "regime":   regime.astype(np.int8),
        "hurst":    hurst.astype(np.float32),
        "atr_norm": atr_norm.astype(np.float32),
        "vol_pct":  vol_pct.astype(np.float32),
    }, index=df.index)
    return ctx


# ── Trade ID: deterministic hash so re-runs don't duplicate rows ──────────────

def _trade_id(symbol: str, strategy: str, w_start: str, params: SweepParams,
               entry_bar: int) -> str:
    key = f"{symbol}|{strategy}|{w_start}|{params.stop_atr_mult}|{params.tp_rr}|{entry_bar}"
    return hashlib.md5(key.encode()).hexdigest()


# ── Per-window worker ─────────────────────────────────────────────────────────

def _process_window(
    symbol: str,
    strategy: str,
    window_df: pd.DataFrame,
    ctx: pd.DataFrame,
    params: SweepParams,
    w_start: str,
    w_end: str,
) -> List[dict]:
    """Run one backtest on one window, return list of trade dicts."""
    if len(window_df) < MIN_BARS:
        return []

    sig_fn = SIGNAL_FNS[strategy]
    try:
        signals, confidence = sig_fn(window_df)
    except Exception:
        return []

    try:
        engine = FastBacktestEngine.from_signals(window_df, signals, confidence)
        detail = engine.run_detailed(params)
    except Exception:
        return []

    if detail["trade_count"] == 0:
        return []

    entry_idx = detail["entry_idx"]
    exit_idx  = detail["exit_idx"]
    pnl       = detail["pnl"]
    pnl_r     = detail["pnl_r"]
    direction = detail["direction"]
    is_stop   = detail["is_stop"]

    ctx_arr = ctx.values  # regime, hurst, atr_norm, vol_pct
    now = datetime.utcnow()

    records = []
    for k in range(len(entry_idx)):
        ei = int(entry_idx[k])
        xi = int(exit_idx[k])
        if ei >= len(ctx_arr):
            continue

        regime_val, hurst_val, atr_norm_val, vol_pct_val = ctx_arr[ei]
        bar_ts = window_df.index[ei]

        records.append({
            "trade_id":      _trade_id(symbol, strategy, w_start, params, ei),
            "symbol":        symbol,
            "strategy":      strategy,
            "window_start":  w_start,
            "window_end":    w_end,
            "stop_atr_mult": params.stop_atr_mult,
            "tp_rr":         params.tp_rr,
            "atr_period":    params.atr_period,
            "entry_bar":     ei,
            "exit_bar":      xi,
            "direction":     int(direction[k]),
            "pnl":           float(pnl[k]),
            "pnl_r":         float(pnl_r[k]),
            "is_stop":       int(is_stop[k]),
            "is_profitable": 1 if pnl[k] > 0 else 0,
            "regime":        int(regime_val),
            "hurst":         float(hurst_val),
            "atr_norm":      float(atr_norm_val),
            "vol_pct":       float(vol_pct_val),
            "month":         bar_ts.month if hasattr(bar_ts, "month") else 0,
            "day_of_week":   bar_ts.dayofweek if hasattr(bar_ts, "dayofweek") else 0,
            "harvested_at":  now,
        })
    return records


# ── Main harvest loop ─────────────────────────────────────────────────────────

def _generate_windows(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Return (start_idx, end_idx) pairs for walk-forward windows."""
    n = len(df)
    windows = []
    i = 0
    while i + WINDOW_DAYS <= n:
        windows.append((i, i + WINDOW_DAYS))
        i += STEP_DAYS
    return windows


def _db_write(records: List[dict]) -> int:
    """Open a short-lived connection, write records, close immediately.
    Releasing the lock after each symbol lets the retrain loop grab a read window."""
    if not records:
        return 0
    batch = pd.DataFrame(records)
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("INSERT OR IGNORE INTO trades SELECT * FROM batch")
        return len(records)
    except Exception as e:
        log.debug(f"Insert error: {e}")
        return 0
    finally:
        con.close()


def harvest_symbol(symbol: str) -> int:
    """Harvest all windows for one symbol. Opens/closes DB per symbol."""
    df = load_ohlcv(symbol)
    if df is None or len(df) < WINDOW_DAYS + 10:
        log.warning(f"  {symbol}: no data or too short, skip")
        return 0

    ctx = _compute_context(df)
    windows = _generate_windows(df)
    all_records: List[dict] = []

    for (si, ei) in windows:
        window_df = df.iloc[si:ei].copy()
        window_ctx = ctx.iloc[si:ei].reset_index(drop=True)
        w_start = str(df.index[si])[:10]
        w_end   = str(df.index[ei - 1])[:10]

        for strategy in STRATEGIES:
            for params in PARAM_GRID:
                records = _process_window(
                    symbol, strategy, window_df, window_ctx,
                    params, w_start, w_end
                )
                all_records.extend(records)

    # Single open-write-close per symbol — lock held for milliseconds
    return _db_write(all_records)


def run_harvest(symbols: List[str], max_passes: int = 0) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Init schema once (short-lived connection)
    con = duckdb.connect(str(DB_PATH))
    init_db(con)
    con.close()

    pass_num = 0
    while True:
        pass_num += 1
        t0 = time.time()
        log.info(f"=== PASS {pass_num} START — {len(symbols)} symbols ===")

        total_trades = 0
        for i, sym in enumerate(symbols):
            sym_t0 = time.time()
            n = harvest_symbol(sym)
            elapsed = time.time() - sym_t0
            total_trades += n
            log.info(f"  [{i+1}/{len(symbols)}] {sym:8s}  {n:6,} trades  {elapsed:.1f}s")

        elapsed_pass = time.time() - t0

        # Summary stats (brief read connection)
        con = duckdb.connect(str(DB_PATH))
        con.execute("""
            INSERT OR REPLACE INTO harvest_progress VALUES
            ('last_pass_num',    ?),
            ('last_pass_trades', ?),
            ('last_pass_ts',     ?)
        """, [str(pass_num), str(total_trades), str(datetime.utcnow())])
        row = con.execute("SELECT COUNT(*), SUM(pnl), AVG(CAST(is_profitable AS FLOAT)) FROM trades").fetchone()
        con.close()

        total_db, total_pnl, win_rate = row
        log.info(
            f"=== PASS {pass_num} DONE  {total_trades:,} new trades  {elapsed_pass/60:.1f}m  "
            f"| DB total: {total_db:,} trades  WR={win_rate*100:.1f}%  PnL=${total_pnl:,.0f} ==="
        )

        if max_passes and pass_num >= max_passes:
            log.info("Max passes reached, exiting.")
            break

        time.sleep(1)


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=0,
                        help="Number of full passes (0=infinite)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Subset of symbols (default: full universe)")
    args = parser.parse_args()

    symbols = args.symbols or UNIVERSE
    log.info(f"Harvester starting — {len(symbols)} symbols, {len(STRATEGIES)} strategies, "
             f"{len(PARAM_GRID)} param combos, window={WINDOW_DAYS}d step={STEP_DAYS}d")
    run_harvest(symbols, max_passes=args.passes)
