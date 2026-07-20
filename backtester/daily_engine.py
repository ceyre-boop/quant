"""Daily-bar backtest engine + scanner for multi-day-hold strategies.

The intraday engine (engine.py) trades entry->exit within one session. Families
C (RSI mean reversion), D (index dip), E (gap follow/fade) hold across days on
daily bars. This module provides a signal->trade backtest on daily OHLCV plus a
vectorised, FWER-corrected scanner over signal families.

Bias controls carried over:
  - Entry at NEXT bar's open after a signal (no same-bar look-ahead).
  - Stop checked on each held bar's low(long)/high(short); gap-through fills at
    that bar's open, else at the stop level. Never at trigger on a gap.
  - Exit at hold_days later close OR stop, whichever first.
  - Per-trade slippage + cost charged.
  - Real 12-month holdout split enforced by the caller (date filter).
  - Permutation test shuffles signal dates; Bonferroni + Holm across all configs.
"""
from __future__ import annotations

import itertools
import multiprocessing as _mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import holdout_guard as _hg

REPO = Path(__file__).resolve().parents[1]
UNIVERSE = REPO / "data/cache/daily_universe"
_CTX = _mp.get_context("fork" if os.name == "posix" else "spawn")
N_PERM = 100
SEED = 42
COST = 0.0005  # 5 bps per side, liquid ETFs/large caps


def load_daily(ticker: str) -> pd.DataFrame | None:
    p = UNIVERSE / f"{ticker}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    rs = np.divide(ru, rd, out=np.full_like(ru, np.inf), where=rd != 0)
    return 100 - 100 / (1 + rs)


def _signal_dates(df: pd.DataFrame, fam: str, cfg: dict) -> np.ndarray:
    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    idx = np.arange(len(df))
    if fam == "rsi":
        r = rsi(c, cfg.get("rsi_n", 14))
        if cfg["direction"] == "long":
            sig = (r[:-1] >= cfg["thr"]) & (r[1:] < cfg["thr"])  # crossing down into oversold
        else:
            sig = (r[:-1] <= cfg["thr"]) & (r[1:] > cfg["thr"])  # crossing up into overbought
        return idx[1:][sig]
    if fam == "gap":
        prev_close = c[:-1]
        gap = o[1:] / prev_close - 1
        if cfg["gap_dir"] == "up":
            hit = gap >= cfg["thr"]
        else:
            hit = gap <= -cfg["thr"]
        return idx[1:][hit]
    if fam == "dip":  # N-day drawdown reversion (index-dip family / HYP-095 spirit)
        ret = c[cfg["look"]:] / c[:-cfg["look"]] - 1
        hit = ret <= -cfg["thr"]
        return idx[cfg["look"]:][hit]
    if fam == "breakout":  # close breaks N-day high (long) / low (short)
        n = cfg["look"]
        roll_hi = pd.Series(c).rolling(n).max().to_numpy()
        roll_lo = pd.Series(c).rolling(n).min().to_numpy()
        if cfg["direction"] == "long":
            hit = c[n:] >= roll_hi[:-n] if False else (c >= roll_hi)
        else:
            hit = (c <= roll_lo)
        hit = hit.copy()
        hit[:n] = False
        return idx[hit]
    return np.array([], dtype=int)


def backtest_daily(df: pd.DataFrame, fam: str, cfg: dict,
                   date_lo="0000", date_hi="9999", check_holdout: bool = True) -> dict:
    """Return per-trade net returns + aggregate metrics on [date_lo, date_hi).

    date_hi is exclusive, so the guard is asked about the last tradeable date.
    A sanctioned verdict runner (prereg verified) sets ALLOW_HOLDOUT_ACCESS or
    calls holdout_guard.sanction(); check_holdout=False is for unit tests on
    synthetic frames only.
    """
    if check_holdout:
        _hg.validate_date_range(date_lo, str(date_hi), context="daily_engine.backtest_daily",
                                dataset="equities_daily")
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    dates = df["date"].to_numpy()
    sig_idx = _signal_dates(df, fam, cfg)
    direction = cfg.get("trade_dir", cfg.get("direction", "long"))
    stop_pct = cfg["stop_pct"]
    hold = cfg.get("hold_days", 3)

    rets, tdates = [], []
    for si in sig_idx:
        ei = si + 1                      # enter next bar open (no look-ahead)
        if ei >= len(df):
            continue
        if not (date_lo <= dates[ei] < date_hi):
            continue
        entry = o[ei]
        if entry <= 0:
            continue
        exit_i = min(ei + hold, len(df) - 1)
        fill = None
        if direction == "long":
            trig = entry * (1 - stop_pct)
            for j in range(ei, exit_i + 1):
                if lo[j] <= trig:
                    fill = min(o[j], trig) if o[j] <= trig else trig
                    break
            exitp = fill if fill is not None else c[exit_i]
            gross = exitp / entry - 1
        else:
            trig = entry * (1 + stop_pct)
            for j in range(ei, exit_i + 1):
                if h[j] >= trig:
                    fill = max(o[j], trig) if o[j] >= trig else trig
                    break
            exitp = fill if fill is not None else c[exit_i]
            gross = entry / exitp - 1 if exitp > 0 else -1.0
        rets.append(gross - 2 * COST)
        tdates.append(dates[ei])
    return {"rets": rets, "dates": tdates}


def _metrics(rets, years, sizing=0.1):
    """Event-strategy metrics, calendar-aware.

    annual   : mean per-trade net * trades_per_year * sizing (arithmetic — avoids
               the overlapping-position compounding artifact of pooled events).
    sharpe   : per-trade Sharpe annualised by sqrt(trades_per_year).
    max_dd   : running drawdown of the sized, time-ordered trade series.
    """
    if not rets or years <= 0:
        return dict(annual=0.0, sharpe=0.0, max_dd=0.0, n=0, win=0.0,
                    per_year=0.0)
    r = np.array(rets)
    tpy = len(r) / years
    mean, sd = float(r.mean()), float(r.std())
    annual = mean * tpy * sizing
    sharpe = (mean / sd * np.sqrt(tpy)) if sd > 0 else 0.0
    eq, peak, mdd = 1.0, 1.0, 0.0
    for x in r * sizing:
        eq = max(eq * (1 + max(x, -1.0)), 0.0)
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak if peak > 0 else 1.0)
    return dict(annual=round(annual, 4), sharpe=round(sharpe, 3),
                max_dd=round(mdd, 4), n=len(r), win=round(float((r > 0).mean()), 4),
                per_year=round(tpy, 1))
