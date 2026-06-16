"""VRP data loader — the ONLY impure module in the vrp package.

Pulls free yfinance series (no historical option chains exist in this system; see
strategy_simulator for the DATA_INSUFFICIENT boundary). It also READS the OUTPUT of the
live forex backtest (logs/forex_backtest_trades.json) for the recent-window carry
secondary — it imports NO forex/ict modules (NN#1; enforced by test_vrp_isolation.py).

yfinance is imported lazily inside functions so importing this module stays light and
network-free (the isolation/causality tests import it without hitting the network).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]   # vrp -> research -> sovereign -> repo root
logging.getLogger("yfinance").setLevel(logging.ERROR)

FOREX_TRADES = ROOT / "logs" / "forex_backtest_trades.json"


def _hist(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    if h is None or len(h) == 0:
        return pd.DataFrame()
    h.index = h.index.tz_localize(None)
    return h


def load_underlying(symbol: str) -> pd.DataFrame:
    """Daily OHLC for SPY/QQQ, corporate-action adjusted (auto_adjust)."""
    h = _hist(symbol)
    if len(h) < 500:
        raise SystemExit(f"FATAL: {symbol} daily history unavailable/too short ({len(h)}) — refusing thin data.")
    return h[["Open", "High", "Low", "Close"]]


def load_vol_index(ticker: str) -> pd.Series:
    """Implied-vol index close (^VIX, ^VXN, ...) in vol POINTS (e.g. 20.0 = 20%)."""
    h = _hist(ticker)
    if len(h) < 500:
        raise SystemExit(f"FATAL: {ticker} history unavailable/too short ({len(h)}) — refusing thin data.")
    return h["Close"].rename(ticker)


def load_carry_proxy() -> pd.Series:
    """DBV (Invesco G10 carry-factor ETF) daily returns, 2006->2023-03. The standard
    tradeable carry proxy — covers GFC/COVID/2022 (the v015 forex log is recent/OOS-only)."""
    h = _hist("DBV")
    if len(h) < 250:
        return pd.Series(dtype=float)
    return h["Close"].pct_change().dropna().rename("carry_dbv")


def load_forex_log_carry() -> pd.Series:
    """Recent-window secondary: daily summed pnl_pct from the live forex backtest OUTPUT.
    Reads a JSON artifact only — imports no forex module (NN#1)."""
    try:
        raw = json.loads(FOREX_TRADES.read_text())
    except Exception:
        return pd.Series(dtype=float)
    by_date: dict = {}
    for lst in raw.values():
        if not isinstance(lst, list):
            continue
        for t in lst:
            stamp = str(t.get("entry_date", t.get("entry", "")))[:10]
            d = pd.Timestamp(stamp) if stamp else pd.NaT
            if pd.isna(d):
                continue
            by_date[d] = by_date.get(d, 0.0) + float(t.get("pnl_pct", 0.0))
    if not by_date:
        return pd.Series(dtype=float)
    return pd.Series(by_date).sort_index().rename("carry_v015")


def load_overnight_qqq(qqq: pd.DataFrame | None = None) -> pd.Series:
    """Re-derive the overnight-QQQ return (open/prior-close) from QQQ OHLC. Mirrors the
    live edge's definition WITHOUT importing it (NN#1)."""
    if qqq is None:
        qqq = load_underlying("QQQ")
    return (qqq["Open"] / qqq["Close"].shift(1) - 1.0).dropna().rename("overnight_qqq")
