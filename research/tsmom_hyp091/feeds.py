"""Read-only data feeds for HYP-091 (TICK-027).

- Prices: yfinance daily Close, SAME convention as v015's
  forex_backtester._download_price (auto_adjust=True), pulled from WARMUP_START so
  the 252d signal exists across the full eval window.
- Rate differentials: FRED policy-rate differential per pair via
  sovereign.forex.data_fetcher.get_pair_differentials — drives the correct
  financing model. Fails LOUD if FRED is unavailable (no silent synthetic).
- v015 carry: monthly return series = sum(risk_adjusted_pnl_pct) by EXIT-month
  from the git-tracked decade CSV (the field v015's own Sharpe uses).
- Swap calibration: the 2026 OANDA snapshot from TICK-024's swap_calibration.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.tsmom_hyp091._lib import (
    EVAL_END, PAIR_COUNTRIES, PAIRS, ROOT, SWAP_CALIB_PATH, V015_DECADE_CSV,
    WARMUP_START,
)


def load_prices() -> dict[str, pd.Series]:
    """Daily Close per pair (yfinance, auto_adjust), WARMUP_START..EVAL_END+1.
    Mirrors forex_backtester._download_price so the series matches v015's."""
    import yfinance as yf
    end = (pd.Timestamp(EVAL_END) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    out: dict[str, pd.Series] = {}
    degraded = []
    for pair in PAIRS:
        df = yf.download(pair, start=WARMUP_START, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            degraded.append(pair)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        out[pair] = close
    if degraded:
        raise SystemExit(f"FATAL: yfinance returned no data for {degraded} — refusing to "
                         f"proceed on partial price data (no silent mocking).")
    return out


def load_rate_differentials() -> dict[str, pd.Series]:
    """Daily FRED rate_differential (percentage points, base_rate - quote_rate) per pair.
    Fails loud if the series is synthetic/constant (FRED unavailable)."""
    from sovereign.forex.data_fetcher import ForexDataFetcher
    fetcher = ForexDataFetcher()
    out: dict[str, pd.Series] = {}
    for pair in PAIRS:
        base, quote = PAIR_COUNTRIES[pair]
        df = fetcher.get_pair_differentials(base, quote, start=WARMUP_START)
        series = df["rate_differential"].copy()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        # Guard: a real policy-rate differential MUST vary over 2013-2024 (hiking cycles).
        if series.dropna().nunique() < 5:
            raise SystemExit(
                f"FATAL: rate_differential for {pair} is near-constant "
                f"(nunique={series.dropna().nunique()}) — FRED likely unavailable and the "
                f"fetcher fell back to synthetic rates. Correct financing is impossible; "
                f"set the FRED key in ~/quant/.env and re-run (no silent synthetic).")
        out[pair] = series
    return out


def load_v015_monthly() -> pd.Series:
    """v015 carry monthly return: sum(risk_adjusted_pnl_pct) by exit-month.
    Indexed by month-end Timestamp. This is the ACTUAL carry book (not a proxy)."""
    if not V015_DECADE_CSV.exists():
        raise SystemExit(f"FATAL: v015 decade CSV missing: {V015_DECADE_CSV}")
    df = pd.read_csv(V015_DECADE_CSV)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["risk_adjusted_pnl_pct"] = pd.to_numeric(df["risk_adjusted_pnl_pct"], errors="coerce")
    df = df.dropna(subset=["risk_adjusted_pnl_pct"])
    monthly = df.groupby(df["exit_date"].dt.to_period("M"))["risk_adjusted_pnl_pct"].sum()
    monthly.index = monthly.index.to_timestamp("M")  # month-END timestamps
    return monthly.astype(float)


def load_swap_calibration() -> dict:
    """2026 OANDA per-pair-per-side annual financing snapshot (TICK-024).
    Returns {pair: {'LONG': float, 'SHORT': float}} in fraction/yr."""
    if not SWAP_CALIB_PATH.exists():
        raise SystemExit(f"FATAL: TICK-024 swap calibration missing: {SWAP_CALIB_PATH}")
    raw = json.loads(SWAP_CALIB_PATH.read_text())
    out = {}
    for pair in PAIRS:
        entry = raw["pairs"][pair]
        out[pair] = {"LONG": float(entry["LONG"]["oanda_annual"]),
                     "SHORT": float(entry["SHORT"]["oanda_annual"])}
    return out
