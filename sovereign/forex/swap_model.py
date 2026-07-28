"""TICK-024 corrected swap/financing model.

SWAP_RATES_ANNUAL (the static table in forex_backtester.py) understates realized
OANDA financing by ~9x on all 4 live pairs (median across 24 real trades, see
research/TICK-024_cost_measurement.md) and has a sign flip on EURUSD SHORT
(model charges a cost; OANDA actually pays a credit).

This module promotes the rate-differential-derived model already built and used
by research/tsmom_hyp091/financing.py (operator decision 2026-07-12) from
research-only to a shared module the live backtester can import. Same math:

    financing_LONG(t)  = oanda_LONG_now  + (diff(t) - diff_now)
    financing_SHORT(t) = oanda_SHORT_now - (diff(t) - diff_now)

anchored to the OANDA snapshot in data/research/swap_calibration.json and varied
across history by the CHANGE in the FRED policy-rate differential (base - quote,
percentage points -> fraction/yr) since the snapshot date — so 2015-2024 history
is not flattened to today's OANDA rate. Returns None (never a fabricated number)
if the differential series or calibration is unavailable for the pair; the
caller falls back to the broken static table, loudly.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SWAP_CALIB_PATH = ROOT / "data" / "research" / "swap_calibration.json"

# base/quote country codes for sovereign.forex.data_fetcher.get_pair_differentials.
# Mirrors research/tsmom_hyp091/_lib.py::PAIR_COUNTRIES.
PAIR_COUNTRIES = {
    "EURUSD=X": ("EU", "US"),
    "GBPUSD=X": ("UK", "US"),
    "USDJPY=X": ("US", "JP"),
    "AUDUSD=X": ("AU", "US"),
}


@lru_cache(maxsize=1)
def _load_calibration() -> dict:
    if not SWAP_CALIB_PATH.exists():
        return {}
    raw = json.loads(SWAP_CALIB_PATH.read_text())
    out = {}
    for pair, entry in raw.get("pairs", {}).items():
        out[pair] = {
            "LONG": float(entry["LONG"]["oanda_annual"]),
            "SHORT": float(entry["SHORT"]["oanda_annual"]),
        }
    return out


@lru_cache(maxsize=len(PAIR_COUNTRIES))
def _load_differential_series(pair: str) -> Optional[pd.Series]:
    if pair not in PAIR_COUNTRIES:
        return None
    from sovereign.forex.data_fetcher import ForexDataFetcher
    base, quote = PAIR_COUNTRIES[pair]
    fetcher = ForexDataFetcher()
    try:
        df = fetcher.get_pair_differentials(base, quote, start="2013-06-01")
    except Exception:
        return None
    series = df["rate_differential"].astype(float) / 100.0   # pct points -> fraction/yr
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.sort_index()
    if series.dropna().nunique() < 5:
        return None   # near-constant => FRED unavailable / synthetic fallback, don't trust it
    return series


def ratediff_financing_rate(pair: str, side: str, entry_date) -> Optional[float]:
    """Annual financing rate (fraction/yr) for `pair`/`side` on `entry_date`,
    anchored to the OANDA snapshot in data/research/swap_calibration.json and
    varied by the FRED rate-differential change since the snapshot date.
    Returns None if the pair has no differential series or no calibration
    entry (caller falls back to the broken static table with a loud warning,
    never silently)."""
    calib = _load_calibration()
    if pair not in calib:
        return None
    series = _load_differential_series(pair)
    if series is None or series.empty:
        return None

    diff_now = float(series.dropna().iloc[-1])
    ts = pd.Timestamp(entry_date).tz_localize(None) if pd.Timestamp(entry_date).tzinfo else pd.Timestamp(entry_date)
    asof = series.loc[:ts]
    diff_t = float(asof.dropna().iloc[-1]) if not asof.dropna().empty else diff_now
    delta = diff_t - diff_now

    long_now = calib[pair]["LONG"]
    short_now = calib[pair]["SHORT"]
    if side == "LONG":
        return long_now + delta
    if side == "SHORT":
        return short_now - delta
    return None
