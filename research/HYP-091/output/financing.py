"""Correct rate-differential-derived financing for HYP-091 (operator decision).

TICK-024 proves SWAP_RATES_ANNUAL is ~10x too small on all 4 pairs with one sign
flip, and it is Colin-gated (feeds the 0.6886 reconcile anchor). So the PRIMARY
financing here does NOT use that table. Instead: an anchored differential-tracking
model that reproduces the 2026 OANDA snapshot AND the trade-227 anchor at t=now,
and varies across 2015-2024 by the CHANGE in the FRED policy-rate differential —
the economically correct driver of financing.

    financing_LONG(t)  = oanda_LONG_now  + (diff(t) - diff_now)
    financing_SHORT(t) = oanda_SHORT_now - (diff(t) - diff_now)

diff is the FRED rate_differential (base_rate - quote_rate) in FRACTION/yr
(get_pair_differentials returns percentage points -> /100 here). diff_now = the
differential at the calibration/snapshot date (last available). oanda_*_now =
data/research/swap_calibration.json (TICK-024). Sign convention matches v015's
_apply_costs: positive = EARN carry, negative = PAY.
"""
from __future__ import annotations

import pandas as pd

# Robustness leg (i): the BROKEN table, read-only, for the apples-to-apples
# cross-check against the v015 CSV (which was costed with it). NOT the primary.
from sovereign.forex.forex_backtester import SWAP_RATES_ANNUAL


def ratediff_financing(pair: str, diff_series: pd.Series, calib: dict) -> pd.DataFrame:
    """Daily LONG/SHORT annual financing rates (fraction/yr), primary model."""
    diff_frac = diff_series.astype(float) / 100.0          # pct points -> fraction/yr
    diff_now = float(diff_frac.dropna().iloc[-1])          # snapshot-date differential
    d = diff_frac - diff_now
    long_now = calib[pair]["LONG"]
    short_now = calib[pair]["SHORT"]
    return pd.DataFrame({
        "LONG": long_now + d,
        "SHORT": short_now - d,
    })


def broken_financing(pair: str, index: pd.Index) -> pd.DataFrame:
    """Robustness leg (i): constant SWAP_RATES_ANNUAL (the broken model v015 used)."""
    tbl = SWAP_RATES_ANNUAL.get(pair, {"LONG": -0.0010, "SHORT": -0.0010})
    return pd.DataFrame({"LONG": tbl["LONG"], "SHORT": tbl["SHORT"]}, index=index)


def zero_financing(index: pd.Index) -> pd.DataFrame:
    """Robustness leg (ii): price-only."""
    return pd.DataFrame({"LONG": 0.0, "SHORT": 0.0}, index=index)


def build_financing(mode: str, pair: str, diff_series: pd.Series,
                    calib: dict, price_index: pd.Index) -> pd.DataFrame:
    """Return a daily financing frame (LONG/SHORT annual rates) reindexed to price
    dates and forward-filled. mode in {'ratediff','broken','none'}."""
    if mode == "ratediff":
        fin = ratediff_financing(pair, diff_series, calib)
    elif mode == "broken":
        fin = broken_financing(pair, diff_series.index)
    elif mode == "none":
        fin = zero_financing(diff_series.index)
    else:
        raise ValueError(f"unknown financing mode {mode!r}")
    fin.index = pd.to_datetime(fin.index)
    return fin.reindex(price_index.union(fin.index)).ffill().reindex(price_index)
