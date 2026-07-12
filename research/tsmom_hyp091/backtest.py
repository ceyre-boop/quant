"""TSMOM backtest for HYP-091 — monthly rebalance, inverse-vol sizing, correct
rate-differential financing. Produces a monthly portfolio return series.

No look-ahead: the weight for month m+1 is set from the signal and ex-ante vol
observed at the last close of month m (data <= t only). Financing accrues on
calendar days at the rate-differential-derived annual rate for the held side.

The per-month/per-pair decomposition (absw, spot, finL/finS per unit |w|, spread
unit) is exposed so the gauntlet can recompute Sharpe under permuted momentum
signs financing-consistently — actual signs must reproduce backtest() exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.tsmom_hyp091 import feeds, financing
from research.tsmom_hyp091._lib import (
    EVAL_END, EVAL_START, LEV_CAP, LOOKBACK_DAYS, PAIRS, TARGET_VOL_ANN,
    TRADING_YEAR, VOL_COM,
)
from sovereign.forex.forex_backtester import SLIPPAGE_PER_SIDE, SPREAD_COST, _DEFAULT_SPREAD


def _month_end_dates(index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    idx = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    s = pd.Series(idx, index=idx)
    return list(s.groupby([idx.year, idx.month]).max().values)


def _signal_and_absw(close: pd.Series, vol_ann: pd.Series, d: pd.Timestamp) -> tuple[float, float]:
    """Momentum sign and inverse-vol MAGNITUDE at rebalance close d (info <= d)."""
    pos = close.index.get_loc(d)
    if pos < LOOKBACK_DAYS:
        return 0.0, 0.0
    sign = float(np.sign(close.iloc[pos] / close.iloc[pos - LOOKBACK_DAYS] - 1.0))
    v = float(vol_ann.loc[d])
    absw = 0.0 if (not np.isfinite(v) or v <= 1e-9) else min(TARGET_VOL_ANN / v, LEV_CAP)
    return sign, absw


def decompose(mode: str = "ratediff", prices: dict | None = None,
              diffs: dict | None = None, calib: dict | None = None) -> dict:
    """Per-pair, per-realized-month sign-independent components."""
    prices = prices if prices is not None else feeds.load_prices()
    diffs = diffs if diffs is not None else feeds.load_rate_differentials()
    calib = calib if calib is not None else feeds.load_swap_calibration()

    calendar = prices[PAIRS[0]].index
    for p in PAIRS[1:]:
        calendar = calendar.union(prices[p].index)
    reb = _month_end_dates(calendar, EVAL_START, EVAL_END)

    logret, vol_ann, fin = {}, {}, {}
    for p in PAIRS:
        c = prices[p]
        logret[p] = np.log(c / c.shift(1))
        vol_ann[p] = logret[p].ewm(com=VOL_COM).std() * np.sqrt(TRADING_YEAR)
        fin[p] = financing.build_financing(mode, p, diffs[p], calib, c.index)
    spread_px = {p: (SPREAD_COST.get(p, _DEFAULT_SPREAD) + 2 * SLIPPAGE_PER_SIDE) for p in PAIRS}

    months = [pd.Timestamp(reb[k + 1]) for k in range(len(reb) - 1)]
    comp = {p: {"sign": [], "absw": [], "spot": [], "finL": [], "finS": [], "spread_unit": []}
            for p in PAIRS}
    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]
        for p in PAIRS:
            c = prices[p]
            if d0 not in c.index or d1 not in c.index:
                for key, val in (("sign", 0.0), ("absw", 0.0), ("spot", 0.0),
                                 ("finL", 0.0), ("finS", 0.0), ("spread_unit", 0.0)):
                    comp[p][key].append(val)
                continue
            sign, absw = _signal_and_absw(c, vol_ann[p], d0)
            spot = float(c.loc[d1] / c.loc[d0] - 1.0)
            seg = fin[p].loc[(fin[p].index > d0) & (fin[p].index <= d1)]
            if len(seg):
                gaps = seg.index.to_series().diff().dt.days.fillna(1.0).clip(lower=1.0).values
                finL = float(np.sum(seg["LONG"].values * gaps)) / 365.0
                finS = float(np.sum(seg["SHORT"].values * gaps)) / 365.0
            else:
                finL = finS = 0.0
            comp[p]["sign"].append(sign)
            comp[p]["absw"].append(absw)
            comp[p]["spot"].append(spot)
            comp[p]["finL"].append(finL)
            comp[p]["finS"].append(finS)
            comp[p]["spread_unit"].append(spread_px[p] / float(c.loc[d0]))
    for p in PAIRS:
        comp[p] = {k: np.asarray(v, dtype=float) for k, v in comp[p].items()}
    return {"mode": mode, "months": months, "comp": comp}


def _pair_returns(cp: dict, signs: np.ndarray) -> np.ndarray:
    """Realized monthly returns for one pair given a sign vector (financing-consistent)."""
    absw = cp["absw"]
    w = signs * absw
    prev = np.concatenate([[0.0], w[:-1]])
    turnover = np.abs(w - prev)
    fin = absw * np.where(signs > 0, cp["finL"], cp["finS"])
    return w * cp["spot"] + fin - turnover * cp["spread_unit"]


def portfolio_returns(dec: dict, signs_by_pair: dict | None = None) -> pd.Series:
    """Equal-weight portfolio monthly return. Default signs = actual momentum signs."""
    per = []
    for p in PAIRS:
        cp = dec["comp"][p]
        signs = cp["sign"] if signs_by_pair is None else signs_by_pair[p]
        per.append(_pair_returns(cp, signs))
    port = np.nanmean(np.vstack(per), axis=0)
    return pd.Series(port, index=pd.DatetimeIndex(dec["months"]))


def backtest(mode: str = "ratediff", prices: dict | None = None,
             diffs: dict | None = None, calib: dict | None = None) -> dict:
    """Run the monthly TSMOM backtest. mode in {'ratediff','broken','none'}."""
    dec = decompose(mode, prices, diffs, calib)
    port = portfolio_returns(dec).dropna()
    per_pair = {p: pd.Series(_pair_returns(dec["comp"][p], dec["comp"][p]["sign"]),
                             index=pd.DatetimeIndex(dec["months"])) for p in PAIRS}
    return {
        "mode": mode,
        "dec": dec,
        "monthly": port,
        "per_pair": pd.DataFrame(per_pair),
        "n_months": int(len(port)),
        "first_month": str(port.index.min().date()),
        "last_month": str(port.index.max().date()),
    }
