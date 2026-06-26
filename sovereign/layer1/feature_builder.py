"""sovereign/layer1/feature_builder.py — assemble the Layer-1 feature panel (HYP-064, Phase 2).

Builds a MultiIndex (date, pair) panel of the declared features (docs/layer1/feature_windows.json)
from the historical loaders in data_loader.py. All feature computation uses data on or before t0
(no forward-looking windows). Forward-fill is explicit and documented per cadence. A feature that
cannot be sourced is EXCLUDED and recorded in the LoadReport — never zero-filled.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign.layer1 import data_loader as dl
from sovereign.layer1.data_loader import LoadReport

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs" / "layer1" / "feature_windows.json"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
TRADING_YEAR = 252


# ─── small windowed helpers (all use data up to t0 only) ────────────────────────

def _mom(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n)

def _z(s: pd.Series, n: int) -> pd.Series:
    return (s - s.rolling(n).mean()) / s.rolling(n).std()

def _vs_sma(s: pd.Series, n: int) -> pd.Series:
    return s / s.rolling(n).mean() - 1.0

def _pos_range(s: pd.Series, n: int) -> pd.Series:
    lo, hi = s.rolling(n).min(), s.rolling(n).max()
    return (s - lo) / (hi - lo)

def _ffill_to(s: pd.Series | None, idx: pd.DatetimeIndex, limit: int = 7) -> pd.Series | None:
    """Reindex a daily/irregular series onto the trading-day index, ffilling gaps up to `limit`
    days (bridges weekends/holidays). Past-only: ffill never uses future values."""
    if s is None:
        return None
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(idx)).sort_index().ffill(limit=limit).reindex(idx)

def _cot_to_daily(weekly: pd.Series, idx: pd.DatetimeIndex, lag_bdays: int = 3) -> pd.Series:
    """Weekly COT → daily with a publication lag: a report dated d only becomes usable d+lag
    business days later (CFTC publishes Friday for Tuesday data). Then forward-fill. No peeking."""
    avail = weekly.copy()
    avail.index = avail.index + pd.tseries.offsets.BDay(lag_bdays)
    return avail.reindex(avail.index.union(idx)).sort_index().ffill().reindex(idx)


# ─── global (cross-pair) features ───────────────────────────────────────────────

def build_global_features(report: LoadReport) -> dict[str, pd.Series]:
    """Compute every global feature on its native index. Returns {name: series}; a feature whose
    source failed is simply absent (and recorded in the report by the loader)."""
    g: dict[str, pd.Series] = {}

    dxy = dl.get_yf_close("dxy", dl.YF_TICKERS["dxy"], report)
    vix = dl.get_yf_close("vix", dl.YF_TICKERS["vix"], report)
    vix9d = dl.get_yf_close("vix9d", dl.YF_TICKERS["vix9d"], report)
    vix3m = dl.get_yf_close("vix3m", dl.YF_TICKERS["vix3m"], report)
    spy = dl.get_yf_close("spy", dl.YF_TICKERS["spy"], report)
    oil = dl.get_yf_close("oil", dl.YF_TICKERS["oil"], report)
    gold = dl.get_yf_close("gold", dl.YF_TICKERS["gold"], report)
    crb = dl.get_yf_close("crb", dl.YF_TICKERS["crb"], report)
    hyg = dl.get_yf_close("hyg", dl.YF_TICKERS["hyg"], report)
    ief = dl.get_yf_close("ief", dl.YF_TICKERS["ief"], report)
    tlt = dl.get_yf_close("tlt", dl.YF_TICKERS["tlt"], report)
    nq = dl.get_yf_close("nq", "NQ=F", report)
    es = dl.get_yf_close("es", "ES=F", report)

    dgs2 = dl.get_fred_series("us_2y", report)
    dgs5 = dl.get_fred_series("us_5y", report)
    dgs10 = dl.get_fred_series("us_10y", report)
    t10y2y = dl.get_fred_series("us_2s10s", report)
    breakeven = dl.get_fred_series("breakeven_10y", report)
    fedfunds = dl.get_fred_series("fed_funds", report)
    hy_oas = dl.get_fred_series("BAMLH0A0HYM2", report)  # FRED HY OAS

    if dxy is not None:
        g["dxy_level"] = dxy
        g["dxy_mom_1m"] = _mom(dxy, 21)
        g["dxy_mom_3m"] = _mom(dxy, 63)
        g["dxy_vs_200sma"] = _vs_sma(dxy, 200)
        g["dxy_pos_52w"] = _pos_range(dxy, TRADING_YEAR)
        if spy is not None:
            # smile proxy: rolling 63d corr(dxy_ret, spy_ret); negative ≈ risk-off USD-strength regime
            j = pd.concat([dxy.pct_change(), spy.pct_change()], axis=1).dropna()
            g["dxy_smile_regime"] = j.iloc[:, 0].rolling(63).corr(j.iloc[:, 1]).reindex(dxy.index)
    if vix is not None:
        g["vix_level"] = vix
        g["vix_z_1y"] = _z(vix, TRADING_YEAR)
        if vix9d is not None:
            g["vix_term_9d_1m"] = (vix9d / vix).reindex(vix.index)
        if vix3m is not None:
            g["vix_term_1m_3m"] = (vix / vix3m).reindex(vix.index)
    if spy is not None:
        g["spy_vs_200sma"] = _vs_sma(spy, 200)
        g["spy_mom_1m"] = _mom(spy, 21)
        g["spy_mom_3m"] = _mom(spy, 63)
    if t10y2y is not None:
        g["us_2s10s"] = t10y2y
    if dgs5 is not None and dgs2 is not None:
        g["us_2s5s"] = (dgs5 - dgs2)
    if dgs10 is not None:
        g["us_10y_mom_1m"] = dgs10.diff(21)
    if dgs2 is not None:
        g["us_2y_mom_1m"] = dgs2.diff(21)
    if breakeven is not None:
        g["us_breakeven_10y"] = breakeven
    if fedfunds is not None:
        g["fed_funds_level"] = fedfunds
    if oil is not None:
        g["oil_mom_1m"] = _mom(oil, 21)
        g["oil_mom_3m"] = _mom(oil, 63)
    if gold is not None:
        g["gold_mom_1m"] = _mom(gold, 21)
    if crb is not None:
        g["crb_mom_3m"] = _mom(crb, 63)
    if hy_oas is not None:
        g["credit_hy_oas"] = hy_oas
    if hyg is not None and ief is not None:
        g["risk_on_off_hyg_ief"] = _mom((hyg / ief).dropna(), 21)
    if spy is not None and tlt is not None:
        j = pd.concat([spy.pct_change(), tlt.pct_change()], axis=1).dropna()
        g["bond_equity_corr_63d"] = j.iloc[:, 0].rolling(63).corr(j.iloc[:, 1])
    if nq is not None and es is not None:
        # nqes regime proxy: NQ 5d momentum minus ES 5d momentum (risk leadership)
        g["nqes_regime_flag"] = (_mom(nq, 5) - _mom(es, 5)).dropna()

    return g


def calendar_features(idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    ev = dl.historical_event_dates()
    out = {
        "day_of_week": pd.Series(idx.dayofweek, index=idx, dtype=float),
        "month_of_year": pd.Series(idx.month, index=idx, dtype=float),
    }
    for key in ("fomc", "nfp", "ecb"):
        dates = np.array([d.value for d in ev[key]])
        days = []
        for t in idx:
            future = dates[dates >= t.value]
            days.append((future.min() - t.value) / 86_400_000_000_000 if len(future) else np.nan)
        out[f"days_to_{key}"] = pd.Series(days, index=idx, dtype=float)
    return out


# ─── pair-specific features ─────────────────────────────────────────────────────

def build_pair_features(pair: str, idx: pd.DatetimeIndex, report: LoadReport) -> dict[str, pd.Series]:
    p: dict[str, pd.Series] = {}

    diffs = dl.get_pair_differentials(pair, report)
    if diffs is not None:
        rd = diffs.get("rate_differential")
        rrd = diffs.get("real_rate_differential")
        if rd is not None:
            p["pair_2y_rate_diff"] = rd
            p["pair_rate_diff_mom_1m"] = rd.diff(21)
            p["pair_rate_diff_mom_3m"] = rd.diff(63)
            p["pair_rate_diff_mom_6m"] = rd.diff(126)
        if rrd is not None:
            p["pair_real_rate_diff"] = rrd

    px = dl.get_pair_prices(pair, report)
    if px is not None:
        ret = px.pct_change()
        p["pair_realized_vol_20d"] = ret.rolling(20).std() * np.sqrt(TRADING_YEAR)
        p["pair_realized_vol_60d"] = ret.rolling(60).std() * np.sqrt(TRADING_YEAR)
        p["pair_price_mom_20d"] = _mom(px, 20)
        p["pair_price_mom_60d"] = _mom(px, 60)

    cot = dl.get_cot_net(pair, report)
    if cot is not None and len(cot) > 5:
        net_daily = _cot_to_daily(cot, idx)
        p["pair_cot_net_spec"] = net_daily
        # 156-week z and 1y percentile computed on the weekly series, then lagged to daily.
        wk = cot.sort_index()
        z156 = (wk - wk.rolling(156, min_periods=52).mean()) / wk.rolling(156, min_periods=52).std()
        pct1y = wk.rolling(52, min_periods=26).apply(lambda w: (w.rank(pct=True).iloc[-1]), raw=False)
        p["pair_cot_z_156w"] = _cot_to_daily(z156.dropna(), idx)
        p["pair_cot_percentile_1y"] = _cot_to_daily(pct1y.dropna(), idx)
        ext = ((pct1y > 0.9) | (pct1y < 0.1)).astype(float)
        p["pair_cot_extreme_flag"] = _cot_to_daily(ext, idx)

    return p


# ─── panel assembly ─────────────────────────────────────────────────────────────

def declared_features() -> list[str]:
    spec = json.loads(SPEC.read_text())
    return [f["name"] for f in spec["features"]]


def build_panel(pairs: list[str] = PAIRS, report: LoadReport | None = None):
    """Return (panel_df with MultiIndex [date, pair], report, coverage dict)."""
    report = report or LoadReport()
    declared = declared_features()

    g = build_global_features(report)

    frames = []
    for pair in pairs:
        px = dl.get_pair_prices(pair, report)   # also defines this pair's trading-day index
        if px is None:
            continue
        idx = px.index
        cols: dict[str, pd.Series] = {}
        # global features → reindex onto this pair's index (documented ffill)
        for name, s in g.items():
            cols[name] = _ffill_to(s, idx)
        # calendar
        for name, s in calendar_features(idx).items():
            cols[name] = s
        # pair-specific
        for name, s in build_pair_features(pair, idx, report).items():
            cols[name] = _ffill_to(s, idx) if name.startswith("pair_") else s
        df = pd.DataFrame(cols, index=idx)
        df["pair"] = pair
        df = df.set_index("pair", append=True)        # MultiIndex (date, pair)
        frames.append(df)

    panel = pd.concat(frames).sort_index() if frames else pd.DataFrame()

    built = [c for c in panel.columns if c in declared]
    missing = [f for f in declared if f not in panel.columns]
    coverage = {
        "declared": len(declared),
        "built": len(built),
        "missing": missing,
        "extra_undeclared": [c for c in panel.columns if c not in declared],
    }
    # keep only declared feature columns (drop any helper columns), preserve order
    keep = [f for f in declared if f in panel.columns]
    panel = panel[keep]
    return panel, report, coverage
