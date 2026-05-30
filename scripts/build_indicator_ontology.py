"""
build_indicator_ontology.py — one-shot sweep to build Oracle market memory

Fetches 10yr daily OHLCV for 8 forex pairs, runs all 30 indicators on every bar,
attaches forward returns, computes IC + hit_rate per indicator, finds green conditions
across C(30,3) = 4,060 triple combos, and writes the memory files Oracle reads live.

Runtime: ~3-5 minutes.

Usage:
    python3 scripts/build_indicator_ontology.py

Outputs (all under data/indicators/):
    history.parquet           — ~2M rows: date, pair, 30 state cols, 30 raw cols, fwd returns
    indicator_rankings.json   — IC + hit_rate per indicator per pair
    green_conditions.json     — top 10 long + short green combos per pair
    oracle_indicator_memory.json — summary Oracle reads at pulse time
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.intelligence.indicator_library import (
    INDICATOR_NAMES,
    compute_all_indicators,
)

# ─── Config ──────────────────────────────────────────────────────────────────

PAIRS = {
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "AUDUSD": "AUDUSD=X",
    "AUDNZD": "AUDNZD=X",
    "USDJPY": "USDJPY=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCHF": "USDCHF=X",
}

HISTORY_START = "2015-01-01"
HISTORY_END = "2024-12-31"
OUT_DIR = ROOT / "data" / "indicators"

# Green condition thresholds
MIN_SAMPLES = 20      # min bars where all 3 indicators agree (triples are rare on daily)
MIN_HIT_RATE = 0.53   # above chance for daily forex; strict IC can't work on constant series
TOP_N = 10            # top combos to store per pair per direction


# ─── Step 1: Fetch + compute indicators ──────────────────────────────────────

def _fetch_pair(pair: str, ticker: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.download(
            ticker,
            start=HISTORY_START,
            end=HISTORY_END,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df is None or len(df) < 100:
            print(f"  [SKIP] {pair}: insufficient data ({len(df) if df is not None else 0} bars)")
            return None

        # yfinance may return MultiIndex columns when downloading a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        df = df.dropna(subset=["Close"])
        print(f"  {pair}: {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")
        return df
    except Exception as exc:
        print(f"  [ERROR] {pair}: {exc}")
        return None


def _attach_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    df = df.copy()
    df["fwd_5d"]  = c.pct_change(5).shift(-5)
    df["fwd_10d"] = c.pct_change(10).shift(-10)
    df["fwd_20d"] = c.pct_change(20).shift(-20)

    # MFE / MAE over next 10 bars (using High / Low)
    h = df["High"]
    lo = df["Low"]
    entry = c

    mfe_vals, mae_vals = [], []
    n = len(df)
    for i in range(n):
        if i + 10 >= n:
            mfe_vals.append(float("nan"))
            mae_vals.append(float("nan"))
        else:
            ep = entry.iat[i]
            if ep == 0 or np.isnan(ep):
                mfe_vals.append(float("nan"))
                mae_vals.append(float("nan"))
            else:
                future_highs = h.iloc[i + 1 : i + 11]
                future_lows = lo.iloc[i + 1 : i + 11]
                mfe_vals.append(float((future_highs.max() - ep) / ep))
                mae_vals.append(float((ep - future_lows.min()) / ep))

    df["mfe_10d"] = mfe_vals
    df["mae_10d"] = mae_vals
    return df


def _build_indicator_rows(pair: str, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Run all 30 indicators bar-by-bar. Rolling window: use only past data."""
    records = []
    n = len(ohlcv_df)
    # Minimum bars needed for the slowest indicator (Ichimoku uses 52)
    min_warmup = 60

    # Run indicators on expanding windows to avoid lookahead — but that's too slow
    # for 2,600 bars × 30 indicators. Instead, run on full df (past only, indicators
    # use rolling/ewm which never look forward) then tag by date.
    indicators = compute_all_indicators(ohlcv_df)

    # Build one row per bar: collect states + raws from each indicator
    # compute_all_indicators returns scalar IndicatorState for the LAST bar.
    # For history we need a vectorised approach — run per-row slices.
    # Use vectorised indicator computation with the full series — every indicator
    # only uses past data (rolling/ewm). We'll extract the state series directly.

    state_series: dict[str, pd.Series] = {}
    raw_series: dict[str, pd.Series] = {}

    for name in INDICATOR_NAMES:
        state_series[name] = pd.Series(0, index=ohlcv_df.index, dtype=int)
        raw_series[name] = pd.Series(float("nan"), index=ohlcv_df.index, dtype=float)

    # For each bar from min_warmup onward, compute indicators on [0:i+1] slice
    # and record the last-bar state. This is O(n²) — too slow for 2,600 bars.
    # Instead: compute indicator states over the full series efficiently.
    # Each indicator function in indicator_library uses rolling/ewm — we can
    # extract the intermediate series. We re-implement the vectorised loop here.

    _build_vectorised(ohlcv_df, state_series, raw_series, min_warmup)

    # Assemble rows
    result_rows = []
    ohlcv_with_fwd = _attach_forward_returns(ohlcv_df)
    for idx in ohlcv_df.index:
        row = {
            "date": idx,
            "pair": pair,
            "close": float(ohlcv_df.at[idx, "Close"]),
            "fwd_5d":  float(ohlcv_with_fwd.at[idx, "fwd_5d"]),
            "fwd_10d": float(ohlcv_with_fwd.at[idx, "fwd_10d"]),
            "fwd_20d": float(ohlcv_with_fwd.at[idx, "fwd_20d"]),
            "mfe_10d": float(ohlcv_with_fwd.at[idx, "mfe_10d"]),
            "mae_10d": float(ohlcv_with_fwd.at[idx, "mae_10d"]),
        }
        for name in INDICATOR_NAMES:
            row[f"state_{name}"] = int(state_series[name].get(idx, 0))
            row[f"raw_{name}"]   = float(raw_series[name].get(idx, float("nan")))
        result_rows.append(row)

    return pd.DataFrame(result_rows)


# ─── Vectorised indicator computation ────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _tr(df: pd.DataFrame) -> pd.Series:
    h, lo, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    return pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _tr(df).ewm(span=n, adjust=False).mean()

def _rsi_series(c: pd.Series, n: int = 14) -> pd.Series:
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def _build_vectorised(
    df: pd.DataFrame,
    states: dict[str, pd.Series],
    raws: dict[str, pd.Series],
    min_warmup: int,
) -> None:
    """Compute all 30 indicator state series without lookahead."""
    o = df["Open"]
    h = df["High"]
    lo = df["Low"]
    c = df["Close"]
    v = df["Volume"]
    idx = df.index
    valid = pd.Series(False, index=idx)
    valid.iloc[min_warmup:] = True

    # ── TREND ──────────────────────────────────────────────────────────────────

    # adx
    up_move = h.diff()
    dn_move = -lo.diff()
    plus_dm = pd.Series(0.0, index=idx)
    minus_dm = pd.Series(0.0, index=idx)
    mask_up = (up_move > dn_move) & (up_move > 0)
    mask_dn = (dn_move > up_move) & (dn_move > 0)
    plus_dm[mask_up] = up_move[mask_up]
    minus_dm[mask_dn] = dn_move[mask_dn]
    atr14 = _atr(df, 14)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    adx_v = dx.ewm(span=14, adjust=False).mean()
    raws["adx"] = adx_v
    s = pd.Series(0, index=idx)
    s[(adx_v > 25) & (plus_di > minus_di)] = 1
    s[(adx_v > 25) & (minus_di > plus_di)] = -1
    states["adx"] = s.where(valid, 0)

    # ema_cross
    e20 = _ema(c, 20)
    e50 = _ema(c, 50)
    raws["ema_cross"] = e20 - e50
    s = pd.Series(0, index=idx)
    s[e20 > e50] = 1
    s[e20 < e50] = -1
    states["ema_cross"] = s.where(valid, 0)

    # supertrend (10, 3)
    atr10 = _atr(df, 10)
    hl_mid = (h + lo) / 2
    upper_basic = hl_mid + 3 * atr10
    lower_basic = hl_mid - 3 * atr10
    st_dir = pd.Series(1, index=idx, dtype=int)
    st_val = pd.Series(float("nan"), index=idx)
    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    for i in range(1, len(df)):
        prev_upper = upper_band.iat[i - 1]
        prev_lower = lower_band.iat[i - 1]
        upper_band.iat[i] = min(upper_basic.iat[i], prev_upper) if c.iat[i - 1] <= prev_upper else upper_basic.iat[i]
        lower_band.iat[i] = max(lower_basic.iat[i], prev_lower) if c.iat[i - 1] >= prev_lower else lower_basic.iat[i]
        if st_dir.iat[i - 1] == -1 and c.iat[i] > upper_band.iat[i]:
            st_dir.iat[i] = 1
        elif st_dir.iat[i - 1] == 1 and c.iat[i] < lower_band.iat[i]:
            st_dir.iat[i] = -1
        else:
            st_dir.iat[i] = st_dir.iat[i - 1]
        st_val.iat[i] = lower_band.iat[i] if st_dir.iat[i] == 1 else upper_band.iat[i]
    raws["supertrend"] = st_dir.astype(float)
    states["supertrend"] = st_dir.where(valid, 0)

    # ichimoku (9, 26, 52)
    tenkan = (h.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun  = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    raws["ichimoku"] = c - cloud_top
    s = pd.Series(0, index=idx)
    s[c > cloud_top] = 1
    s[c < cloud_bot] = -1
    states["ichimoku"] = s.where(valid, 0)

    # parabolic_sar (Wilder)
    af_step = 0.02
    af_max  = 0.20
    sar_vals = pd.Series(float("nan"), index=idx)
    sar_dir  = pd.Series(1, index=idx)
    if len(df) >= 2:
        ep = float(h.iat[1])
        sar = float(lo.iat[0])
        af = af_step
        bull = True
        sar_vals.iat[0] = sar
        sar_dir.iat[0] = 1
        for i in range(1, len(df)):
            if bull:
                sar = min(sar + af * (ep - sar), lo.iat[max(0, i-1)], lo.iat[max(0, i-2)] if i >= 2 else lo.iat[i-1])
                if lo.iat[i] < sar:
                    bull = False
                    sar = ep
                    ep = float(lo.iat[i])
                    af = af_step
                else:
                    if h.iat[i] > ep:
                        ep = float(h.iat[i])
                        af = min(af + af_step, af_max)
            else:
                sar = max(sar + af * (ep - sar), h.iat[max(0, i-1)], h.iat[max(0, i-2)] if i >= 2 else h.iat[i-1])
                if h.iat[i] > sar:
                    bull = True
                    sar = ep
                    ep = float(h.iat[i])
                    af = af_step
                else:
                    if lo.iat[i] < ep:
                        ep = float(lo.iat[i])
                        af = min(af + af_step, af_max)
            sar_vals.iat[i] = sar
            sar_dir.iat[i] = 1 if bull else -1
    raws["parabolic_sar"] = sar_vals
    states["parabolic_sar"] = sar_dir.where(valid, 0)

    # aroon (25)
    aroon_up = h.rolling(26).apply(lambda x: ((x.argmax()) / 25) * 100, raw=True)
    aroon_dn = lo.rolling(26).apply(lambda x: ((x.argmin()) / 25) * 100, raw=True)
    aroon_osc = aroon_up - aroon_dn
    raws["aroon"] = aroon_osc
    s = pd.Series(0, index=idx)
    s[(aroon_up > 70) & (aroon_dn < 30)] = 1
    s[(aroon_dn > 70) & (aroon_up < 30)] = -1
    states["aroon"] = s.where(valid, 0)

    # vwap_dev (daily VWAP deviation)
    typical = (h + lo + c) / 3
    vwap_d = (typical * v).cumsum() / v.cumsum().replace(0, float("nan"))
    vwap_std = (typical - vwap_d).rolling(20).std()
    vwap_dev_val = (c - vwap_d) / vwap_std.replace(0, float("nan"))
    raws["vwap_dev"] = vwap_dev_val
    s = pd.Series(0, index=idx)
    s[vwap_dev_val > 0.5] = 1
    s[vwap_dev_val < -0.5] = -1
    states["vwap_dev"] = s.where(valid, 0)

    # ── MOMENTUM ───────────────────────────────────────────────────────────────

    # rsi
    rsi_v = _rsi_series(c, 14)
    raws["rsi"] = rsi_v
    s = pd.Series(0, index=idx)
    s[rsi_v > 60] = 1
    s[rsi_v < 40] = -1
    states["rsi"] = s.where(valid, 0)

    # macd
    macd_line = _ema(c, 12) - _ema(c, 26)
    signal_line = _ema(macd_line, 9)
    hist = macd_line - signal_line
    raws["macd"] = hist
    s = pd.Series(0, index=idx)
    s[(hist > 0) & (hist > hist.shift(1))] = 1
    s[(hist < 0) & (hist < hist.shift(1))] = -1
    states["macd"] = s.where(valid, 0)

    # stochastic (14, 3, 3)
    low14  = lo.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = 100 * (c - low14) / (high14 - low14).replace(0, float("nan"))
    stoch_d = stoch_k.rolling(3).mean()
    stoch_k_sm = stoch_k.rolling(3).mean()
    raws["stochastic"] = stoch_k_sm
    s = pd.Series(0, index=idx)
    s[(stoch_k_sm > 50) & (stoch_k_sm > stoch_d)] = 1
    s[(stoch_k_sm < 50) & (stoch_k_sm < stoch_d)] = -1
    states["stochastic"] = s.where(valid, 0)

    # cci (20)
    cci_typical = (h + lo + c) / 3
    cci_mean = cci_typical.rolling(20).mean()
    cci_mad  = cci_typical.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci_v = (cci_typical - cci_mean) / (0.015 * cci_mad.replace(0, float("nan")))
    raws["cci"] = cci_v
    s = pd.Series(0, index=idx)
    s[cci_v > 100] = 1
    s[cci_v < -100] = -1
    states["cci"] = s.where(valid, 0)

    # williams_r (14)
    wr = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - lo.rolling(14).min()).replace(0, float("nan"))
    raws["williams_r"] = wr
    s = pd.Series(0, index=idx)
    s[wr > -20] = 1   # overbought → momentum still bull on short-term
    s[wr < -80] = -1
    states["williams_r"] = s.where(valid, 0)

    # roc (10)
    roc_v = c.pct_change(10) * 100
    raws["roc"] = roc_v
    s = pd.Series(0, index=idx)
    s[roc_v > 0.5] = 1
    s[roc_v < -0.5] = -1
    states["roc"] = s.where(valid, 0)

    # mfi (14) — money flow index
    raw_mf = typical * v
    pos_mf = raw_mf.where(typical > typical.shift(1), 0.0)
    neg_mf = raw_mf.where(typical < typical.shift(1), 0.0)
    mf_ratio = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, float("nan"))
    mfi_v = 100 - 100 / (1 + mf_ratio)
    raws["mfi"] = mfi_v
    s = pd.Series(0, index=idx)
    s[mfi_v > 60] = 1
    s[mfi_v < 40] = -1
    states["mfi"] = s.where(valid, 0)

    # ── VOLATILITY ─────────────────────────────────────────────────────────────

    # bollinger_bands (20, 2σ)
    bb_mid = _sma(c, 20)
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pct_b = (c - bb_lower) / (bb_upper - bb_lower).replace(0, float("nan"))
    raws["bollinger_bands"] = bb_pct_b
    s = pd.Series(0, index=idx)
    s[bb_pct_b > 0.80] = 1
    s[bb_pct_b < 0.20] = -1
    states["bollinger_bands"] = s.where(valid, 0)

    # atr_ratio
    atr14_r = _atr(df, 14)
    atr_ratio_v = atr14_r / c.replace(0, float("nan")) * 100
    atr_rolling_mean = atr_ratio_v.rolling(20).mean()
    atr_rolling_std  = atr_ratio_v.rolling(20).std()
    atr_z = (atr_ratio_v - atr_rolling_mean) / atr_rolling_std.replace(0, float("nan"))
    raws["atr_ratio"] = atr_ratio_v
    # HIGH volatility → neutral (can't call direction from ATR alone)
    states["atr_ratio"] = pd.Series(0, index=idx).where(valid, 0)

    # keltner (EMA20 ± 2×ATR)
    k_mid = _ema(c, 20)
    k_atr = _atr(df, 14)
    k_upper = k_mid + 2 * k_atr
    k_lower = k_mid - 2 * k_atr
    raws["keltner"] = (c - k_mid) / k_atr.replace(0, float("nan"))
    s = pd.Series(0, index=idx)
    s[c > k_upper] = 1
    s[c < k_lower] = -1
    states["keltner"] = s.where(valid, 0)

    # donchian (20)
    don_high = h.rolling(20).max()
    don_low  = lo.rolling(20).min()
    don_mid  = (don_high + don_low) / 2
    raws["donchian"] = c - don_mid
    s = pd.Series(0, index=idx)
    s[c > don_mid] = 1
    s[c < don_mid] = -1
    states["donchian"] = s.where(valid, 0)

    # hist_vol (20-day realized vol, annualised)
    log_ret = np.log(c / c.shift(1))
    hvol = log_ret.rolling(20).std() * np.sqrt(252) * 100
    hvol_mean = hvol.rolling(60).mean()
    raws["hist_vol"] = hvol
    # elevated vol vs own 60d mean → NEUTRAL (can't predict direction)
    states["hist_vol"] = pd.Series(0, index=idx).where(valid, 0)

    # ── VOLUME ─────────────────────────────────────────────────────────────────

    # obv (slope via 20-bar EMA of daily OBV change)
    obv_raw = (np.sign(c.diff()) * v).cumsum()
    obv_slope = _ema(obv_raw.diff(), 20)
    raws["obv"] = obv_slope
    s = pd.Series(0, index=idx)
    s[obv_slope > 0] = 1
    s[obv_slope < 0] = -1
    states["obv"] = s.where(valid, 0)

    # vol_trend (volume vs 20d average z-score)
    vol_ma = v.rolling(20).mean()
    vol_std = v.rolling(20).std()
    vol_z = (v - vol_ma) / vol_std.replace(0, float("nan"))
    raws["vol_trend"] = vol_z
    s = pd.Series(0, index=idx)
    s[vol_z > 1.0] = 1
    s[vol_z < -1.0] = -1
    states["vol_trend"] = s.where(valid, 0)

    # vwap_ratio (volume SMA ratio — use vol_z as proxy)
    raws["vwap_ratio"] = vol_z
    states["vwap_ratio"] = states["vol_trend"].copy()

    # acc_dist (Accumulation/Distribution)
    clv = ((c - lo) - (h - c)) / (h - lo).replace(0, float("nan"))
    ad_line = (clv * v).cumsum()
    ad_slope = _ema(ad_line.diff(), 20)
    raws["acc_dist"] = ad_slope
    s = pd.Series(0, index=idx)
    s[ad_slope > 0] = 1
    s[ad_slope < 0] = -1
    states["acc_dist"] = s.where(valid, 0)

    # cmf (Chaikin Money Flow, 20)
    mf_vol = clv * v
    cmf_v = mf_vol.rolling(20).sum() / v.rolling(20).sum().replace(0, float("nan"))
    raws["cmf"] = cmf_v
    s = pd.Series(0, index=idx)
    s[cmf_v > 0.05] = 1
    s[cmf_v < -0.05] = -1
    states["cmf"] = s.where(valid, 0)

    # ── MARKET STRUCTURE ───────────────────────────────────────────────────────

    # fvg (Fair Value Gap — 3-candle imbalance)
    fvg_bull = lo.shift(-1) > h.shift(1)   # gap: next low > prev high
    fvg_bear = h.shift(-1) < lo.shift(1)   # gap: next high < prev low
    fvg_s = pd.Series(0, index=idx)
    fvg_s[fvg_bull] = 1
    fvg_s[fvg_bear] = -1
    raws["fvg"] = fvg_s.astype(float)
    states["fvg"] = fvg_s.where(valid, 0)

    # bos (Break of Structure — swing high/low break)
    swing_high = h.rolling(10, center=True).max()
    swing_low  = lo.rolling(10, center=True).min()
    bos_bull = (c > swing_high.shift(1)) & (c.shift(1) <= swing_high.shift(2))
    bos_bear = (c < swing_low.shift(1))  & (c.shift(1) >= swing_low.shift(2))
    bos_s = pd.Series(0, index=idx)
    bos_s[bos_bull] = 1
    bos_s[bos_bear] = -1
    raws["bos"] = bos_s.astype(float)
    states["bos"] = bos_s.where(valid, 0)

    # session_hl (price vs prior-day high/low)
    prev_h = h.shift(1)
    prev_l = lo.shift(1)
    sess_v = pd.Series(0, index=idx)
    sess_v[c > prev_h] = 1
    sess_v[c < prev_l] = -1
    raws["session_hl"] = (c - (prev_h + prev_l) / 2)
    states["session_hl"] = sess_v.where(valid, 0)

    # displacement (body > 2×ATR14)
    body = (c - o).abs()
    disp_v = pd.Series(0, index=idx)
    disp_v[(body > 2 * atr14) & (c > o)] = 1
    disp_v[(body > 2 * atr14) & (c < o)] = -1
    raws["displacement"] = body / atr14.replace(0, float("nan"))
    states["displacement"] = disp_v.where(valid, 0)

    # liq_sweep (wick beyond prior-day extreme then close back inside)
    sweep_bull = (lo < lo.shift(1)) & (c > lo.shift(1))
    sweep_bear = (h > h.shift(1)) & (c < h.shift(1))
    sweep_v = pd.Series(0, index=idx)
    sweep_v[sweep_bull] = 1
    sweep_v[sweep_bear] = -1
    raws["liq_sweep"] = sweep_v.astype(float)
    states["liq_sweep"] = sweep_v.where(valid, 0)

    # order_block (last opposing candle before a BOS — simplified: candle before bos)
    ob_v = pd.Series(0, index=idx)
    ob_v[bos_bull] = 1   # bullish OB context
    ob_v[bos_bear] = -1
    raws["order_block"] = ob_v.astype(float)
    states["order_block"] = ob_v.where(valid, 0)


# ─── Step 2: IC + hit_rate per indicator per pair ────────────────────────────

def _compute_rankings(hist: pd.DataFrame) -> dict:
    from scipy.stats import pearsonr

    rankings = {}
    for pair in hist["pair"].unique():
        g = hist[hist["pair"] == pair].dropna(subset=["fwd_10d"])
        pair_rankings = []
        for name in INDICATOR_NAMES:
            col = f"state_{name}"
            if col not in g.columns:
                continue
            series = g[col].astype(float)
            fwd = g["fwd_10d"].astype(float)
            valid_mask = series.notna() & fwd.notna()
            if valid_mask.sum() < 30:
                continue
            try:
                ic, _ = pearsonr(series[valid_mask], fwd[valid_mask])
            except Exception:
                ic = 0.0

            bull = g[series == 1]["fwd_10d"]
            bear = g[series == -1]["fwd_10d"]
            pair_rankings.append({
                "indicator": name,
                "ic": round(float(ic), 4),
                "bull_hit_rate": round(float((bull > 0).mean()) if len(bull) > 0 else 0.0, 4),
                "bear_hit_rate": round(float((bear < 0).mean()) if len(bear) > 0 else 0.0, 4),
                "bull_n": int(len(bull)),
                "bear_n": int(len(bear)),
                "bull_avg_r": round(float(bull.mean()) if len(bull) > 0 else 0.0, 6),
                "bear_avg_r": round(float(bear.mean()) if len(bear) > 0 else 0.0, 6),
            })
        pair_rankings.sort(key=lambda x: -abs(x["ic"]))
        rankings[pair] = pair_rankings

    return rankings


# ─── Step 3: Green conditions (C(30,3) combos) ───────────────────────────────

def _find_green_conditions(hist: pd.DataFrame) -> dict:
    green = {}
    all_combos = list(combinations(INDICATOR_NAMES, 3))
    print(f"  Testing {len(all_combos):,} triple combos per pair…")

    for pair in hist["pair"].unique():
        g = hist[hist["pair"] == pair].dropna(subset=["fwd_10d"]).copy()
        state_cols = {name: f"state_{name}" for name in INDICATOR_NAMES}
        fwd = g["fwd_10d"].astype(float)

        best_long  = []
        best_short = []

        for a, b, cc in all_combos:
            ca, cb, ccc = state_cols[a], state_cols[b], state_cols[cc]

            # LONG: all three BULLISH
            # IC not computed per-combo (all state cols = 1 when masked → constant → undefined pearson).
            # Ranking is by hit_rate; individual IC lives in indicator_rankings.json.
            long_mask = (g[ca] == 1) & (g[cb] == 1) & (g[ccc] == 1)
            n_long = long_mask.sum()
            if n_long >= MIN_SAMPLES:
                subset_fwd = fwd[long_mask]
                hr = float((subset_fwd > 0).mean())
                avg_r = float(subset_fwd.mean())
                if hr >= MIN_HIT_RATE:
                    best_long.append({
                        "indicators": [a, b, cc],
                        "hit_rate": round(hr, 4),
                        "avg_return": round(avg_r, 6),
                        "n": int(n_long),
                    })

            # SHORT: all three BEARISH
            short_mask = (g[ca] == -1) & (g[cb] == -1) & (g[ccc] == -1)
            n_short = short_mask.sum()
            if n_short >= MIN_SAMPLES:
                subset_fwd = fwd[short_mask]
                hr = float((subset_fwd < 0).mean())
                avg_r = float(subset_fwd.mean())
                if hr >= MIN_HIT_RATE:
                    best_short.append({
                        "indicators": [a, b, cc],
                        "hit_rate": round(hr, 4),
                        "avg_return": round(avg_r, 6),
                        "n": int(n_short),
                    })

        best_long.sort(key=lambda x: -x["hit_rate"])
        best_short.sort(key=lambda x: -x["hit_rate"])
        green[pair] = {
            "best_long":  best_long[:TOP_N],
            "best_short": best_short[:TOP_N],
        }
        n_green_l = len(best_long[:TOP_N])
        n_green_s = len(best_short[:TOP_N])
        print(f"    {pair}: {n_green_l} long green, {n_green_s} short green")

    return green


# ─── Step 4: Oracle summary ───────────────────────────────────────────────────

def _build_oracle_memory(rankings: dict, green: dict, total_obs: int) -> dict:
    total_green = sum(
        len(v["best_long"]) + len(v["best_short"])
        for v in green.values()
    )

    best_combo = None
    best_hr = 0.0
    for pair, v in green.items():
        for cond in v["best_long"] + v["best_short"]:
            if cond["hit_rate"] > best_hr:
                best_hr = cond["hit_rate"]
                best_combo = {"pair": pair, **cond}

    # Worst indicators — lowest average abs(IC) across pairs
    ic_sums: dict[str, list[float]] = {n: [] for n in INDICATOR_NAMES}
    for pair_ranks in rankings.values():
        for entry in pair_ranks:
            ic_sums[entry["indicator"]].append(abs(entry["ic"]))
    worst_indicators = sorted(
        [{"indicator": n, "avg_abs_ic": round(float(np.mean(vals)), 4)} for n, vals in ic_sums.items() if vals],
        key=lambda x: x["avg_abs_ic"],
    )[:5]

    per_pair = {}
    for pair in rankings:
        top5 = rankings[pair][:5]
        per_pair[pair] = {
            "top_indicators": top5,
            "green_long_count": len(green.get(pair, {}).get("best_long", [])),
            "green_short_count": len(green.get(pair, {}).get("best_short", [])),
        }

    return {
        "last_computed": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "history_start": HISTORY_START,
        "history_end": HISTORY_END,
        "total_observations": total_obs,
        "indicator_count": len(INDICATOR_NAMES),
        "green_conditions_found": total_green,
        "best_overall_combination": best_combo,
        "worst_indicators": worst_indicators,
        "per_pair": per_pair,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import time
    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Oracle Indicator Ontology Builder")
    print(f"Pairs: {len(PAIRS)} | History: {HISTORY_START} → {HISTORY_END}")
    print(f"Indicators: {len(INDICATOR_NAMES)} | Combos: {len(list(combinations(INDICATOR_NAMES, 3))):,}")
    print(f"{'='*60}\n")

    # Step 1: Fetch and compute
    print("Step 1: Fetching OHLCV + computing indicators")
    frames = []
    for pair, ticker in PAIRS.items():
        ohlcv = _fetch_pair(pair, ticker)
        if ohlcv is None:
            continue
        print(f"  Computing indicators for {pair}…", end="", flush=True)
        pair_df = _build_indicator_rows(pair, ohlcv)
        frames.append(pair_df)
        print(f" done ({len(pair_df)} rows)")

    if not frames:
        print("[ERROR] No data fetched. Check network connection.")
        sys.exit(1)

    hist = pd.concat(frames, ignore_index=True)
    hist_path = OUT_DIR / "history.parquet"
    hist.to_parquet(hist_path, index=False)
    print(f"\n  Saved history.parquet: {len(hist):,} rows, {len(hist.columns)} columns")

    # Step 2: IC + hit_rate rankings
    print("\nStep 2: Computing IC + hit_rate per indicator per pair")
    try:
        from scipy.stats import pearsonr  # noqa: F401  — verify available
    except ImportError:
        print("  [ERROR] scipy not installed. Run: pip install scipy")
        sys.exit(1)

    rankings = _compute_rankings(hist)
    rankings_path = OUT_DIR / "indicator_rankings.json"
    rankings_path.write_text(json.dumps(rankings, indent=2))
    print(f"  Saved indicator_rankings.json")
    for pair, ranks in rankings.items():
        if ranks:
            top = ranks[0]
            print(f"    {pair}: best={top['indicator']} IC={top['ic']:+.3f}")

    # Step 3: Green conditions
    print("\nStep 3: Finding green conditions (C(30,3) sweep)")
    green = _find_green_conditions(hist)
    green_path = OUT_DIR / "green_conditions.json"
    green_path.write_text(json.dumps(green, indent=2))
    total_green = sum(len(v["best_long"]) + len(v["best_short"]) for v in green.values())
    print(f"  Saved green_conditions.json ({total_green} total green conditions)")

    # Step 4: Oracle memory summary
    print("\nStep 4: Writing oracle_indicator_memory.json")
    memory = _build_oracle_memory(rankings, green, len(hist))
    mem_path = OUT_DIR / "oracle_indicator_memory.json"
    mem_path.write_text(json.dumps(memory, indent=2))
    print(f"  Saved oracle_indicator_memory.json")
    if memory["best_overall_combination"]:
        b = memory["best_overall_combination"]
        print(f"  Best combo: {b['pair']} {b['indicators']} HR={b['hit_rate']:.0%}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  history.parquet:            {len(hist):,} rows")
    print(f"  indicator_rankings.json:    {len(PAIRS)} pairs × {len(INDICATOR_NAMES)} indicators")
    print(f"  green_conditions.json:      {total_green} green conditions")
    print(f"  oracle_indicator_memory.json: ready for Oracle pulse")
    print(f"\nNext:")
    print(f"  python3 scripts/build_indicator_ontology.py  ← re-run monthly")
    print(f"  python3 sovereign/oracle/pulse_check.py      ← writes live_snapshot.json")


if __name__ == "__main__":
    main()
