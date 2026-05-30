"""
sovereign/intelligence/indicator_library.py

Unified 30-indicator API for the Oracle market memory layer.

Each indicator → IndicatorState(state: -1|0|1, raw: float, label: str)
  +1 = BULLISH   0 = NEUTRAL   -1 = BEARISH

Input: OHLCV DataFrame with columns Open/High/Low/Close/Volume (yfinance style).
       Normalizes to lower-case internally so column case is irrelevant.

Public API:
  compute_all_indicators(df)  → dict[str, IndicatorState]
  INDICATOR_NAMES             → list[str]  (30 names, stable order)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ─── Types ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IndicatorState:
    state: int    # 1=BULLISH, 0=NEUTRAL, -1=BEARISH
    raw: float    # numeric value of the indicator
    label: str    # human-readable, e.g. "RSI=67.2 (BULLISH)"


def _neutral(label: str = "N/A") -> IndicatorState:
    return IndicatorState(state=0, raw=float("nan"), label=label)


# ─── Column normalizer ────────────────────────────────────────────────────────

def _cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lowercase. Returns a view with renamed columns."""
    return df.rename(columns={c: c.lower() for c in df.columns})


# ─── Primitive building blocks ────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    h, l, cp = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / period, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.where(d > 0, 0).rolling(period).mean()
    loss = (-d.where(d < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ─── TREND INDICATORS (7) ────────────────────────────────────────────────────

def ind_adx(df: pd.DataFrame, period: int = 14) -> IndicatorState:
    """ADX: > 25 trending, use ±DI for direction."""
    d = _cols(df)
    h, l, c = d["high"], d["low"], d["close"]
    up = h - h.shift(1)
    dn = l.shift(1) - l
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    a = 1 / period
    atr = _true_range(d).ewm(alpha=a, min_periods=period).mean()
    pdi = 100 * pdm.ewm(alpha=a, min_periods=period).mean() / atr
    mdi = 100 * mdm.ewm(alpha=a, min_periods=period).mean() / atr
    dsum = pdi + mdi
    dx = 100 * (pdi - mdi).abs() / dsum.replace(0, np.nan)
    adx = dx.ewm(alpha=a, min_periods=period).mean()
    v = float(adx.iloc[-1])
    p = float(pdi.iloc[-1])
    m = float(mdi.iloc[-1])
    if math.isnan(v) or v < 20:
        return IndicatorState(0, v, f"ADX={v:.1f} (NEUTRAL — weak trend)")
    state = 1 if p > m else -1
    word = "BULLISH" if state == 1 else "BEARISH"
    return IndicatorState(state, v, f"ADX={v:.1f} +DI={p:.1f} -DI={m:.1f} ({word})")


def ind_ema_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> IndicatorState:
    """EMA20 vs EMA50 crossover."""
    c = _cols(df)["close"]
    e20 = _ema(c, fast).iloc[-1]
    e50 = _ema(c, slow).iloc[-1]
    if math.isnan(e20) or math.isnan(e50):
        return _neutral("EMA Cross=N/A")
    diff_pct = (e20 - e50) / e50 * 100
    if abs(diff_pct) < 0.05:
        return IndicatorState(0, diff_pct, f"EMA Cross={diff_pct:.3f}% (NEUTRAL — near cross)")
    state = 1 if e20 > e50 else -1
    word = "BULLISH" if state == 1 else "BEARISH"
    return IndicatorState(state, diff_pct, f"EMA Cross={diff_pct:.3f}% ({word})")


def ind_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> IndicatorState:
    """Supertrend direction."""
    d = _cols(df)
    if len(d) < period + 5:
        return _neutral("Supertrend=N/A")
    atr = _atr(d, period)
    hl2 = (d["high"] + d["low"]) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    # Compute Supertrend
    trend = pd.Series(np.nan, index=d.index)
    direction = pd.Series(0, index=d.index)
    close = d["close"]
    for i in range(1, len(d)):
        prev_trend = trend.iloc[i - 1] if not math.isnan(trend.iloc[i - 1]) else lower.iloc[i]
        u = upper.iloc[i]
        lo = lower.iloc[i]
        # Adjust upper/lower
        if lo > prev_trend or close.iloc[i - 1] < prev_trend:
            lo = lo
        else:
            lo = max(lo, prev_trend)
        if u < prev_trend or close.iloc[i - 1] > prev_trend:
            u = u
        else:
            u = min(u, prev_trend)
        if close.iloc[i] > u:
            trend.iloc[i] = lo
            direction.iloc[i] = 1
        elif close.iloc[i] < lo:
            trend.iloc[i] = u
            direction.iloc[i] = -1
        else:
            trend.iloc[i] = prev_trend
            direction.iloc[i] = direction.iloc[i - 1]

    v = int(direction.iloc[-1])
    raw = float(trend.iloc[-1]) if not math.isnan(trend.iloc[-1]) else 0.0
    word = "BULLISH" if v == 1 else "BEARISH" if v == -1 else "NEUTRAL"
    return IndicatorState(v if v in (-1, 0, 1) else 0, raw, f"Supertrend={raw:.5f} ({word})")


def ind_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> IndicatorState:
    """Price vs Ichimoku cloud: above=bullish, below=bearish, inside=neutral."""
    d = _cols(df)
    if len(d) < senkou:
        return _neutral("Ichimoku=N/A")
    high, low, close = d["high"], d["low"], d["close"]
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen  = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2
    span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    span_b = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).shift(kijun)
    price = float(close.iloc[-1])
    a = float(span_a.iloc[-1]) if not math.isnan(span_a.iloc[-1]) else price
    b = float(span_b.iloc[-1]) if not math.isnan(span_b.iloc[-1]) else price
    top_cloud = max(a, b)
    bot_cloud = min(a, b)
    if price > top_cloud:
        return IndicatorState(1, price, f"Ichimoku: price above cloud (BULLISH)")
    if price < bot_cloud:
        return IndicatorState(-1, price, f"Ichimoku: price below cloud (BEARISH)")
    return IndicatorState(0, price, f"Ichimoku: price inside cloud (NEUTRAL)")


def ind_parabolic_sar(df: pd.DataFrame, af0: float = 0.02, af_max: float = 0.2) -> IndicatorState:
    """Parabolic SAR: price above SAR = bullish."""
    d = _cols(df)
    if len(d) < 5:
        return _neutral("ParSAR=N/A")
    high, low, close = d["high"].values, d["low"].values, d["close"].values
    af, ep = af0, high[0]
    sar = low[0]
    bull = True
    for i in range(1, len(high)):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, low[i - 1], low[max(i - 2, 0)])
            if low[i] < sar:
                bull = False
                sar = ep
                ep = low[i]
                af = af0
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af0, af_max)
        else:
            sar = max(sar, high[i - 1], high[max(i - 2, 0)])
            if high[i] > sar:
                bull = True
                sar = ep
                ep = high[i]
                af = af0
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af0, af_max)
    state = 1 if bull else -1
    word = "BULLISH" if bull else "BEARISH"
    return IndicatorState(state, float(sar), f"ParSAR={sar:.5f} price={close[-1]:.5f} ({word})")


def ind_aroon(df: pd.DataFrame, period: int = 25) -> IndicatorState:
    """Aroon Oscillator (Up - Down): > 50 bullish, < -50 bearish."""
    d = _cols(df)
    if len(d) < period + 1:
        return _neutral("Aroon=N/A")
    high, low = d["high"], d["low"]
    aroon_up  = high.rolling(period + 1).apply(lambda x: (period - x.argmax()) / period * 100, raw=True)
    aroon_dn  = low.rolling(period + 1).apply(lambda x: (period - x.argmin()) / period * 100, raw=True)
    osc = float((aroon_up - aroon_dn).iloc[-1])
    if math.isnan(osc):
        return _neutral("Aroon=N/A")
    if osc > 50:
        return IndicatorState(1, osc, f"Aroon={osc:.1f} (BULLISH)")
    if osc < -50:
        return IndicatorState(-1, osc, f"Aroon={osc:.1f} (BEARISH)")
    return IndicatorState(0, osc, f"Aroon={osc:.1f} (NEUTRAL)")


def ind_vwap_dev(df: pd.DataFrame, period: int = 20) -> IndicatorState:
    """VWAP deviation: rolling VWAP vs close, normalised by σ."""
    d = _cols(df)
    if len(d) < period or "volume" not in d.columns or d["volume"].sum() == 0:
        return _neutral("VWAP Dev=N/A")
    tp = (d["high"] + d["low"] + d["close"]) / 3
    vol = d["volume"].replace(0, np.nan)
    vwap = (tp * vol).rolling(period).sum() / vol.rolling(period).sum()
    dev  = (d["close"] - vwap) / vwap.rolling(period).std()
    v = float(dev.iloc[-1])
    if math.isnan(v):
        return _neutral("VWAP Dev=N/A")
    if v > 1.0:
        return IndicatorState(1, v, f"VWAP Dev={v:.2f}σ (BULLISH — above VWAP)")
    if v < -1.0:
        return IndicatorState(-1, v, f"VWAP Dev={v:.2f}σ (BEARISH — below VWAP)")
    return IndicatorState(0, v, f"VWAP Dev={v:.2f}σ (NEUTRAL)")


# ─── MOMENTUM INDICATORS (7) ─────────────────────────────────────────────────

def ind_rsi(df: pd.DataFrame, period: int = 14) -> IndicatorState:
    """RSI: > 60 bull, < 40 bear, 40-60 neutral."""
    d = _cols(df)
    v = float(_rsi(d["close"], period).iloc[-1])
    if math.isnan(v):
        return _neutral("RSI=N/A")
    if v > 60:
        return IndicatorState(1, v, f"RSI={v:.1f} (BULLISH)")
    if v < 40:
        return IndicatorState(-1, v, f"RSI={v:.1f} (BEARISH)")
    return IndicatorState(0, v, f"RSI={v:.1f} (NEUTRAL)")


def ind_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> IndicatorState:
    """MACD histogram: positive and rising = bullish."""
    d = _cols(df)
    close = d["close"]
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig_line  = _ema(macd_line, signal)
    hist = macd_line - sig_line
    h = hist.iloc[-1]
    hp = hist.iloc[-2] if len(hist) > 1 else h
    if math.isnan(h):
        return _neutral("MACD=N/A")
    rising = h > hp
    if h > 0 and rising:
        return IndicatorState(1, float(h), f"MACD hist={h:.5f} rising (BULLISH)")
    if h < 0 and not rising:
        return IndicatorState(-1, float(h), f"MACD hist={h:.5f} falling (BEARISH)")
    return IndicatorState(0, float(h), f"MACD hist={h:.5f} (NEUTRAL)")


def ind_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> IndicatorState:
    """Stochastic K: K > D and both > 50 = bull; both < 50 and K < D = bear."""
    df2 = _cols(df)
    low_min = df2["low"].rolling(k).min()
    high_max = df2["high"].rolling(k).max()
    k_line = 100 * (df2["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d_line = k_line.rolling(d).mean()
    kv = float(k_line.iloc[-1])
    dv = float(d_line.iloc[-1])
    if math.isnan(kv) or math.isnan(dv):
        return _neutral("Stoch=N/A")
    if kv > 50 and dv > 50 and kv > dv:
        return IndicatorState(1, kv, f"Stoch K={kv:.1f} D={dv:.1f} (BULLISH)")
    if kv < 50 and dv < 50 and kv < dv:
        return IndicatorState(-1, kv, f"Stoch K={kv:.1f} D={dv:.1f} (BEARISH)")
    return IndicatorState(0, kv, f"Stoch K={kv:.1f} D={dv:.1f} (NEUTRAL)")


def ind_cci(df: pd.DataFrame, period: int = 20) -> IndicatorState:
    """CCI: > +100 bull, < -100 bear."""
    d = _cols(df)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    v = float(cci.iloc[-1])
    if math.isnan(v):
        return _neutral("CCI=N/A")
    if v > 100:
        return IndicatorState(1, v, f"CCI={v:.1f} (BULLISH)")
    if v < -100:
        return IndicatorState(-1, v, f"CCI={v:.1f} (BEARISH)")
    return IndicatorState(0, v, f"CCI={v:.1f} (NEUTRAL)")


def ind_williams_r(df: pd.DataFrame, period: int = 14) -> IndicatorState:
    """Williams %R: > -20 overbought (BULLISH momentum), < -80 oversold (BEARISH)."""
    d = _cols(df)
    hh = d["high"].rolling(period).max()
    ll = d["low"].rolling(period).min()
    wr = -100 * (hh - d["close"]) / (hh - ll).replace(0, np.nan)
    v = float(wr.iloc[-1])
    if math.isnan(v):
        return _neutral("WR=N/A")
    if v > -20:
        return IndicatorState(1, v, f"Williams %R={v:.1f} (BULLISH — overbought)")
    if v < -80:
        return IndicatorState(-1, v, f"Williams %R={v:.1f} (BEARISH — oversold)")
    return IndicatorState(0, v, f"Williams %R={v:.1f} (NEUTRAL)")


def ind_roc(df: pd.DataFrame, period: int = 10) -> IndicatorState:
    """Rate of Change: positive = bullish, negative = bearish."""
    d = _cols(df)
    close = d["close"]
    roc = (close - close.shift(period)) / close.shift(period) * 100
    v = float(roc.iloc[-1])
    if math.isnan(v):
        return _neutral("ROC=N/A")
    if v > 0.5:
        return IndicatorState(1, v, f"ROC={v:.2f}% (BULLISH)")
    if v < -0.5:
        return IndicatorState(-1, v, f"ROC={v:.2f}% (BEARISH)")
    return IndicatorState(0, v, f"ROC={v:.2f}% (NEUTRAL)")


def ind_mfi(df: pd.DataFrame, period: int = 14) -> IndicatorState:
    """Money Flow Index (volume-weighted RSI): > 60 bull, < 40 bear."""
    d = _cols(df)
    if "volume" not in d.columns or d["volume"].sum() == 0:
        return _neutral("MFI=N/A (no volume)")
    tp = (d["high"] + d["low"] + d["close"]) / 3
    mf = tp * d["volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    mfi = 100 - 100 / (1 + mfr)
    v = float(mfi.iloc[-1])
    if math.isnan(v):
        return _neutral("MFI=N/A")
    if v > 60:
        return IndicatorState(1, v, f"MFI={v:.1f} (BULLISH)")
    if v < 40:
        return IndicatorState(-1, v, f"MFI={v:.1f} (BEARISH)")
    return IndicatorState(0, v, f"MFI={v:.1f} (NEUTRAL)")


# ─── VOLATILITY INDICATORS (5) ───────────────────────────────────────────────

def ind_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> IndicatorState:
    """BB %B: > 0.8 bull breakout, < 0.2 bear breakdown, 0.2-0.8 neutral."""
    d = _cols(df)
    close = d["close"]
    mid = _sma(close, period)
    sigma = close.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    v = float(pct_b.iloc[-1])
    if math.isnan(v):
        return _neutral("BB=N/A")
    if v > 0.8:
        return IndicatorState(1, v, f"BB %B={v:.2f} (BULLISH — upper band)")
    if v < 0.2:
        return IndicatorState(-1, v, f"BB %B={v:.2f} (BEARISH — lower band)")
    return IndicatorState(0, v, f"BB %B={v:.2f} (NEUTRAL)")


def ind_atr_ratio(df: pd.DataFrame, period: int = 14, atr_lookback: int = 90) -> IndicatorState:
    """ATR as % of price vs its own 90d history: HIGH = volatile NEUTRAL."""
    d = _cols(df)
    atr = _atr(d, period)
    atr_pct = atr / d["close"]
    zscore = (atr_pct - atr_pct.rolling(atr_lookback).mean()) / atr_pct.rolling(atr_lookback).std()
    v = float(zscore.iloc[-1])
    raw = float(atr_pct.iloc[-1]) * 100
    if math.isnan(v):
        return _neutral(f"ATR ratio={raw:.3f}% (N/A)")
    # Very high ATR = uncertain market
    if v > 1.5:
        return IndicatorState(0, raw, f"ATR ratio={raw:.3f}% z={v:.2f} (NEUTRAL — high vol)")
    # Normal or low ATR: direction from close vs rolling median
    if math.isnan(raw):
        return _neutral("ATR=N/A")
    return IndicatorState(0, raw, f"ATR ratio={raw:.3f}% (NEUTRAL)")


def ind_keltner(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> IndicatorState:
    """Keltner Channel: above upper = bullish breakout, below lower = bearish."""
    d = _cols(df)
    close = d["close"]
    mid   = _ema(close, period)
    atr   = _atr(d, 14)
    upper = mid + mult * atr
    lower = mid - mult * atr
    price = float(close.iloc[-1])
    u = float(upper.iloc[-1])
    lo = float(lower.iloc[-1])
    if math.isnan(u):
        return _neutral("Keltner=N/A")
    if price > u:
        return IndicatorState(1, price, f"Keltner: above upper {u:.5f} (BULLISH)")
    if price < lo:
        return IndicatorState(-1, price, f"Keltner: below lower {lo:.5f} (BEARISH)")
    return IndicatorState(0, price, f"Keltner: inside channel (NEUTRAL)")


def ind_donchian(df: pd.DataFrame, period: int = 20) -> IndicatorState:
    """Donchian: price vs midline (halfway between N-period high and low)."""
    d = _cols(df)
    hh = d["high"].rolling(period).max()
    ll = d["low"].rolling(period).min()
    mid = (hh + ll) / 2
    price = float(d["close"].iloc[-1])
    m = float(mid.iloc[-1])
    if math.isnan(m):
        return _neutral("Donchian=N/A")
    diff = price - m
    pct = diff / m
    if pct > 0.001:
        return IndicatorState(1, price, f"Donchian: above mid {m:.5f} (BULLISH)")
    if pct < -0.001:
        return IndicatorState(-1, price, f"Donchian: below mid {m:.5f} (BEARISH)")
    return IndicatorState(0, price, f"Donchian: at mid {m:.5f} (NEUTRAL)")


def ind_hist_vol(df: pd.DataFrame, period: int = 20, thresh_high: float = 0.12) -> IndicatorState:
    """Realized volatility (annualized). Extreme vol = uncertain (NEUTRAL)."""
    d = _cols(df)
    log_ret = np.log(d["close"] / d["close"].shift(1))
    hvol = log_ret.rolling(period).std() * math.sqrt(252)
    v = float(hvol.iloc[-1])
    if math.isnan(v):
        return _neutral("HistVol=N/A")
    # Very high volatility = no directional edge
    if v > thresh_high:
        return IndicatorState(0, v, f"HistVol={v:.1%} (NEUTRAL — elevated)")
    # Low vol trending market: use close direction
    close = d["close"]
    ret_20 = float((close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]) if len(close) >= period else 0
    if ret_20 > 0.005:
        return IndicatorState(1, v, f"HistVol={v:.1%} low — uptrend (BULLISH)")
    if ret_20 < -0.005:
        return IndicatorState(-1, v, f"HistVol={v:.1%} low — downtrend (BEARISH)")
    return IndicatorState(0, v, f"HistVol={v:.1%} (NEUTRAL)")


# ─── VOLUME INDICATORS (5) ───────────────────────────────────────────────────

def ind_obv(df: pd.DataFrame, slope_period: int = 20) -> IndicatorState:
    """OBV slope: rising OBV = bullish."""
    d = _cols(df)
    if "volume" not in d.columns:
        return _neutral("OBV=N/A")
    delta = d["close"].diff()
    sign = pd.Series(np.where(delta > 0, 1, np.where(delta < 0, -1, 0)), index=d.index)
    obv = (sign * d["volume"]).cumsum()
    obv_ema = _ema(obv, slope_period)
    slope = float((obv_ema.iloc[-1] - obv_ema.iloc[-slope_period]) if len(obv_ema) >= slope_period else 0)
    v = float(obv.iloc[-1])
    if slope > 0:
        return IndicatorState(1, v, f"OBV slope rising (BULLISH)")
    if slope < 0:
        return IndicatorState(-1, v, f"OBV slope falling (BEARISH)")
    return IndicatorState(0, v, f"OBV flat (NEUTRAL)")


def ind_vol_trend(df: pd.DataFrame, period: int = 20) -> IndicatorState:
    """Volume z-score: very high volume with price up = bullish confirmation."""
    d = _cols(df)
    if "volume" not in d.columns or d["volume"].sum() == 0:
        return _neutral("VolTrend=N/A")
    vol = d["volume"]
    zscore = (vol - vol.rolling(period).mean()) / vol.rolling(period).std()
    z = float(zscore.iloc[-1])
    price_up = float(d["close"].iloc[-1]) > float(d["close"].iloc[-2]) if len(d) > 1 else True
    if math.isnan(z):
        return _neutral("VolTrend=N/A")
    if z > 1.5 and price_up:
        return IndicatorState(1, z, f"VolTrend z={z:.2f} + price up (BULLISH)")
    if z > 1.5 and not price_up:
        return IndicatorState(-1, z, f"VolTrend z={z:.2f} + price down (BEARISH)")
    return IndicatorState(0, z, f"VolTrend z={z:.2f} (NEUTRAL)")


def ind_vwap_ratio(df: pd.DataFrame, period: int = 20) -> IndicatorState:
    """Rolling VWAP ratio: price / VWAP. Same as VWAP deviation but ratio-based."""
    d = _cols(df)
    if "volume" not in d.columns or d["volume"].sum() == 0:
        return _neutral("VWAP ratio=N/A")
    tp = (d["high"] + d["low"] + d["close"]) / 3
    vol = d["volume"].replace(0, np.nan)
    vwap = (tp * vol).rolling(period).sum() / vol.rolling(period).sum()
    ratio = d["close"] / vwap.replace(0, np.nan)
    v = float(ratio.iloc[-1])
    if math.isnan(v):
        return _neutral("VWAP ratio=N/A")
    if v > 1.001:
        return IndicatorState(1, v, f"VWAP ratio={v:.4f} (BULLISH — above VWAP)")
    if v < 0.999:
        return IndicatorState(-1, v, f"VWAP ratio={v:.4f} (BEARISH — below VWAP)")
    return IndicatorState(0, v, f"VWAP ratio={v:.4f} (NEUTRAL)")


def ind_acc_dist(df: pd.DataFrame, slope_period: int = 20) -> IndicatorState:
    """Accumulation/Distribution: rising A/D = bullish."""
    d = _cols(df)
    if "volume" not in d.columns:
        return _neutral("A/D=N/A")
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    clv = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / rng
    ad = (clv * d["volume"]).cumsum()
    ad_ema = _ema(ad, slope_period)
    slope = float(ad_ema.iloc[-1] - ad_ema.iloc[-slope_period]) if len(ad_ema) >= slope_period else 0
    v = float(ad.iloc[-1])
    if slope > 0:
        return IndicatorState(1, v, f"A/D rising (BULLISH accumulation)")
    if slope < 0:
        return IndicatorState(-1, v, f"A/D falling (BEARISH distribution)")
    return IndicatorState(0, v, f"A/D flat (NEUTRAL)")


def ind_cmf(df: pd.DataFrame, period: int = 20) -> IndicatorState:
    """Chaikin Money Flow: > 0 = accumulation (bull), < 0 = distribution (bear)."""
    d = _cols(df)
    if "volume" not in d.columns or d["volume"].sum() == 0:
        return _neutral("CMF=N/A")
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    clv = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / rng
    cmf = (clv * d["volume"]).rolling(period).sum() / d["volume"].rolling(period).sum().replace(0, np.nan)
    v = float(cmf.iloc[-1])
    if math.isnan(v):
        return _neutral("CMF=N/A")
    if v > 0.05:
        return IndicatorState(1, v, f"CMF={v:.3f} (BULLISH — accumulation)")
    if v < -0.05:
        return IndicatorState(-1, v, f"CMF={v:.3f} (BEARISH — distribution)")
    return IndicatorState(0, v, f"CMF={v:.3f} (NEUTRAL)")


# ─── MARKET STRUCTURE INDICATORS (6) ─────────────────────────────────────────

def ind_fvg(df: pd.DataFrame, lookback: int = 10) -> IndicatorState:
    """Fair Value Gap: 3-candle imbalance. Bullish FVG = H[i-2] < L[i] (gap up)."""
    d = _cols(df)
    if len(d) < 3:
        return _neutral("FVG=N/A")
    h, l = d["high"], d["low"]
    bull_fvg = bear_fvg = False
    for i in range(len(d) - 1, max(len(d) - lookback - 1, 1), -1):
        if l.iloc[i] > h.iloc[i - 2]:
            bull_fvg = True
            break
        if h.iloc[i] < l.iloc[i - 2]:
            bear_fvg = True
            break
    if bull_fvg:
        return IndicatorState(1, 1.0, "FVG: bullish imbalance present (BULLISH)")
    if bear_fvg:
        return IndicatorState(-1, -1.0, "FVG: bearish imbalance present (BEARISH)")
    return IndicatorState(0, 0.0, "FVG: no recent imbalance (NEUTRAL)")


def ind_bos(df: pd.DataFrame, swing_period: int = 10) -> IndicatorState:
    """Break of Structure: recent swing high/low broken with confirmation."""
    d = _cols(df)
    if len(d) < swing_period * 2:
        return _neutral("BOS=N/A")
    h, l, c = d["high"], d["low"], d["close"]
    swing_high = h.iloc[-swing_period - 5 : -5].max()
    swing_low  = l.iloc[-swing_period - 5 : -5].min()
    price = float(c.iloc[-1])
    if price > float(swing_high):
        return IndicatorState(1, price, f"BOS: broke swing high {swing_high:.5f} (BULLISH)")
    if price < float(swing_low):
        return IndicatorState(-1, price, f"BOS: broke swing low {swing_low:.5f} (BEARISH)")
    return IndicatorState(0, price, f"BOS: within swing range (NEUTRAL)")


def ind_session_hl(df: pd.DataFrame, asian_period: int = 5) -> IndicatorState:
    """Price vs recent session high/low (proxy: last N bars as 'Asian range')."""
    d = _cols(df)
    if len(d) < asian_period + 5:
        return _neutral("Session HL=N/A")
    # Use bars [−10 : −5] as "prior session" and last bar as current
    prior = d.iloc[-10:-5]
    session_high = float(prior["high"].max())
    session_low  = float(prior["low"].min())
    price = float(d["close"].iloc[-1])
    if price > session_high:
        return IndicatorState(1, price, f"Above session high {session_high:.5f} (BULLISH)")
    if price < session_low:
        return IndicatorState(-1, price, f"Below session low {session_low:.5f} (BEARISH)")
    return IndicatorState(0, price, f"Inside session range (NEUTRAL)")


def ind_displacement(df: pd.DataFrame, atr_mult: float = 2.0) -> IndicatorState:
    """Large body candle (> 2×ATR body): displacement = momentum confirmation."""
    d = _cols(df)
    if len(d) < 20:
        return _neutral("Displacement=N/A")
    body = (d["close"] - d["open"]).abs()
    atr  = _atr(d, 14)
    disp = body / atr.replace(0, np.nan)
    v = float(disp.iloc[-1])
    direction = float(d["close"].iloc[-1]) - float(d["open"].iloc[-1])
    if math.isnan(v) or v < atr_mult:
        return IndicatorState(0, v if not math.isnan(v) else 0.0, f"Displacement={v:.2f}×ATR (NEUTRAL)")
    if direction > 0:
        return IndicatorState(1, v, f"Displacement={v:.2f}×ATR up (BULLISH)")
    return IndicatorState(-1, v, f"Displacement={v:.2f}×ATR down (BEARISH)")


def ind_liq_sweep(df: pd.DataFrame, lookback: int = 10, confirm_bars: int = 3) -> IndicatorState:
    """Liquidity sweep: wick beyond session extreme with reversal."""
    d = _cols(df)
    if len(d) < lookback + confirm_bars + 2:
        return _neutral("LiqSweep=N/A")
    prior = d.iloc[-(lookback + confirm_bars + 2):-(confirm_bars + 2)]
    sweep_high = float(prior["high"].max())
    sweep_low  = float(prior["low"].min())
    recent = d.iloc[-(confirm_bars + 2):]
    swept_high = (recent["high"] > sweep_high).any()
    swept_low  = (recent["low"]  < sweep_low).any()
    price = float(d["close"].iloc[-1])
    if swept_high and price < sweep_high:
        return IndicatorState(-1, price, f"Swept high {sweep_high:.5f} then reversed (BEARISH)")
    if swept_low and price > sweep_low:
        return IndicatorState(1, price, f"Swept low {sweep_low:.5f} then reversed (BULLISH)")
    return IndicatorState(0, price, "No sweep detected (NEUTRAL)")


def ind_order_block(df: pd.DataFrame, lookback: int = 20) -> IndicatorState:
    """Order block: last opposing candle before recent BOS, price returning to it."""
    d = _cols(df)
    if len(d) < lookback + 5:
        return _neutral("OB=N/A")
    h, l, o, c = d["high"], d["low"], d["open"], d["close"]
    swing_high = float(h.iloc[-lookback:-5].max())
    swing_low  = float(l.iloc[-lookback:-5].min())
    price = float(c.iloc[-1])
    # Find last bearish candle before bullish BOS
    bos_bull = price > swing_high
    bos_bear = price < swing_low
    if bos_bull:
        # Look for last bearish OB in lookback window
        for i in range(len(d) - 5, len(d) - lookback, -1):
            if float(c.iloc[i]) < float(o.iloc[i]):
                ob_top = float(h.iloc[i])
                ob_bot = float(l.iloc[i])
                if ob_bot < price < ob_top * 1.002:
                    return IndicatorState(1, price, f"OB bullish {ob_bot:.5f}-{ob_top:.5f} (BULLISH)")
                break
    if bos_bear:
        for i in range(len(d) - 5, len(d) - lookback, -1):
            if float(c.iloc[i]) > float(o.iloc[i]):
                ob_top = float(h.iloc[i])
                ob_bot = float(l.iloc[i])
                if ob_bot * 0.998 < price < ob_top:
                    return IndicatorState(-1, price, f"OB bearish {ob_bot:.5f}-{ob_top:.5f} (BEARISH)")
                break
    return IndicatorState(0, price, "No order block confluence (NEUTRAL)")


# ─── Master compute function ──────────────────────────────────────────────────

INDICATOR_NAMES = [
    # TREND
    "adx", "ema_cross", "supertrend", "ichimoku", "parabolic_sar", "aroon", "vwap_dev",
    # MOMENTUM
    "rsi", "macd", "stochastic", "cci", "williams_r", "roc", "mfi",
    # VOLATILITY
    "bollinger_bands", "atr_ratio", "keltner", "donchian", "hist_vol",
    # VOLUME
    "obv", "vol_trend", "vwap_ratio", "acc_dist", "cmf",
    # MARKET STRUCTURE
    "fvg", "bos", "session_hl", "displacement", "liq_sweep", "order_block",
]

_COMPUTE_MAP = {
    "adx":           ind_adx,
    "ema_cross":     ind_ema_cross,
    "supertrend":    ind_supertrend,
    "ichimoku":      ind_ichimoku,
    "parabolic_sar": ind_parabolic_sar,
    "aroon":         ind_aroon,
    "vwap_dev":      ind_vwap_dev,
    "rsi":           ind_rsi,
    "macd":          ind_macd,
    "stochastic":    ind_stochastic,
    "cci":           ind_cci,
    "williams_r":    ind_williams_r,
    "roc":           ind_roc,
    "mfi":           ind_mfi,
    "bollinger_bands": ind_bollinger_bands,
    "atr_ratio":     ind_atr_ratio,
    "keltner":       ind_keltner,
    "donchian":      ind_donchian,
    "hist_vol":      ind_hist_vol,
    "obv":           ind_obv,
    "vol_trend":     ind_vol_trend,
    "vwap_ratio":    ind_vwap_ratio,
    "acc_dist":      ind_acc_dist,
    "cmf":           ind_cmf,
    "fvg":           ind_fvg,
    "bos":           ind_bos,
    "session_hl":    ind_session_hl,
    "displacement":  ind_displacement,
    "liq_sweep":     ind_liq_sweep,
    "order_block":   ind_order_block,
}


def compute_all_indicators(df: pd.DataFrame) -> dict[str, IndicatorState]:
    """
    Run all 30 indicators on an OHLCV DataFrame.
    Accepts yfinance output (Open/High/Low/Close/Volume) or lowercase columns.
    Returns {name: IndicatorState} for all 30. Never raises.
    """
    result: dict[str, IndicatorState] = {}
    for name in INDICATOR_NAMES:
        try:
            result[name] = _COMPUTE_MAP[name](df)
        except Exception:
            result[name] = _neutral(f"{name}=error")
    return result
