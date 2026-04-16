"""
Universe Sweep — 4 strategies × 46 assets, 2020-2024

Part 1  UNIVERSE SWEEP — strategy × asset backtest table
Part 2  FAILURE MAP    — per-losing-trade records with regime/ATR/market context
Part 3  CLUSTERING     — KMeans(k=5) on failure conditions → avoid-condition labels

Run:  python3 universe_sweep.py

Output:
    logs/universe_backtest_results.json
    logs/failure_map.csv
    logs/failure_clusters.json
"""

from __future__ import annotations

import json
import time
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Output directory ─────────────────────────────────────────────────────────

LOGS = Path("logs")

# ── Asset universe (46 symbols) ──────────────────────────────────────────────

ASSET_GROUPS: Dict[str, List[str]] = {
    "TECH":          ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD", "AVGO", "ASML"],
    "FINANCE":       ["JPM", "GS", "BAC", "MS", "BLK", "V", "MA"],
    "HEALTH":        ["UNH", "JNJ", "PFE", "ABBV", "LLY", "MRK"],
    "ENERGY":        ["XOM", "CVX", "OXY", "SLB"],
    "MACRO_ETF":     ["SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "USO", "UUP"],
    "VOLATILITY":    ["VXX"],
    "INTERNATIONAL": ["EEM", "EFA", "FXI"],
    "SECTOR":        ["XLF", "XLK", "XLE", "XLV", "XLU"],
}
ALL_ASSETS: List[str] = [s for grp in ASSET_GROUPS.values() for s in grp]
SECTOR_MAP: Dict[str, str] = {s: grp for grp, syms in ASSET_GROUPS.items() for s in syms}

START = "2020-01-01"
END   = "2024-12-31"

# ── Default sweep params ──────────────────────────────────────────────────────

DEFAULT_STOP_MULT = 2.0
DEFAULT_TP_RR     = 2.0
DEFAULT_ATR_PERIOD = 14
DEFAULT_COMMISSION = 2.5
DEFAULT_SLIPPAGE   = 0.0001
INITIAL_CAPITAL    = 50_000.0


# ═════════════════════════════════════════════════════════════════════════════
# Vectorized indicators
# ═════════════════════════════════════════════════════════════════════════════

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    out = arr.copy()
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rolling_hurst(closes: np.ndarray, window: int = 63) -> np.ndarray:
    """Vectorized Hurst approximation via variance ratio method.

    Uses numpy stride tricks — no Python loop over bars.
    H > 0.55 → MOMENTUM   (trending)
    H < 0.45 → REVERSION  (mean-reverting)
    else     → NEUTRAL
    """
    from numpy.lib.stride_tricks import as_strided

    n = len(closes)
    hurst = np.full(n, 0.5, dtype=np.float32)
    if n < window + 4:
        return hurst

    closes_f = closes.astype(np.float64)
    returns = np.log(closes_f[1:] / (closes_f[:-1] + 1e-10))  # length n-1

    # Rolling windows of 1-period returns
    n_w = len(returns) - window + 1         # n - window
    strd = returns.strides[0]
    w1 = as_strided(returns, shape=(n_w, window), strides=(strd, strd))
    m1 = w1.mean(axis=1, keepdims=True)
    var1 = ((w1 - m1) ** 2).mean(axis=1)   # shape (n_w,)

    # 2-period overlapping returns
    returns2 = returns[:-1] + returns[1:]   # length n-2
    n_w2 = len(returns2) - window + 1       # n - window - 1
    if n_w2 < 1:
        return hurst
    strd2 = returns2.strides[0]
    w2 = as_strided(returns2, shape=(n_w2, window), strides=(strd2, strd2))
    m2 = w2.mean(axis=1, keepdims=True)
    var2 = ((w2 - m2) ** 2).mean(axis=1)   # shape (n_w2,)

    min_len = min(n_w, n_w2)
    ratio = var2[:min_len] / (2.0 * var1[:min_len] + 1e-12)
    ratio = np.clip(ratio, 1e-6, 100.0)
    h = 0.5 + 0.5 * np.log(ratio) / np.log(2.0)
    h = np.clip(h, 0.1, 0.9).astype(np.float32)

    end_idx = window + min_len
    hurst[window:end_idx] = h[: end_idx - window]
    return hurst


def _atr_series(highs, lows, closes, period=14):
    n = len(closes)
    tr = np.empty(n, dtype=np.float32)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    return _ema(tr, period).astype(np.float32)


def _hurst_label(h: float) -> str:
    if h > 0.55:
        return "MOMENTUM"
    if h < 0.45:
        return "REVERSION"
    return "NEUTRAL"


# ═════════════════════════════════════════════════════════════════════════════
# 4 Strategy signal generators
# Each returns (signals: np.int8[n], confidence: np.float32[n])
# ═════════════════════════════════════════════════════════════════════════════

def momentum_sma_signals(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volume: np.ndarray, hurst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """EMA(9/21) crossover — gated to trending regime (Hurst > 0.52)."""
    fast = _ema(closes, 9)
    slow = _ema(closes, 21)
    diff = fast - slow
    prev = np.roll(diff, 1); prev[0] = diff[0]

    sigs = np.zeros(len(closes), dtype=np.int8)
    sigs[(diff > 0) & (prev <= 0)] = 1
    sigs[(diff < 0) & (prev >= 0)] = -1
    sigs[hurst < 0.52] = 0                          # regime gate

    conf = np.clip(np.abs(diff) / (slow + 1e-10) * 500.0, 0, 1).astype(np.float32)
    sigs[:21] = 0
    return sigs, conf


def bb_reversion_signals(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volume: np.ndarray, hurst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bollinger Band reversion — gated to mean-reverting regime (Hurst < 0.48)."""
    period = 20
    n = len(closes)
    cs = np.cumsum(closes)
    cs2 = np.cumsum(closes ** 2)
    roll_sum = cs[period:] - cs[:-period]
    roll_sum2 = cs2[period:] - cs2[:-period]
    mid = np.empty(n); mid[:period] = closes[:period].mean()
    mid[period:] = roll_sum / period
    var = roll_sum2 / period - (roll_sum / period) ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    std_full = np.empty(n); std_full[:period] = std[0] if len(std) else 0.0
    std_full[period:] = std

    upper = mid + 2.0 * std_full
    lower = mid - 2.0 * std_full

    prev_c = np.roll(closes, 1); prev_c[0] = closes[0]
    prev_l = np.roll(lower, 1);  prev_l[0] = lower[0]
    prev_u = np.roll(upper, 1);  prev_u[0] = upper[0]

    sigs = np.zeros(n, dtype=np.int8)
    sigs[(closes < lower) & (prev_c >= prev_l)] = 1   # bounce long
    sigs[(closes > upper) & (prev_c <= prev_u)] = -1  # fade short
    sigs[hurst > 0.48] = 0                             # regime gate

    z = (closes - mid) / (std_full + 1e-10)
    conf = np.clip(np.abs(z) / 3.0, 0, 1).astype(np.float32)
    sigs[:period] = 0
    return sigs, conf


def donchian_breakout_signals(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volume: np.ndarray, hurst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Donchian channel breakout (20-day high/low). No regime gate — works across regimes."""
    period = 20
    n = len(closes)
    sigs = np.zeros(n, dtype=np.int8)

    # Rolling max/min via cumulative approach
    for i in range(period, n):
        w = closes[i - period: i]
        if closes[i] > w.max():
            sigs[i] = 1
        elif closes[i] < w.min():
            sigs[i] = -1

    # Confidence: volume surge relative to 20-day average
    vol_ma = np.ones(n, dtype=np.float64)
    for i in range(period, n):
        vol_ma[i] = volume[i - period: i].mean()
    conf = np.clip(volume / (vol_ma + 1e-10) / 3.0, 0, 1).astype(np.float32)
    return sigs, conf


def atr_channel_signals(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volume: np.ndarray, hurst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """EMA(50) ± 1.5×ATR(14) channel breakout — gated to non-mean-reverting (Hurst > 0.50)."""
    ema50 = _ema(closes, 50)
    atr   = _atr_series(highs, lows, closes, 14)
    upper = ema50 + 1.5 * atr
    lower = ema50 - 1.5 * atr

    prev_c = np.roll(closes, 1); prev_c[0] = closes[0]
    prev_u = np.roll(upper, 1);  prev_u[0] = upper[0]
    prev_l = np.roll(lower, 1);  prev_l[0] = lower[0]

    sigs = np.zeros(len(closes), dtype=np.int8)
    sigs[(closes > upper) & (prev_c <= prev_u)] = 1
    sigs[(closes < lower) & (prev_c >= prev_l)] = -1
    sigs[hurst < 0.50] = 0                           # regime gate

    conf = np.full(len(closes), 0.65, dtype=np.float32)
    sigs[:50] = 0
    return sigs, conf


STRATEGIES = [
    ("momentum_sma",      momentum_sma_signals),
    ("bb_reversion",      bb_reversion_signals),
    ("donchian_breakout", donchian_breakout_signals),
    ("atr_channel",       atr_channel_signals),
]


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def _download_all(tickers: List[str], spy_weekly_returns: Dict[str, float]) -> Dict[str, dict]:
    """Bulk yfinance download → dict of prepared per-asset data."""
    print(f"[Data] Downloading {len(tickers)} assets from yfinance ({START} → {END})...")
    t0 = time.perf_counter()

    # Do NOT pass group_by="ticker" — that flips the MultiIndex to (Ticker, Price).
    # Default gives (Price, Ticker), which xs(sym, level=1) handles correctly.
    raw = yf.download(
        tickers, start=START, end=END,
        auto_adjust=True, progress=False,
    )

    elapsed = time.perf_counter() - t0
    print(f"[Data] Download complete in {elapsed:.1f}s")

    asset_data: Dict[str, dict] = {}

    for sym in tickers:
        try:
            # yfinance ≥ 0.2 always returns (Field, Symbol) MultiIndex
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw.xs(sym, axis=1, level=1).copy()
            else:
                df = raw[sym].copy() if len(tickers) > 1 else raw.copy()
            df.columns = [str(c).lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            if len(df) < 100:
                continue

            closes = df["close"].to_numpy(dtype=np.float64)
            highs  = df["high"].to_numpy(dtype=np.float64)
            lows   = df["low"].to_numpy(dtype=np.float64)
            volume = df["volume"].fillna(1e6).to_numpy(dtype=np.float64)
            dates  = df.index  # DatetimeIndex

            hurst   = _rolling_hurst(closes)
            atr_arr = _atr_series(highs.astype(np.float32),
                                   lows.astype(np.float32),
                                   closes.astype(np.float32))
            atr_pct = (atr_arr / (closes.astype(np.float32) + 1e-8)).astype(np.float32)

            # SPY weekly return aligned to each bar's date
            spy_weekly_at_bar = np.zeros(len(dates), dtype=np.float32)
            for k, d in enumerate(dates):
                key = str(d.date())
                spy_weekly_at_bar[k] = spy_weekly_returns.get(key, 0.0)

            asset_data[sym] = {
                "df":              df,
                "closes":          closes,
                "highs":           highs,
                "lows":            lows,
                "volume":          volume,
                "dates":           dates,
                "hurst":           hurst,
                "atr_pct":         atr_pct,
                "spy_weekly":      spy_weekly_at_bar,
            }
        except Exception as e:
            print(f"  [warn] {sym}: {e}")

    print(f"[Data] Prepared {len(asset_data)}/{len(tickers)} assets")
    return asset_data


def _build_spy_weekly(spy_df: pd.DataFrame) -> Dict[str, float]:
    """Map each calendar date → SPY return for the trailing 5 trading days."""
    closes = spy_df["close"].astype(np.float64)
    weekly_ret = closes.pct_change(5).fillna(0.0)
    return {str(d.date()): float(r) for d, r in zip(spy_df.index, weekly_ret)}


# ═════════════════════════════════════════════════════════════════════════════
# Multiprocessing worker
# ═════════════════════════════════════════════════════════════════════════════

_GLOBAL_DATA: Dict[str, dict] = {}   # inherited via fork


def _init_worker(data: Dict[str, dict]) -> None:
    global _GLOBAL_DATA
    _GLOBAL_DATA = data


def _combo_worker(args: tuple) -> Tuple[dict, List[dict]]:
    """Run one strategy × asset combo. Returns (summary_dict, failure_records)."""
    symbol, strategy_name, strat_idx = args

    import sys
    sys.path.insert(0, ".")
    from backtest.fast_engine import FastBacktestEngine, SweepParams

    d = _GLOBAL_DATA.get(symbol)
    if d is None:
        return {}, []

    closes  = d["closes"]
    highs   = d["highs"]
    lows    = d["lows"]
    volume  = d["volume"]
    hurst   = d["hurst"]
    dates   = d["dates"]
    atr_pct = d["atr_pct"]
    spy_w   = d["spy_weekly"]

    strat_fns = [
        momentum_sma_signals,
        bb_reversion_signals,
        donchian_breakout_signals,
        atr_channel_signals,
    ]
    sig_fn = strat_fns[strat_idx]

    try:
        signals, confidence = sig_fn(closes, highs, lows, volume, hurst)
    except Exception as e:
        return {}, []

    if signals.sum() == 0 and (signals == 0).all():
        return {}, []

    try:
        engine = FastBacktestEngine.from_signals(
            d["df"], signals, confidence
        )
        params = SweepParams(
            stop_atr_mult=DEFAULT_STOP_MULT,
            tp_rr=DEFAULT_TP_RR,
            atr_period=DEFAULT_ATR_PERIOD,
            commission_per_side=DEFAULT_COMMISSION,
            slippage_pct=DEFAULT_SLIPPAGE,
            initial_capital=INITIAL_CAPITAL,
        )
        res = engine.run_detailed(params)
    except Exception as e:
        return {}, []

    tc = res["trade_count"]
    if tc == 0:
        return {}, []

    entry_idx = res["entry_idx"]
    pnl_arr   = res["pnl"]
    pnl_r     = res["pnl_r"]
    dir_arr   = res["direction"]
    is_stop   = res["is_stop"]

    # ── Best regime ──────────────────────────────────────────────────────────
    regime_pnl: Dict[str, float] = {}
    for i in range(tc):
        ei = entry_idx[i]
        if ei < len(hurst):
            lbl = _hurst_label(float(hurst[ei]))
            regime_pnl[lbl] = regime_pnl.get(lbl, 0.0) + float(pnl_arr[i])
    best_regime = max(regime_pnl, key=regime_pnl.get) if regime_pnl else "NEUTRAL"

    # ── Best/worst year ───────────────────────────────────────────────────────
    year_pnl: Dict[int, List[float]] = {}
    for i in range(tc):
        ei = entry_idx[i]
        if ei < len(dates):
            yr = int(dates[ei].year)
            year_pnl.setdefault(yr, []).append(float(pnl_arr[i]))

    year_sharpe: Dict[int, float] = {}
    for yr, pnls in year_pnl.items():
        arr = np.array(pnls)
        std = arr.std()
        year_sharpe[yr] = float((arr.mean() / (std + 1e-8)) * np.sqrt(252)) if std > 0 else 0.0

    best_year  = max(year_sharpe, key=year_sharpe.get) if year_sharpe else None
    worst_year = min(year_sharpe, key=year_sharpe.get) if year_sharpe else None

    # ── Trades per year ───────────────────────────────────────────────────────
    n_years = (dates[-1].year - dates[0].year + 1) if len(dates) > 1 else 1
    trades_per_year = tc / max(n_years, 1)

    # ── avg_r ─────────────────────────────────────────────────────────────────
    avg_r = float(np.mean(pnl_r)) if tc > 0 else 0.0

    # ── Sharpe (annualized from trade-level PnL) ──────────────────────────────
    pnl_s = pnl_arr.astype(np.float64)
    sharpe = 0.0
    if len(pnl_s) > 1 and pnl_s.std() > 0:
        sharpe = float((pnl_s.mean() / pnl_s.std()) * np.sqrt(252))

    summary = {
        "strategy":         strategy_name,
        "asset":            symbol,
        "sector":           SECTOR_MAP.get(symbol, "UNKNOWN"),
        "total_pnl":        round(float(res["total_pnl"]), 2),
        "total_return_pct": round(float(res["total_return_pct"]), 3),
        "trade_count":      tc,
        "win_rate":         round(float(res["win_rate"]), 4),
        "profit_factor":    round(float(res["profit_factor"]), 4),
        "sharpe":           round(sharpe, 4),
        "max_drawdown_pct": round(float(res["max_drawdown_pct"]), 3),
        "avg_r":            round(avg_r, 4),
        "trades_per_year":  round(trades_per_year, 1),
        "best_regime":      best_regime,
        "best_year":        best_year,
        "worst_year":       worst_year,
    }

    # ── Failure records (losing trades only) ─────────────────────────────────
    failure_records: List[dict] = []
    for i in range(tc):
        if pnl_arr[i] >= 0:
            continue
        ei = entry_idx[i]
        if ei >= len(dates):
            continue
        hurst_val = float(hurst[ei]) if ei < len(hurst) else 0.5
        atr_p     = float(atr_pct[ei]) if ei < len(atr_pct) else 0.0
        spy_ret   = float(spy_w[ei]) if ei < len(spy_w) else 0.0
        loss_r    = float(pnl_r[i])

        failure_records.append({
            "date":             str(dates[ei].date()),
            "asset":            symbol,
            "sector":           SECTOR_MAP.get(symbol, "UNKNOWN"),
            "strategy":         strategy_name,
            "regime_at_entry":  _hurst_label(hurst_val),
            "hurst":            round(hurst_val, 3),
            "atr_pct":          round(atr_p * 100, 4),   # as percent
            "market_condition": round(spy_ret * 100, 4),  # SPY 5-day % return
            "loss_r":           round(loss_r, 4),
            "direction":        "LONG" if dir_arr[i] == 1 else "SHORT",
        })

    return summary, failure_records


# ═════════════════════════════════════════════════════════════════════════════
# Part 3 — KMeans failure clustering
# ═════════════════════════════════════════════════════════════════════════════

def _cluster_failures(failure_map: pd.DataFrame, k: int = 5) -> Tuple[pd.DataFrame, List[dict]]:
    """Fit KMeans(k=5) on failure conditions. Return labelled df + cluster descriptions."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans

    df = failure_map.copy()

    le_regime   = LabelEncoder().fit(["MOMENTUM", "REVERSION", "NEUTRAL"])
    le_strategy = LabelEncoder().fit([s for s, _ in STRATEGIES])

    df["regime_enc"]   = le_regime.transform(df["regime_at_entry"].fillna("NEUTRAL"))
    df["strategy_enc"] = le_strategy.transform(df["strategy"].fillna("momentum_sma"))

    features = ["regime_enc", "atr_pct", "market_condition", "strategy_enc"]
    X = df[features].fillna(0).to_numpy(dtype=np.float32)

    # Normalise each feature to [0, 1] range for equal weighting
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    col_range = col_max - col_min + 1e-8
    X_norm = (X - col_min) / col_range

    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    df["failure_cluster"] = km.fit_predict(X_norm)

    cluster_descriptions: List[dict] = []
    strategy_names = [s for s, _ in STRATEGIES]

    for c in range(k):
        sub = df[df["failure_cluster"] == c]
        if len(sub) == 0:
            continue

        dom_regime   = sub["regime_at_entry"].mode()[0]
        dom_strategy = sub["strategy"].mode()[0]
        dom_assets   = sub["asset"].value_counts().head(3).index.tolist()
        mean_atr     = sub["atr_pct"].mean()
        mean_spy     = sub["market_condition"].mean()
        mean_loss_r  = sub["loss_r"].mean()
        size         = len(sub)

        # Plain-English avoid label
        conditions = []
        if dom_regime == "MOMENTUM":
            conditions.append("trending market")
        elif dom_regime == "REVERSION":
            conditions.append("mean-reverting market")
        else:
            conditions.append("choppy/neutral market")

        if mean_atr > df["atr_pct"].quantile(0.66):
            conditions.append("high ATR (volatility spike)")
        elif mean_atr < df["atr_pct"].quantile(0.33):
            conditions.append("low ATR (compressed range)")

        if mean_spy < -1.0:
            conditions.append("weak/falling SPY week")
        elif mean_spy > 1.0:
            conditions.append("strong/rising SPY week")

        avoid_label = (
            f"{dom_strategy} losses when: "
            + " + ".join(conditions)
            + f" → avg {mean_loss_r:.2f}R lost"
        )

        cluster_descriptions.append({
            "cluster":        c,
            "size":           size,
            "dominant_strategy": dom_strategy,
            "dominant_regime":   dom_regime,
            "mean_atr_pct":      round(float(mean_atr), 4),
            "mean_spy_week_pct": round(float(mean_spy), 4),
            "mean_loss_r":       round(float(mean_loss_r), 4),
            "assets_most_affected": dom_assets,
            "avoid_condition":   avoid_label,
        })

    return df.drop(columns=["regime_enc", "strategy_enc"]), cluster_descriptions


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()
    LOGS.mkdir(exist_ok=True)

    # ── Load SPY first for weekly return context ──────────────────────────────
    print("[Data] Fetching SPY for market condition baseline...")
    spy_raw = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
    # yfinance ≥ 0.2: always (Field, Symbol) MultiIndex, even for single tickers
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw = spy_raw.xs("SPY", axis=1, level=1)
    spy_raw.columns = [str(c).lower() for c in spy_raw.columns]
    spy_weekly_map = _build_spy_weekly(spy_raw)

    # ── Load all 46 assets ────────────────────────────────────────────────────
    asset_data = _download_all(ALL_ASSETS, spy_weekly_map)

    if not asset_data:
        print("[Error] No data loaded. Check API/internet connection.")
        return

    # ── Warm JIT (main process) so workers skip compilation ──────────────────
    print("[JIT] Warming Numba cache...")
    import sys; sys.path.insert(0, ".")
    from backtest.fast_engine import FastBacktestEngine, SweepParams
    sample_sym = next(iter(asset_data))
    sample_d   = asset_data[sample_sym]
    dummy_sigs = np.zeros(len(sample_d["closes"]), dtype=np.int8)
    dummy_sigs[30] = 1
    dummy_conf = np.ones(len(dummy_sigs), dtype=np.float32)
    eng_warm = FastBacktestEngine.from_signals(sample_d["df"], dummy_sigs, dummy_conf)
    eng_warm.warmup()
    print("[JIT] Cache warm.\n")

    # ── Build combos ──────────────────────────────────────────────────────────
    combos = [
        (sym, strat_name, strat_idx)
        for sym in asset_data
        for strat_idx, (strat_name, _) in enumerate(STRATEGIES)
    ]
    n_combos = len(combos)
    n_cores  = _physical_cores()
    print(f"[Sweep] {n_combos} combos ({len(asset_data)} assets × {len(STRATEGIES)} strategies) "
          f"across {n_cores} cores...")

    t1 = time.perf_counter()
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=n_cores,
        initializer=_init_worker,
        initargs=(asset_data,),
    ) as pool:
        raw_results = pool.map(_combo_worker, combos, chunksize=max(1, n_combos // (n_cores * 4)))

    sweep_elapsed = time.perf_counter() - t1

    # ── Unpack results ────────────────────────────────────────────────────────
    all_summaries: List[dict] = []
    all_failures:  List[dict] = []
    for summary, failures in raw_results:
        if summary:
            all_summaries.append(summary)
        all_failures.extend(failures)

    # ═══ Part 1: Save universe backtest results ═══════════════════════════════
    winning = [r for r in all_summaries if r.get("sharpe", 0) > 1.0]

    universe_output = {
        "meta": {
            "strategies": [s for s, _ in STRATEGIES],
            "assets":     list(asset_data.keys()),
            "start":      START,
            "end":        END,
            "total_combos":   n_combos,
            "combos_run":     len(all_summaries),
            "winning_combos": len(winning),
            "sweep_time_sec": round(sweep_elapsed, 2),
        },
        "results": sorted(all_summaries, key=lambda x: x.get("sharpe", 0), reverse=True),
    }

    out1 = LOGS / "universe_backtest_results.json"
    out1.write_text(json.dumps(universe_output, indent=2))

    # ═══ Part 2: Save failure map ═════════════════════════════════════════════
    out2 = LOGS / "failure_map.csv"
    if all_failures:
        fail_df = pd.DataFrame(all_failures)
        fail_df.to_csv(out2, index=False)
    else:
        fail_df = pd.DataFrame()
        out2.write_text("")

    # ═══ Part 3: Failure clustering ══════════════════════════════════════════
    out3_csv  = LOGS / "failure_map.csv"   # overwritten with cluster column
    out3_json = LOGS / "failure_clusters.json"

    cluster_descriptions: List[dict] = []
    if len(fail_df) >= 5:
        k = min(5, len(fail_df) // 3)
        fail_labelled, cluster_descriptions = _cluster_failures(fail_df, k=k)
        fail_labelled.to_csv(out3_csv, index=False)
        out3_json.write_text(json.dumps(cluster_descriptions, indent=2))
    else:
        out3_json.write_text("[]")

    # ═══ Final ranked table ═══════════════════════════════════════════════════
    total_elapsed = time.perf_counter() - t_start

    print("\n" + "═" * 78)
    print(" FINAL RANKED TABLE — strategy | best_asset | sharpe | win_rate | cluster_to_avoid")
    print("═" * 78)

    # Find best asset per strategy
    strat_best: Dict[str, dict] = {}
    for r in all_summaries:
        sn = r["strategy"]
        if sn not in strat_best or r["sharpe"] > strat_best[sn]["sharpe"]:
            strat_best[sn] = r

    # Find worst cluster per strategy
    strat_cluster: Dict[str, str] = {}
    for cd in cluster_descriptions:
        sn = cd["dominant_strategy"]
        if sn not in strat_cluster:
            strat_cluster[sn] = cd["avoid_condition"]

    header = f"{'Strategy':<22} {'Best Asset':<8} {'Sharpe':>7} {'WinRate':>8} {'Failure Cluster to Avoid'}"
    print(header)
    print("-" * 78)
    for sn, _ in STRATEGIES:
        b = strat_best.get(sn, {})
        avoid = strat_cluster.get(sn, "—")
        # Trim avoid string to fit terminal
        avoid_short = (avoid[:40] + "…") if len(avoid) > 41 else avoid
        print(
            f"{sn:<22} {b.get('asset','—'):<8} "
            f"{b.get('sharpe',0):>7.2f} "
            f"{b.get('win_rate',0):>8.1%} "
            f"{avoid_short}"
        )

    print("═" * 78)
    print(f"\nUNIVERSE SWEEP: complete — {len(winning)} winning combinations (sharpe > 1.0)")
    print(f"FAILURE MAP:    {len(all_failures)} losing trades logged → {out2}")
    print(f"CLUSTERS:       {len(cluster_descriptions)} failure conditions → {out3_json}")
    print(f"Total time:     {total_elapsed:.1f}s")


def _physical_cores() -> int:
    try:
        import subprocess
        out = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip()
        return int(out)
    except Exception:
        import os
        return os.cpu_count() or 4


if __name__ == "__main__":
    main()
