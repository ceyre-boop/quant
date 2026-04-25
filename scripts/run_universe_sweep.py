"""
Full Universe Sweep + Statistical Mapping
==========================================
Steps:
  1. Fetch 7 years of daily OHLCV via yfinance (parallel)
  2. Run 4 strategies × all assets × 25 param combos via fast_engine
  3. Build full_trade_map.csv (every trade, every condition)
  4. KMeans cluster wins (k=8) and losses (k=8)
  5. Build regime_inference_table.json
  6. AI pure-play lifetime analysis

Outputs:
  logs/full_universe_results.json
  logs/full_trade_map.csv
  logs/win_clusters.json
  logs/loss_clusters.json
  logs/regime_inference_table.json
  logs/ai_company_analysis.json
"""

from __future__ import annotations

import json
import sys
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

from backtest.fast_engine import FastBacktestEngine, SweepParams
from config.universe import UNIVERSE, AI_BUCKET

START = "2018-01-01"
END   = "2024-12-31"
N_CORES = min(mp.cpu_count(), 12)

STRATEGIES = ["momentum_sma", "donchian_breakout", "bb_reversion", "atr_channel"]

PARAM_GRID = {
    "stop_atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
    "tp_rr":         [1.5, 2.0, 2.5, 3.0, 4.0],
}  # 25 combos


# ── Signal generators ────────────────────────────────────────────────────────

def _sma_signals(df: pd.DataFrame, fast=9, slow=21):
    c = df["close"].to_numpy(np.float32)
    def sma(a, w):
        out = np.empty_like(a); cs = np.cumsum(a)
        out[:w] = a[:w].mean()
        out[w:] = (cs[w:] - cs[:-w]) / w
        return out
    fm, sm = sma(c, fast), sma(c, slow)
    d = fm - sm; pd_ = np.roll(d, 1); pd_[0] = d[0]
    sig = np.zeros(len(c), np.int8)
    sig[(d > 0) & (pd_ <= 0)] = 1
    sig[(d < 0) & (pd_ >= 0)] = -1
    slope = np.abs(d - pd_) / (sm + 1e-9)
    conf = np.clip(slope * 1000, 0, 1).astype(np.float32)
    return sig, conf


def _donchian_signals(df: pd.DataFrame, period=20):
    h = df["high"].to_numpy(np.float32)
    l = df["low"].to_numpy(np.float32)
    c = df["close"].to_numpy(np.float32)
    n = len(c)
    sig = np.zeros(n, np.int8)
    conf = np.full(n, 0.6, np.float32)
    for i in range(period, n):
        hi = h[i-period:i].max(); lo = l[i-period:i].min()
        if c[i] > hi:
            sig[i] = 1
        elif c[i] < lo:
            sig[i] = -1
    return sig, conf


def _bb_reversion_signals(df: pd.DataFrame, period=20, n_std=2.0):
    c = df["close"].to_numpy(np.float32)
    n = len(c)
    sig = np.zeros(n, np.int8)
    conf = np.full(n, 0.6, np.float32)
    for i in range(period, n):
        win = c[i-period:i]
        mid = win.mean(); std = win.std()
        upper = mid + n_std * std; lower = mid - n_std * std
        if c[i] <= lower:
            sig[i] = 1   # reversion long
        elif c[i] >= upper:
            sig[i] = -1  # reversion short
        if std > 0:
            dist = abs(c[i] - mid) / (n_std * std + 1e-9)
            conf[i] = float(np.clip(dist, 0, 1))
    return sig, conf


def _atr_channel_signals(df: pd.DataFrame, period=20, mult=2.0):
    c = df["close"].to_numpy(np.float32)
    h = df["high"].to_numpy(np.float32)
    lo = df["low"].to_numpy(np.float32)
    n = len(c)
    # Compute ATR
    atr = np.zeros(n, np.float32)
    for i in range(1, n):
        tr = max(h[i]-lo[i], abs(h[i]-c[i-1]), abs(lo[i]-c[i-1]))
        atr[i] = atr[i-1] * (1 - 1/period) + tr * (1/period) if i >= period else tr
    sig = np.zeros(n, np.int8)
    conf = np.full(n, 0.6, np.float32)
    for i in range(period, n):
        mid = c[i-period:i].mean()
        upper = mid + mult * atr[i]; lower = mid - mult * atr[i]
        if c[i] > upper:
            sig[i] = 1
        elif c[i] < lower:
            sig[i] = -1
        if atr[i] > 0:
            conf[i] = float(np.clip(abs(c[i] - mid) / (mult * atr[i] + 1e-9), 0, 1))
    return sig, conf


SIGNAL_FNS = {
    "momentum_sma":    _sma_signals,
    "donchian_breakout": _donchian_signals,
    "bb_reversion":    _bb_reversion_signals,
    "atr_channel":     _atr_channel_signals,
}


# ── Feature helpers ──────────────────────────────────────────────────────────

def _rolling_hurst(closes: np.ndarray, window: int = 60) -> np.ndarray:
    """Fast R/S Hurst using log returns, rolling window."""
    n = len(closes)
    out = np.full(n, 0.5, np.float64)
    log_c = np.log(np.maximum(closes, 1e-9))
    for i in range(window, n):
        seg = log_c[i-window:i]
        ret = np.diff(seg)
        if len(ret) < 4:
            continue
        mean_r = ret.mean()
        dev = np.cumsum(ret - mean_r)
        r = dev.max() - dev.min()
        s = ret.std()
        if s < 1e-12:
            continue
        rs = r / s
        if rs <= 0:
            continue
        out[i] = np.log(rs) / np.log(window)
    return out.astype(np.float32)


def _rolling_adx(h: np.ndarray, lo: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorised ADX. Inputs must be 1-D float64 arrays."""
    h  = np.asarray(h,  dtype=np.float64).ravel()
    lo = np.asarray(lo, dtype=np.float64).ravel()
    c  = np.asarray(c,  dtype=np.float64).ravel()
    n  = len(c)
    adx_out = np.full(n, 25.0, np.float32)
    if n < period * 2 + 1:
        return adx_out

    # True range (vectorised)
    prev_c = np.empty(n); prev_c[0] = c[0]; prev_c[1:] = c[:-1]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c)))

    # Directional movement (vectorised)
    prev_h = np.empty(n); prev_h[0] = h[0]; prev_h[1:] = h[:-1]
    prev_l = np.empty(n); prev_l[0] = lo[0]; prev_l[1:] = lo[:-1]
    up = h - prev_h; dn = prev_l - lo
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)

    # Wilder smooth (loop only over scalar sums — still fast)
    atr_w = np.zeros(n); pdi = np.zeros(n); ndi = np.zeros(n)
    atr_w[period] = tr[1:period+1].sum()
    pdi[period]   = pdm[1:period+1].sum()
    ndi[period]   = ndm[1:period+1].sum()
    inv = 1.0 / period
    for i in range(period + 1, n):
        atr_w[i] = atr_w[i-1] - atr_w[i-1]*inv + tr[i]
        pdi[i]   = pdi[i-1]   - pdi[i-1]*inv   + pdm[i]
        ndi[i]   = ndi[i-1]   - ndi[i-1]*inv   + ndm[i]

    # DX (vectorised)
    safe_atr = np.where(atr_w > 0, atr_w, 1.0)
    pdi_pct  = 100.0 * pdi / safe_atr
    ndi_pct  = 100.0 * ndi / safe_atr
    denom    = pdi_pct + ndi_pct
    dx       = np.where(denom > 0, 100.0 * np.abs(pdi_pct - ndi_pct) / denom, 0.0)

    # Smooth into ADX
    adx_v = np.zeros(n)
    adx_v[period*2] = dx[period:period*2].mean()
    for i in range(period*2 + 1, n):
        adx_v[i] = (adx_v[i-1]*(period-1) + dx[i]) * inv
    adx_out[period*2:] = adx_v[period*2:].astype(np.float32)
    return adx_out


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_one(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
    import yfinance as yf
    try:
        # Use Ticker.history() — avoids yf.download() session-sharing bug under
        # ThreadPoolExecutor (different tickers can receive each other's cached data)
        ticker = yf.Ticker(sym)
        raw = ticker.history(start=START, end=END, auto_adjust=True)
        if raw.empty or len(raw) < 100:
            return sym, None
        # history() returns flat columns; lowercase and select OHLCV
        raw.columns = [str(c).lower() for c in raw.columns]
        df = raw[["open", "high", "low", "close", "volume"]].dropna()
        # Drop timezone so downstream numpy ops stay clean
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index)
        return sym, df
    except Exception:
        return sym, None


def fetch_all(universe: List[str]) -> Dict[str, pd.DataFrame]:
    print(f"[Data] Fetching {len(universe)} assets via yfinance ({START}→{END})...")
    data = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_fetch_one, s): s for s in universe}
        for fut in as_completed(futs):
            sym, df = fut.result()
            if df is not None:
                data[sym] = df
                print(f"  {sym}: {len(df)} bars", end="\r", flush=True)
    print(f"\n[Data] Loaded {len(data)}/{len(universe)} assets")
    return data


# ── Per-asset sweep ──────────────────────────────────────────────────────────

def sweep_asset(sym: str, df: pd.DataFrame) -> Dict:
    """Run all 4 strategies × 25 param combos. Return best result per strategy."""
    results = {}
    for strat in STRATEGIES:
        sig_fn = SIGNAL_FNS[strat]
        try:
            engine = FastBacktestEngine.from_dataframe(df, signal_fn=sig_fn)
            rows = []
            for stop_m in PARAM_GRID["stop_atr_mult"]:
                for tp_r in PARAM_GRID["tp_rr"]:
                    p = SweepParams(stop_atr_mult=stop_m, tp_rr=tp_r, signal_min_confidence=0.0)
                    res = engine.run_single(p)
                    if res.trade_count >= 5:
                        rows.append(res.to_dict())
            if not rows:
                continue
            df_res = pd.DataFrame(rows).sort_values("profit_factor", ascending=False)
            best = df_res.iloc[0].to_dict()
            # Compute Sharpe from pnl curve — approximate from available stats
            # Sharpe ≈ (ann_return / ann_vol); proxy with pnl / (capital × max_dd + 1e-9)
            sharpe_proxy = best["total_return_pct"] / (best["max_dd_pct"] + 1.0)
            trades_py = best["trades"] / 7.0
            results[strat] = {
                "win_rate":        round(best["win_rate"], 4),
                "profit_factor":   round(best["profit_factor"], 4),
                "max_drawdown":    round(best["max_dd_pct"], 2),
                "total_return_pct": round(best["total_return_pct"], 2),
                "sharpe_proxy":    round(sharpe_proxy, 3),
                "trades_per_year": round(trades_py, 1),
                "best_param_combo": {
                    "stop_atr_mult": best["stop_atr_mult"],
                    "tp_rr":         best["tp_rr"],
                },
                "all_results_count": len(df_res),
            }
        except Exception as e:
            results[strat] = {"error": str(e)}
    return results


# ── Trade map builder ─────────────────────────────────────────────────────────

def build_trade_records(sym: str, df: pd.DataFrame,
                        spy_5d: pd.Series, vxx: pd.Series) -> List[dict]:
    """For each strategy, run detailed sim on best params, record every trade."""
    records = []
    closes = df["close"].to_numpy(np.float32)
    highs  = df["high"].to_numpy(np.float32)
    lows   = df["low"].to_numpy(np.float32)
    dates  = df.index

    closes_1d = closes.ravel().astype(np.float64)
    highs_1d  = highs.ravel().astype(np.float64)
    lows_1d   = lows.ravel().astype(np.float64)

    hurst = _rolling_hurst(closes_1d)
    adx   = _rolling_adx(highs_1d, lows_1d, closes_1d)

    # Wilder ATR-14 (vectorised)
    prev_c = np.empty(len(closes_1d)); prev_c[0] = closes_1d[0]; prev_c[1:] = closes_1d[:-1]
    tr_arr = np.maximum(highs_1d - lows_1d,
                        np.maximum(np.abs(highs_1d - prev_c), np.abs(lows_1d - prev_c)))
    atr14 = np.zeros(len(closes_1d), np.float32)
    for i in range(1, len(closes_1d)):
        atr14[i] = float(atr14[i-1]*0.929 + tr_arr[i]*0.071)

    closes = closes_1d.astype(np.float32)
    highs  = highs_1d.astype(np.float32)
    lows   = lows_1d.astype(np.float32)

    for strat in STRATEGIES:
        sig_fn = SIGNAL_FNS[strat]
        try:
            engine = FastBacktestEngine.from_dataframe(df, signal_fn=sig_fn)
            qualifying_params = []
            for stop_m in PARAM_GRID["stop_atr_mult"]:
                for tp_r in PARAM_GRID["tp_rr"]:
                    p = SweepParams(stop_atr_mult=stop_m, tp_rr=tp_r, signal_min_confidence=0.0)
                    res = engine.run_single(p)
                    if res.trade_count >= 5:
                        qualifying_params.append(p)
            if not qualifying_params:
                continue
        except Exception:
            continue

        # Run detailed sim for ALL qualifying param combos — gives statistical depth
        for params in qualifying_params:
            try:
                det = engine.run_detailed(params)
            except Exception:
                continue

            entry_idx = det["entry_idx"]
            exit_idx  = det["exit_idx"]
            pnl_r     = det["pnl_r"]
            direction = det["direction"]
            is_stop   = det["is_stop"]

            for k in range(len(entry_idx)):
                ei = int(entry_idx[k]); xi = int(exit_idx[k])
                if ei >= len(dates) or xi >= len(dates):
                    continue
                entry_date = dates[ei]
                h_val  = float(hurst[ei])
                adx_v  = float(adx[ei])
                ep     = float(closes[ei])
                atr_p  = float(atr14[ei] / ep * 100) if ep > 0 else 0.0
                dir_   = int(direction[k])

                # Regime
                if h_val > 0.52:
                    regime = "MOMENTUM"
                elif h_val < 0.45:
                    regime = "REVERSION"
                else:
                    regime = "FLAT"

                # SPY 5d return — align to nearest available date
                spy_ret = 0.0
                try:
                    spy_ret = float(spy_5d.asof(entry_date)) if len(spy_5d) > 0 else 0.0
                except Exception:
                    pass

                # VXX level
                vxx_lv = 0.0
                try:
                    vxx_lv = float(vxx.asof(entry_date)) if len(vxx) > 0 else 0.0
                except Exception:
                    pass

                # MFE / MAE in the trade window
                trade_slice_h = highs[ei:xi+1]
                trade_slice_l = lows[ei:xi+1]
                if len(trade_slice_h) == 0:
                    mfe_pct = mae_pct = 0.0
                else:
                    if dir_ == 1:
                        mfe_pct = float((trade_slice_h.max() - ep) / (ep+1e-9) * 100)
                        mae_pct = float((ep - trade_slice_l.min()) / (ep+1e-9) * 100)
                    else:
                        mfe_pct = float((ep - trade_slice_l.min()) / (ep+1e-9) * 100)
                        mae_pct = float((trade_slice_h.max() - ep) / (ep+1e-9) * 100)
                mae_mfe_ratio = mae_pct / (mfe_pct + 1e-9)

                r_val = float(pnl_r[k])
                result = "WIN" if r_val > 0 else ("LOSS" if int(is_stop[k]) == 1 else "TIMEOUT")

                records.append({
                    "date":          entry_date.strftime("%Y-%m-%d"),
                    "asset":         sym,
                    "strategy":      strat,
                    "direction":     "LONG" if dir_ == 1 else "SHORT",
                    "regime":        regime,
                    "hurst_at_entry": round(h_val, 4),
                    "atr_pct":       round(atr_p, 4),
                    "adx_at_entry":  round(adx_v, 2),
                    "spy_5d_return": round(spy_ret, 5),
                    "vix_level":     round(vxx_lv, 2),
                    "entry_r":       0.0,
                    "exit_r":        round(r_val, 4),
                    "result":        result,
                    "hold_bars":     xi - ei,
                    "mfe_pct":       round(mfe_pct, 4),
                    "mae_pct":       round(mae_pct, 4),
                    "mae_mfe_ratio": round(mae_mfe_ratio, 4),
                    "year":          entry_date.year,
                    "month":         entry_date.month,
                    "day_of_week":   entry_date.dayofweek,
                    "hour_of_day":   0,
                })
    return records


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_trades(trade_df: pd.DataFrame, k: int = 8, subset: str = "LOSS") -> List[dict]:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans

    mask = trade_df["result"] == ("WIN" if subset == "WIN" else "LOSS")
    sub  = trade_df[mask].copy()
    if len(sub) < k * 5:
        return []

    le_r = LabelEncoder(); le_s = LabelEncoder()
    sub["regime_enc"]   = le_r.fit_transform(sub["regime"].fillna("FLAT"))
    sub["strategy_enc"] = le_s.fit_transform(sub["strategy"].fillna("momentum_sma"))

    feat_cols = ["regime_enc","hurst_at_entry","atr_pct","adx_at_entry",
                 "spy_5d_return","mae_mfe_ratio","strategy_enc"]
    X = sub[feat_cols].fillna(0).to_numpy()

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    sub = sub.copy(); sub["cluster"] = km.fit_predict(X)

    clusters = []
    for cid in range(k):
        c = sub[sub["cluster"] == cid]
        if len(c) == 0:
            continue

        dom_strat  = c["strategy"].value_counts().index[0]
        dom_regime = c["regime"].value_counts().index[0]
        avg_hurst  = c["hurst_at_entry"].mean()
        avg_atr    = c["atr_pct"].mean()
        avg_spy    = c["spy_5d_return"].mean()
        avg_r      = c["exit_r"].mean()

        # Best/worst assets by win rate within cluster
        asset_wr = c.groupby("asset").apply(
            lambda x: (x["result"]=="WIN").mean()
        ).sort_values(ascending=False)
        best_assets  = list(asset_wr.head(3).index)
        worst_assets = list(asset_wr.tail(3).index)

        # Peak years
        year_counts = c["year"].value_counts().head(3)
        peak_years  = list(year_counts.index.astype(int))

        # Plain-english label
        sma_r   = f"h={avg_hurst:.2f}"
        atr_l   = "low-ATR" if avg_atr < 1.0 else ("high-ATR" if avg_atr > 2.5 else "med-ATR")
        spy_l   = "down-SPY" if avg_spy < -0.01 else ("up-SPY" if avg_spy > 0.01 else "flat-SPY")
        label   = f"{dom_regime}+{dom_strat}+{atr_l}+{spy_l} → avg {avg_r:.2f}R"

        clusters.append({
            "cluster_id":        cid,
            "size":              int(len(c)),
            "avg_r":             round(avg_r, 3),
            "dominant_strategy": dom_strat,
            "dominant_regime":   dom_regime,
            "avg_hurst":         round(avg_hurst, 3),
            "avg_atr_pct":       round(avg_atr, 3),
            "avg_spy_return":    round(avg_spy, 5),
            "label":             label,
            "best_assets":       best_assets,
            "worst_assets":      worst_assets,
            "peak_years":        peak_years,
        })

    clusters.sort(key=lambda x: x["avg_r"])
    return clusters


# ── Inference table ───────────────────────────────────────────────────────────

def build_inference_table(trade_df: pd.DataFrame) -> List[dict]:
    df = trade_df.copy()
    df["atr_bucket"] = pd.cut(df["atr_pct"], bins=[0, 1.0, 2.5, 999],
                              labels=["low", "med", "high"])
    df["spy_trend"]  = pd.cut(df["spy_5d_return"],
                              bins=[-999, -0.01, 0.01, 999],
                              labels=["down", "flat", "up"])
    df["is_win"] = (df["result"] == "WIN").astype(int)

    rows = []
    for regime in ["MOMENTUM","REVERSION","FLAT"]:
        for strat in STRATEGIES:
            for atr_b in ["low","med","high"]:
                for spy_t in ["down","flat","up"]:
                    mask = ((df["regime"] == regime) &
                            (df["strategy"] == strat) &
                            (df["atr_bucket"] == atr_b) &
                            (df["spy_trend"] == spy_t))
                    sub = df[mask]
                    n = len(sub)
                    if n == 0:
                        continue
                    wr  = float(sub["is_win"].mean())
                    wins_r  = sub.loc[sub["is_win"]==1, "exit_r"]
                    loss_r  = sub.loc[sub["is_win"]==0, "exit_r"]
                    gp = wins_r.sum() if len(wins_r) > 0 else 0.0
                    gl = abs(loss_r.sum()) if len(loss_r) > 0 else 0.0
                    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
                    recommended = bool(pf > 1.5 and n >= 30)
                    avoid       = bool(pf < 0.8 and n >= 30)
                    rows.append({
                        "regime":       regime,
                        "strategy":     strat,
                        "atr_bucket":   atr_b,
                        "spy_trend":    spy_t,
                        "historical_win_rate":      round(wr, 4),
                        "historical_profit_factor": round(pf, 4),
                        "sample_size":  n,
                        "recommended":  recommended,
                        "avoid":        avoid,
                    })
    return rows


# ── AI bucket analysis ────────────────────────────────────────────────────────

def ai_bucket_analysis(ai_data: Dict[str, pd.DataFrame],
                        trade_df: pd.DataFrame) -> List[dict]:
    results = []
    for sym in AI_BUCKET:
        df = ai_data.get(sym)
        if df is None or len(df) < 100:
            results.append({"symbol": sym, "status": "no_data"})
            continue

        # Buy-and-hold return
        cl = df["close"].to_numpy(dtype=np.float64).ravel()
        bh_ret = float(cl[-1] / cl[0] - 1) * 100
        years  = (df.index[-1] - df.index[0]).days / 365.25
        bh_ann = float((1 + bh_ret/100)**(1/max(years,1)) - 1) * 100

        # Strategy returns from trade_df
        sym_trades = trade_df[trade_df["asset"] == sym]
        strat_stats = {}
        for strat in STRATEGIES:
            sub = sym_trades[sym_trades["strategy"] == strat]
            if len(sub) < 5:
                continue
            wins = sub[sub["result"]=="WIN"]
            loss = sub[sub["result"]!="WIN"]
            gp = wins["exit_r"].sum(); gl = abs(loss["exit_r"].sum())
            pf = gp/gl if gl > 0 else (999.0 if gp > 0 else 0.0)
            wr = (sub["result"]=="WIN").mean()
            strat_stats[strat] = {
                "win_rate": round(float(wr), 3),
                "profit_factor": round(pf, 3),
                "trade_count": len(sub),
            }

        # Best entry regime (most wins)
        if len(sym_trades) > 0:
            regime_wr = sym_trades.groupby("regime").apply(
                lambda x: (x["result"]=="WIN").mean()
            ).sort_values(ascending=False)
            best_regime = str(regime_wr.index[0]) if len(regime_wr) > 0 else "N/A"
        else:
            best_regime = "N/A"

        # Worst drawdown period
        cl_dd = df["close"].to_numpy(dtype=np.float64).ravel()
        roll_max_arr = np.maximum.accumulate(cl_dd)
        dd_arr = (cl_dd - roll_max_arr) / (roll_max_arr + 1e-9) * 100
        worst_dd = float(dd_arr.min())
        worst_dd_idx = int(np.argmin(dd_arr))
        worst_dd_date = str(df.index[worst_dd_idx].date()) if len(cl_dd) > 0 else "N/A"

        # Strategy alpha vs buy-and-hold (approximate)
        # Alpha = strategy total_return_pct - bh_ret (annualised)
        best_strat = max(strat_stats.items(),
                         key=lambda x: x[1]["profit_factor"]) if strat_stats else (None, {})

        results.append({
            "symbol":              sym,
            "buy_hold_total_pct":  round(bh_ret, 2),
            "buy_hold_annual_pct": round(bh_ann, 2),
            "years_of_data":       round(years, 1),
            "best_entry_regime":   best_regime,
            "worst_drawdown_pct":  round(worst_dd, 2),
            "worst_dd_date":       worst_dd_date,
            "strategy_stats":      strat_stats,
            "best_strategy":       best_strat[0],
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_total = time.perf_counter()

    # ── Step 0: warmup JIT ───────────────────────────────────────────────
    print("[JIT] Warming up Numba kernels...")
    dummy = pd.DataFrame({
        "open": np.random.rand(200).astype(np.float32)+10,
        "high": np.random.rand(200).astype(np.float32)+10.5,
        "low":  np.random.rand(200).astype(np.float32)+9.5,
        "close":np.random.rand(200).astype(np.float32)+10,
        "volume": np.ones(200),
    })
    FastBacktestEngine.from_dataframe(dummy).warmup()
    print("[JIT] Warm.")

    # ── Step 1: Fetch data ────────────────────────────────────────────────
    data = fetch_all(UNIVERSE)
    assets = list(data.keys())

    # Fetch SPY & VXX separately for enrichment
    _, spy_df = _fetch_one("SPY")
    _, vxx_df = _fetch_one("VXX")
    spy_5d = pd.Series(dtype=float)
    vxx_cl = pd.Series(dtype=float)
    if spy_df is not None:
        spy_5d = spy_df["close"].pct_change(5).sort_index()
    if vxx_df is not None:
        vxx_cl = vxx_df["close"].sort_index()

    # ── Step 2: Sweep ─────────────────────────────────────────────────────
    print(f"\n[Sweep] Running 4 strategies × {len(PARAM_GRID['stop_atr_mult'])*len(PARAM_GRID['tp_rr'])} combos × {len(assets)} assets...")
    t_sweep = time.perf_counter()
    universe_results = {}
    total_combos = 0

    for i, sym in enumerate(assets):
        df = data[sym]
        r = sweep_asset(sym, df)
        universe_results[sym] = r
        n_c = sum(v.get("all_results_count",0) for v in r.values() if isinstance(v,dict))
        total_combos += n_c
        print(f"  [{i+1}/{len(assets)}] {sym}: {n_c} combos", end="\r", flush=True)

    sweep_time = time.perf_counter() - t_sweep
    print(f"\n[Sweep] Done: {total_combos:,} combos in {sweep_time:.1f}s")

    # Save universe results
    out_path = LOGS / "full_universe_results.json"
    out_path.write_text(json.dumps(universe_results, indent=2))
    print(f"[Saved] {out_path}")

    # ── Step 3: Build trade map ───────────────────────────────────────────
    print(f"\n[TradeMap] Building per-trade records for {len(assets)} assets...")
    t_map = time.perf_counter()
    all_trades: List[dict] = []

    for i, sym in enumerate(assets):
        recs = build_trade_records(sym, data[sym], spy_5d, vxx_cl)
        all_trades.extend(recs)
        print(f"  [{i+1}/{len(assets)}] {sym}: {len(recs)} trades", end="\r", flush=True)

    map_time = time.perf_counter() - t_map
    print(f"\n[TradeMap] {len(all_trades):,} trades in {map_time:.1f}s")

    trade_df = pd.DataFrame(all_trades)
    trade_df.to_csv(LOGS / "full_trade_map.csv", index=False)
    print(f"[Saved] {LOGS / 'full_trade_map.csv'}")

    # ── Step 4: Cluster ───────────────────────────────────────────────────
    print("\n[Cluster] Running KMeans k=8 on losses and wins...")
    loss_clusters = cluster_trades(trade_df, k=8, subset="LOSS")
    win_clusters  = cluster_trades(trade_df, k=8, subset="WIN")

    (LOGS / "loss_clusters.json").write_text(json.dumps(loss_clusters, indent=2))
    (LOGS / "win_clusters.json").write_text(json.dumps(win_clusters, indent=2))
    print(f"[Cluster] {len(loss_clusters)} loss clusters, {len(win_clusters)} win clusters")

    # ── Step 5: Inference table ───────────────────────────────────────────
    print("\n[Inference] Building regime inference table...")
    inf_table = build_inference_table(trade_df)
    (LOGS / "regime_inference_table.json").write_text(json.dumps(inf_table, indent=2))
    recommended = sum(1 for r in inf_table if r["recommended"])
    avoid       = sum(1 for r in inf_table if r["avoid"])
    print(f"[Inference] {len(inf_table)} conditions | {recommended} recommended | {avoid} avoid")

    # ── Step 6: AI bucket analysis ────────────────────────────────────────
    print("\n[AI] Running lifetime analysis on AI pure-play bucket...")
    ai_data = {s: data[s] for s in AI_BUCKET if s in data}
    ai_analysis = ai_bucket_analysis(ai_data, trade_df)
    (LOGS / "ai_company_analysis.json").write_text(json.dumps(ai_analysis, indent=2))

    # ── Final summary ─────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total

    # Top 5 by sharpe proxy across all asset×strategy
    all_perf = []
    for sym, strats in universe_results.items():
        for strat, res in strats.items():
            if isinstance(res, dict) and "sharpe_proxy" in res:
                all_perf.append({
                    "asset": sym, "strategy": strat,
                    "sharpe": res["sharpe_proxy"],
                    "pf": res["profit_factor"],
                    "win_rate": res["win_rate"],
                    "best_params": res["best_param_combo"],
                })
    all_perf.sort(key=lambda x: x["sharpe"], reverse=True)
    top5    = all_perf[:5]
    bottom5 = [p for p in all_perf if p["pf"] < 1.0][-5:]

    print("\n" + "═"*60)
    print(f"UNIVERSE: {len(assets)} assets")
    print(f"COMBOS TESTED: {total_combos:,}")
    print(f"RUNTIME: {total_time:.1f}s")
    print(f"TRADES MAPPED: {len(all_trades):,}")
    print(f"WIN CLUSTERS: {len(win_clusters)} identified")
    print(f"LOSS CLUSTERS: {len(loss_clusters)} identified")
    print(f"INFERENCE TABLE: {len(inf_table)} condition combinations mapped")
    print(f"AI ANALYSIS: complete ({len(ai_analysis)} stocks)")
    print()
    print("TOP 5 COMBINATIONS BY SHARPE:")
    for r in top5:
        print(f"  {r['asset']:6s} × {r['strategy']:20s}  sharpe={r['sharpe']:.3f}  "
              f"pf={r['pf']:.2f}  wr={r['win_rate']:.2%}  params={r['best_params']}")
    print()
    print("WORST 5 COMBINATIONS TO AVOID:")
    for r in bottom5:
        print(f"  {r['asset']:6s} × {r['strategy']:20s}  sharpe={r['sharpe']:.3f}  "
              f"pf={r['pf']:.2f}  wr={r['win_rate']:.2%}  params={r['best_params']}")
    print()
    print("[AI] Alpha vs buy-and-hold:")
    for a in ai_analysis:
        if a.get("status") == "no_data":
            print(f"  {a['symbol']:6s}: NO DATA")
        else:
            best_s = a.get("best_strategy", "N/A")
            ss = a["strategy_stats"].get(best_s, {})
            pf = ss.get("profit_factor", 0)
            print(f"  {a['symbol']:6s}: B&H {a['buy_hold_annual_pct']:+.1f}%/yr | "
                  f"best={best_s} pf={pf:.2f} | "
                  f"worst_dd={a['worst_drawdown_pct']:.1f}% ({a['worst_dd_date']})")
    print("═"*60)


if __name__ == "__main__":
    main()
