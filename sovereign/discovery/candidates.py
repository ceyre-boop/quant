"""
sovereign/discovery/candidates.py
=================================
Generate a BOUNDED, pre-registerable set of candidate setups. Each candidate is a
function emitting a {-1,0,+1} signal array; the gate decides which are real.

Two families:
  • rule-based   — economically-motivated threshold rules over the feature matrix
                   (trend, mean-revert, MTF alignment, regime, quarter-end, etc.)
  • cluster      — unsupervised KMeans on TRAIN-window feature vectors; clusters with
                   a consistent forward-return skew become candidates (sklearn-guarded).

Bounded by design (~20-30 candidates, not millions) — the count feeds the Deflated
Sharpe / BH multiple-testing correction in gate.py. This is "systematic but honest",
the opposite of the factor-zoo scan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign.discovery.features import FEATURE_COLUMNS
from sovereign.discovery.gate import Candidate


# ─── rule-based ───────────────────────────────────────────────────────────────

def _rule_fn(long_expr, short_expr):
    def fn(pair, pdf, fdf):
        s = np.zeros(len(fdf), dtype=np.int8)
        lc = long_expr(fdf)
        sc = short_expr(fdf)
        lc = lc.fillna(False).to_numpy() if hasattr(lc, "fillna") else np.asarray(lc)
        sc = sc.fillna(False).to_numpy() if hasattr(sc, "fillna") else np.asarray(sc)
        s[lc] = 1
        s[sc & ~lc] = -1
        return s
    return fn


_RULES = [
    ("trend_mom20", "Trend-follow: 20-bar momentum sign",
     lambda f: f["ret20"] > 0, lambda f: f["ret20"] < 0),
    ("trend_filtered", "Trend-follow gated by 200-SMA regime",
     lambda f: (f["ret20"] > 0) & (f["above_sma200"] > 0),
     lambda f: (f["ret20"] < 0) & (f["above_sma200"] < 1)),
    ("rsi_meanrevert", "Mean-revert RSI extremes",
     lambda f: f["rsi14"] < 30, lambda f: f["rsi14"] > 70),
    ("macd_cross", "MACD histogram sign",
     lambda f: f["macd_hist"] > 0, lambda f: f["macd_hist"] < 0),
    ("mtf_align", "Multi-timeframe alignment (price vs 50 vs 200)",
     lambda f: f["mtf_align"] >= 1, lambda f: f["mtf_align"] <= -1),
    ("bb_breakout", "Bollinger breakout (momentum)",
     lambda f: f["bb_pos"] > 1, lambda f: f["bb_pos"] < -1),
    ("bb_fade", "Bollinger fade (mean-revert)",
     lambda f: f["bb_pos"] < -1, lambda f: f["bb_pos"] > 1),
    ("sma50_dist", "Distance from 50-SMA momentum",
     lambda f: f["dist_sma50"] > 0, lambda f: f["dist_sma50"] < 0),
    ("adx_trend", "ADX-confirmed momentum",
     lambda f: (f["adx14"] > 25) & (f["mom_sign20"] > 0),
     lambda f: (f["adx14"] > 25) & (f["mom_sign20"] < 0)),
    ("hurst_trend", "Trend-follow only in trending regime (Hurst>0.55)",
     lambda f: (f["hurst"] > 0.55) & (f["ret5"] > 0),
     lambda f: (f["hurst"] > 0.55) & (f["ret5"] < 0)),
    ("hurst_revert", "Mean-revert only in reverting regime (Hurst<0.45)",
     lambda f: (f["hurst"] < 0.45) & (f["rsi14"] < 40),
     lambda f: (f["hurst"] < 0.45) & (f["rsi14"] > 60)),
    ("quarter_end_long", "Quarter-end rebalancing drift (long)",
     lambda f: f["is_quarter_end"] > 0, lambda f: pd.Series(False, index=f.index)),
    ("lowvol_trend", "Low-vol expansion: enter trend when vol compressed",
     lambda f: (f["vol_z"] < -0.5) & (f["ret20"] > 0),
     lambda f: (f["vol_z"] < -0.5) & (f["ret20"] < 0)),
    ("long_only_regime", "Long-only above 200-SMA",
     lambda f: f["above_sma200"] > 0, lambda f: pd.Series(False, index=f.index)),
    ("rsi_trend_combo", "RSI pullback in uptrend",
     lambda f: (f["above_sma200"] > 0) & (f["rsi14"] < 45),
     lambda f: (f["above_sma200"] < 1) & (f["rsi14"] > 55)),
]


def _rule_candidates() -> list[Candidate]:
    out = []
    for cid, desc, long_e, short_e in _RULES:
        out.append(Candidate(
            id=f"rule_{cid}", name=cid, description=desc,
            signal_for=_rule_fn(long_e, short_e),
            meta={"family": "rule", "spec": desc},
        ))
    return out


# ─── cluster-derived (sklearn-guarded) ────────────────────────────────────────

def _cluster_candidates(adapter, features_by_pair: dict, train_window: tuple,
                        k: int = 6, fwd: int = 5, min_abs_ret: float = 0.0015) -> list[Candidate]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return []

    cols = FEATURE_COLUMNS
    X_rows, y_rows = [], []
    for pair in adapter.pairs:
        fdf = features_by_pair.get(pair)
        pdf = adapter.price_df(pair)
        if fdf is None or pdf is None:
            continue
        close = pdf["Close"] if "Close" in pdf.columns else pdf.iloc[:, 0]
        fret = close.shift(-fwd) / close - 1.0
        mask = (fdf.index >= pd.Timestamp(train_window[0])) & (fdf.index <= pd.Timestamp(train_window[1]))
        sub = fdf.loc[mask, cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X_rows.append(sub.to_numpy())
        y_rows.append(fret.loc[mask].fillna(0.0).to_numpy())
    if not X_rows:
        return []
    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    if len(X) < 500:
        return []

    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaler.transform(X))
    labels = km.labels_
    # direction per cluster from TRAIN forward returns
    directions = {}
    for c in range(k):
        m = labels == c
        if m.sum() < 30:
            continue
        avg = float(np.mean(y[m]))
        if abs(avg) >= min_abs_ret:
            directions[c] = 1 if avg > 0 else -1

    out = []
    for c, d in directions.items():
        def make_fn(cluster_id, direction):
            def fn(pair, pdf, fdf):
                Xb = scaler.transform(fdf[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy())
                lab = km.predict(Xb)
                s = np.zeros(len(fdf), dtype=np.int8)
                s[lab == cluster_id] = direction
                return s
            return fn
        out.append(Candidate(
            id=f"cluster_{c}", name=f"cluster{c}_{'long' if d > 0 else 'short'}",
            description=f"KMeans cluster {c} (train fwd-return {'+' if d > 0 else '-'}), k={k}",
            signal_for=make_fn(c, d), meta={"family": "cluster", "spec": f"kmeans k={k} cluster {c} dir={d}"},
        ))
    return out


def generate(adapter, features_by_pair: dict, train_window: tuple,
             include_clusters: bool = True) -> list[Candidate]:
    cands = _rule_candidates()
    if include_clusters:
        try:
            cands += _cluster_candidates(adapter, features_by_pair, train_window)
        except Exception:
            pass
    return cands
