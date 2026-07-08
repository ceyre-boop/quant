"""
RQ-REST-011 — 90-day rolling rate-spread velocity as a position-size throttle.

Pure overlay test on the canonical v003 trade log (2015-2024, 4 pairs, costed).
The gate NEVER drops a trade; it only scales each trade's fractional return by a
size multiplier derived from where portfolio rate-spread velocity sits at entry.

Adversarial design (the operation killed a 2023 artifact last cycle, REST-007):
  - CAUSAL percentiles (expanding window, min 252 obs) vs full-sample (look-ahead).
    Only the causal number is tradable; full-sample is reported to expose inflation.
  - |velocity| (trend magnitude) is the primary hypothesis; signed velocity tested.
  - 90d lookback primary; 60d/120d robustness.
  - Per-year Sharpe, portfolio Sharpe (sqrt-n weighted, matches engine), max DD.

Baseline Sharpe is reproduced EXACTLY by calibrating each pair's annualization
factor to the tracked metadata (v003 metadata.json), then reusing that same factor
on gated returns -> apples-to-apples.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/sessions/dreamy-zen-lovelace/mnt/quant")
TRADES = json.load(open(ROOT / "logs/research/v003/backtest_trades.json"))
META = json.load(open(ROOT / "logs/research/v003/metadata.json"))
META_SHARPE = META["metrics"]["sharpe_by_pair"]

PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
# pair -> (base ccy file, quote ccy file)
RATE_MAP = {
    "EURUSD=X": ("EU", "US"),
    "GBPUSD=X": ("GB", "US"),
    "USDJPY=X": ("US", "JP"),
    "AUDUSD=X": ("AU", "US"),
}

def load_rate(ccy):
    return pd.read_parquet(ROOT / f"data/cache/macro/{ccy}_rates.parquet")["rate"]

# ---- daily calendar of per-pair rate spreads (base - quote) -----------------
cal = pd.date_range("2014-01-01", "2025-12-31", freq="D")
rates = {c: load_rate(c).reindex(cal).ffill() for c in
         set(sum([list(v) for v in RATE_MAP.values()], []))}
spread = pd.DataFrame({p: rates[b] - rates[q] for p, (b, q) in RATE_MAP.items()},
                      index=cal)

def portfolio_velocity(lookback_days, signed=False):
    """Sum across pairs of the spread's change over `lookback_days`.
    signed=False -> sum of |changes| (trend-magnitude). signed=True -> sum of changes."""
    chg = spread - spread.shift(lookback_days)
    chg = chg if signed else chg.abs()
    return chg.sum(axis=1)

def multipliers(vel, causal=True, lo_q=0.25, hi_q=0.50):
    """Map each day's velocity -> {0.25,0.5,1.0} via percentile buckets.
    causal=True uses an expanding-window percentile rank (min 252 prior obs),
    so the threshold at day t uses only data up to t (no look-ahead)."""
    v = vel.dropna()
    mult = pd.Series(index=v.index, dtype=float)
    if causal:
        for i, (dt, x) in enumerate(v.items()):
            hist = v.iloc[:i+1]
            if len(hist) < 252:
                mult.loc[dt] = 1.0           # warm-up: no throttle (neutral)
                continue
            rank = (hist < x).mean()         # percentile rank of today vs history
            mult.loc[dt] = 1.0 if rank >= hi_q else (0.5 if rank >= lo_q else 0.25)
    else:
        lo, hi = v.quantile(lo_q), v.quantile(hi_q)
        mult = v.apply(lambda x: 1.0 if x >= hi else (0.5 if x >= lo else 0.25))
    return mult.reindex(cal).ffill()

# ---- per-trade frame --------------------------------------------------------
rows = []
for p in PAIRS:
    for t in TRADES[p]:
        rows.append({"pair": p,
                     "entry": pd.Timestamp(t["entry_date"]),
                     "year": pd.Timestamp(t["entry_date"]).year,
                     "pnl": t["pnl_pct"]})          # already net of costs
trades = pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)

# ---- Sharpe machinery (matches forex_backtester._compute_stats) -------------
def raw_sharpe(pnls):
    pnls = np.asarray(pnls, float)
    if len(pnls) < 2:
        return 0.0
    equity = np.cumprod(1 + pnls)
    rets = np.diff(np.log(equity), prepend=0)
    return np.mean(rets) / (np.std(rets) + 1e-9)

# calibrate per-pair annualization factor to reproduce tracked metadata Sharpe
ANN = {}
for p in PAIRS:
    base = [t["pnl_pct"] for t in TRADES[p]]
    rs = raw_sharpe(base)
    ANN[p] = META_SHARPE[p] / rs if rs else 0.0

def pair_sharpe(pnls, pair):
    return raw_sharpe(pnls) * ANN[pair]

def max_dd(pnls):
    eq = np.cumprod(1 + np.asarray(pnls, float))
    rm = np.maximum.accumulate(eq)
    return float(((eq - rm) / rm).min()) if len(eq) else 0.0

def portfolio_sharpe(df, col):
    """sqrt-n weighted mean of per-pair Sharpes (engine convention)."""
    num = den = 0.0
    for p in PAIRS:
        sub = df[df.pair == p]
        if len(sub) < 2:
            continue
        s = pair_sharpe(sub[col].tolist(), p)
        w = np.sqrt(len(sub))
        num += s * w; den += w
    return num / den if den else 0.0

# ---- evaluate one configuration --------------------------------------------
def evaluate(lookback=90, signed=False, causal=True, lo_q=0.25, hi_q=0.50):
    vel = portfolio_velocity(lookback, signed=signed)
    mult = multipliers(vel, causal=causal, lo_q=lo_q, hi_q=hi_q)
    df = trades.copy()
    df["mult"] = df["entry"].map(lambda d: mult.get(d, 1.0)).fillna(1.0)
    df["pnl_gated"] = df["pnl"] * df["mult"]
    out = {
        "base_portfolio_sharpe": round(portfolio_sharpe(df, "pnl"), 4),
        "gated_portfolio_sharpe": round(portfolio_sharpe(df, "pnl_gated"), 4),
        "base_maxdd": round(max_dd(df.sort_values('entry')["pnl"].tolist()), 4),
        "gated_maxdd": round(max_dd(df.sort_values('entry')["pnl_gated"].tolist()), 4),
        "mult_dist": {f"{k:.2f}": int(v) for k, v in df["mult"].value_counts().sort_index().items()},
        "avg_mult": round(df["mult"].mean(), 3),
        "per_year": {},
    }
    for y in range(2015, 2025):
        sub = df[df.year == y]
        if len(sub) < 2:
            continue
        out["per_year"][y] = {
            "n": int(len(sub)),
            "base_sharpe": round(portfolio_sharpe(sub, "pnl"), 3),
            "gated_sharpe": round(portfolio_sharpe(sub, "pnl_gated"), 3),
            "avg_mult": round(sub["mult"].mean(), 2),
        }
    return out

# ---- primary + robustness grid ---------------------------------------------
results = {
    "baseline_metadata_sharpe_by_pair": META_SHARPE,
    "baseline_portfolio_sharpe_reproduced": round(portfolio_sharpe(trades.assign(x=trades.pnl).rename(columns={}), "pnl"), 4),
    "ann_factors": {k: round(v, 3) for k, v in ANN.items()},
    "n_trades": int(len(trades)),
    "PRIMARY_causal_abs_90d": evaluate(90, signed=False, causal=True),
    "robust_fullsample_abs_90d_LOOKAHEAD": evaluate(90, signed=False, causal=False),
    "robust_causal_abs_60d": evaluate(60, signed=False, causal=True),
    "robust_causal_abs_120d": evaluate(120, signed=False, causal=True),
    "robust_causal_signed_90d": evaluate(90, signed=True, causal=True),
}
print(json.dumps(results, indent=2, default=str))
Path("/sessions/dreamy-zen-lovelace/mnt/outputs/rq_rest_011_results.json").write_text(
    json.dumps(results, indent=2, default=str))
