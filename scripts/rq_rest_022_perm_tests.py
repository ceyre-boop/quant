#!/usr/bin/env python3
"""
RQ-REST-022 — Offline label-permutation significance tests for four
confirmed/deployed gates that had NO recorded p-value (FIND-REST-021-c).

Gates:
  HYP-047  score inversion (DEPLOYED)  -- ICT graded trades
  HYP-050  Tue+Thu DOW veto            -- ICT pattern trades
  HYP-052c rate-trend widening gate    -- 4-pair forex forensic
  HYP-054  rate-level gate |rd|>1.0%   -- 4-pair forex forensic

Method: fix the per-trade outcome_r vector, permute the *label* (score order /
weekday / gate membership) N times, recompute the gate's own effect statistic,
report two-sided p = (1 + #{|perm| >= |obs|}) / (N+1).  No network, no live change.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
ICT  = ROOT / "data" / "forensics" / "trade_forensics.jsonl"
FX   = ROOT / "data" / "research"  / "trade_forensics.json"
N    = 20000
RNG  = np.random.default_rng(42)
LIVE_PAIRS = ("GBPUSD", "EURUSD", "AUDUSD", "GBPJPY")  # v015 4-pair (AUDNZD excluded)

def two_sided_p(obs, null):
    null = np.asarray(null)
    return (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (len(null) + 1)

def sharpe(r):
    r = np.asarray(r, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1))

def rank(a):
    # average-rank (handles ties) for Spearman
    a = np.asarray(a, float)
    order = a.argsort()
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a))
    # tie correction
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.concatenate([[0], np.cumsum(cnt)])
    avg = {i: (csum[i] + csum[i+1] - 1) / 2.0 for i in range(len(cnt))}
    return np.array([avg[i] for i in inv])

def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x*x).sum() * (y*y).sum())
    return float((x*y).sum() / d) if d else float("nan")

def spearman(x, y):
    return pearson(rank(x), rank(y))

results = {}

# ---------- load ----------
ict = pd.DataFrame([json.loads(l) for l in ICT.read_text().splitlines() if l.strip()])
ict["wd"] = pd.to_datetime(ict["entry_date"], errors="coerce").dt.dayofweek  # 0=Mon
fx  = pd.DataFrame(json.loads(FX.read_text()))
fx["pair_s"] = fx["pair"].str.replace("=X", "", regex=False)
fx["dt"] = pd.to_datetime(fx["entry_date"], errors="coerce")
fx4 = fx[fx["pair_s"].isin(LIVE_PAIRS)].copy()

# ================= HYP-047: score inversion (ICT graded) =================
g = ict[(ict["session"].isin(["London", "NY_PM"])) & ict["score"].notna() & ict["pnl_r"].notna()]
s = g["score"].to_numpy(float); r = g["pnl_r"].to_numpy(float)
obs = spearman(s, r)
null = np.array([spearman(s, RNG.permutation(r)) for _ in range(N)])
p = two_sided_p(obs, null)
# A vs A+ contrast (deployed gate prefers A over A+)
a  = g[g["grade"] == "A"]["pnl_r"].to_numpy(float)
ap = g[g["grade"] == "A+"]["pnl_r"].to_numpy(float)
results["HYP-047"] = {
    "test": "Spearman(score, pnl_r) on ICT graded trades; expect NEGATIVE (inversion)",
    "n": int(len(g)), "spearman": round(obs, 4), "p_two_sided": round(float(p), 5),
    "A_meanR": round(float(a.mean()), 3), "A_n": int(len(a)),
    "Aplus_meanR": round(float(ap.mean()), 3), "Aplus_n": int(len(ap)),
    "A_minus_Aplus_meanR": round(float(a.mean() - ap.mean()), 3),
    "status": "DEPLOYED",
}

# ================= HYP-050: Tue+Thu DOW veto (ICT pattern) =================
g = ict[ict["session"].isin(["London", "NY_PM"]) & ict["pnl_r"].notna() & ict["wd"].notna()]
r = g["pnl_r"].to_numpy(float); is_tt = g["wd"].isin([1, 3]).to_numpy()  # Tue=1,Thu=3
def diff_tt(mask):
    return r[mask].mean() - r[~mask].mean()
obs = diff_tt(is_tt)
k = is_tt.sum()
null = np.empty(N)
idx = np.arange(len(r))
for i in range(N):
    m = np.zeros(len(r), bool); m[RNG.choice(idx, k, replace=False)] = True
    null[i] = diff_tt(m)
p = two_sided_p(obs, null)
wr = lambda x: float((x > 0).mean())
results["HYP-050"] = {
    "test": "meanR(Tue+Thu) - meanR(other) on ICT pattern trades; expect NEGATIVE",
    "n": int(len(g)), "n_tue_thu": int(k),
    "meanR_tuethu": round(float(r[is_tt].mean()), 3), "WR_tuethu": round(wr(r[is_tt]), 3),
    "meanR_other": round(float(r[~is_tt].mean()), 3), "WR_other": round(wr(r[~is_tt]), 3),
    "diff": round(float(obs), 3), "p_two_sided": round(float(p), 5),
    "status": "CONFIRMED",
}

# ================= HYP-052c: rate-trend widening gate (4-pair) =================
g = fx4[fx4["outcome_r"].notna()].sort_values(["pair_s", "dt"]).copy()
g["abs_rd"] = g["real_rate_diff"].abs()
g["prev_abs"] = g.groupby("pair_s")["abs_rd"].shift(1)
g = g[g["prev_abs"].notna()]                       # need a previous trade for the pair
widen = (g["abs_rd"] > g["prev_abs"]).to_numpy()
r = g["outcome_r"].to_numpy(float)
obs = sharpe(r[widen]) - sharpe(r[~widen])
k = widen.sum(); idx = np.arange(len(r))
null = np.empty(N)
for i in range(N):
    m = np.zeros(len(r), bool); m[RNG.choice(idx, k, replace=False)] = True
    null[i] = sharpe(r[m]) - sharpe(r[~m])
p = two_sided_p(obs, null)
results["HYP-052c"] = {
    "test": "Sharpe(widening |rd|) - Sharpe(not) on 4-pair forex; expect POSITIVE",
    "n": int(len(g)), "n_widen": int(k),
    "sharpe_widen": round(sharpe(r[widen]), 3), "meanR_widen": round(float(r[widen].mean()), 3),
    "sharpe_not": round(sharpe(r[~widen]), 3), "meanR_not": round(float(r[~widen].mean()), 3),
    "delta_sharpe": round(float(obs), 3), "p_two_sided": round(float(p), 5),
    "status": "CONFIRMED",
}

# ================= HYP-054: rate-level gate |rd|>1.0% (4-pair) =================
g = fx4[fx4["outcome_r"].notna()].copy()
passg = (g["real_rate_diff"].abs() > 1.0).to_numpy()
r = g["outcome_r"].to_numpy(float)
obs = sharpe(r[passg]) - sharpe(r[~passg])
k = passg.sum(); idx = np.arange(len(r))
null = np.empty(N)
for i in range(N):
    m = np.zeros(len(r), bool); m[RNG.choice(idx, k, replace=False)] = True
    null[i] = sharpe(r[m]) - sharpe(r[~m])
p = two_sided_p(obs, null)
results["HYP-054"] = {
    "test": "Sharpe(|rd|>1.0) - Sharpe(|rd|<=1.0) on 4-pair forex; expect POSITIVE",
    "n": int(len(g)), "n_pass": int(k),
    "sharpe_pass": round(sharpe(r[passg]), 3), "meanR_pass": round(float(r[passg].mean()), 3),
    "sharpe_fail": round(sharpe(r[~passg]), 3), "meanR_fail": round(float(r[~passg].mean()), 3),
    "delta_sharpe": round(float(obs), 3), "p_two_sided": round(float(p), 5),
    "status": "CONFIRMED",
}

out = {"perm_N": N, "seed": 42, "live_pairs": LIVE_PAIRS, "results": results}
(ROOT / "data" / "research" / "rq_rest_022_perm_tests.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
