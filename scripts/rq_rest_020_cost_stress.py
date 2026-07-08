#!/usr/bin/env python3
"""
RQ-REST-020 — Cost-stress / break-even robustness of the 4-pair forex edge.

Fully offline. Reads logs/forex_backtest_trades.json (the costed 4-pair log,
n=318, 2015-2022). Baseline net R reconstructed as:
    net_R = (pnl_pct - k*spread_frac + swap_frac) / risk_pct
with k=1 the live cost assumption (verified to reproduce the recorded
baseline meanR=0.2998, Sharpe=0.1758, sumR=95.346).

Question: how much would transaction (spread) cost have to rise before the
edge dies? This bounds execution-quality fragility WITHOUT needing price
paths (unlike the blocked RQ-REST-013 exit re-sim).

Writes artifact to data/agent/rq_rest_020_cost_stress_results.json.
"""
import json, os, statistics as st, random

random.seed(7)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(HERE, "logs", "forex_backtest_trades.json")
OUT = os.path.join(HERE, "data", "agent", "rq_rest_020_cost_stress_results.json")

d = json.load(open(LOG))
trades = []
for pair, ts in d.items():
    for t in ts:
        t["pair"] = pair
        trades.append(t)

def netR(t, k):
    """net R with spread cost multiplied by k (k=1 = live)."""
    return (t["pnl_pct"] - k * t["cost_spread_frac"] + t["cost_swap_frac"]) / t["risk_pct"]

def grossR(t):
    """price-only R, no spread, no swap."""
    return t["pnl_pct"] / t["risk_pct"]

def carryR(t):
    return t["cost_swap_frac"] / t["risk_pct"]

def spreadR(t):
    return t["cost_spread_frac"] / t["risk_pct"]

def summ(Rs):
    n = len(Rs)
    if n == 0: return {"n": 0}
    m = st.mean(Rs); sd = st.pstdev(Rs)
    return {"n": n, "wr": round(100*sum(1 for r in Rs if r>0)/n,2),
            "meanR": round(m,4), "sumR": round(sum(Rs),3),
            "sharpe_per_trade": round(m/sd,4) if sd else None}

def breakeven_k(ts):
    """k at which sum(netR)=0. sumR(k) = sum(grossR+carryR) - k*sum(spreadR)."""
    base = sum(grossR(t) + carryR(t) for t in ts)
    spr  = sum(spreadR(t) for t in ts)
    if spr <= 0: return None
    return base / spr  # k* where total net = 0

def boot_ci_meanR(ts, k, B=5000):
    n = len(ts)
    Rs = [netR(t, k) for t in ts]
    means = []
    for _ in range(B):
        s = [Rs[random.randrange(n)] for _ in range(n)]
        means.append(sum(s)/n)
    means.sort()
    return means[int(0.025*B)], means[int(0.975*B)]

def boot_breakeven_k(ts, B=4000):
    """Smallest k (search) at which 2.5% bootstrap quantile of meanR hits 0.
    i.e. k where the edge stops being significantly > 0 at 95%."""
    lo, hi = 1.0, 30.0
    # ensure bracket: at k=lo CI lower may be >0 or <0
    def ci_lo(k):
        return boot_ci_meanR(ts, k, B)[0]
    if ci_lo(lo) <= 0:
        return ("<=1.0 (not significant even at live cost)", round(ci_lo(1.0),4))
    if ci_lo(hi) > 0:
        return (">30", round(ci_lo(hi),4))
    for _ in range(40):
        mid = (lo+hi)/2
        if ci_lo(mid) > 0: lo = mid
        else: hi = mid
    return (round((lo+hi)/2,2), 0.0)

# median spread in price-fraction -> pips conversion per pair
def frac_to_pips(pair, frac, entry):
    # JPY pairs: pip = 0.01 ; others pip = 0.0001
    price_move = frac * entry
    pip = 0.01 if "JPY" in pair else 0.0001
    return price_move / pip

result = {
    "meta": {
        "task": "RQ-REST-020 cost-stress / break-even robustness",
        "src": "logs/forex_backtest_trades.json",
        "n_trades": len(trades),
        "model": "net_R = (pnl_pct - k*spread_frac + swap_frac)/risk_pct ; k=1 live",
        "note": "Spread is the stressed transaction cost. Swap=carry kept signed. Offline; no price paths needed.",
    },
    "baseline_k1": summ([netR(t,1) for t in trades]),
    "gross_price_only": summ([grossR(t) for t in trades]),
    "decomposition": {
        "price_sumR": round(sum(grossR(t) for t in trades),3),
        "carry_swap_sumR": round(sum(carryR(t) for t in trades),3),
        "spread_cost_sumR_at_k1": round(-sum(spreadR(t) for t in trades),3),
        "net_sumR_k1": round(sum(netR(t,1) for t in trades),3),
    },
}

# Portfolio break-even multiple and stress curve
result["portfolio_breakeven_k"] = round(breakeven_k(trades),3)
result["stress_curve"] = {}
for k in [1,2,3,5,8,10,15]:
    result["stress_curve"][f"k={k}"] = summ([netR(t,k) for t in trades])

# Bootstrap significance break-even
sig_k, sig_val = boot_breakeven_k(trades)
ci_lo1, ci_hi1 = boot_ci_meanR(trades, 1)
result["bootstrap"] = {
    "meanR_95ci_at_k1": [round(ci_lo1,4), round(ci_hi1,4)],
    "sig_breakeven_k": sig_k,  # k where 95% CI lower bound of meanR hits 0
    "interpretation": "Largest spread-cost multiple at which the edge stays significantly>0 (95%).",
}

# Per-pair
result["per_pair"] = {}
for pair in d:
    ts = [t for t in trades if t["pair"]==pair]
    bk = breakeven_k(ts)
    med_spread_frac = st.median([t["cost_spread_frac"] for t in ts])
    med_entry = st.median([t["entry"] for t in ts])
    live_pips = frac_to_pips(pair, med_spread_frac, med_entry)
    result["per_pair"][pair] = {
        "baseline_k1": summ([netR(t,1) for t in ts]),
        "gross_price_only_sumR": round(sum(grossR(t) for t in ts),3),
        "carry_sumR": round(sum(carryR(t) for t in ts),3),
        "breakeven_k": round(bk,3) if bk else None,
        "live_modeled_spread_pips": round(live_pips,2),
        "breakeven_spread_pips": round(live_pips*bk,2) if bk else None,
    }

# Per-year break-even (regime fragility lens)
from collections import defaultdict
yr = defaultdict(list)
for t in trades:
    yr[t["entry_date"][:4]].append(t)
result["per_year"] = {}
for y in sorted(yr):
    ts = yr[y]; bk = breakeven_k(ts)
    result["per_year"][y] = {
        "baseline_k1": summ([netR(t,1) for t in ts]),
        "breakeven_k": round(bk,3) if bk else None,
    }

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(result, open(OUT,"w"), indent=2)
print(json.dumps(result, indent=2))
print("\nWROTE", OUT)
