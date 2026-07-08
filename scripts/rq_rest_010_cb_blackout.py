"""
RQ-REST-010 / HYP-039(CB-blackout): MC validation of a "days 3-14 before CB" gate
on London ICT trades.

Methodology note: the cached calendar data/cache/cb_decisions.json is contaminated
for 2023-2024 (FED = 1st-of-month placeholders, BOE = spurious consecutive-day
clusters). So we rebuild a CLEAN scheduled BOE-MPC + FOMC calendar. The 2025 portion
matches sovereign/forex/cb_calendar.py exactly, which validates the reconstruction.

Universe: GBPUSD London-session trades (dominant single pair; relevant CBs = BOE+FED
only, so no multi-bank reconstruction error). Pulled from the two 1Y hourly backtest
windows so coverage spans 2023-08 .. 2025-04.
"""
import json, datetime as dt
import numpy as np

rng = np.random.default_rng(42)

# ---- clean scheduled decision dates (announcement day) ----
FOMC = [ "2023-07-26","2023-09-20","2023-11-01","2023-12-13",
         "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
         "2024-09-18","2024-11-07","2024-12-18",
         "2025-01-29","2025-03-19","2025-05-07" ]
BOE  = [ "2023-08-03","2023-09-21","2023-11-02","2023-12-14",
         "2024-02-01","2024-03-21","2024-05-09","2024-06-20","2024-08-01",
         "2024-09-19","2024-11-07","2024-12-19",
         "2025-02-06","2025-03-20","2025-05-08" ]
CB = sorted(dt.date.fromisoformat(x) for x in (FOMC+BOE))

def days_to_next_cb(entry_date):
    fut = [(m-entry_date).days for m in CB if (m-entry_date).days >= 0]
    return min(fut) if fut else None

# ---- load trades ----
trades=[]
seen=set()
for f in ("logs/ict_backtest_window_A.json","logs/ict_backtest_window_B.json"):
    d=json.load(open(f))
    for t in d["trades"]:
        if t.get("session")!="London" or t.get("pair")!="GBPUSD":
            continue
        key=(t["entry_dt"], round(t["entry"],5))
        if key in seen:           # dedup small May-Jun 2024 overlap
            continue
        seen.add(key)
        ed=dt.date.fromisoformat(t["entry_dt"][:10])
        dtn=days_to_next_cb(ed)
        if dtn is None:
            continue
        trades.append({"entry_dt":t["entry_dt"],"r":float(t["pnl_r"]),
                       "grade":t.get("grade"),"dtn":dtn})

R=np.array([t["r"] for t in trades])
DTN=np.array([t["dtn"] for t in trades])
N=len(trades)
print(f"GBPUSD London trades tagged: N={N}  span {min(t['entry_dt'] for t in trades)[:10]} .. {max(t['entry_dt'] for t in trades)[:10]}")

def stats(r):
    r=np.asarray(r,float)
    if len(r)==0: return dict(n=0)
    return dict(n=len(r), wr=float((r>0).mean()), avgR=float(r.mean()),
                sumR=float(r.sum()), sharpe=float(r.mean()/r.std(ddof=1)) if len(r)>1 and r.std(ddof=1)>0 else 0.0)

print("\n--- proximity buckets (days before next BOE/FED meeting) ---")
buckets={"1-2":(1,2),"3-6":(3,6),"7-14":(7,14),"15+":(15,9999),"0":(0,0)}
for name,(lo,hi) in buckets.items():
    mask=(DTN>=lo)&(DTN<=hi)
    print(f"  {name:5} {stats(R[mask])}")

print("\nbaseline (all):", stats(R))

# ---- the gate: veto trades with dtn in [3,14] ----
veto=(DTN>=3)&(DTN<=14)
kept=~veto
print(f"\nveto window 3-14d: vetoes {veto.sum()}/{N} trades ({veto.mean()*100:.0f}%)")
base=stats(R); keptS=stats(R[kept]); vetoS=stats(R[veto])
print("  vetoed-out trades:", vetoS)
print("  kept (post-gate) :", keptS)
dSharpe=keptS["sharpe"]-base["sharpe"]
dAvgR =keptS["avgR"]-base["avgR"]
print(f"  delta Sharpe (kept - baseline) = {dSharpe:+.3f}")
print(f"  delta avgR   (kept - baseline) = {dAvgR:+.3f}")
print(f"  sumR baseline {base['sumR']:+.2f}  ->  kept {keptS['sumR']:+.2f}  (forgone {keptS['sumR']-base['sumR']:+.2f}R)")

# ---- MC 1: permutation test. Is the vetoed bucket's badness distinguishable from
#            randomly removing the same number of trades? ----
k=int(veto.sum())
obs_gap = R[veto].mean() - R.mean()        # how much worse vetoed trades are
B=20000
perm=np.empty(B)
idx=np.arange(N)
for b in range(B):
    sel=rng.choice(idx,size=k,replace=False)
    perm[b]=R[sel].mean()-R.mean()
p_perm=float((perm<=obs_gap).mean())       # P(random subset as bad or worse)
print(f"\n[MC1 permutation] vetoed avgR gap vs all = {obs_gap:+.3f}R; "
      f"P(random {k}-subset this bad) = {p_perm:.3f}")

# ---- MC 2: bootstrap CI on delta-Sharpe of the gate ----
dS=np.empty(B)
for b in range(B):
    s=rng.integers(0,N,N)
    rb=R[s]; vb=veto[s]
    base_s = rb.mean()/rb.std(ddof=1) if rb.std(ddof=1)>0 else 0
    kr=rb[~vb]
    kept_s = kr.mean()/kr.std(ddof=1) if len(kr)>1 and kr.std(ddof=1)>0 else 0
    dS[b]=kept_s-base_s
lo,hi=np.percentile(dS,[2.5,97.5])
p_dSpos=float((dS>0).mean())
print(f"[MC2 bootstrap] delta-Sharpe mean {dS.mean():+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  P(>0)={p_dSpos:.3f}")

# ---- MC 3: prop-challenge pass rate, baseline vs gated ----
# Lucid/MFF style: +8% profit target, -6% EOD-equivalent max drawdown (peak-to-trough),
# risk 1% of equity per 1R, sequence length = one "phase" of 40 trades, bootstrap order.
def pass_rate(pool, target=0.08, maxdd=0.06, risk=0.01, length=40, sims=4000):
    pool=np.asarray(pool,float)
    if len(pool)==0: return 0.0
    wins=0
    for _ in range(sims):
        seq=pool[rng.integers(0,len(pool),length)]
        eq=1.0; peak=1.0; passed=False
        for r in seq:
            eq*= (1+risk*r)
            peak=max(peak,eq)
            if (eq-peak)/peak <= -maxdd:    # breached
                break
            if eq-1.0 >= target:
                passed=True; break
        wins+=passed
    return wins/sims

pr_base=pass_rate(R)
pr_gate=pass_rate(R[kept])
print(f"\n[MC3 prop-challenge] pass rate  baseline {pr_base*100:.1f}%   gated {pr_gate*100:.1f}%   delta {(pr_gate-pr_base)*100:+.1f}pp")

# ---- decision rule from RQ-REST-010 ----
print("\n=== DECISION ===")
if dSharpe>0.10 and p_perm<0.05 and lo>0:
    verd="WIRE HARD VETO (dSharpe>0.10, robust)"
elif dSharpe>0 and p_perm<0.10:
    verd="WIRE 0.5x SIZE REDUCER (positive but modest/borderline)"
else:
    verd="REJECT / DO NOT WIRE (effect not robust)"
print(verd)

out=dict(universe="GBPUSD London", N=N, baseline=base, kept=keptS, vetoed=vetoS,
         delta_sharpe=dSharpe, delta_avgR=dAvgR, forgone_sumR=keptS["sumR"]-base["sumR"],
         mc1_perm_p=p_perm, mc1_obs_gap=obs_gap,
         mc2_dSharpe_mean=float(dS.mean()), mc2_ci=[float(lo),float(hi)], mc2_p_pos=p_dSpos,
         mc3_pass_base=pr_base, mc3_pass_gate=pr_gate,
         verdict=verd, cb_calendar="reconstructed clean BOE+FOMC (2025 matches cb_calendar.py)",
         note="cb_decisions.json contaminated 2023-24; original n=172 finding used dirty calendar")
json.dump(out, open("data/agent/rq_rest_010_results.json","w"), indent=1)
print("\nwrote data/agent/rq_rest_010_results.json")

# ================= VERIFICATION =================
print("\n\n================ VERIFICATION ================")
ED=np.array([dt.date.fromisoformat(t["entry_dt"][:10]) for t in trades])
split=dt.date(2024,7,1)
for label,m in [("2023-08..2024-06 (windowB era)", ED<split),
                ("2024-07..2025-03 (windowA era)", ED>=split)]:
    rr=R[m]; vv=veto[m]
    b=stats(rr); kp=stats(rr[~vv])
    dsh = (kp["sharpe"]-b["sharpe"]) if b.get("n",0)>1 else float('nan')
    print(f"\n {label}: N={m.sum()}  vetoed={vv.sum()}")
    print(f"   baseline avgR {b.get('avgR'):+.3f} Sharpe {b.get('sharpe'):+.3f} | "
          f"kept avgR {kp.get('avgR'):+.3f} Sharpe {kp.get('sharpe'):+.3f} | dSharpe {dsh:+.3f}")

print("\n--- window-bound sensitivity (avgR of vetoed-out set, want very negative) ---")
for lo,hi in [(1,14),(3,14),(7,14),(3,6),(0,14),(1,7)]:
    v=(DTN>=lo)&(DTN<=hi)
    kp=stats(R[~v])
    print(f"   veto [{lo:>2},{hi:>2}]: removes {v.sum():>2}  vetoedAvgR {R[v].mean():+.3f}  keptSharpe {kp['sharpe']:+.3f}  keptAvgR {kp['avgR']:+.3f}")

print("\n--- caveat check: is the edge just the 15+ winners? grade mix of vetoed set ---")
from collections import Counter
print("   vetoed grades:", dict(Counter(trades[i]['grade'] for i in range(N) if veto[i])))
print("   kept   grades:", dict(Counter(trades[i]['grade'] for i in range(N) if not veto[i])))
