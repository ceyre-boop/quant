"""
RQ-REST-007 / REST-007 cycle (2026-06-08): HYP-056 robustness battery.

HYP-056 (counter-momentum edge): trades entered AGAINST 63-day price momentum
(direction * momentum_63d < 0) outperform momentum-aligned entries.
REST-006 flagged it CANDIDATE: full-period perm p=0.185 (NS), OOS p=0.039 (sig
but 2023-driven), 3-gate OOS Sharpe 0.499 but perm p=0.065-0.10. Underpowered.

This script does NOT try to confirm the edge. It tries to BREAK it. A real edge
survives: bootstrap CIs that exclude zero, per-year sign stability, insensitivity
to the exact momentum cutoff, and multiple-testing correction. E1/E2/E3 all looked
real once too. The job is to find out which bucket HYP-056 belongs in.

Pure forensics — no network. Reads data/research/trade_forensics.json.
"""
import json, numpy as np
from collections import defaultdict, Counter

FORENSICS_PATH = 'data/research/trade_forensics.json'
LIVE_PAIRS = {'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCAD=X'}  # HYP-045 live universe
RNG = np.random.default_rng(20260608)

def sharpe(returns):
    r = np.asarray(returns, dtype=float)
    if len(r) < 3 or np.std(r) < 1e-9:
        return 0.0
    return float(np.mean(r) / np.std(r))

def perm_test(a, b, n=10000):
    """Two-sided-ish: p = P(|perm delta| >= |obs delta|). Sharpe(counter) - Sharpe(aligned)."""
    obs = sharpe(a) - sharpe(b)
    combined = np.array(list(a) + list(b), dtype=float)
    na = len(a)
    cnt = 0
    for _ in range(n):
        idx = RNG.permutation(len(combined))
        d = sharpe(combined[idx[:na]]) - sharpe(combined[idx[na:]])
        if abs(d) >= abs(obs):
            cnt += 1
    return obs, cnt / n

def bootstrap_delta_ci(counter, aligned, n=5000, alpha=0.05):
    """Bootstrap the Sharpe delta (counter - aligned). Returns (median, lo, hi, P(delta>0))."""
    c = np.asarray(counter, float); a = np.asarray(aligned, float)
    deltas = np.empty(n)
    for i in range(n):
        cs = c[RNG.integers(0, len(c), len(c))]
        as_ = a[RNG.integers(0, len(a), len(a))]
        deltas[i] = sharpe(cs) - sharpe(as_)
    lo, hi = np.quantile(deltas, [alpha/2, 1-alpha/2])
    return float(np.median(deltas)), float(lo), float(hi), float(np.mean(deltas > 0))

def split(trades, thresh=0.0):
    """Counter: dir*mom < -thresh. Aligned: dir*mom > +thresh. Drop the band in between."""
    counter, aligned = [], []
    for t in trades:
        s = t['direction'] * t['momentum_63d']
        if s < -thresh: counter.append(t['outcome_r'])
        elif s > thresh: aligned.append(t['outcome_r'])
    return counter, aligned

def year_of(t):
    return int(t['entry_date'][:4])

# ── load ──────────────────────────────────────────────────────────────────────
all_trades = json.load(open(FORENSICS_PATH))
live = [t for t in all_trades if t['pair'] in LIVE_PAIRS]
print(f"Loaded {len(all_trades)} total, {len(live)} live-universe (4-pair) trades")
print(f"Years: {sorted(set(year_of(t) for t in live))}")

out = {"cycle": "REST-007", "date": "2026-06-08", "hypothesis": "HYP-056"}

# ── TEST 1: headline counter vs aligned, full + OOS, with bootstrap CI ─────────
print("\n=== TEST 1: Counter vs Aligned (live 4-pair) ===")
c_full, a_full = split(live)
obs, p = perm_test(c_full, a_full)
med, lo, hi, ppos = bootstrap_delta_ci(c_full, a_full)
print(f"FULL: counter n={len(c_full)} Sharpe={sharpe(c_full):.3f} | aligned n={len(a_full)} Sharpe={sharpe(a_full):.3f}")
print(f"  delta={obs:+.3f} perm_p={p:.3f} | bootstrap delta CI[{lo:+.3f},{hi:+.3f}] P(>0)={ppos:.2f}")

oos = [t for t in live if year_of(t) >= 2023]
c_oos, a_oos = split(oos)
obs_o, p_o = perm_test(c_oos, a_oos)
med_o, lo_o, hi_o, ppos_o = bootstrap_delta_ci(c_oos, a_oos)
print(f"OOS(>=2023): counter n={len(c_oos)} Sharpe={sharpe(c_oos):.3f} | aligned n={len(a_oos)} Sharpe={sharpe(a_oos):.3f}")
print(f"  delta={obs_o:+.3f} perm_p={p_o:.3f} | bootstrap delta CI[{lo_o:+.3f},{hi_o:+.3f}] P(>0)={ppos_o:.2f}")
out["test1"] = {"full": {"n_counter": len(c_full), "n_aligned": len(a_full),
                          "sharpe_counter": sharpe(c_full), "sharpe_aligned": sharpe(a_full),
                          "delta": obs, "perm_p": p, "boot_ci": [lo, hi], "p_delta_gt0": ppos},
                "oos": {"n_counter": len(c_oos), "n_aligned": len(a_oos),
                        "sharpe_counter": sharpe(c_oos), "sharpe_aligned": sharpe(a_oos),
                        "delta": obs_o, "perm_p": p_o, "boot_ci": [lo_o, hi_o], "p_delta_gt0": ppos_o}}

# ── TEST 2: per-year walk-forward sign stability ──────────────────────────────
print("\n=== TEST 2: Per-year counter vs aligned Sharpe (sign stability) ===")
yearly = {}
for y in sorted(set(year_of(t) for t in live)):
    yt = [t for t in live if year_of(t) == y]
    cy, ay = split(yt)
    sc, sa = sharpe(cy), sharpe(ay)
    yearly[y] = {"n_c": len(cy), "n_a": len(ay), "sharpe_c": sc, "sharpe_a": sa, "delta": sc - sa}
    print(f"  {y}: counter Sharpe={sc:+.3f}(n={len(cy)}) aligned={sa:+.3f}(n={len(ay)}) delta={sc-sa:+.3f}")
pos_years = sum(1 for y in yearly.values() if y["delta"] > 0)
print(f"  Years counter beats aligned: {pos_years}/{len(yearly)}")
out["test2_yearly"] = yearly
out["test2_pos_years"] = f"{pos_years}/{len(yearly)}"

# ── TEST 3: drop-2023 (is OOS significance a single-year artifact?) ────────────
print("\n=== TEST 3: OOS excluding 2023 ===")
oos_no23 = [t for t in oos if year_of(t) != 2023]
c3, a3 = split(oos_no23)
obs3, p3 = perm_test(c3, a3)
print(f"  OOS w/o 2023: counter Sharpe={sharpe(c3):.3f}(n={len(c3)}) aligned={sharpe(a3):.3f}(n={len(a3)}) delta={obs3:+.3f} perm_p={p3:.3f}")
out["test3_oos_drop2023"] = {"sharpe_counter": sharpe(c3), "sharpe_aligned": sharpe(a3),
                              "delta": obs3, "perm_p": p3, "n_counter": len(c3), "n_aligned": len(a3)}

# ── TEST 4: threshold sensitivity (is the edge knife-edge at exactly mom=0?) ───
print("\n=== TEST 4: Momentum threshold sensitivity (full live universe) ===")
thr_res = {}
for thr in [0.0, 0.01, 0.02, 0.03, 0.05]:
    ct, at = split(live, thresh=thr)
    d = sharpe(ct) - sharpe(at)
    thr_res[thr] = {"delta": d, "n_c": len(ct), "n_a": len(at)}
    print(f"  thresh={thr:.2f}: delta={d:+.3f} (counter n={len(ct)}, aligned n={len(at)})")
out["test4_threshold"] = thr_res

# ── TEST 5: per-pair with multiple-testing (BH) correction ────────────────────
print("\n=== TEST 5: Per-pair counter vs aligned + BH correction ===")
pair_p = {}
for pair in sorted(LIVE_PAIRS):
    pt = [t for t in live if t['pair'] == pair]
    cp, ap = split(pt)
    if len(cp) < 5 or len(ap) < 5:
        print(f"  {pair}: underpowered (n_c={len(cp)}, n_a={len(ap)})");
        pair_p[pair] = {"delta": None, "p": None, "n_c": len(cp), "n_a": len(ap)}
        continue
    dp, pp = perm_test(cp, ap, n=5000)
    pair_p[pair] = {"delta": dp, "p": pp, "n_c": len(cp), "n_a": len(ap),
                    "sharpe_c": sharpe(cp), "sharpe_a": sharpe(ap)}
    print(f"  {pair}: counter={sharpe(cp):+.3f} aligned={sharpe(ap):+.3f} delta={dp:+.3f} perm_p={pp:.3f} (n_c={len(cp)},n_a={len(ap)})")
# BH on available p-values
ps = [(k, v["p"]) for k, v in pair_p.items() if v["p"] is not None]
ps_sorted = sorted(ps, key=lambda x: x[1])
m = len(ps_sorted)
print("  --- Benjamini-Hochberg (alpha=0.10) ---")
bh_any = False
for i, (k, pv) in enumerate(ps_sorted, 1):
    crit = 0.10 * i / m
    passed = pv <= crit
    bh_any = bh_any or passed
    print(f"    {k}: p={pv:.3f} vs BH crit={crit:.3f} -> {'PASS' if passed else 'fail'}")
out["test5_perpair"] = pair_p
out["test5_bh_any_significant"] = bh_any

# ── VERDICT ───────────────────────────────────────────────────────────────────
print("\n=== VERDICT ===")
checks = {
    "full_boot_excludes_0": out["test1"]["full"]["boot_ci"][0] > 0,
    "oos_boot_excludes_0": out["test1"]["oos"]["boot_ci"][0] > 0,
    "year_sign_stable_>=3of4_oos": sum(1 for y, v in yearly.items() if y >= 2023 and v["delta"] > 0) >= 3,
    "survives_drop_2023": p3 < 0.10 and obs3 > 0,
    "threshold_robust": all(v["delta"] > 0 for v in thr_res.values()),
    "bh_significant": bh_any,
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
n_pass = sum(checks.values())
verdict = "CONFIRM" if n_pass >= 5 else ("CANDIDATE" if n_pass >= 3 else "REJECT/UNPROVEN")
print(f"  => {n_pass}/6 checks pass -> {verdict}")
out["checks"] = checks
out["verdict"] = verdict
out["n_pass"] = f"{n_pass}/6"

json.dump(out, open('data/agent/rq_rest_007_hyp056_robustness.json', 'w'), indent=2)
print("\nSaved data/agent/rq_rest_007_hyp056_robustness.json")
