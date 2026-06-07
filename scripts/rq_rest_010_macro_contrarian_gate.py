"""
RQ-REST-010: Formal permutation test — HYP-055 USDJPY+USDCAD macro contrarian gate
REST-006 cycle, 2026-06-07

Hypothesis: For USDJPY and USDCAD, only entering when macro_vs_direction=-1
(macro signal OPPOSES trade direction) yields significantly better outcomes.
REST-005 discovered per-pair signals. This script formally tests statistical significance.

Also tests:
  - IS vs OOS split
  - Per-year breakdown
  - EURUSD reversed pattern (MVD=+1 better) — is that significant too?
  - 3-gate stack: widening + rate level + HYP-055
"""

import json
import numpy as np
from collections import defaultdict

FORENSICS_PATH = '/sessions/great-nifty-pasteur/mnt/quant/data/research/trade_forensics.json'

def sharpe(returns):
    r = np.array(returns, dtype=float)
    if len(r) < 3 or np.std(r) < 1e-9:
        return 0.0
    return np.mean(r) / np.std(r)

def permutation_test(group_a, group_b, n_perm=10000, seed=42):
    """
    Two-group permutation test on Sharpe difference.
    H0: The split is random — the Sharpe difference between group A and group B
        is consistent with random label assignment.
    Returns: observed_delta, p_value, perm_deltas
    """
    rng = np.random.default_rng(seed)
    obs_delta = sharpe(group_a) - sharpe(group_b)
    combined = np.array(group_a + group_b, dtype=float)
    n_a = len(group_a)
    count_ge = 0
    perm_deltas = []
    for _ in range(n_perm):
        idx = rng.permutation(len(combined))
        perm_a = combined[idx[:n_a]]
        perm_b = combined[idx[n_a:]]
        d = sharpe(perm_a) - sharpe(perm_b)
        perm_deltas.append(d)
        if d >= obs_delta:
            count_ge += 1
    p_val = count_ge / n_perm
    return obs_delta, p_val, perm_deltas

def load_4pair_trades():
    with open(FORENSICS_PATH) as f:
        trades = json.load(f)
    live_pairs = {'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCAD=X'}
    return [t for t in trades if t['pair'] in live_pairs]

def apply_widening_gate(trades):
    """Apply HYP-052c: each trade passes only if |rrd| >= prev |rrd| for that pair."""
    by_pair = defaultdict(list)
    for t in sorted(trades, key=lambda x: x['entry_date']):
        by_pair[t['pair']].append(t)
    
    widening = []
    for pair_trades in by_pair.values():
        prev_rrd = None
        for t in pair_trades:
            curr_rrd = abs(t.get('real_rate_diff', 0))
            if prev_rrd is None or curr_rrd >= prev_rrd:
                widening.append(t)
            prev_rrd = curr_rrd
    return widening

def apply_rate_level_gate(trades, threshold=1.0):
    """Apply HYP-054: require |real_rate_diff| >= threshold."""
    return [t for t in trades if abs(t.get('real_rate_diff', 0)) >= threshold]

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
trades = load_4pair_trades()
print(f"4-pair trades loaded: {len(trades)}")

is_trades = [t for t in trades if int(t['entry_date'][:4]) < 2023]
oos_trades = [t for t in trades if int(t['entry_date'][:4]) >= 2023]
print(f"IS (2015-2022): {len(is_trades)}, OOS (2023-2024): {len(oos_trades)}")

# ─── SECTION 1: HYP-055 PERMUTATION TESTS ────────────────────────────────────
print("\n" + "="*60)
print("SECTION 1: HYP-055 — USDJPY+USDCAD MVD=-1 contrarian gate")
print("="*60)

target_pairs = {'USDJPY=X', 'USDCAD=X'}

for label, subset in [('IS (2015-2022)', is_trades), ('OOS (2023-2024)', oos_trades), ('Full period', trades)]:
    target = [t for t in subset if t['pair'] in target_pairs]
    mvd_neg1 = [t['outcome_r'] for t in target if t['macro_vs_direction'] == -1]
    mvd_pos1 = [t['outcome_r'] for t in target if t['macro_vs_direction'] == 1]
    
    if len(mvd_neg1) < 5 or len(mvd_pos1) < 5:
        print(f"\n{label}: insufficient data (n_neg={len(mvd_neg1)}, n_pos={len(mvd_pos1)})")
        continue
    
    obs_delta, p_val, perm_deltas = permutation_test(mvd_neg1, mvd_pos1)
    perm_arr = np.array(perm_deltas)
    ci_lo, ci_hi = np.percentile(perm_arr, [2.5, 97.5])
    
    print(f"\n{label} — USDJPY+USDCAD (n={len(target)} trades):")
    print(f"  MVD=-1 (contrarian): n={len(mvd_neg1):3d}, Sharpe={sharpe(mvd_neg1):.3f}")
    print(f"  MVD=+1 (aligned):    n={len(mvd_pos1):3d}, Sharpe={sharpe(mvd_pos1):.3f}")
    print(f"  Observed delta:  {obs_delta:+.3f}")
    print(f"  Permutation p:   {p_val:.4f}  (N=10,000)")
    print(f"  Null 95% CI:     [{ci_lo:.3f}, {ci_hi:.3f}]")
    verdict = "SIGNIFICANT" if p_val < 0.05 else ("MARGINAL" if p_val < 0.10 else "NOT SIGNIFICANT")
    print(f"  Verdict:         {verdict}")

# ─── SECTION 1b: Per-pair breakdown for USDJPY and USDCAD ────────────────────
print("\n--- Per-pair permutation results ---")
for pair in ['USDJPY=X', 'USDCAD=X']:
    pair_trades = [t for t in trades if t['pair'] == pair]
    mvd_neg1 = [t['outcome_r'] for t in pair_trades if t['macro_vs_direction'] == -1]
    mvd_pos1 = [t['outcome_r'] for t in pair_trades if t['macro_vs_direction'] == 1]
    
    if len(mvd_neg1) < 5 or len(mvd_pos1) < 5:
        print(f"\n{pair}: insufficient data")
        continue
    
    obs_delta, p_val, _ = permutation_test(mvd_neg1, mvd_pos1)
    pname = pair.replace('=X','')
    print(f"\n{pname}: MVD=-1 n={len(mvd_neg1)}, Sharpe={sharpe(mvd_neg1):.3f} | MVD=+1 n={len(mvd_pos1)}, Sharpe={sharpe(mvd_pos1):.3f} | delta={obs_delta:+.3f} | p={p_val:.4f}")

# ─── SECTION 2: EURUSD reversed pattern ──────────────────────────────────────
print("\n" + "="*60)
print("SECTION 2: EURUSD reversed pattern (MVD=+1 better?)")
print("="*60)
eur_trades = [t for t in trades if t['pair'] == 'EURUSD=X']
eur_mvd_pos1 = [t['outcome_r'] for t in eur_trades if t['macro_vs_direction'] == 1]
eur_mvd_neg1 = [t['outcome_r'] for t in eur_trades if t['macro_vs_direction'] == -1]

obs_delta_eur, p_val_eur, _ = permutation_test(eur_mvd_pos1, eur_mvd_neg1)
print(f"EURUSD MVD=+1: n={len(eur_mvd_pos1)}, Sharpe={sharpe(eur_mvd_pos1):.3f}")
print(f"EURUSD MVD=-1: n={len(eur_mvd_neg1)}, Sharpe={sharpe(eur_mvd_neg1):.3f}")
print(f"Delta (pos1-neg1): {obs_delta_eur:+.3f}, p={p_val_eur:.4f}")

# ─── SECTION 3: Year-by-year for USDJPY+USDCAD MVD filter ───────────────────
print("\n" + "="*60)
print("SECTION 3: Year-by-year Sharpe — USDJPY+USDCAD MVD=-1 vs all")
print("="*60)
print(f"{'Year':<6} {'All_n':>6} {'All_S':>7} {'MVD-1_n':>8} {'MVD-1_S':>8} {'Delta':>7}")
for yr in range(2015, 2025):
    yr_all = [t['outcome_r'] for t in trades if t['pair'] in target_pairs and int(t['entry_date'][:4]) == yr]
    yr_neg1 = [t['outcome_r'] for t in trades if t['pair'] in target_pairs and int(t['entry_date'][:4]) == yr and t['macro_vs_direction'] == -1]
    if not yr_all:
        continue
    s_all = sharpe(yr_all)
    s_neg1 = sharpe(yr_neg1) if yr_neg1 else float('nan')
    delta = s_neg1 - s_all if yr_neg1 else float('nan')
    marker = "<-- OOS" if yr >= 2023 else ""
    print(f"{yr:<6} {len(yr_all):>6} {s_all:>7.3f} {len(yr_neg1):>8} {s_neg1:>8.3f} {delta:>7.3f} {marker}")

# ─── SECTION 4: Three-gate stack ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 4: Three-gate stack — widening + rate level + HYP-055")
print("="*60)

# Build widening tags
by_pair_sorted = defaultdict(list)
for t in sorted(trades, key=lambda x: x['entry_date']):
    by_pair_sorted[t['pair']].append(t)

widening_ids = set()
for pair_trades in by_pair_sorted.values():
    prev_rrd = None
    for t in pair_trades:
        curr_rrd = abs(t.get('real_rate_diff', 0))
        trade_id = (t['pair'], t['entry_date'])
        if prev_rrd is None or curr_rrd >= prev_rrd:
            widening_ids.add(trade_id)
        prev_rrd = curr_rrd

# Gate definitions
def gate_widening(t):
    return (t['pair'], t['entry_date']) in widening_ids

def gate_rate_level(t, thresh=1.0):
    return abs(t.get('real_rate_diff', 0)) >= thresh

def gate_hyp055(t):
    """USDJPY and USDCAD: only MVD=-1. EURUSD/GBPUSD: no filter."""
    pair = t['pair']
    if pair in {'USDJPY=X', 'USDCAD=X'}:
        return t['macro_vs_direction'] == -1
    return True  # EURUSD and GBPUSD pass through

def gate_hyp055_extended(t):
    """Extended: USDJPY+USDCAD MVD=-1, EURUSD MVD=+1, GBPUSD unchanged."""
    pair = t['pair']
    if pair in {'USDJPY=X', 'USDCAD=X'}:
        return t['macro_vs_direction'] == -1
    if pair == 'EURUSD=X':
        return t['macro_vs_direction'] == 1
    return True

combos = [
    ("2-gate (widening + rate level)", lambda t: gate_widening(t) and gate_rate_level(t)),
    ("3-gate (+ HYP-055 USDJPY/USDCAD)", lambda t: gate_widening(t) and gate_rate_level(t) and gate_hyp055(t)),
    ("3-gate-ext (+ EURUSD MVD=+1 too)", lambda t: gate_widening(t) and gate_rate_level(t) and gate_hyp055_extended(t)),
]

for name, gate_fn in combos:
    gated = [t for t in trades if gate_fn(t)]
    gated_is = [t for t in gated if int(t['entry_date'][:4]) < 2023]
    gated_oos = [t for t in gated if int(t['entry_date'][:4]) >= 2023]
    
    s_all = sharpe([t['outcome_r'] for t in gated])
    s_is = sharpe([t['outcome_r'] for t in gated_is])
    s_oos = sharpe([t['outcome_r'] for t in gated_oos])
    
    print(f"\n{name}:")
    print(f"  n={len(gated)} ({len(gated_is)} IS, {len(gated_oos)} OOS)")
    print(f"  IS={s_is:.3f}, OOS={s_oos:.3f}, Full={s_all:.3f}")
    
    # Year-by-year
    for yr in [2021, 2022, 2023, 2024]:
        yr_r = [t['outcome_r'] for t in gated if int(t['entry_date'][:4]) == yr]
        marker = "<-- OOS" if yr >= 2023 else ""
        print(f"    {yr}: n={len(yr_r):3d}, Sharpe={sharpe(yr_r):.3f} {marker}")

# ─── SECTION 5: MVD filter on EURUSD IS vs OOS ───────────────────────────────
print("\n" + "="*60)
print("SECTION 5: EURUSD MVD=+1 filter IS vs OOS breakdown")
print("="*60)
for label, subset in [('IS', is_trades), ('OOS', oos_trades)]:
    eur = [t for t in subset if t['pair'] == 'EURUSD=X']
    for mvd in [-1, 1]:
        rs = [t['outcome_r'] for t in eur if t['macro_vs_direction'] == mvd]
        if rs:
            print(f"EURUSD {label} MVD={mvd:+d}: n={len(rs):3d}, Sharpe={sharpe(rs):.3f}")

print("\nDone.")
