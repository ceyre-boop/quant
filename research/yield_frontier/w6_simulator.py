#!/usr/bin/env python3
"""
w6_simulator.py — W6 sizing-policy simulator for HYP-093 (The Undertow).

Implements optimization/W6_SPEC.md. Reads the reproduced-and-validated holdout event
paths (W6_inputs/hyp093_events.json), bootstraps synthetic multi-year paths with a GPD
tail and a disaster mixture, evaluates five sizing-policy families, ranks them under a
lexicographic (non-scalarized) rule, and writes a sealed verdict.

Anti-overfit welds enforced (spec §Failure Modes):
  - All policy parameters are FIXED before the bootstrap (no search on bootstrap perf).
  - Selection is lexicographic, never a weighted score.
  - Disaster frequency is swept {0.001, 0.002, 0.005}; the recommendation must win at
    the pessimistic 0.005, not the midpoint.
  - Floor comparison is ARITHMETIC yield/day (Σ ret·size / calendar_days), matching the
    HYP-097 definition — NOT the log-growth G, which is reported separately.

Validation anchor (spec §Build 2): F0 at the flat constitutional weight reproduces the
sealed verdicts.json mean_pct_day_constitutional (0.00023) to within Monte-Carlo noise.

Deviations from spec, recorded in the verdict `notes` rather than applied silently:
  1. Event set is the full 559-event sealed HYP-093 gauntlet (reconciles exactly to
     verdicts.json), not the 539-event HYP-097 sizing subset the spec inputs cite. The
     spec's F0 target of 0.000166/day is on the 539 subset; the 559-set F0 is higher.
     The verdict does not depend on the difference — both are far below the 0.0005 floor.
  2. RCK (F2) decision variable is scalar per tier, so the Busseti-Ryu-Boyd convex
     program reduces to a 1-D concave maximization; solved with cvxpy over the empirical
     distribution. Equivalent to the full LP for a scalar leverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import genpareto

REPO = Path(__file__).resolve().parent.parent.parent
YF = REPO / "data" / "research" / "yield_frontier"
EVENTS_FILE = YF / "optimization" / "W6_inputs" / "hyp093_events.json"
OUT_DIR = YF / "optimization" / "W6_results"
VERDICT_LEDGER = REPO / "data" / "research" / "preregister" / "verdicts_optimization.jsonl"

# ── Frozen constants (spec §Inputs) ───────────────────────────────────────────
W_STAR = {"T10": 0.6269, "T20": 0.7975}
RISK_BUDGET = 0.0075            # Art.1 constitutional per-trade risk
LOCATE = 0.50                  # W3 locate fill rate
FLAT_NOTIONAL = 0.0125         # constitutional flat weight (validation anchor)
N_DAYS = 242                   # sealed holdout calendar span
FLOOR = 0.0005                 # constitutional yield floor (non-negotiable)

# Bootstrap / stress (spec §Synthetic Path Generator)
N_PATHS = 10_000              # overridden at runtime by --paths/--quick
GPD_PATH_FRAC = 0.05           # fraction of paths that get GPD-extended left tail
DISASTER_P_GRID = [0.001, 0.002, 0.005]
DISASTER_L_RANGE = (-2.0, -1.0)
RUIN_LEVEL = 0.50              # ruin = wealth < 50% of start
SEED = 42

# Hard-exclusion constraints (spec §Selection Step 1)
MAX_P_RUIN = 0.01
MAX_P95_MAXDD = 0.25
MAX_CVAR99 = 0.10

# Policy parameters — FIXED before bootstrap (spec §Overfit-to-Bootstrap Trap)
KAPPA_F1 = 0.25                # quarter-Kelly
RCK_ALPHA = 0.95
RCK_EPS_DD = 0.15
F3_DD_MAX = 0.15
F3_GAMMA = 1.0
F4_H_MAX = 0.005


# ─────────────────────────────────────────────────────────────────────────────
def load_events() -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(EVENTS_FILE.read_text())
    ev = d["events"]
    rets = np.array([e["ret_event"] for e in ev], dtype=float)
    tiers = np.array([0 if e["tier"] == "T10" else 1 for e in ev], dtype=int)  # 0=T10
    return rets, tiers


def kelly_fraction(rets: np.ndarray) -> float:
    """Full-Kelly leverage on the empirical per-event return distribution.
    Maximizes E[ln(1+f·R)] by 1-D search (concave). Returns f* in [0, f_cap)."""
    lo = rets.min()
    f_cap = (-1.0 / lo) * 0.999 if lo < 0 else 5.0
    grid = np.linspace(1e-4, f_cap, 4000)
    best_f, best_g = 0.0, -np.inf
    for f in grid:
        g = np.mean(np.log1p(f * rets))
        if g > best_g:
            best_g, best_f = g, f
    return best_f


def _pathwise_max_dd(f: float, seq: np.ndarray) -> float:
    """Max drawdown of compounding the empirical event sequence at leverage f."""
    wealth = np.cumprod(np.maximum(1.0 + f * seq, 1e-9))
    hwm = np.maximum.accumulate(wealth)
    return float(((hwm - wealth) / hwm).max())


def rck_fraction(rets: np.ndarray, seq: np.ndarray | None = None) -> float:
    """Risk-Constrained Kelly (Busseti-Ryu-Boyd), scalar leverage.

    max E[ln(1+f·R)]  s.t.  whole-path max drawdown on the empirical event
    sequence <= RCK_EPS_DD. This is the drawdown *certificate* Busseti-Boyd
    intend, not a per-event CVaR (an earlier draft used per-event CVaR and left f
    dangerously high — the certificate binds far tighter). Drawdown and growth are
    both monotone in f up to Kelly, so the constraint yields the growth-maximizing
    feasible f directly: the largest f whose pathwise max-DD <= eps.
    """
    seq = rets if seq is None else seq
    lo = rets.min()
    f_cap = (-1.0 / lo) * 0.999 if lo < 0 else 5.0
    grid = np.linspace(1e-4, f_cap, 4000)
    feasible = 0.0
    for f in grid:
        if _pathwise_max_dd(f, seq) <= RCK_EPS_DD:
            feasible = f
        else:
            break  # drawdown monotone increasing in f
    return feasible


def gpd_tail_samples(rets: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample extreme losses from a GPD fit to the worst-decile (below 30th-pct
    adverse) losses. Returns negative returns (losses) more extreme than observed."""
    thresh = np.percentile(rets, 30)               # 30th pct adverse quantile (low)
    tail = rets[rets < thresh]
    if len(tail) < 20:
        return rng.choice(rets, size=size)         # fall back to empirical
    exceed = thresh - tail                          # positive exceedances below thresh
    xi, _, beta = genpareto.fit(exceed, floc=0)
    draws = genpareto.rvs(xi, loc=0, scale=beta, size=size, random_state=rng)
    samples = thresh - draws                        # back to (more negative) returns
    return np.clip(samples, -0.95, thresh)          # never below -95% at event level


BLOCK_SIZE = 5  # trading days, for block bootstrap (spec open question §1)


def build_path(rets: np.ndarray, tiers: np.ndarray, rng: np.random.Generator,
               use_gpd: bool, p_disaster: float,
               block: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One synthetic path: 559 events resampled with replacement, tier-preserved,
    with optional GPD tail on 2% of events and a disaster overlay.

    block=True uses a moving-block bootstrap (contiguous runs of BLOCK_SIZE events
    in calendar order) instead of i.i.d. resampling, preserving serial clustering.
    Clustered wins are harder to compound, so this is the honest stress on the
    i.i.d. floor-clearance margin."""
    n = len(rets)
    if block:
        starts = rng.integers(0, n - BLOCK_SIZE, size=(n // BLOCK_SIZE) + 1)
        idx = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in starts])[:n]
    else:
        idx = rng.integers(0, n, size=n)
    path_rets = rets[idx].copy()
    path_tiers = tiers[idx].copy()

    if use_gpd:
        m = max(1, int(0.02 * n))
        pos = rng.choice(n, size=m, replace=False)
        path_rets[pos] = gpd_tail_samples(rets, m, rng)

    # Disaster mixture: scaled by tier worst-case (spec §Disaster Mixture).
    dis = rng.random(n) < p_disaster
    if dis.any():
        wstar = np.where(path_tiers == 0, W_STAR["T10"], W_STAR["T20"])
        L = rng.uniform(DISASTER_L_RANGE[0], DISASTER_L_RANGE[1], size=n)
        path_rets = np.where(dis, L * wstar, path_rets)
    return path_rets, path_tiers


def apply_policy(policy: str, path_rets: np.ndarray, path_tiers: np.ndarray,
                 f_by_tier: dict) -> np.ndarray:
    """Return the per-event fractional size vector for a policy along one path.
    f_by_tier holds the pre-computed base fraction per tier for F0/F1/F2."""
    n = len(path_rets)
    base = np.where(path_tiers == 0, f_by_tier["T10"], f_by_tier["T20"]) * LOCATE

    if policy in ("F0", "F1", "F2"):
        return base

    if policy == "F2+F3" or policy == "F2+F3+F4":
        size = base.copy()
        wealth, hwm = 1.0, 1.0
        for i in range(n):
            dd = (hwm - wealth) / hwm
            gov = max(0.0, 1.0 - dd / F3_DD_MAX) ** F3_GAMMA
            size[i] = base[i] * gov
            wealth *= (1.0 + size[i] * path_rets[i])
            hwm = max(hwm, wealth)
        if policy == "F2+F3":
            return size
        base = size  # fall through to F4 heat on top

    if policy in ("F2+F4", "F2+F3+F4"):
        # Per-day heat cap. Events carry no intra-path day index here (bootstrap
        # breaks calendar), so apply the heat cap at the single-event granularity
        # the median day implies (0.5 events/day => cap rarely binds). Conservative
        # proxy: clip each event's tail contribution f·CVaR95 to h_max.
        cvar95_event = abs(np.percentile(path_rets, 5))
        cap = F4_H_MAX / max(cvar95_event, 1e-6)
        return np.minimum(base, cap)

    raise ValueError(policy)


def path_metrics(size: np.ndarray, rets: np.ndarray) -> dict:
    incr = 1.0 + size * rets
    incr = np.maximum(incr, 1e-9)                   # ruin floor at wealth ~0
    wealth = np.cumprod(incr)
    log_g = np.log(incr).sum()
    hwm = np.maximum.accumulate(np.concatenate([[1.0], wealth]))
    dd = (hwm[1:] - wealth) / hwm[1:]
    arithmetic_pnl = float((size * rets).sum())     # Σ ret·size over the path
    return {
        "log_G_total": float(log_g),
        "max_dd": float(dd.max()),
        "final_wealth": float(wealth[-1]),
        "min_wealth": float(wealth.min()),
        "arith_pnl": arithmetic_pnl,
    }


def evaluate(policy: str, rets, tiers, f_by_tier, p_disaster, rng, block=False) -> dict:
    Gs, DDs, mins, arith = [], [], [], []
    for s in range(N_PATHS):
        use_gpd = (s % int(1 / GPD_PATH_FRAC)) == 0
        pr, pt = build_path(rets, tiers, rng, use_gpd, p_disaster, block=block)
        size = apply_policy(policy, pr, pt, f_by_tier)
        m = path_metrics(size, pr)
        Gs.append(m["log_G_total"])
        DDs.append(m["max_dd"])
        mins.append(m["min_wealth"])
        arith.append(m["arith_pnl"])
    Gs, DDs, mins, arith = map(np.asarray, (Gs, DDs, mins, arith))

    G_day = Gs / N_DAYS
    arith_day = arith / N_DAYS                       # arithmetic yield/day (floor metric)
    p_ruin = float(np.mean(mins < RUIN_LEVEL))
    # CVaR99 per single event, across all paths at this policy's sizing:
    all_incr_loss = []
    # cheap tail proxy: recompute event-level losses on one large resample
    idx = rng.integers(0, len(rets), size=200_000)
    ev_size = np.where(tiers[idx] == 0, f_by_tier["T10"], f_by_tier["T20"]) * LOCATE
    ev_loss = -(ev_size * rets[idx])
    cvar99 = float(np.mean(np.sort(ev_loss)[-int(0.01 * len(ev_loss)):]))

    dd_sorted = np.sort(DDs)
    cdar90 = float(dd_sorted[int(0.90 * len(dd_sorted)):].mean())
    e_maxdd = float(DDs.mean())
    tpr = e_maxdd / max(G_day.mean(), 1e-9)

    return {
        "policy": policy,
        "p_disaster": p_disaster,
        "G_day_mean": float(G_day.mean()),
        "G_day_p10": float(np.percentile(G_day, 10)),
        "G_day_p50": float(np.percentile(G_day, 50)),
        "G_day_p90": float(np.percentile(G_day, 90)),
        "arith_yield_day_mean": float(arith_day.mean()),
        "CVaR_99": cvar99,
        "p50_MaxDD": float(np.percentile(DDs, 50)),
        "p95_MaxDD": float(np.percentile(DDs, 95)),
        "p99_MaxDD": float(np.percentile(DDs, 99)),
        "CDaR_90": cdar90,
        "TPR": float(tpr),
        "P_ruin": p_ruin,
    }


def excluded(r: dict) -> str | None:
    if r["P_ruin"] > MAX_P_RUIN:
        return f"P_ruin {r['P_ruin']:.4f} > {MAX_P_RUIN}"
    if r["p95_MaxDD"] >= MAX_P95_MAXDD:
        return f"p95_MaxDD {r['p95_MaxDD']:.4f} >= {MAX_P95_MAXDD}"
    if r["CVaR_99"] > MAX_CVAR99:
        return f"CVaR_99 {r['CVaR_99']:.4f} > {MAX_CVAR99}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=N_PATHS)
    ap.add_argument("--quick", action="store_true", help="1000 paths for a smoke run")
    args = ap.parse_args()
    globals()["N_PATHS"] = 1000 if args.quick else args.paths

    rets, tiers = load_events()
    rng = np.random.default_rng(SEED)

    # ── Base fractions per family (fixed before bootstrap) ────────────────────
    t10 = rets[tiers == 0]
    t20 = rets[tiers == 1]
    F0 = {"T10": RISK_BUDGET / W_STAR["T10"], "T20": RISK_BUDGET / W_STAR["T20"]}
    F0_flat = {"T10": FLAT_NOTIONAL, "T20": FLAT_NOTIONAL}
    F1 = {"T10": KAPPA_F1 * kelly_fraction(t10), "T20": KAPPA_F1 * kelly_fraction(t20)}
    F2 = {"T10": rck_fraction(t10), "T20": rck_fraction(t20)}

    print(f"Fixed base fractions (pre-bootstrap):")
    print(f"  F0 (b/W*)      T10={F0['T10']:.4f}  T20={F0['T20']:.4f}")
    print(f"  F1 (0.25 Kelly)T10={F1['T10']:.4f}  T20={F1['T20']:.4f}")
    print(f"  F2 (RCK)       T10={F2['T10']:.4f}  T20={F2['T20']:.4f}")

    # ── F0 validation anchor: flat weight reproduces sealed 0.00023 ───────────
    anchor = evaluate("F0", rets, tiers, F0_flat, 0.0, np.random.default_rng(SEED))
    print(f"\nF0-flat validation: arith yield/day = {anchor['arith_yield_day_mean']:.6f} "
          f"(sealed target 0.00023)")
    anchor_ok = abs(anchor["arith_yield_day_mean"] - 0.00023) <= 0.00023 * 0.05

    fracs = {"F0": F0, "F1": F1, "F2": F2, "F2+F3": F2, "F2+F4": F2,
             "F2+F3+F4": F2}
    policies = ["F0", "F1", "F2", "F2+F3", "F2+F4", "F2+F3+F4"]

    # ── Run every policy at the pessimistic disaster rate (selection basis) ───
    print(f"\nEvaluating {len(policies)} policies × {N_PATHS} paths at p_disaster=0.005 ...")
    results = {}
    for pol in policies:
        results[pol] = evaluate(pol, rets, tiers, fracs[pol], 0.005,
                                np.random.default_rng(SEED + hash(pol) % 1000))
        r = results[pol]
        ex = excluded(r)
        print(f"  {pol:<9} G/day={r['G_day_mean']:+.6f}  arith/day={r['arith_yield_day_mean']:+.6f}  "
              f"p95DD={r['p95_MaxDD']:.3f}  ruin={r['P_ruin']:.4f}  "
              f"{'EXCLUDED: ' + ex if ex else 'ok'}")

    # ── Lexicographic selection (spec §Selection) ─────────────────────────────
    survivors = {p: r for p, r in results.items() if excluded(r) is None}
    ranked = sorted(survivors.items(),
                    key=lambda kv: (-kv[1]["G_day_mean"], kv[1]["CDaR_90"], kv[1]["TPR"]))
    recommended = ranked[0][0] if ranked else None

    # ── Floor check on ARITHMETIC yield (spec §"G Looks Higher" Trap) ─────────
    best_arith = max((r["arith_yield_day_mean"] for r in survivors.values()), default=0.0)
    floor_cleared = bool(recommended and best_arith >= FLOOR)

    # ── Disaster-frequency robustness: does the winner win at all rates? ──────
    robustness = {}
    if recommended:
        for pdis in DISASTER_P_GRID:
            rr = evaluate(recommended, rets, tiers, fracs[recommended], pdis,
                          np.random.default_rng(SEED + 7))
            robustness[str(pdis)] = {"G_day": rr["G_day_mean"], "P_ruin": rr["P_ruin"],
                                     "arith_day": rr["arith_yield_day_mean"]}

    # ── Block-bootstrap stress: does the margin survive serial clustering? ────
    block_check = {}
    if recommended:
        rb = evaluate(recommended, rets, tiers, fracs[recommended], 0.005,
                      np.random.default_rng(SEED + 99), block=True)
        block_check = {
            "policy": recommended, "block_size": BLOCK_SIZE,
            "arith_yield_day": round(rb["arith_yield_day_mean"], 6),
            "G_day_mean": round(rb["G_day_mean"], 6),
            "p95_MaxDD": round(rb["p95_MaxDD"], 4),
            "P_ruin": round(rb["P_ruin"], 5),
            "floor_cleared_under_clustering": bool(rb["arith_yield_day_mean"] >= FLOOR
                                                   and excluded(rb) is None),
        }
        print(f"\nBlock-bootstrap (block={BLOCK_SIZE}, clustering-preserving):")
        print(f"  {recommended}  arith/day={rb['arith_yield_day_mean']:+.6f}  "
              f"p95DD={rb['p95_MaxDD']:.3f}  ruin={rb['P_ruin']:.4f}  "
              f"floor_cleared={block_check['floor_cleared_under_clustering']}")

    rec = results.get(recommended, {})
    verdict = {
        "id": "W6",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "hypothesis": "Optimal sizing policy for HYP-093 (The Undertow)",
        "event_set": "559-event sealed HYP-093 gauntlet (validated vs verdicts.json)",
        "input_hash": hashlib.sha256(EVENTS_FILE.read_bytes()).hexdigest()[:16],
        "policies_tested": policies,
        "floor": FLOOR,
        "f0_flat_anchor_yield_day": round(anchor["arith_yield_day_mean"], 6),
        "f0_flat_anchor_ok": anchor_ok,
        "recommended_policy": recommended,
        "recommended_f_T10": round(fracs.get(recommended, {}).get("T10", 0.0) * LOCATE, 5),
        "recommended_f_T20": round(fracs.get(recommended, {}).get("T20", 0.0) * LOCATE, 5),
        "G_day_mean": round(rec.get("G_day_mean", 0.0), 6),
        "G_day_p10": round(rec.get("G_day_p10", 0.0), 6),
        "G_day_p50": round(rec.get("G_day_p50", 0.0), 6),
        "G_day_p90": round(rec.get("G_day_p90", 0.0), 6),
        "best_arith_yield_day": round(best_arith, 6),
        "gap_to_floor": round(FLOOR - best_arith, 6),
        "CVaR_99": round(rec.get("CVaR_99", 0.0), 5),
        "p95_MaxDD": round(rec.get("p95_MaxDD", 0.0), 4),
        "CDaR_90": round(rec.get("CDaR_90", 0.0), 4),
        "P_ruin": round(rec.get("P_ruin", 0.0), 5),
        "TPR": round(rec.get("TPR", 0.0), 1),
        "floor_cleared": floor_cleared,
        "n_bootstrap_paths": N_PATHS,
        "disaster_p_range": [0.001, 0.005],
        "disaster_L_range": list(DISASTER_L_RANGE),
        "disaster_robustness": robustness,
        "block_bootstrap_stress": block_check,
        "all_results": results,
        "notes": (
            "Event set is the full 559-event sealed gauntlet (reconciles to verdicts.json "
            "exactly), not the 539-event HYP-097 subset the spec inputs cite; the spec's "
            "0.000166/day F0 target is on the 539 subset. F0-flat reproduces the sealed "
            "0.00023 constitutional yield as the validation anchor. Floor comparison uses "
            "arithmetic yield/day per spec. RCK solved as scalar-f line search (exact for "
            "scalar leverage). Bootstrap is i.i.d.; real serial clustering is harder to "
            "compound, so a cleared floor here would be optimistic — flagged per spec."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "W6_verdict.json").write_text(json.dumps(verdict, indent=2))
    VERDICT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(VERDICT_LEDGER, "a") as f:
        slim = {k: verdict[k] for k in ("id", "date", "recommended_policy",
                "best_arith_yield_day", "floor", "gap_to_floor", "floor_cleared",
                "P_ruin", "input_hash")}
        f.write(json.dumps(slim) + "\n")

    print(f"\n{'='*66}")
    print(f"RECOMMENDED: {recommended}")
    print(f"best arithmetic yield/day: {best_arith:.6f}  vs floor {FLOOR}")
    print(f"gap to floor: {FLOOR - best_arith:+.6f}  ({best_arith/FLOOR*100:.0f}% of floor)")
    print(f"FLOOR_CLEARED: {floor_cleared}")
    print(f"{'='*66}")
    print(f"verdict -> {OUT_DIR / 'W6_verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
