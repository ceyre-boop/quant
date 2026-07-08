#!/usr/bin/env python3
"""
RQ-REST-023 — Reconcile + re-validate the two rate gates (HYP-052c, HYP-054)
============================================================================

Motivation (FIND-REST-022-c/d):
  HYP-054's recorded delta-Sharpe (pass 0.650 vs fail 0.199, delta 0.451) is ~4x
  the per-trade label-permutation measurement from RQ-REST-022 (0.201 vs 0.084,
  delta 0.117). Both scripts use the SAME per-trade mean/std Sharpe, so the gap
  must be DATA UNIVERSE (rq_rest_009 = all 7 pairs; rq_rest_022 = 4 live pairs),
  THRESHOLD (>=1.0 vs >1.0), or an equity-curve vs per-trade Sharpe definition.

This script pins down the source by computing the gate effect under every
combination, then attaches a bootstrap CI + label-permutation p to the ORIGINAL
(rq_rest_009 native) statistic so HYP-052c/HYP-054 can be honestly graded.

Offline. No network. No live change. Reads cached forensic json only.
Output: data/research/rq_rest_023_rate_gate_reconcile.json
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
FX = ROOT / "data" / "research" / "trade_forensics.json"
OUT = ROOT / "data" / "research" / "rq_rest_023_rate_gate_reconcile.json"
LIVE4 = ("GBPUSD", "EURUSD", "AUDUSD", "GBPJPY")
N_PERM = 20000
N_BOOT = 10000
RNG = np.random.default_rng(42)


def per_trade_sharpe(r):
    r = np.asarray(r, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1))


def curve_sharpe_annualized(trades, r_col="outcome_r", date_col="entry_date",
                            periods_per_year=252):
    """Equity-curve Sharpe: aggregate per-trade R into a daily PnL series and
    annualize. Tests the 'recorded number is a portfolio-curve Sharpe' theory."""
    df = pd.DataFrame(trades)
    df = df[df[r_col].notna()].copy()
    if len(df) < 2:
        return float("nan")
    df["d"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    daily = df.groupby("d")[r_col].sum().sort_index()
    if daily.std(ddof=1) == 0 or len(daily) < 2:
        return float("nan")
    return float(daily.mean() / daily.std(ddof=1) * np.sqrt(periods_per_year))


def two_sided_p(obs, null):
    null = np.asarray(null)
    return float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (len(null) + 1))


def perm_p_sharpe_diff(r, mask, n=N_PERM):
    r = np.asarray(r, float)
    obs = per_trade_sharpe(r[mask]) - per_trade_sharpe(r[~mask])
    k = int(mask.sum())
    idx = np.arange(len(r))
    null = np.empty(n)
    for i in range(n):
        m = np.zeros(len(r), bool)
        m[RNG.choice(idx, k, replace=False)] = True
        null[i] = per_trade_sharpe(r[m]) - per_trade_sharpe(r[~m])
    return obs, two_sided_p(obs, null)


def boot_ci_sharpe_diff(r, mask, n=N_BOOT, lo=2.5, hi=97.5):
    """Bootstrap CI of the per-trade Sharpe difference by resampling WITHIN each
    arm (preserves arm sizes)."""
    r = np.asarray(r, float)
    a = r[mask]
    b = r[~mask]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    diffs = np.empty(n)
    ia = np.arange(len(a))
    ib = np.arange(len(b))
    for i in range(n):
        sa = a[RNG.choice(ia, len(a), replace=True)]
        sb = b[RNG.choice(ib, len(b), replace=True)]
        diffs[i] = per_trade_sharpe(sa) - per_trade_sharpe(sb)
    return (float(np.percentile(diffs, lo)), float(np.percentile(diffs, hi)))


def arm_stats(r):
    r = np.asarray(r, float)
    return {
        "n": int(len(r)),
        "sharpe": round(per_trade_sharpe(r), 4),
        "mean_r": round(float(r.mean()), 4) if len(r) else float("nan"),
        "win_rate": round(float((r > 0).mean()), 4) if len(r) else float("nan"),
    }


def load(universe):
    fx = pd.DataFrame(json.load(open(FX)))
    fx["pair_s"] = fx["pair"].str.replace("=X", "", regex=False)
    fx["dt"] = pd.to_datetime(fx["entry_date"], errors="coerce")
    fx = fx[fx["outcome_r"].notna()].copy()
    if universe == "live4":
        fx = fx[fx["pair_s"].isin(LIVE4)].copy()
    return fx


def hyp054_masks(fx, op):
    rd = fx["real_rate_diff"].abs().to_numpy()
    if op == "ge":
        return rd >= 1.0
    return rd > 1.0


def hyp052c_setup(fx):
    g = fx.sort_values(["pair_s", "dt"]).copy()
    g["abs_rd"] = g["real_rate_diff"].abs()
    g["prev_abs"] = g.groupby("pair_s")["abs_rd"].shift(1)
    g = g[g["prev_abs"].notna()].copy()
    widen = (g["abs_rd"] > g["prev_abs"]).to_numpy()
    return g, widen


def run_gate(name, fx, mask, do_perm=True):
    r = fx["outcome_r"].to_numpy(float)
    pass_s = arm_stats(r[mask])
    fail_s = arm_stats(r[~mask])
    delta = pass_s["sharpe"] - fail_s["sharpe"]
    out = {
        "n_total": int(len(r)),
        "pass": pass_s,
        "fail": fail_s,
        "delta_sharpe_per_trade": round(float(delta), 4),
        "mean_r_diff": round(float(r[mask].mean() - r[~mask].mean()), 4),
        "curve_sharpe_pass_annualized": round(
            curve_sharpe_annualized(fx[mask].to_dict("records")), 4),
        "curve_sharpe_fail_annualized": round(
            curve_sharpe_annualized(fx[~mask].to_dict("records")), 4),
    }
    if do_perm:
        obs, p = perm_p_sharpe_diff(r, mask)
        ci = boot_ci_sharpe_diff(r, mask)
        out["perm_p_two_sided"] = round(p, 5)
        out["boot_ci95_delta_sharpe"] = [round(ci[0], 4), round(ci[1], 4)]
    return out


def main():
    report = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "task": "RQ-REST-023",
        "purpose": "Reconcile HYP-054/HYP-052c magnitude discrepancy; add bootstrap CI + permutation p on native statistic.",
        "data": str(FX.relative_to(ROOT)),
        "scenarios": {},
    }

    # ---- HYP-054 across universes x threshold operators ----
    for universe in ("all7", "live4"):
        fx = load(universe)
        for op in ("ge", "gt"):
            mask = hyp054_masks(fx, op)
            key = f"HYP-054__{universe}__rd_{'ge' if op=='ge' else 'gt'}_1.0"
            report["scenarios"][key] = run_gate(key, fx, mask)

    # ---- HYP-052c across universes ----
    for universe in ("all7", "live4"):
        fx = load(universe)
        g, widen = hyp052c_setup(fx)
        key = f"HYP-052c__{universe}__widening"
        report["scenarios"][key] = run_gate(key, g, widen)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    # ---- console summary ----
    print("=" * 78)
    print("RQ-REST-023 RATE-GATE RECONCILIATION")
    print("=" * 78)
    for k, v in report["scenarios"].items():
        print(f"\n{k}")
        print(f"  pass: n={v['pass']['n']:>4} Sharpe(pt)={v['pass']['sharpe']:+.3f} "
              f"meanR={v['pass']['mean_r']:+.3f} WR={v['pass']['win_rate']:.1%} "
              f"| curveSharpe(ann)={v['curve_sharpe_pass_annualized']:+.3f}")
        print(f"  fail: n={v['fail']['n']:>4} Sharpe(pt)={v['fail']['sharpe']:+.3f} "
              f"meanR={v['fail']['mean_r']:+.3f} WR={v['fail']['win_rate']:.1%} "
              f"| curveSharpe(ann)={v['curve_sharpe_fail_annualized']:+.3f}")
        print(f"  delta_sharpe(per-trade)={v['delta_sharpe_per_trade']:+.3f} "
              f"meanR_diff={v['mean_r_diff']:+.3f} "
              f"perm_p={v.get('perm_p_two_sided')} "
              f"boot95={v.get('boot_ci95_delta_sharpe')}")
        # curve-delta for the discrepancy hunt
        cp = v["curve_sharpe_pass_annualized"]
        cf = v["curve_sharpe_fail_annualized"]
        if cp == cp and cf == cf:
            print(f"  >> curve_delta(ann) = {cp - cf:+.3f}")
    print(f"\nWritten: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
