"""HYP-090 orchestrator (TICK-023 P4).

Order is law: gate_zero FIRST (prereg + ledger intact), reconcile band checked,
then replay -> gauntlet -> verdict seal -> report.

Run: python3 -m research.modern.run_all [--seed 42] [--smoke] [--no-seal]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="50 placebos + 1k bootstrap")
    ap.add_argument("--no-seal", action="store_true", help="skip the ledger write")
    args = ap.parse_args()

    from research.modern._lib import OUT_DIR, canonical_dumps, env_record, gate_zero, write_json
    prereg = gate_zero()                       # ← FIRST: nothing runs on a broken lock

    recon = json.loads((OUT_DIR / "reconcile_report.json").read_text())
    if not recon.get("in_band"):
        raise SystemExit("RECONCILE ABORT recorded — study halted (see reconcile_report.json)")

    import numpy as np

    from research.modern import gauntlet, replay, report, selection as sel

    t0 = time.time()
    n_placebo = 50 if args.smoke else replay.N_PLACEBO
    n_boot = 1_000 if args.smoke else 10_000

    print("── universe + replayer ──")
    uni = sel.VariantUniverse()
    rp = replay.Replayer(uni)
    eval_dates = uni.index[rp.eval_days]
    print(f"  {uni.n_variants} variants x {uni.D} days; replay {eval_dates[0].date()} "
          f"→ {eval_dates[-1].date()} ({len(rp.eval_days)} days)")

    print("── arms ──")
    runs = [rp.run_arm(arm, W) for arm in ("A1", "A2") for W in sel.WINDOWS]
    a0 = rp.a0_returns()

    print("── placebo floor ──")
    placebo = {W: rp.placebo_sharpes(W, n_placebo) for W in sel.WINDOWS}
    placebo_p95 = {W: float(np.percentile(placebo[W], 95)) for W in sel.WINDOWS}
    print("  p95:", {k: round(v, 3) for k, v in placebo_p95.items()})

    print("── gauntlet ──")
    adjud = gauntlet.adjudicate(runs, a0, eval_dates, placebo_p95,
                                reconcile_in_band=bool(recon["in_band"]), n_boot=n_boot)
    for r in adjud["rows"]:
        print(f"  {r['run']:8s} sharpe={r['sharpe']:+.3f} (A0 {r['a0_sharpe']:+.3f}) "
              f"p={r['p_vs_a0']:.4f} placebo95={r['placebo_p95']:+.3f} "
              f"dsr={r['dsr']:+.3f} yr={'ok' if r['c4_per_year_nondegrade'] else 'FAIL'} "
              f"{'ALL-PASS' if r['all_pass'] else ''}")
    print(f"VERDICT: {adjud['verdict']}")

    print("── report ──")
    report.equity_chart(a0, runs, placebo_p95, eval_dates)
    best = max(runs, key=lambda run: gauntlet.daily_sharpe(run["returns"]))
    report.selection_timeline(best, uni, eval_dates,
                              f"selection_timeline_{best['arm']}_W{best['window']}.png")
    report.per_year_chart(adjud)
    report.write_summary(adjud, prereg, args.seed, env_record(), time.time() - t0)

    results = {
        "ticket": "TICK-023", "hyp": "HYP-090", "seed": args.seed, "smoke": args.smoke,
        "env": env_record(), "runtime_seconds": round(time.time() - t0, 1),
        "prereg_hash": prereg["hash_lock"], "reconcile": recon,
        "verdict": adjud["verdict"],
        "adjudication": {k: v for k, v in adjud.items() if k != "rows"},
        "rows": [{k: v for k, v in r.items() if k != "boot"} for r in adjud["rows"]],
        "placebo_sharpe_p95": placebo_p95,
        "placebo_sharpe_mean": {W: float(np.mean(placebo[W])) for W in sel.WINDOWS},
        "n_placebo": n_placebo, "n_boot": n_boot,
    }
    write_json(OUT_DIR / ("results_smoke.json" if args.smoke else "results.json"), results)
    (OUT_DIR / "results_canonical.txt").write_text(canonical_dumps(results))

    if not args.smoke and not args.no_seal:
        report.seal_ledger(adjud)
        gate_zero()                            # hash + entry still intact after seal
    print(f"DONE in {time.time()-t0:.0f}s -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
