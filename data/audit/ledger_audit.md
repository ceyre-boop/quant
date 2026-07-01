# Hypothesis Ledger Audit
Generated: 2026-06-07 17:12 UTC
Total: 36 hypotheses

## Action Required

### NEEDS_RETEST — run through canonical runner
```
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-046 --name "Displacement Gate >= 1.5 improves London trade quality" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-047 --name "ICT Score Inversion — high score anti-edge within high-displacement trades" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-049 --name "Short natural-duration trades underperform — entries before commitment formed" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-050 --name "Tuesday+Thursday DOW veto — London ICT" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-052c --name "Pair-level rate trend gate (>50% days widening per pair)" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-054 --name "Rate level gate: require |real_rate_diff| > 1.0% before macro swing entry" --perms 500
```

## Status Summary

| Classification | Count | Action |
|---------------|-------|--------|
| HAS_OOS_PVALUE | 8 | ✓ None |
| NEEDS_RETEST | 6 | Re-run via canonical runner |
| METHODOLOGY_INVALID | 12 | Re-run if claimed confirmed |
| IN_SAMPLE_ONLY | 10 | None (rejected) |

## Why methodology_invalid?

The batch backtester (`ForexBatchBacktester`) and fast backtester (`ForexFastBacktester.run()`)
do **not** apply `_apply_costs()`. All Sharpe numbers from these paths are pre-cost,
daily-annualized, and in-sample only. The canonical runner (`scripts/run_hypothesis.py`)
uses `ForexBacktester` with costs, √(n/years) annualization, and mandatory IS/OOS split.