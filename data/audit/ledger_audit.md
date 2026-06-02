# Hypothesis Ledger Audit
Generated: 2026-06-02 22:23 UTC
Total: 20 hypotheses

## Action Required

### NEEDS_RETEST — run through canonical runner
```
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-046 --name "Displacement Gate >= 1.5 improves London trade quality" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-047 --name "ICT Score Inversion — high score anti-edge within high-displacement trades" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-049 --name "Short natural-duration trades underperform — entries before commitment formed" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-050 --name "Tuesday+Thursday DOW veto — London ICT" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-027 --name "USDJPY regime gate: suppress signals in bull+elevated VIX" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-028 --name "US10Y divergence from EUR/USD as size modifier (not standalone)" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-034 --name "market_structure score is an anti-edge — mirrors HYP-024" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-030 --name "Three-confirmation gate (term structure + options skew) for commodity-linked pairs" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-031 --name "USDCAD portfolio re-entry with term structure gate" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-035 --name "Random Matrix Theory Noise Filtering — RMT covariance cleaning" --perms 500
PYTHONPATH=. python3 scripts/run_hypothesis.py --id HYP-037 --name "fvg_tap anti-edge is timing-conditional" --perms 500
```

## Status Summary

| Classification | Count | Action |
|---------------|-------|--------|
| HAS_OOS_PVALUE | 0 | ✓ None |
| NEEDS_RETEST | 4 | Re-run via canonical runner |
| METHODOLOGY_INVALID | 12 | Re-run if claimed confirmed |
| IN_SAMPLE_ONLY | 4 | None (rejected) |

## Why methodology_invalid?

The batch backtester (`ForexBatchBacktester`) and fast backtester (`ForexFastBacktester.run()`)
do **not** apply `_apply_costs()`. All Sharpe numbers from these paths are pre-cost,
daily-annualized, and in-sample only. The canonical runner (`scripts/run_hypothesis.py`)
uses `ForexBacktester` with costs, √(n/years) annualization, and mandatory IS/OOS split.