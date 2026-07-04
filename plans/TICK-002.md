# Plan — TICK-002: VRP Stage 2/3 on real FXE chains

Authorized by the approved Day-2 operating plan:
`Plans/context-day-2-imperative-stonebraker.md` §E1 (mandate supersedes
`pre_approved:false` for this run). Ticket ACs govern; this file is the pointer.

Execution sequence (no deviations):
1. Pre-verified 2026-07-03: loader bodies FILLED (stub markers gone);
   `vrp_schema_verify --symbol FXE` and `SPY` both exit 0 (terminal healthy).
2. `python3 scripts/vrp_sign_prereg.py --check` → OK, else STOP (prereg violation).
3. `pytest tests/unit/test_vrp_options_backtest.py tests/unit/test_vrp_isolation.py -q` green.
4. `validate_vrp.py --stage 2` — IS sanity. NO_TRADES interpretation gated on the
   $100k account-size note in NEXT.md history (2026-06-16 finding).
5. `validate_vrp.py --stage 3` — OOS. No param changes after seeing it. Stage-4
   holdout untouched.
6. Ledger `VRP-001-OPTIONS` with full artifacts + coverage stamp
   ("chains 2020-01-03+, Value-tier depth — six years, not the decade").
7. CONFIRMED → E4 protocol (report `python -m factory.train --hyp VRP-001-OPTIONS`,
   DO NOT run, halt Track E). Any verdict → NEXT.md; Colin reviews before any deploy.

Risks: terminal stalls mid-run (feeder has skip-and-count + circuit breaker; retries
only advance via parquet cache); spot sourced from yfinance (loader NOTE) — record in
artifacts.
