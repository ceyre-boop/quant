# VRP Iron-Condor — Activation Runbook

The moment the **ThetaData Options Value** subscription is active, this is the exact sequence
from "key in hand" to "real verdict." No new architecture — the scaffolding (loader contract,
backtest, pre-registration, CLI, tests) is already built and committed `[VRP-PLAN]`.

Phase I (free-data inverted gauntlet, commit `aec72d5`) already returned **VRP-001 =
DATA_INSUFFICIENT**: VRP exists and is orthogonal to carry in the proxy gate — which is what
justified buying real chains. This runbook is Stages 2–4 on those chains.

---

## 0. Prereqs
- ThetaData Options Value subscription active.
- **ThetaTerminal running locally** (it exposes the REST gateway, default `127.0.0.1:25503` (code default; older docs said 25510 — 25503 is what data_loader/vrp_feed/schema_verify all use)).
- Pre-registration is signed and frozen: `python3 scripts/vrp_sign_prereg.py --check` → `OK`.

## 1. Add credentials
Copy `.env.example` → `.env` (if not already) and set:
```
THETADATA_API_KEY=...            # if your ThetaTerminal login requires it
THETADATA_BASE_URL=http://127.0.0.1:25503
```

## 2. Verify the schema (one request — the only API touch before coding)
```
python3 scripts/vrp_schema_verify.py --symbol SPY
```
- Prints the **real** response schema and the loader contract columns.
- **Boundary check:** note the earliest expiration ThetaData serves. If it is **after
  2022-01-01**, the pre-registered IS window (`2022-01-01..2022-12-31`) is partly unavailable
  → log a `data/agent/param_change_log.jsonl` entry and shift IS forward **before** running.
  Do not silently move it.

## 3. Fill the loader bodies (<30 min, mechanical)
In `sovereign/research/vrp/data_loader.py::ThetaDataLoader`, implement the four methods marked
`# VERIFY SCHEMA AGAINST LIVE RESPONSE BEFORE IMPLEMENTING`, mapping raw fields onto
`OPTION_CHAIN_COLUMNS`. **Only the parsing changes — the contract (method signatures +
columns) stays fixed**, so nothing downstream moves. Re-run step 2 until columns line up.
Confirm the architecture still passes against the mock:
```
python3 -m pytest tests/unit/test_vrp_options_backtest.py -q
```

## 4. IS run (parameter sanity — NOT a verdict)
```
python3 scripts/validate_vrp.py --stage 2
```
Pulls + caches IS chains (`data/research/vrp_data_cache/`), runs the backtest, writes the IS
report. **Sanity gate** before going further: no insane Sharpe, no \$0 P&L, not all-stops,
trade count plausible (~52 Mondays/yr minus skips), costs non-trivial.

## 5. OOS run (the primary validation)
```
python3 scripts/validate_vrp.py --stage 3
```
Out-of-sample `2023-01-01..2024-06-30`. This is the number that matters.

## 6. Holdout (touch ONCE)
```
python3 scripts/validate_vrp.py --stage 4 --confirm-once
```
- Requires `--confirm-once` and refuses if `data/research/.vrp_holdout_touched` exists.
- `2024-07-01..2026-06-15`. Runs once. The result is the answer — **do not adjust params and
  re-run.** A bad result is logged to `hypothesis_ledger.json` (`VRP-001-OPTIONS`) and buried.

---

## Discipline (non-negotiable)
- The signed `options_backtest` block in `vrp_preregistration.json` is read-only. Any change
  fails `vrp_sign_prereg.py --check` and the options tripwire test — that is intended.
- No re-optimization after seeing OOS/holdout. Log it, bury it, move on.
- `vrp/` imports nothing from `sovereign/forex` or `sovereign/ict` (enforced by
  `test_vrp_isolation.py`). Live system untouched; no auto-deploy on a positive verdict —
  Colin reviews first.
