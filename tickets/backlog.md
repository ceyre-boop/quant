# Backlog (repo-native ticket fallback)

Used when Linear MCP isn't connected in a session. Same schema either way — switching to real
Linear later is a port, not a rewrite.

Schema per ticket:
- id: TICK-XXX
- title:
- description:
- depends_on: [ticket ids]
- blocks: [ticket ids]
- acceptance_criteria: [bullet list]
- status: backlog | ready | in_progress | in_review | done
- pre_approved: true | false   # true = plan may proceed straight to build without a stop

---

## TICK-001
**title:** Fix `vrp_schema_verify.py` expiration-selection probe bug
**description:** `scripts/vrp_schema_verify.py:89` selects the *median* of the full 2012→present expiration list (an already-expired contract, ~2019) then queries a hardcoded `2022-03-07` EOD at line 92 → ThetaData returns HTTP `472 NO_DATA`. This makes the runbook's "one API touch" schema check falsely fail even though entitlement is fine (proven 2026-07-02: SPY + FXE `list/expirations` HTTP 200; FXE EOD `2022-03-18` on `2022-03-07` returns a full chain). Fix: pick an expiration `>= start_date` (nearest listed expiration on/after the probe date) instead of the median.
**depends_on:** []
**blocks:** [TICK-002]
**acceptance_criteria:**
- [ ] `python3 scripts/vrp_schema_verify.py --symbol FXE` exits 0 and prints the loader-contract columns
- [ ] `--symbol SPY` also passes
- [ ] Expiration chosen for the EOD probe is >= the probe date (no expired-contract queries)
- [ ] No change to `THETADATA_BASE_URL` handling or the local-no-auth path
**status:** ready
**pre_approved:** true

## TICK-002
**title:** VRP Stage 2/3 — fill `ThetaDataLoader` bodies and run IS/OOS on real FXE chains
**description:** ThetaData gateway is live on `127.0.0.1:25503` and the Options **Value** tier serves FXE option chains ($0 ask, confirmed 2026-07-02). Per `docs/vrp_activation.md` §3–6: fill the 4 `ThetaDataLoader` methods marked `# VERIFY SCHEMA AGAINST LIVE RESPONSE` (`sovereign/research/vrp/data_loader.py`), keeping the frozen `OPTION_CHAIN_COLUMNS` contract, then run `validate_vrp.py --stage 2` (IS sanity) and `--stage 3` (OOS — the number that matters). NOTE: this session's memory (`project_vrp.md`) says the bodies were filled 2026-06-16 but the runbook still lists them as TODO — **verify the live state of `data_loader.py` on `sovereign-v2` first**; the fill may already exist. Holdout (`--stage 4`) is touch-once and stays out of scope until OOS is reviewed by Colin. VRP isolation (`test_vrp_isolation.py`) and the signed pre-registration (`vrp_sign_prereg.py --check`) are non-negotiable.
**depends_on:** [TICK-001]
**blocks:** []
**acceptance_criteria:**
- [ ] `data_loader.py` loader bodies confirmed filled + `pytest tests/unit/test_vrp_options_backtest.py -q` green
- [ ] `vrp_sign_prereg.py --check` → OK before any run
- [ ] `validate_vrp.py --stage 2` completes with a plausible IS report (not NO_TRADES for the wrong reason — the 2026-06-16 NO_TRADES was a real $100k-account sizing finding; re-check account-size decision in NEXT.md before interpreting)
- [ ] OOS (`--stage 3`) result logged to `hypothesis_ledger.json` (VRP-001-OPTIONS); no param re-optimization after seeing it
- [ ] Verdict written to repo-root `NEXT.md`; Colin reviews before any deploy
**status:** backlog
**pre_approved:** false
