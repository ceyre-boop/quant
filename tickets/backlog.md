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
- [x] `python3 scripts/vrp_schema_verify.py --symbol FXE` exits 0 and prints the loader-contract columns (2026-07-02: probe expiration 2022-03-11, MATCH True)
- [x] `--symbol SPY` also passes (probe 2022-03-07, MATCH True)
- [x] Expiration chosen for the EOD probe is >= the probe date (nearest listed on/after)
- [x] No change to `THETADATA_BASE_URL` handling or the local-no-auth path
**status:** done (2026-07-02, pre_approved build)
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

## TICK-003
**title:** Options-leg family run — VRP-001, HYP-074/075/076/078/079 + HYP-077 full composite, then family BH
**description:** Real surface data landed 2026-07-02 (sentiment_options_surface: 1,306 rows, 2020-01-03→2026-07-02, rr25/bf25 96.5% of post-2020 board rows, 0 look-ahead violations, all bs_invert). Extend scripts/research/run_positioning_family.py with the six options-leg members under the SAME locked protocol (gate-zero hash verify; rr25_z/bf25_z = trailing-252-obs z on the weekly surface series; HYP-076 needs econ_surprise_z×crowding; HYP-080 stays blocked on the GDELT backfill — family BH cannot run until 080 has a primary p or the family documents its handling). Re-run HYP-077 with the FULL composite (COT + rr25 alignment) superseding the COT-only interim seal. VRP-001 first per the standing V4 mandate: TICK-002's stage 2/3 on real chains (verify loader bodies, prereg check, account-size note in NEXT.md before interpreting NO_TRADES). Coverage caveat to stamp on every seal: options history starts 2020-01-03 (Value-tier depth) — six years, not the decade.
**depends_on:** [TICK-002]
**blocks:** []
**acceptance_criteria:**
- [ ] rr25_z/bf25_z computed trailing-only (truncation-invariance test, same standard as cot features)
- [ ] Six new primaries sealed as dated interim annotations (or UNDERPOWERED/BLOCKED stamps where data forbids), statuses stay PREREGISTERED
- [ ] HYP-077 full-composite seal supersedes the COT-only interim (both annotations remain)
- [ ] Family BH runs ONLY when all 10 primaries exist; otherwise the blocker (HYP-080/GDELT) is stamped
- [ ] VRP-001 stage 2/3 per TICK-002; no param re-optimization after seeing OOS
**status:** ready
**pre_approved:** false

## TICK-004
**title:** Standing Adversarial Invariant Guard — close the weak (Layer-4) correctness layer
**description:** A 4-layer audit rated Adversarial WEAK: RED-1 Oracle contamination + the rogue USD_CAD writer were caught only by manual audit, no standing detector/test. Built `audit/invariant_guard.py` (read-only, spec-first, self-escalating; I1 Oracle-reflection purity, I2 no rogue/sentinel OANDA writes, I3 forbidden-pair guard), `audit/invariants_spec.md` (hashed single-fence contract), `audit/CORRECTNESS_LAYERS.md` (the map), `tests/test_invariant_guard.py` (19/19), `scripts/com.alta.invariant_guard.plist` (daily 09:20). Detect-only — does NOT fix RED-1 (that's the pre-registered Blue change); the guard is its regression test.
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [x] `python3 audit/invariant_guard.py --run` = FAIL on the live contamination (I1=5 exact RED-1 records, I2 caught USD_CAD units=1 sentinels), 3 URGENT escalations, report written, exit 1
- [x] 19/19 tests green incl. I1 exclusion test, independence cross-check, no-execution-import AST guard
- [x] spec is a single hashed `yaml audit-spec` fence; guard imports nothing from `sovereign/execution/`
- [ ] **operator:** load the launchd job to make it standing (`launchctl load ~/Library/LaunchAgents/com.alta.invariant_guard.plist`) — blocked in-session as unauthorized persistence
- [ ] follow-on: land the pre-registered Blue fix in `reflect_cycle` (guard turns green when contamination stops)
**status:** done (2026-07-03, build) — pending operator load
**pre_approved:** false
