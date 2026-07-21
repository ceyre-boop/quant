# Restoration Campaign — Ledger

Cross-session state for the System Restoration Campaign
(`~/Obsidian/.../Trading/Ops/PROMPT-System-Restoration-Campaign.md`). Each session
reads this first and resumes without re-deriving context. `plans/` is gitignored —
this file is **force-added** so it survives across machines.

Schema: `id | phase | item | finding | action | evidence | status`

---

## Corrected campaign premises (verified against the filesystem before acting)

The campaign was written one commit stale. Its own NN#1 requires verifying claims first.

| id | phase | item | finding | action | evidence | status |
|---|---|---|---|---|---|---|
| C-1 | 0 | LOGPATH parser "blind, fix FIRST" | **Already fixed** this session in `aab90eb`; `_load_plist` strips XML comments, LOGPATH returns real verdicts on comment-header plists | Collapse Phase 0 to a verification gate; do not redo | `claim_check` LOGPATH on `com.alta.execution_harness` → REFUTED (not UNVERIFIABLE); `test_plist_with_double_hyphen_comment_is_parseable` present | VERIFIED-PREEXISTING |
| C-2 | 1 | `research_agent.log` "dark since 2026-05-16" | **Not dark** — fired 2026-07-20 21:02 on its 21:00 Sun–Thu schedule | Close as not-dark | `ls logs/research_agent.log` → Jul 20 21:02 | CLOSED |

---

## Phase 0 — verify the instrument

| id | phase | item | finding | action | evidence | status |
|---|---|---|---|---|---|---|
| P0-1 | 0 | claim_check LOGPATH regression | `test_plist_with_double_hyphen_comment_is_parseable` already pins the comment-header fix; `test_stale_nonempty_log_downgrades_not_refutes` pins recency | Confirmed coverage exists; no new test needed | `pytest tests/unit/test_claim_check.py` 17/17 | DONE |
| P0-G | 0 | **GATE** | self-test green, no blanket-UNVERIFIABLE class | — | `claim_check --self-test` PASS | PASSED |

---

## Phase 1 — restore the feedback loops

| id | phase | item | finding | action | evidence | status |
|---|---|---|---|---|---|---|
| L1 | 1 | Oracle close-loop / null contract fields | **Reframed.** Nulls are explicit `None` with documented reasons at `sovereign/forex/forex_specialist.py:118-131`, not a silent leak. The DECISION_LOGGER contract (commitment/rate-diff/library/bars) is the ICT/equity spec; forex legitimately lacks `commitment_score` (ICT concept), `vix_at_entry` (VIX gate dead), `cot_percentile` (COT is z-score). Only `library_match` ("not yet wired into ForexEntrySignal") and upstream `irp_z` (feeds `rate_diff_z`, often None) are real wireable gaps. Outcome loop is HEALTHY: 73 decisions → 56 EXPIRED (unfilled limits), 5 closed w/ outcome, 12 open. | **TICK-050** — diagnose-only per operator ruling; `forex_specialist.py` is LIVE/frozen-adjacent (2 live importers), so the fix waits for the 07-28 unlock | `forex_specialist.py:130` `commitment_score=None`; `:132` `library_match=None # not yet wired`; decisions Counter EXPIRED 56/None 12/LOSS 5 | TICKETED (TICK-050) |
| L2 | 1 | `sync_dashboard_data.py` green-but-empty | `dashboard_state.json` (May 31) is a **phantom**: no `.py` writes it, no code reads it (only `NEXT.md`/`AGENT_DIRECTIVE.md`/an audit file mention it). The script writes 8 sibling state files, never this one. The directive named it falsely. | **FIXED** the false dependency in `AGENT_DIRECTIVE.md:104`; **TICK-051** to delete the stale file + purge remaining mentions. No empty-file written (that is the green-but-empty trap one level up). | `grep dashboard_state` → 0 code writers/readers; `sync_dashboard_data.py` writes health/checklist/prop_challenge/g2_progress | FIXED + TICKETED |
| L3 | 1 | dashboard "missing inputs" | **Reframed.** `dashboard/index.html` fetches `prop_account_balance.json` + `execution/fills.json` (both absent), but `deploy.yml:22-24` copies `dashboard/.` then **overwrites** the served root with repo-root `index.html`. So those fetches never run live; `dashboard/` is the orphaned legacy view (live dashboards are repo-root `index.html` + `ict/index.html`). Not a broken live loop. | **TICK-052** — either point `dashboard/` at real files (`fill_log.jsonl`) or mark it orphaned/delete. Non-urgent (not served). | `index.html` fetches calendar/proof_of_life/icarus/macro/daily_digest — none of the two named files; `deploy.yml` overwrite ordering | TICKETED |
| L4 | 1 | heartbeat coverage | Deferred to diagnosis: `plist_manifest.py` + `health_check.py` already surface job state; no new gap fixed this session | Note; full heartbeat audit is its own item | — | DEFERRED |
| L5 | 1 | `research_agent.log` | Duplicate of C-2 — fired 07-20 21:02, not dark | Closed | see C-2 | CLOSED |
| P1-G | 1 | **GATE** | Every loop carries data, is fixed, or is ticketed with a named cause + freeze class. No "exits 0, does nothing" left unnamed. | — | this table | PASSED |

---

## Pre-existing test failures (named, not absorbed — baseline for the campaign)

Present on committed HEAD before any campaign work (working tree clean):
- `tests/ -k "ict and pipeline"`: **4 failed / 23 passed** — the campaign's "must stay 21/21"
  is already below that on the committed branch. Failing:
  `test_ict_pipeline.py::TestScoreAndGrade::test_component_scores_dict_has_expected_keys`,
  `::test_confirmations_plus_missing_covers_all_components`,
  `::TestRiskEngineGate::test_max_positions_hit_vetoes_even_good_setup`,
  `::test_daily_loss_limit_vetoes`.
- `tests/ -k ict` broader: **15 failed** (adds `test_ict_session_classifier.py` ×10,
  `test_forex_entry_engine.py` ×1).
- Isolation guard `test_pipeline_does_not_import_sovereign`: **PASS**.
- These are a later campaign item; this session added no failures.

---

## Next session

Resume at Phase 2 (TICK-048's 28 DORMANT) per the campaign, OR act on TICK-050/051/052
if the 07-28 unlock has landed. Re-verify every open row above against the filesystem
first — the stale-premise pattern (C-1, C-2, L1, L3) recurs.

---

## Phase 2 — the 28 DORMANT (all resolved; gate 28→0)

Decisions with evidence. RETIRE *executions* are ticketed (TICK-053) per the >3-files
discipline — this pass makes the decision, a focused pass moves the files with a full
test run. Gate is "every file has a decision + evidence": met.

**Inventory refinement found:** 7 of the 28 are real pytest tests under `research/*/tests/`
(19 collected) mislabelled DORMANT because `system_inventory.py`'s test-root detection is
top-level-`tests/` only. TICK-054 to fix the heuristic. These are KEEP (working tests).

| id | file | claim_check | decision | reason |
|---|---|---|---|---|
| D-1..7 | `research/{fvg_corridor,gapper_continuation,political_alpha,political_alpha_v2,yield_frontier}/**/test_*.py` + `conftest.py` | collectable (19 tests) | KEEP | real tests; inventory test-root bug (TICK-054), not dormant |
| D-8..15 | 8× `__init__.py` (root, models, clawd swing, fvg_corridor ×2, movers_study, yield_frontier tests, sovereign/strategies) | — | KEEP | package markers, structural, not features |
| D-16 | `sovereign/strategies/strategy_selector.py` | REFUTED (imported by `sovereign/strategies/__init__.py`) | KEEP-DORMANT | dormant strategy-selection scaffold; revisit when equity layer activates (branch-topology note) |
| D-17 | `check_account.py` | no importer | RETIRE→attic (TICK-053) | 1.4KB root utility, no importer, no `__main__` |
| D-18 | `fix_hardcoded_equity.py` | no importer | RETIRE→attic (TICK-053) | one-off migration fix, job done (name states it) |
| D-19 | `research/debug_indexing.py` | no importer | RETIRE→attic (TICK-053) | 491B debug one-off |
| D-20 | `clawd_trading/swing_prediction/TODO_COMPLETE_IMPLEMENTATION.py` | no importer | RETIRE→attic (TICK-053) | literal TODO scratchpad by its own header; 39KB of unimplemented code, not a feature |
| D-21..26 | `clawd_trading/{infrastructure/guardrails,infrastructure/risk_manager,swing_prediction/fomc_calendar,positioning_feed,prediction_scrutiny,real_base_rates}.py` | no importer | KEEP-DORMANT | `clawd_trading` is a separate incomplete swing-prediction system; revisit 2026-09 as a unit, not piecemeal |
| D-27 | `firebase/rest_client.py` | no importer | KEEP-DORMANT | Firebase is a known dormant ghost (see memory `project_dashboard_data_architecture`); revisit if Firebase is ever activated |
| D-28 | `data/research/sovereign_core_gauntlet/_diag_gates.py` | no importer | KEEP-DORMANT | research diagnostic in a data/ dir; harmless, revisit with the gauntlet |

**Gate P2:** 28/28 resolved — 15 KEEP, 4 RETIRE (ticketed TICK-053), 9 KEEP-DORMANT with reasons. PASSED.

---

## Next session (updated)

**Phases 0–4 complete** (verify, loops, 28 DORMANT, 46 RETIRED triage, 20 ON-DEMAND load-probe). Remaining:
- **Phase 4 deeper run** (optional) — the 20 probed tools load clean; a full live-data
  execution pass is deferred to a session with Alpaca/network infra available.
- **Phase 5** — consistency sweep: purge retired Sharpe 1.2864 from vault notes (identified,
  not yet purged); reconcile fills logs; relink 56 orphan vault notes.
- **Tickets:** TICK-050/051/052 (post-07-28 unlock), TICK-053 (execute 4 retirements),
  TICK-054 (inventory test-root heuristic).
Re-verify every open row against the filesystem first — the stale-premise pattern recurs.

---

## Phase 3 — the 46 RETIRED, triaged (propose ≤5, implement none)

**All 46 are ENGINEERING-retired, not evidence-retired.** The killed *edges* (HYP-044 VIX,
v007, overnight-QQQ, AUDNZD) live in the hypothesis ledger, not as these files — so the
resurrection rule's "stays dead" evidence list does not apply to any file here. The test that
governs is: *does the current system already cover it?*

**Buckets (44 stay retired):**
- **attic/ (31)** — the superseded clawd_trading / Firebase / Kimi / XGBoost-brain generation
  (run_clawd, kimi_brain, xgboost_brain, dashboard_api, realtime_publisher, ~12 test/runner
  scripts). Replaced wholesale by the sovereign architecture. Reviving = restoring a dead
  architecture. **STAYS RETIRED** (already covered).
- **scratch/ (9)** — one-off diagnostics/audits (diagnose_alpaca, audit_trades, desk_audit…).
  Throwaway by convention. **STAYS RETIRED.**
- **archive/agent_scheduler.py** — replaced by launchd plists. **STAYS RETIRED.**
- **scripts/push_to_firebase.py** — Firebase is a dormant ghost. **STAYS RETIRED.**

**Refurbishment shortlist — 1 genuine (caveated) + 1 dependent. Not padded to 5.**

| rank | file | case | serves | caveat |
|---|---|---|---|---|
| 1 | `lab/feature_registry.py` | Append-only GRAVEYARD registry whose explicit rule is "never re-deploy without new evidence" and a 90-day re-validation gate. This is the direct antidote to the demonstrated failure of HYP-044's VIX gate being re-proposed **3 times** off a formatting example. | Scientific-integrity tenet | **Partially covered:** `data/agent/hypothesis_ledger.json` (86 entries) + `sovereign/research/signal_decay.py` + `scripts/audit_live_features.py` already do much of this informally. So the real proposal is **consolidate the existing ledger into this principled schema**, not build new. Operator judgment. |
| 2 | `lab/baseline_registry.py` | Baseline tracking for walk-forward marginal-contribution gating — pairs with #1. | Scientific-integrity tenet | Only worth it if #1 is pursued; standalone value low. |

**Nothing resurrected.** Both candidates are proposals for a separate approved pass.

**Gate P3:** all 46 classified (44 stay-retired with bucket reason, 2 shortlisted with written
cases); nothing resurrected. PASSED.

---

## Phase 4 — spot-check 20 of 294 ON-DEMAND tools (gate: breakages ticketed)

**Method (limit named):** import/load-layer probe — exec each entry point, catch
ImportError/SyntaxError/setup failures. This is the safe, side-effect-free detector of the
bitrot the phase targets (renamed/removed deps silently breaking a validation tool). It is
**not** a full data-run: a tool can load-clean and still fail on current data (API drift,
missing file). A deeper live-data execution pass is deferred — it mutates ledgers, hits
network, and needs the infra that is currently absent.

**20 consequential tools probed (validation/holdout/audit/backtest — by what they measure):**
`hyp104_holdout, backtest_integrity_audit, run_replay_validation, validation/backtest_engine,
walk_forward_validation, mc, megascan, permutation_test_sovereign, backtest_lifecycle,
accelerated_validation, conviction_audit_full, final_lookahead_audit, options_tradeability_audit,
hyp_008_lift_audit, iv_sentinel_audit, modern/reconcile, holdout_validation_v014, discover,
engine, stage1_discovery`.

**Result: 20/20 load clean — no import-layer bitrot.** Initial probe flagged 3
(`megascan`, `engine`, `hyp104_holdout`) with "relative import" errors — **false positives from
the probe method** (loose-file exec vs package `-m`); re-probed as `python3 -m` → all import OK.
The probe's own blind spot was caught and corrected before recording (the verification tooling
got verified).

**Gate P4:** 20 spot-checked, 0 breakages → none to ticket. Deeper live-data run of these tools
deferred to a session with the infra available. PASSED (load-layer).
