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
