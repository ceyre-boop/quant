# System Restoration Campaign — session 1: verify Phase 0, restore the feedback loops

**Ticket:** TICK-049 (proposed) · **Date:** 2026-07-21 · **Status:** PLAN — awaiting approval
*(Replaces the TICK-047 inventory plan, which shipped 90b22c2. Executes the campaign in
`~/Obsidian/.../Trading/Ops/PROMPT-System-Restoration-Campaign.md`, session 1 of N.)*

---

## Context

Another session wrote a strong multi-session restoration campaign: make every feature work,
be honestly retired, or be honestly named as unwired — no fourth category — before Goal 1's
cost cascade makes numbers "trustworthy" on top of a half-wired machine. This plan executes
**session 1** and adopts the campaign's doctrine wholesale. Each session resumes from
`plans/restoration-ledger.md`.

**But the campaign was written one commit stale, and its own Non-Negotiable #1 requires
verifying claims before acting — so its premises were checked against the filesystem first.**
Two are already overtaken by this session's own work:

| campaign claim | verified reality |
|---|---|
| Phase 0: "claim_check LOGPATH parser is blind on plist comment headers — fix FIRST, everything depends on it" | **Already fixed** in `aab90eb` today. `_load_plist` strips XML comments; LOGPATH returns real verdicts on comment-header plists. Phase 0 collapses to a verification gate. |
| Phase 1.5: "`research_agent.log` dark since 2026-05-16" | **Not dark.** Log is dated **Jul 20 21:02** — fired on its 21:00 Sun–Thu schedule. Stale claim; will be recorded corrected. |

Three Phase-1 claims are **confirmed real** and are the actual work:

| claim | verified |
|---|---|
| Oracle close-loop: DECISION_LOGGER contract fields arriving null | **CONFIRMED, serious.** 0/73 of the 2026-07 decisions have all four fields; `commitment_score` and `library_match` are null in **every** record, `rate_differential_zscore` in all but one. Oracle is learning from near-blank records. |
| `sync_dashboard_data.py` green-but-empty | **CONFIRMED.** `data/agent/dashboard_state.json` dated **May 31**; no `.py` in scripts/sovereign/execution writes it. |
| dashboard missing inputs | **CONFIRMED.** `data/agent/prop_account_balance.json` and `data/execution/fills.json` do not exist; only `fill_log.jsonl` does — a filename mismatch. |

---

## Operator decisions (2026-07-21)

1. **Decision-logger null fields: diagnose + ticket, do NOT fix this session.** Trace exactly
   which caller passes `None` and whether the origin is frozen-path; write the fix as a ticket
   with the precise change; touch no execution-path code during the shadow window (closes
   07-28). Oracle learns from blanks a few more days — acceptable versus disturbing the freeze.
2. **Session scope: Phase 0 verify + Phase 1 loops.** Create the ledger, verify Phase 0, fix
   the safe non-frozen loops, ticket the rest, stop at the Phase 1 gate.

---

## Non-negotiables (carried verbatim from the campaign — these override everything)

- **claim_check before any deadness/absence/silent-crash/citation claim** (`AGENT_DIRECTIVE`
  rule 10). UNVERIFIABLE is not permission to proceed.
- **Execution path frozen until 2026-07-28:** `forex_exit_manager`, `decide_exit`,
  `execution/harness.py`, `carry_engine`, anything importable by the live/backtest exit path.
  No changes without a `NEXT.md` unlock.
- **ICT/sovereign isolation:** verify `pytest tests/ -k test_pipeline_does_not_import_sovereign`.
- **No training, no new edges, no strategy changes.** Repair and wiring only.
- **Git: `git add <explicit paths> && git commit` as ONE command.** Never `-A`. Push each session.
- **No silent mocking.** Missing infra → stop and name what's needed.

## Resurrection rule (carried verbatim — governs Phase 3, not this session, but recorded in the ledger now)

Retired-for-ENGINEERING → refurbishment candidate (judge on: serves a `TRADING_PHILOSOPHY.md`
tenet? rewire cheaper than rebuild? already covered?). Retired-for-EVIDENCE → **stays dead**:
- **HYP-044 VIX gate** — REJECTED_OOS p=0.50. Re-proposed 3× off a CLAUDE.md formatting example. Not a work item.
- **v007 per-pair holds** — NOT_SIGNIFICANT, fails walk-forward, rolled back.
- **Overnight-QQQ as a diversifier** — recouples with carry in crisis (ρ=0.42, BH p=0.007).
- **AUDNZD** — both legs RBA-driven, no independent differential.
- **Sharpe 1.2864 / "~1.25 OOS"** — retired; live costed reference is **0.6886**. Purge on sight.

---

## Work — session 1

Create `plans/restoration-ledger.md` first (schema: `id | phase | item | finding | action |
evidence | status`). Record the two corrected premises above as the first rows.

### Phase 0 — verify the instrument (already fixed; prove it)
1. Add a regression test asserting LOGPATH parses a real comment-header plist
   (`com.alta.execution_harness`) and returns a non-UNVERIFIABLE verdict — pins `aab90eb` so it
   can't silently regress. (`tests/unit/test_claim_check.py` already has
   `test_plist_with_double_hyphen_comment_is_parseable` — confirm it covers this; extend only
   if it doesn't.)
2. **Gate:** `claim_check --self-test` passes; no claim class returns blanket UNVERIFIABLE.
   Ledger Phase 0 as VERIFIED-PREEXISTING, not re-done.

### Phase 1 — restore the feedback loops (the leverage)
For each loop: establish what it consumes, what it produces, and whether output actually
reaches its consumer — not merely exit 0.

1. **Oracle close-loop — DIAGNOSE + TICKET ONLY (operator ruling).** Trace why
   `commitment_score` / `library_match` / `rate_differential_zscore` are null across 73
   records. Candidate origins from exploration: `commitment_score` is set at
   `ict/pipeline.py:657` from a `commitment_result` the orchestrator pre-computes
   (`:575`); `library_match` relates to `AlexandrianLibrary` in `sovereign/orchestrator.py`.
   But the null records are **FOREX** (`EURUSD=X` …), so the forex scan path — not ICT — is
   the likely caller passing `None`. Identify the exact call site, classify it frozen vs not
   (run `claim_check` on the file's execution-path reachability), and write **TICK-050** with
   the precise one-line-per-field fix and the freeze classification. **No code change.**
   Also verify `update_outcome()` coverage: count decisions logged vs outcomes recorded.
2. **`sync_dashboard_data.py` green-but-empty — FIX (non-frozen).** Confirm nothing writes
   `dashboard_state.json` and that `AGENT_DIRECTIVE.md:104,:218` name it falsely. Then EITHER
   wire a real writer in `sync_dashboard_data.py` (it already writes sibling state files —
   follow that pattern) OR, if no consumer truly needs it, correct the directive and remove the
   false dependency. Decide by finding the reader (`dashboard/index.html`?). Doc + leaf-script
   only — not execution path.
3. **Dashboard missing inputs — FIX (non-frozen).** `prop_account_balance.json` and
   `data/execution/fills.json` don't exist. Resolve the `fills.json` vs `fill_log.jsonl`
   mismatch (rename in the fetcher, or point the dashboard at the real file). For
   `prop_account_balance.json`, find the intended writer or mark the panel honestly absent.
4. **Heartbeat coverage — DIAGNOSE.** List scheduled jobs lacking a heartbeat a health check
   sees; make gaps loud. Reuse `scripts/health_check.py` + `plist_manifest.py`. Ticket fixes
   that touch frozen jobs.
5. **`research_agent.log` — RESOLVE THE STALE CLAIM.** Record it fired 07-20 21:02; confirm the
   plist schedule; close the item as not-dark. If it's producing nothing useful despite firing,
   that's a separate finding — note, don't fix.

**Gate:** every loop either demonstrably carries data end-to-end, is fixed, or is ticketed with
a named cause and freeze classification. None left in "exits 0, does nothing."

### Discipline
- Ticket anything >3 files or non-trivial before building; plan in `plans/<ticket-id>.md`.
- Guard tests before AND after, counts reported, pre-existing failures named not absorbed:
  `pytest tests/ -q` · `pytest tests/ -k ict -v` (must stay 21/21) ·
  `pytest tests/ -k test_pipeline_does_not_import_sovereign`.
- `NEXT.md` entry at session end; commit + push.

---

## Explicitly NOT doing this session

- No decision-logger code change (operator ruling — diagnose + TICK-050 only).
- No execution-path edits; no touching the shadow window before 07-28.
- Phases 2–5 (28 DORMANT, 46 RETIRED triage, 392 ON-DEMAND, consistency sweep) — later sessions.
- No resurrection of anything; no implementation of refurbishments.
- No `git add -A`.

---

## Verification

1. `plans/restoration-ledger.md` exists, schema-correct, with the two corrected premises and
   every Phase-1 loop as a row with evidence.
2. Phase 0 gate: `claim_check --self-test` green; the LOGPATH comment-header regression test
   passes.
3. Each fixed loop demonstrated end-to-end: dashboard writer produces a fresh
   `dashboard_state.json` (or the false dependency is removed and the directive corrected); the
   dashboard input filenames resolve to files that exist.
4. TICK-050 written with the exact null-field fix and a freeze classification backed by a
   `claim_check` reachability result — not a guess.
5. Guard tests: before/after counts equal or better; ICT 21/21; isolation green; any
   pre-existing failures named.
6. Commit(s) by explicit path, pushed; `NEXT.md` updated.

---

## Risks

- **The decision-logger origin may itself be frozen-path.** Then even the *fix* waits for
  07-28; this session's job is to make that ruling cleanly, with evidence, not to force it.
- **A "fix" to the dashboard writer could mask a deeper break** — if the real problem is that
  the producing pipeline is dead, writing a fresh empty file is the green-but-empty trap one
  level up. The fix must carry real data or the panel is marked honestly absent.
- **Parallel sessions.** Locks collided repeatedly this week; single-command add+commit,
  re-read `NEXT.md`/ledger tails before appending.
- **Stale-premise pattern will recur** across the campaign — every later session must re-verify
  the ledger's open items against the filesystem before acting, exactly as this one did.
