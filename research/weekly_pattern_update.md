# Weekly Pattern Update — Nightly Research Log

Append-only. One entry per night the research agent runs (21:00 ET Sun–Thu).
Written by `com.alta.research_agent` (launchd) per AGENT_DIRECTIVE.md § Research Routine.

**Format:** each entry appended by the autonomous agent; no manual edits.
**Purpose:** running record of what the off-hours scan found, what it queued, and why.
  A blank session ("no new patterns") is as important as a productive one — it
  confirms the system ran and found nothing, rather than failing silently.

---

<!-- Agent appends entries below this line. Newest at bottom. -->

## 2026-07-20 — nightly pattern update
- Sessions reviewed: 1 research-queue pass (5 tasks, `--max 5`). No live backtest executed — 4 tasks are Colin-gated wiring stubs (RQ-REST-006/007/008/015, all `OK: no-op`) and RQ-REST-016 (CB_MEETINGS back-extension, `code_change`) threw `EXCEPTION: s argument must not be None`, its standing blocked state (see FIND-REST-037-a: applying the patch is Colin-gated per NN#4).
- New patterns flagged: none. No exploratory micro-backtest ran this cycle; the queue holds only operator-gated wiring tasks, so there is nothing new to test autonomously.
- Candidates queued for operator review: none.
- Notes: Movers snapshot (50 gainers) captured to `data/research/gapper/movers_recent.json` for tomorrow's scan — single top-gainers snapshot, no lookback. Extreme smallcap gapper ZYBT +1047% @ $8.01 leads; several names are warrants (RNWWW/FGIWW/IVDAW/ACHR.WS) which HYP-093/107 filters exclude. Non-warrant ≥40% gappers: ZYBT, MF, ADVB. Graveyard read clean (27 killed hyps loaded); no re-proposal of any sealed idea. Research loop healthy; RQ-REST-016 remains the one recurring queue error, awaiting operator sign-off — not a new incident.

## 2026-07-21 — nightly pattern update
- Sessions reviewed: 1 (movers snapshot only — execution.alpaca has no multi-session lookback)
- New patterns flagged: none
- Candidates queued for operator review: none (no pattern cleared a first-pass permutation test tonight)
- Notes: Research queue drained 5 tasks — RQ-REST-019b/033/005/009 all OK: no-op (operator-decision items, not runnable); RQ-REST-017 (cb_decisions.json quarantine/rebuild) produced a data audit: 1201 CB decisions, 71 surprises ≥25bp (BOE 18, ECB 15, BOC/FED/RBA 10 each) — data output, not a tested pattern. Movers snapshot: 50 gainers written to data/research/gapper/movers_recent.json. Recent verdicts show repeated BLOCKED_NO_VALIDATOR on HYP-AUTO entries (2026-07-20/21) — pre-existing, noted for triage.

## 2026-08-16 — nightly pattern update
- Sessions reviewed: 1 movers snapshot (single top-gainers snapshot; `execution.alpaca` has no multi-session lookback) + 1 research-queue pass (`--max 5`).
- New patterns flagged: none.
- Candidates queued for operator review: none — nothing was tested tonight, so nothing cleared a first-pass permutation test.
- Notes: Queue held exactly one QUEUED task, RQ-006 (ICT walk-forward stability). It returned `ERROR` (logged `ok:false` as F-20260816210048): `extract_live_edge.py --days 30 --min-trades 20` found **1 paper trade, needs 20+**. That is a data-sufficiency refusal by design, not a crash — the script exited 0 and the walk-forward correctly declined to grade on n=1. Recorded per STANDING RULE 6 (degraded source logged explicitly), not repaired per RULE 9. Root cause is the known ICT fill-rate bottleneck (~98% of setups expire unfilled → ~2 fills/90d), so this task will keep returning ERROR until fill rate is addressed; it is not a new incident. Queue is now empty of QUEUED tasks.
- Movers: 50 gainers written to `data/research/gapper/movers_recent.json`. 15 are warrants/rights/units (HYP-093/107 filters exclude these); 35 common. Only 1 sub-$1 common name. Non-derivative, ≥$1, ≥40% gappers: WETO +128% @ $8.22, CAPR +58% @ $6.65, BANL +56% @ $10.93, HHS +53% @ $4.30, UMAL +51% @ $27.71, VWAV +45% @ $1.78, ETON +44% @ $58.86, DAAQ +44% @ $10.42. Top raw mover AACBR +1233% @ $0.012 is a right, excluded. Snapshot only — no fade test run.
- Graveyard read clean (27 killed hypotheses + confirmed edges loaded); no sealed idea re-proposed.

## 2026-08-17 — nightly pattern update
- Sessions reviewed: 1 (21:00 research routine, run from archive/AGENT_DIRECTIVE.md — file archived to archive/ by commit 7ed778d 2026-08-12; launchd job still points at the archived path, unresolved since flagged 2026-08-16)
- Step 0 brain read: 27 killed hypotheses (graveyard), 17 confirmed edges loaded cleanly. No sealed idea re-proposed.
- Step 1 movers: 50 gainers pulled cleanly, no Alpaca error (credential outage from 08-04/08-05 remains resolved). Heavy warrant/rights/unit noise (~23 of 50 tickers ending W/.WS/.RT, excluded by HYP-093/107 filters). Non-derivative ≥$1 ≥40% gappers: SIC +243.92% ($49.83), IPST +235.89% ($7.39), WETO +199.15% ($24.59), TRUG +58.84% ($1.54), WFF +40.28% ($2.02). Snapshot only, no fade test run.
- Step 2 queue (`--max 5`): 0 QUEUED tasks — queue is empty (still open from 2026-08-16, needs new tasks or continues to no-op nightly).
- New patterns flagged: none
- Candidates queued for operator review: none
- Notes: BLOCKED_NO_VALIDATOR verdicts continue accumulating in auto_hypothesis_results (10 new since 08-16, all HYP-AUTO-* with no validator attached) — still open per 2026-08-05/08-16 NEXT.md entries, unaddressed.

## 2026-08-18 — nightly pattern update
- Sessions reviewed: 1 (21:00 research routine, run from `archive/AGENT_DIRECTIVE.md` — file remains archived, commit 7ed778d 2026-08-12; root-vs-archive ambiguity flagged 08-16/08-17 still unresolved).
- Step 0 brain read: 25 killed hypotheses (graveyard), 17 confirmed edges loaded cleanly via `get_research_context()`. No sealed idea re-proposed.
- Step 1 movers: 50 gainers pulled cleanly (`execution.alpaca.movers(top=50)`), no 401/entitlement error. 24/50 derivative-like (warrants/rights ending W/WS/WW/R/U), 26 non-derivative; 25 sub-$1. Non-derivative ≥$1 ≥40% gappers: PFSA +506.62% ($27.48), CAST +144.24% ($2.10), IPST +113.26% ($15.76), XOS +112.44% ($4.44), AIXC +85.86% ($1.32), SLE +82.12% ($5.50), AMLX +63.84% ($35.11), GNLN +51.74% ($3.05), PSIG +47.22% ($1.59). Snapshot only, no fade test run.
- Step 2 queue (`--max 5`): 0 QUEUED tasks — queue remains empty (open since 2026-08-16, 3rd consecutive no-op night).
- New patterns flagged: none.
- Candidates queued for operator review: none (nothing tested tonight, queue empty).
- Notes: `recent_verdicts` (last 15) = 10 BLOCKED_NO_VALIDATOR + 5 NOT_SIGNIFICANT — BLOCKED_NO_VALIDATOR still accumulating unaddressed. Graveyard count dropped 27→25 since 08-17 read (not investigated — may be a window/lookback effect in `get_research_context()`, flagging for operator awareness rather than treating as an audit finding per STANDING RULE 10).

## 2026-08-31 — nightly pattern update
- Sessions reviewed: 1 (21:00 ET research routine, first successful run since 2026-08-18 — see the blackout finding below).
- **Finding of the night — the agent was dark for 8 scheduled nights and said nothing.** Entries stop at 08-18 and resume tonight. `logs/research_agent.log` shows every intervening run dying on `API Error: 400 You have reached your specified API usage limits. You will regain access on 2026-09-01 at 00:00 UTC` (job exit status 1). Missed firings on the Sun–Thu schedule: 08-19, 08-20, 08-23, 08-24, 08-25, 08-26, 08-27, 08-30. Tonight's run began 2026-08-31 21:00 EDT = 2026-09-01 01:00 UTC — one hour past the quoted reset, so the blackout ended at the quota boundary, not from any repair. Root cause is an account-level API limit; nothing in this repo caused it or can prevent a recurrence. Incident: `logs/incidents/2026-08-31-research-agent-api-limit-blackout.md`.
- **The defect that matters is the invisibility, not the quota.** The 400 lands before the first tool call, so STANDING RULES 3/5/6/9 all no-op — no incident note, no NEXT.md line, no commit. Eight dead nights were indistinguishable from eight quiet nights. The only evidence was a gap in a file nobody diffs. A watchdog that asserts this file gained a section on each scheduled night would have caught it on 08-20; that watchdog must not itself be a Claude agent. Operator decision, not built tonight.
- Step 0 brain read: clean. 25 killed hypotheses (graveyard) + 17 confirmed edges loaded via `get_research_context()`. No sealed idea re-proposed. Graveyard count 25, matching 08-18 (the 27→25 drop flagged 08-18 has not moved further; still un-investigated, still not treated as an audit finding per STANDING RULE 10).
- Step 1 movers: 50 gainers pulled cleanly (`execution.alpaca.movers(top=50)`), no 401/entitlement error. 28/50 derivative-like (warrants/rights/units), 22 non-derivative; 26 sub-$1 — the noisiest snapshot in the recent run of nights. Non-derivative, ≥$1, ≥40% (HYP-093 fade band): **AEHL +83.33% ($6.49), NCRA +49.74% ($2.83)** — only 2, versus 5–9 on 08-16/17/18. HYP-107 band (30–40%): YDDL, USDE, TP. Snapshot only; no fade test run.
- Step 2 queue (`--max 5`): 0 QUEUED. Queue has been empty since 2026-08-16 — but note that only 3 of the intervening nights could have observed that, since 8 were dark. Standing statuses: 35 DONE, 5 RETIRED, 4 COMPLETED, 2 ERROR (RQ-006, RQ-REST-016), 2 DIAGNOSED. `last_updated` on the queue file is 2026-07-09; findings.jsonl last gained a row 2026-08-16. **The nightly research pass has now produced no new test for 15 days.** Empty-queue nights are a valid data point (RULE 5) but a fortnight of them is a supply problem, not a result.
- New patterns flagged: none.
- Candidates queued for operator review: none — nothing was tested tonight, so nothing could clear a first-pass permutation test.
- Notes: the plist/`run_agent.sh` stale directive path (`~/quant/AGENT_DIRECTIVE.md`, archived to `archive/` by 7ed778d on 08-12), flagged unresolved on 08-16, 08-17 and 08-18, is **fixed tonight** in the installed plist, `scripts/com.alta.research_agent.plist` and `scripts/run_agent.sh` (all three routines). The installed plist needs a `launchctl unload/load` to pick it up — deliberately not done from inside the running job. The 2026-08-18 pattern-file section was sitting uncommitted in the working tree; it is committed tonight.

## 2026-09-01 — nightly pattern update
- Sessions reviewed: 1 (21:00 ET research routine, Tue). Ran on `.venv313` (Python 3.13 + `requirements.lock.txt`), not the ambient `python3` (3.14, missing declared deps) — per CLAUDE.md env note.
- **Finding of the night — last night's plist fix is on disk but was never loaded, and tonight proves it.** `~/Library/LaunchAgents/com.alta.research_agent.plist` (mtime Aug 31 21:02) contains the corrected prompt `Read ~/quant/archive/AGENT_DIRECTIVE.md …`. The prompt this run actually received was the *old* string `Read ~/quant/AGENT_DIRECTIVE.md …`, which failed on the first command (`cat: No such file or directory`). launchd is still executing the pre-08-31 in-memory job definition. The 08-31 entry noted the reload was deliberately skipped from inside the running job; it has not been done since. Operator command (must be run from a shell, not from inside a firing agent):
  `launchctl unload ~/Library/LaunchAgents/com.alta.research_agent.plist && launchctl load ~/Library/LaunchAgents/com.alta.research_agent.plist`
  Not automated tonight on purpose: `launchctl unload` terminates the running job's processes, i.e. this session, before it could commit; and a detached reload that half-fails leaves the agent permanently dark — strictly worse than a wrong prompt the agent recovers from in one command. Same applies to `com.alta.morning_agent` and `com.alta.eod_agent`, whose `run_agent.sh` paths were fixed in the same commit.
- Step 0 brain read: clean. 25 killed hypotheses (graveyard), 17 confirmed edges, 15 recent verdicts via `get_research_context()`. Graveyard count 25 — unchanged from 08-18 and 08-31; the 27→25 drop flagged on 08-18 remains un-investigated and is still not being treated as an audit finding (STANDING RULE 10). No sealed idea re-proposed.
- Step 1 movers: 50 gainers pulled cleanly (`execution.alpaca.movers(top=50)`), no 401/entitlement error. 23/50 warrant/unit-like, 27 common; 17 common under $5. Common-stock ≥40% (HYP-093 fade band): **ISRL +98.95% ($24.67), SSM +77.43% ($4.76), HUBCZ +71.76% ($0.0146), FLYE +61.76% ($2.20), BIAF +44.52% ($6.59), RDAC +41.45% ($6.45), GPRO +40.38% ($1.23)** — 7 names, up from 2 on 08-31 and back in line with the 5–9 range of 08-16/17/18. HYP-107 band adds NWGL +38.81%, SWVL +31.62%, CHARR +30.00%. Snapshot only; no fade test run.
  - Housekeeping, no action: the ~46% warrant/unit share of the raw top-50 is already handled downstream — `execution/config.py:78` sets `excluded_last_chars: "WRU"`. The dilution only means `top=50` nets ~27 eligible names, not that unfiltered junk reaches the signal path.
- Step 2 queue (`--max 5`): 0 QUEUED, exit 0. **The queue has now been empty since 2026-08-16 — 16 days, and the nightly pass has produced no new test in that window.** `data/agent/research_queue.json` `last_updated` is 2026-07-09. Statuses unchanged: 35 DONE, 5 RETIRED, 4 COMPLETED, 2 ERROR, 2 DIAGNOSED. Findings cadence by month makes the decay unambiguous: **May 46 → Jun 57 → Jul 16 → Aug 1**. Empty-queue nights are a valid data point (RULE 5); sixteen of them is a supply failure, not a result.
- **RQ-006 is mis-statused and will never clear.** Its ERROR (2026-08-16) is not a code fault — `extract_live_edge.py` refused correctly: *"Only 1 trades found (need 20+)"*. It is blocked on ICT paper-trade volume, which is the known ICT fill-rate bottleneck (~2 fills/90d, TICK-028: setups expire unfilled, ~98%). At the current fill rate a 30-day window will not reach 20 trades in any foreseeable run. It should be RETIRED or re-scoped to a longer window / lower `--min-trades`, not left sitting as ERROR where it reads as a repairable failure. Operator decision — not changed tonight.
- New patterns flagged: none. Nothing was tested, because there was nothing queued to test.
- Candidates queued for operator review: none — no first-pass permutation test could run, so nothing could clear p < 0.10 / n ≥ 20.
- Notes: `recent_verdicts` remains dominated by `BLOCKED_NO_VALIDATOR` (auto-generated HYP-AUTO-* rows, latest batch 2026-08-24) — the hypothesis generator keeps emitting candidates that no validator can score. That is the other half of the same supply problem: the queue is starving while the generator is producing unscoreable rows.
