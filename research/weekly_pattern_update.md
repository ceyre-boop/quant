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
