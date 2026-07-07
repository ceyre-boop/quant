# Plan — TICK-015: close_reason capture (PROP-2026-W27 promotion; Day-3 B1)

Capture-at-write going forward + JOIN-based backfill validation. **Join, never
inference; historical AMBIGUOUS rows stay AMBIGUOUS** (they lack source truth — that
honesty is the machine's first proposal working, not a defect to erase).

## SCOPE SPLIT (Day-3 freeze ruling — read first)
- **SLICE 1 (BUILD NOW, this plan's builder):** the backfill JOIN + tests —
  `experience/backfill.py` + new tests ONLY. experience/ is not the execution path.
- **SLICE 2 (POST-WINDOW, blocked_until: shadow_close — DO NOT BUILD):**
  DecisionRecord.exit_reason field, update_outcome(reason=), and any
  oanda_bridge close-record addition. `decision_logger` and `oanda_bridge` are
  imported by the live path — the shadow freeze (standing constraint 1) covers
  them until ~Jul-28 regardless of how additive the change looks.
- **RELATED, COLIN'S REVIEW BATCH (not this ticket):** pulse_check probe-pair
  pre-filter + adjacent-month match window (OUTCOME_LOOP_STALL fixes) — same
  contamination family as the RED-1 Blue change; they ship together after review
  (see audit/health_diagnosis_2026-07.md).

## Scouted seams (verified 2026-07-06)
- `sovereign/intelligence/decision_logger.py` DecisionRecord (~:96-98): NO exit_reason
  field → add `exit_reason: Optional[str] = None` + accept `reason=` in the
  update_outcome path (find its exact signature in-file; additive params only).
- Shadow log rows (`data/exec/exit_manager_shadow.jsonl`): the `decision` field ALREADY
  carries the taxonomy (HOLD/REVERSAL/INITIAL_STOP/TRAILING_ATR/TIME/CB_REFRESH) —
  mapped today at `experience/backfill.py:103` (_REASON). Do NOT rename shadow fields
  (frozen-path artifact); read-only join.
- `experience/backfill.py:93` hardcodes exit_reason="UNKNOWN" for carry closes → add a
  JOIN: for a closed decision, look up the shadow row by (trade_id if present, else
  normalized pair + close-date) and take its mapped reason; on ambiguity/conflict
  (two candidate reasons) → keep UNKNOWN + log the conflict (never guess).
- `sovereign/execution/oanda_bridge.py` close path: OandaFill is entry-only — add an
  OPTIONAL exit_reason to whatever close-trade record it writes (additive field; if
  the close path writes no record, add a minimal close-event append to the fills
  ledger — additive JSONL, new `kind: "close"` rows; NEVER touch entry-row schema).
- Exit-reason taxonomy: reuse `specs/thesis_exit_spec.md` §taxonomy VERBATIM
  (INITIAL_STOP · TARGET · TIME · thesis_invalidated + the shadow decision values).

## Constraints
- FREEZE: forex_exit_manager / decide_exit / exit_machine untouched. oanda_bridge is
  execution-layer — the close-record addition must be ADDITIVE-ONLY (new optional
  field/new row kind, zero behavior change to order placement); if any test or import
  suggests the close path is exercised by the frozen shadow loop, STOP and report
  instead of editing.
- Historical journal/attribution rows untouched. attribution.py logic untouched.
- Tests: new file tests/test_close_reason_capture.py — join hit, join conflict→UNKNOWN
  +logged, no-shadow-row→UNKNOWN, DecisionRecord round-trip with exit_reason, fixture
  shadow rows (tmp paths, monkeypatched constants per house idiom).

## Acceptance
Suite baseline holds (40 exact / ≥1144 / skips accounted); new tests green; a dry-run
backfill over July fixture data shows joined reasons where shadow truth exists and
UNKNOWN elsewhere; zero mutation of existing jsonl rows (sha test on fixtures).
