# R5 — Futures/Oracle stack portability audit (2026-07-03, read-only)

## Verdicts
- [nightly synthesis reflect_cycle+oracle_agent] → feeds TODAY: its own lesson loop,
  but on CONTAMINATED input (RED-1: FOREX 7-day summary = 7/7 backfilled rows, 5/7
  forbidden pairs, fabricated 1W/6L; no source filter) · could feed: review candidate-
  lessons AFTER the BLUE fix · scout said ATTIC-CANDIDATE — **SYNTHESIS RULING:
  QUARANTINED, not attic**: the organ is sound, its intake is poisoned; RED-1 fix
  (Colin's review) un-quarantines it. DO NOT PORT until then.
- [Big Move Oracle] → dashboard display only (DISPLAY-ONLY until validate gate
  passes; ES/NQ) · LEAVE (unvalidated + wrong market).
- [killzone telemetry] → futures intraday only · LEAVE (wrong market).
- [futures bias log] → futures loop only · LEAVE.
- [lesson_velocity.json] → dashboard panel only · could feed: 1-line read in
  weekly_review → "current lesson + formation stage" context · RETARGET (cheapest
  port in the whole stack).
- [morning briefing latest.json] → reflect_cycle qualitative context · could feed:
  the FRED macro_economic block as a review context section · caveat: regime_read is
  ES/NQ-specific and provenance.verified=false — port the macro snapshot ONLY ·
  RETARGET (partial).

## Headlines
1. Port lesson_velocity (1 line) and the briefing's FRED macro block (discard
   futures regime_read) into the Sunday review.
2. Do NOT port oracle reflection output anywhere until RED-1's source-exclusion fix
   lands (Colin's review) — quarantine, not demolition.
3. Everything ES/NQ-specific (Big Move, killzone, bias log) stays put: wrong market,
   and Big Move is DISPLAY-ONLY pending validation anyway.
