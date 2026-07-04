# R3 — Memory organ retargeting audit (2026-07-03, read-only scout)

## What the review path READS today (file:line)
- weekly_review.py:65-66 — journal.read_all() (60 rows / 5 files) + att.read_attributions()
  (8 rows; 7 AMBIGUOUS); LEDGER (line 45) opened APPEND-ONLY (dedup) — results never read.
- journal_sync.py:32 shadow log · :63-95 decision_logs (extracts only pair/outcome/r;
  discards why_this_trade, signal_layers_active, component_scores, active_lessons,
  commitment_score, grade) · :104 proof_of_life (TODAY only) · :157-164 board_ref lazy fill.
- attribution.py:136-140 own outputs (idempotency) · backfill.py rate parquets + vix_gate.

## Unread inputs (exist on disk, zero reads by the review path)
1. Decision-logger rich context (decisions_*.jsonl, ~60 fields) — journal keeps 3.
2. Veto ledgers forex+ICT (10K+ lines/mo) — zero reads.
3. OANDA fills ledger — no join; backfill.py:93 hardcodes exit_reason="UNKNOWN".
4. Oracle daily reflections — 2026-07-03 carries a CRITICAL integrity flag ("0 logged
   trades in 7 days; closed trades RECONSTRUCTED") — review shipped blind to it.
5. Oracle harvests (33 files) · 6. lesson_velocity.json · 7. audit/reports/* shadow
   divergence (L1/L2 parity) · 8. hypothesis ledger RESULTS (status field) ·
9. proof_of_life full snapshot (historical) — no archive exists.

## Verdicts (schema: feeds TODAY · could feed · gap · cheapest rewiring · verdict)
- [journal] → review counts acted/abstained · could carry decision-logger context ·
  journal_sync discards 95% of fields · preserve why_this_trade + signal_layers_active
  + grade at sync time (3 lines) · RETARGET
- [attribution+rubric] → 7/7 AMBIGUOUS (exit_reason missing) · could classify real
  exits · no fills join · JOIN oanda_fills on trade_id for slippage/validation ·
  RETARGET — but NOTE (synthesis ruling): exit_reason INFERENCE is refused; the
  sanctioned path is PROP-2026-W27 (capture at write time, Colin promotes). Join for
  validation ≠ invention.
- [weekly_review generator] → decision-accounting only · could be decision-forensics ·
  zero reads of oracle/audit/ledger-results/vetoes · 3 one-line JSON reads (reflection
  system_health_note; audit L1/L2 parity; week's tested-status Counter) · RETARGET
- [backfill.py] → attribution driver · could validate realized_r vs fills · no fills
  join · same JOIN as above · RETARGET
- [board context on acted/abstained days] → board_ref lazy fill works; gaps are data
  sparsity not wiring · LEAVE
- [veto ledgers] → none · acted:abstained:vetoed ratio + veto-stage mix in review ·
  no consumer · 3-line monthly-file count/summary · RETARGET
- [oracle reflections] → Oracle self-loop only (and RED-1-contaminated on FOREX
  summaries) · system-health + candidate-lesson section in review · no consumer ·
  1 JSON load into "surprises" · RETARGET
- [shadow-audit reports] → operator eyes only · review-time system-health line
  (determinism/parity) · no consumer · parse audit/reports/{date}.json · RETARGET
- [hypothesis ledger results] → write-only from review · "what THIS week's research
  concluded" section + don't re-propose rejected · never read back · 2-line status
  Counter over date_tested in week · RETARGET
- [proof_of_life historical] → today-only by design; no archive exists · LEAVE
  (POL historization = future ticket if ever needed)

## Headlines
1. Oracle reflection integrity flags never reach the review — W27 shipped blind to
   "all closed trades are RECONSTRUCTED" (which also taints its 7/7 AMBIGUOUS).
2. OANDA fills ledger is the missing validation join for attribution (slippage,
   realized_r checks) — exit_reason capture stays PROP-2026-W27's territory.
3. The review never reads the ledger's RESULTS — the learning loop half-closes:
   proposes but never hears verdicts.
