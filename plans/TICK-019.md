# Plan — TICK-019: geometry fill + Gα/Gβ under the LOCKED specs (the Day-4 spine)

The geometry family's verdict path is the only one fully in a session's hands (no
plist, no external data, no operator dependency). This plan arms it completely.
Specs are LAW: data/research/preregister/HYP-082/083 + GEOMETRY-2026-07 manifest,
hash-locked at 01cacbd BEFORE features existed. Nothing below may reinterpret them.

## Phase 0 — preflight gates (abort on any failure)
1. `git fetch` + clean-enough tree; watchdog GREEN (rebaseline-with-reason first if
   Colin's plist loads landed since the 21-job baseline).
2. No writer on data/sentiment.db (ps for update_sentiment/feeders; if the 07:45 job
   is loaded and near its window, wait it out — single-writer DB).
3. Gate-zero hash verify over ALL FOUR geometry files (082, 083, 084, manifest) +
   re-verify the 10 positioning preregs are untouched (cheap, same helper).

## Phase 1 — real-data fill + audit gate (first live run of geometry_feed)
1. `geometry_feed.update()` against the real board (spot parquets are local; ~3k
   bars × 4 pairs; idempotent upsert). Runtime risk: the rolling linregress loop —
   accept seconds-to-minutes; if >8 min, it is NOT a background gamble: fix the
   vectorization (cumulative-sum regression algebra, already flagged in D2) and
   rerun. Warmup rows must land as NULL (already tested synthetically).
2. `board_state.rebuild()` → report rows.
3. **HARD GATE: `scripts/audit_look_ahead.py` → 0 violations** (now includes the
   geometry provenance block + empirical tuples). Any violation → STOP, report,
   nothing runs. Also spot-check: geometry columns non-NULL share post-2015-07
   (corridor needs 120-bar warmup from the 2015-01 start).
4. Commit checkpoint `[RESEARCH] geometry board fill` (param-free, data + nothing).

## Phase 2 — build `scripts/research/run_geometry_family.py`
Mirror `run_positioning_family_options.py` architecture exactly (gate-zero → loads →
legs → seal() with PREREGISTERED-assert → `--dry-run` → `--adjudicate`). Seed 42,
n_perm 10,000. Interpretations file-header block declared BEFORE results (A1 banner
restated). Data: board geometry columns via store.connect(read_only=True); spot
OHLC/closes from the same parquets the features came from; v015 equity via
vr.load_trades + reconcile guard 0.6886 (same standard).

### Gα (HYP-082) leg — exactly per spec
- Observations: every 5th trading day per pair (de-overlap for h=20), non-NaN
  corridor_dev, from 2015-07 (post-warmup).
- carry_residual fwd return: r_fwd(h=20) − β_t · r_v015(h=20 same window), β_t from
  trailing-252d OLS of daily pair returns on v015 daily returns (trailing-only;
  min_periods 252; obs with undefined β dropped and counted).
- Primary: pooled Spearman IC(corridor_dev, carry_residual_fwd). Null: within-pair
  permutation of the FEATURE vector across eligible obs (10k, seed 42).
  **TWO-SIDED p** (spec-declared deviation from the one-sided family convention).
- Success components computed and reported ALL FOUR: BH-pass (with Gβ, m=2, α .05) ·
  |IC| ≥ 0.05 · CPCV fold-sign consistency (sovereign/discovery/cpcv.py, 6 groups,
  embargo 0.02 — IC recomputed per test fold, ALL same sign) · cost floor (median
  |fwd h20 move| among top-decile |corridor_dev| obs ≥ 3.0 pips = 0.00030 for
  4-decimal pairs, 0.030 for USDJPY — pip definition stated in the artifact).
- Secondaries (exploratory, never adjudicated): ICs of fvg_count_20d, fvg_unfilled,
  range_slope, days_in_consolidation.

### Gβ (HYP-083) leg — exactly per spec
- Events: daily FVG FORMATION days. Seam: the board carries counts, not events —
  the runner re-derives formation dates by calling the REPLICATED kernel per
  trading day on df.iloc[:t+1] (or an additive thin helper
  `geometry_feed.fvg_formation_events(df, cfg)` returning (date, direction) — add
  the helper + a parity-with-counts test; do NOT import ict). Side = gap direction,
  pair space, no flip. De-overlap: one event per pair-day; same-direction
  suppression 5 td. t0 = strictly-after (shift_pub convention).
- Primary: pooled median signed fwd log return h=10, one-sided (continuation);
  per-pair date-shuffle null preserving counts (es.pooled_primary_p verbatim).
- Diversifier gate (JOINT success condition): event-book daily returns
  (corr_to_carry machinery) vs BOTH v015 curve AND DBV (vrp data_loader
  load_carry_proxy; overlap ends 2023-03 — n stated honestly): |ρ_full| < 0.25 AND
  max crisis |ρ| < 0.35 over 2020-02-20→04-30 and 2022 full year.
- N < 50 de-overlapped → UNDERPOWERED stamped (daily-bar honesty per spec).

### Adjudication (same run — the manifest allows it: members run/report together)
`--adjudicate`: BH m=2 α=.05 over the two primary p's → per-member verdicts written
as dated ledger annotations + verdict fields: fail → NOT_SIGNIFICANT (the prior);
Gα BH-pass but any of |IC|/fold-sign/cost-floor fails → NOT_ROBUST /
NOT_SIGNIFICANT per spec wording; Gβ BH-pass but gate fails → NOT_DIVERSIFIER;
N-floor breach → UNDERPOWERED. A CONFIRMED-class survivor (all components green) →
**E4 protocol: ledger + report `python -m factory.train --hyp HYP-08x`, DO NOT run,
halt.** Artifacts: data/research/geometry_family/{HYP-082,HYP-083}.json +
run_manifest. `--dry-run` first, numbers reviewed for MECHANICS only (no threshold
revisits), then the real run in the same sitting.

## Phase 3 — close
Full suite (baseline 40 exact / 1185+new / 1 skipped) · watchdog · push per seal
batch · NEXT.md entry (verdicts, N's, honest stamps) · backlog TICK-019 → done.

## Tests shipped with the runner
Runner-level: gate-zero mismatch aborts (tamper fixture) · Gα residualization is
trailing-only (truncation test on β_t) · pip-floor arithmetic · Gβ event de-overlap
+ suppression on synthetic gaps · formation-events parity with board counts ·
adjudication verdict mapping table (all outcome classes) on fixture p's.

## Risks
- First real-data geometry fill may surface parquet-vs-synthetic surprises (dtype,
  gaps) — the auditor gate catches look-ahead classes; NULL-share sanity catches
  coverage classes; anything else stops the run honestly.
- Gα's β residualization needs v015 daily returns over 2015+ — vr equity curve
  covers it (reconcile-guarded); undefined-β obs counted, never imputed.
- Expect UNDERPOWERED on Gβ (daily FVGs are rare) — the spec pre-accepts it; do
  not widen definitions to buy N.

## NOT in scope
HYP-084/Gγ (dark-month clock, TICK-017 scorer + Colin's flip) · any positioning-
family action (PENDING-080-SCHEDULED) · v2 (Colin's ack) · tier-config clamps
(shadow_close).
