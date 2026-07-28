# HYP-071-v2 Pre-Registration — Exit Value Board, Metric-Fixed (DRAFT)

**Status: DRAFT_PENDING_OPERATOR_APPROVAL.** Not hashed. Not locked. Not filed
in `data/research/preregister/`. No hash appears in this document — Colin
assigns and locks the hash when he approves this design, per the same
protocol as `research/hyp_104_downgap_short_prereg.md` and the original
HYP-071 v1/v2 preregs. This document does not itself open the training gate
(`sovereign/training/gate.py`); it is the artifact the gate's revival guard
requires *in addition to* NET costs and a post-2026-06-30 CONFIRMED
adjudication.

Registered (draft form) 2026-07-28. Distinct from and superseding no prior
prereg — this is a **new hypothesis registration**, not a rerun. It must not
reuse any of the three locked hashes on file:
`c4f29ac3…` (v1), `3d500bda…` (v2), `c1fab807…` (interpretation addendum).

---

## 1. What killed v1 — the specific failure this design must beat

Ledger `HYP-071` (2026-06-30, `METRIC_ARTIFACT`): the per-cell value function
`V(cell, action) = E[R] − λ·DownsideDeviation(R)` compares a **locked**
current-state value (EXIT_NOW, zero forecast variance) against a
**forecast-through-resampling** value (HOLD_AND_TRAIL, real forecast
variance). The λ·DD penalty only bites the side that has variance to
penalize. EXIT_NOW dominance was mechanically guaranteed by that asymmetry,
independent of whether there is any real exit-timing edge. Ledger
`reopen_condition` (verbatim): *"New prereg with symmetric forecast variance
on the EXIT_NOW side, OR a pure E[R] comparison without the lambda*downside
penalty."*

The 2026-07-23 rerun (commit `2be5726`, "PROVISIONAL PASS", 10 CPCV-stable /
9 forward-consistent EXIT_NOW divergences) did **not** touch this flaw — it
reused the same asymmetric metric on a different cost basis, was flagged
governance-invalid same-day (`HYP-071-GOVFLAG`), and correctly failed to
flip the gate's `hyp_071_net_confirmed` guard. Gross costs vs. net costs is
an orthogonal fix (NET is now available post-TICK-024) and is necessary but
**not sufficient** on its own — a v2 that only swaps gross→net without fixing
the metric asymmetry would repeat the same METRIC_ARTIFACT failure with the
existing loose hashes already covering that variant. This is why this
prereg exists: it is a new hypothesis, not a cost-basis rerun.

## 2. Hypothesis

Testing **exit timing within proven carry trades**, not directional
prediction. On the locked 8-dimensional state vector (ATR% tercile ×
excursion-R × hold-fraction × RSI-extreme × carry-alignment, same board
geometry as v1/v2) and on **NET returns** (post-TICK-024 corrected swap
model), a value function with a **metric-symmetric** comparison between
EXIT_NOW and HOLD_AND_TRAIL identifies a non-trivial subset of cells where
exiting materially outperforms the static v015 hold/trail rule — and that
subset is not an artifact of asymmetric forecast variance, is not purely
explained by carry alignment, and is stable enough to be worth a config
change.

## 3. Design fix (locked mechanism — pick ONE, pre-registered before results)

Per the ledger's own reopen condition, the v2 board must implement exactly
one of:

- **(a) Symmetric forecast variance.** Resample EXIT_NOW forward too — i.e.
  score EXIT_NOW not as the locked realized-to-date R but as its own
  regime-conditional bootstrap continuation under a "flat/no-position"
  counterfactual (funding-neutral, zero market exposure after exit), so both
  arms carry real forecast variance and the λ·DD penalty applies to both
  symmetrically; **or**
- **(b) Pure E[R] comparison.** Drop the λ·DD downside-penalty term entirely
  and compare EXIT_NOW vs HOLD_AND_TRAIL on expected return alone,
  eliminating the asymmetric-penalty channel outright.

**This prereg locks choice (a) or (b) BEFORE the board is computed — Colin
selects one at approval time; whichever is selected must be recorded here
before any run, not chosen post-hoc from whichever gives the better number.**

## 4. Cost basis

NET returns, corrected swap model (TICK-024 landed: `SWAP_RATES_ANNUAL` fix +
EURUSD-short sign flip). The v1/v2 boards were GROSS by locked design; v2
board config must carry no `gross_R_caveat` marker (this is also the
`sovereign/training/gate.py::_board_is_net` runtime guard's own check —
consistency between this prereg and the code guard is intentional, not
coincidental).

## 5. Pre-registered pass criteria (decided now, before any results)

**CONFIRMED requires ALL of the following:**

1. **Metric fix applied** — board computed under locked choice (a) or (b)
   from §3, not the asymmetric v1/v2 metric. (Structural gate — if this
   isn't true, stop; nothing else in this list matters.)
2. **CPCV-stable divergences exist** — at least 5 cells where the
   metric-fixed value function disagrees with the static v015 rule
   (EXIT_NOW where static says HOLD_AND_TRAIL, or vice versa), stable under
   combinatorial purged cross-validation (same CPCV harness as the v1/v2
   run) at the ≥0.90 sign-agreement bar — the same bar the v1/v2 run
   **missed** (0.854, flagged as "below robust" in the prior report). 0.90
   is the floor here, not aspirational.
3. **Forward-consistent** — of the CPCV-stable divergences, ≥70% agree in
   direction between the 2023-24 OOS window and the 2025-26 forward window
   (mirrors the v1/v2 "9 of 10 forward-consistent" finding, but now applied
   to a metric that isn't structurally biased toward EXIT_NOW).
4. **Not purely carry-aligned.** The v1/v2 finding was confounded — every
   surviving divergence was a carry-aligned cell, which is exactly the
   subset where the swap-model bug and the metric asymmetry both point the
   same direction (both bias toward under-counting the value of holding).
   CONFIRMED requires that **at least one CPCV-stable, forward-consistent
   divergence is NOT carry-aligned** (i.e., appears in a REVERSAL-side or
   carry-neutral cell). If every surviving divergence is still
   carry-aligned, treat that as the confound reasserting itself under a new
   name — verdict caps at MARGINAL, not CONFIRMED, regardless of criteria
   2-3.
5. **Beats the permutation placebo, out-of-sample.** One-tailed sign-flip
   permutation test (same family as `research/hyp_104_downgap_short_prereg.md`
   — 2000+ perms) on the forward-window (2025-26) divergence-cell returns
   vs. the null of no exit-timing edge. Require **p < 0.05**.
6. **Survives multiple-testing correction.** The board evaluates ~54
   carry-aligned-or-reversal cells (108-cell board minus the 54 N/A-by-
   construction reversal cells, per the v1/v2 split) — each cell's
   EXIT_NOW-vs-HOLD divergence is effectively a separate test. Apply
   Benjamini-Hochberg FDR correction across all evaluated cells at α=0.05
   and require the surviving-CONFIRMED divergences from criterion 2 to
   **still hold after BH correction**, not just on raw per-cell p-values.
   (This ties directly to the TRUST_DECISION_BRIEF — see that document for
   whether BH is also applied retroactively across the full ledger; this
   criterion applies BH within HYP-071-v2 itself regardless of that
   decision, since it is a fresh test, not a past verdict.)
7. **Reconciliation gate holds.** Same non-negotiable as v1/v2: recomputing
   the decade portfolio weighted Sharpe against the frozen 0.6886 baseline
   must reproduce exactly (±0.01), and re-trace parity vs. the canonical
   ledger must be 100%. If either breaks, the run is invalid regardless of
   the exit-value numbers — it means the harness silently diverged from the
   live system it's supposed to be evaluating.

**NOT_ROBUST if:** criteria 2-3 pass but 4, 5, or 6 fail — i.e., real
structure exists but it's either confound-explained, doesn't beat chance, or
doesn't survive correction. This is a *meaningful* outcome, not a null
result — it would mean the metric fix worked (no more tautology) but there
is genuinely no exit-timing edge beyond what carry/regime already explains.

**DATA_INSUFFICIENT if:** forward window (n≈246 members, 2025-26) yields
fewer than 15 common cells for the forward-consistency check (criterion 3)
— the v1/v2 run's forward window was already flagged thin (n=23 common
cells); a metric change could shrink the usable cell count further.

## 6. Explicit non-negotiables (carried from v1/v2, restated so this doc is
self-contained)

- Placebo permutation control is **mandatory**, not optional — no verdict
  above NOT_ROBUST without it passing.
- Applying any resulting rule change to `forex_exit_manager`, `decide_exit`,
  `exit_machine`, or any live config remains **FROZEN** independent of this
  prereg's outcome. A CONFIRMED verdict here unlocks the training-gate
  revival guard in `sovereign/training/gate.py`; it does **not** by itself
  authorize a live exit-rule change — that is a separate decision requiring
  its own logged rationale per CLAUDE.md NN#4.
- This prereg, once hashed and locked by Colin, becomes the fourth entry in
  the HYP-071 prereg family and must be listed alongside the three existing
  locked hashes in `sovereign/training/gate.py::HYP071_LOCKED_PREREG_HASHES`
  — but only as the *new* prereg satisfying the revival guard, never
  retroactively added to the "must not reuse" set (that set stays fixed at
  v1/v2/addendum).

## 7. Hash / lock / adjudication — PENDING COLIN

- Prereg hash: **PENDING** (assigned at lock time, not before).
- Lock timestamp: **PENDING**.
- Board computation: **NOT RUN.**
- Adjudication: **PENDING** — requires a hash-locked run against this exact
  document, dated after 2026-06-30 (trivially true) and after this
  document's own lock date, per `sovereign/training/gate.py::
  _hyp071_revival_confirmed`.
- Metric-fix choice (§3, (a) vs (b)): **PENDING COLIN SELECTION.**

**Status: DRAFT_PENDING_OPERATOR_APPROVAL.**
