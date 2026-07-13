# Execution Engineering of the Gapper-Fade Signal — program charter (TICK-033)

**A new research field for this shop: not finding the signal (done, sealed — HYP-093),
but engineering the exact, precise, best way to run it every day.** Millions of trials
belong HERE — in simulation of execution policies around a FROZEN signal — never in
re-tuning the signal itself.

## The constitution of this program (anti-overfit welds, non-negotiable)

1. **The signal is FROZEN**: ≥50% by 10:30 ET, ≥$2, ≥500K by-10:30 volume, M&A excluded,
   as sealed in HYP-093. Re-tuning signal thresholds at daily resolution re-litigates
   HYP-090's tombstone (adaptive/optimized parameters lost to random) and is BANNED.
   Signal changes require NEW hypotheses on NEW data through the standard gate.
2. **What optimization MAY touch (the execution wrapper):** instrument (stock short vs
   long puts vs spreads), stop design & worst-case geometry, sizing policy under fat
   tails, event-overlap/portfolio-heat rules, locate allocation, entry timing
   microstructure within [10:30, 11:00), exit timing, halt handling.
3. **Trials discipline:** every simulated policy variant counts in an append-only
   trials ledger (mined_n.json pattern). Any policy promoted to a live-candidate gets a
   hash-locked prereg and must clear its floor on data it was not optimized on.
4. **Live gating ladder (no step skips):** winning design CONFIRMED under its own
   prereg → sim/paper daily-operation period → TICK-024 cascade clean → constitutional
   clamps mechanically enforced (July-28) → Colin's explicit go.

## Workstreams

| WS | Question | Method | Status |
|---|---|---|---|
| W1 Mechanism & literature | Why does the fade exist; who is on the other side; known decay | specialist agent (academic lit) | dispatched 2026-07-13 |
| W2 Defined-risk instrument | Are same-day puts on these names buyable at spreads that preserve the edge → HYP-096 spec | specialist agent (options microstructure) + ThetaData probe later | dispatched |
| W3 Short-side plumbing | Locate mechanics, SSR/Reg-SHO 201, LULD reopen behavior, broker landscape for execution (not funding) | specialist agent | dispatched |
| W4 Sizing under jump risk | Kelly/drawdown-constrained sizing with −60% tails; overlap heat math | specialist agent | dispatched |
| W5 Data for millions of trials | Deeper history (paid Polygon/Databento equities), historical borrow rates, halt records — cost & coverage | specialist agent | dispatched |
| W6 Policy simulator | Event-bootstrap Monte Carlo over the sealed event pool: policy grid × 10^6+ resampled years; outputs distribution of %/day, drawdown, ruin per policy | spec-first, then build (spec: `optimization/W6_SPEC.md`, to be written before code) |
| W7 Fresh-data replay | Walk the frozen signal forward (2026-07→) as pure shadow observation — the only new signal data that costs nothing | joins the daily-jobs family after W6 |

## What "exact precise best way to run it every day" will mean at the end

A single operations card: instrument, size formula, stop, overlap cap, locate budget,
entry/exit clock times, halt protocol — each line traceable to a workstream result and
the whole card CONFIRMED as one prereg before a dollar moves.
