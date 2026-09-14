# HYP-119 — the liquidity-cascade state (|kinetic| / dampening, z > 3)

**VERDICT: MAGNITUDE_ONLY** (liquid); microcap INCONCLUSIVE (32 long-side events < 40 minimum).
Sealed `16ecff37d46aaddd` with a Z3 causality proof attached (14 leaves, UNSAT); hash verified before and
after; ledger `ADJUDICATED`. One run (two earlier launches were killed by the host before any statistic
was computed; the fill arithmetic was vectorised — same formulas — so the run fits the foreground window).
First hypothesis through all four stages of the loop.

| claim (liquid, 7,640 events, 23 names, 2024-10 → 2026-09) | result | bar |
|---|---|---|
| (a) forward 30-min range after STATE ÷ non-STATE | **2.71×, CI [2.57, 2.87]** | PASS |
| (b) continuation in the cascade direction, measured spread, halt slip | **−0.276%/trade**, CI [−0.33, −0.22]; hit 41%; placebo p 0.64; perm p 1.0 | FAIL |
| (b) by direction | up-cascades −0.20% (n=3,723) · down-cascades −0.35% (n=3,917) | both lose |
| (c) proven halt bound | worst adverse 22.7% vs proven median bound 10.6%; **3 breaches** in 7,640 | FAIL |

## What it means

1. **The Navier–Stokes reading is right about the blowup and wrong about the ride.** When kinetic
   force overwhelms the bar-derived dampening proxy, the next 30 minutes move 2.7× more than normal
   — the strongest magnitude result in the repo (HYP-109 was 1.36× at weekly). Both priors said
   MAGNITUDE_ONLY, and it is.
2. **Riding the cascade loses in both directions.** −0.28% per trade net, CI entirely negative, worse
   than a coin flip (placebo p 0.64). The move that "breaks the rules" mean-reverts inside the half
   hour on liquid names — the direction the crowd is pushing is, again, the wrong side. This is the
   intraday cousin of every direction null on the ledger. (A *fade* of the cascade is implied, not
   tested: it would be a new prereg at n_trials 1646+, and the daily fade's history — HYP-114 — is the
   prior against it.)
3. **The formal bound caught a model failure, which is its job.** Three events moved 22.7% adverse
   without a halt flag on the tape: either the bar-level halt detector missed a halt, or the LULD
   reference lagged further than five closes, or the print data has a gap. Under the sealed text a
   realized loss beyond the proven bound is a model failure, not bad luck — and it is reported as
   one. The exchange-mechanics prover needs Tier-1 handling (SPY/QQQ/AAPL are 5% names, not 10%)
   and a check against the official halt list before it is relied on for sizing.
4. **The microcap universe was too thin (32 long-side events).** The cache holds 233 event-days; the
   state fired on 50, long on 32. Where LULD actually binds, the question is unanswered, not null.

## What this closes and what it opens

Closes: "ride the blowup" on liquid names, at 30 minutes, from bar-derived state. Opens (not
proposed, only listed): the cascade *fade* as its own prereg; a microcap panel large enough to
adjudicate (needs ≥ 40 long cascades → ~300 event-days); Tier-1 bands and an official halt list in
`research/formal/mechanics.py`.

## Constraints honoured
One run. No z threshold, window, horizon or universe changed. The vectorisation changed no formula
(verified by the formal-stage tests and identical per-event arithmetic).
