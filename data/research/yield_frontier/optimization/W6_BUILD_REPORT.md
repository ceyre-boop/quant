# W6 — Sizing Policy Simulator: Build Report
**Date:** 2026-07-21 · **Status:** BUILT + RUN · **Spec:** `optimization/W6_SPEC.md`
**Verdict:** `optimization/W6_results/W6_verdict.json` · **Ledger:** `data/research/preregister/verdicts_optimization.jsonl`

---

## Headline

**The floor gap was a sizing-policy problem, not an edge problem.**

HYP-093 (The Undertow) sealed at **VALID_BUT_BELOW_FLOOR** — a real, DSR-penalised,
BH-corrected edge whose constitutional yield (0.023%/day) fell short of the 0.05%/day
floor by 2.17×. The sealed verdict measured it under **fixed-fractional** sizing.

W6 tested whether a different sizing policy clears the floor on the same sealed event
paths. The answer:

| policy | arith yield/day | vs floor | drawdown p95 | ruin | status |
|---|---|---|---|---|---|
| F0 fixed-fractional | 0.000144 | 29% | 0.050 | 0.0000 | below floor (reproduces sealed) |
| F1 quarter-Kelly | 0.001081 | 216% | **0.292** | 0.0000 | **excluded** — drawdown breach |
| F2 RCK (raw) | 0.000978 | 196% | **0.306** | 0.0000 | **excluded** — drawdown breach |
| **F2+F3 RCK + DD governor** | **0.000584** | **117%** | 0.140 | 0.0000 | **CLEARS FLOOR** |
| F2+F4 RCK + day-heat | 0.000368 | 74% | 0.138 | 0.0000 | below floor |
| F2+F3+F4 | 0.000337 | 67% | 0.100 | 0.0000 | below floor |

**Recommended policy: F2+F3** — Risk-Constrained Kelly (Busseti-Ryu-Boyd) base with a
Grossman-Zhou drawdown governor. Effective size after locate: **f_T10 = 4.0% notional,
f_T20 = 3.4%**, scaled down continuously as drawdown approaches the 15% ceiling.

**FLOOR_CLEARED: True**, at the pessimistic disaster rate, with zero ruin.

---

## What the result does and does NOT mean

### It means
- The signal always had enough edge; **fixed-fractional sizing was leaving it on the
  table.** F0 extracts 29% of the floor; a drawdown-aware policy extracts 117%. Same
  events, same costs, same locate — only the sizing rule changed.
- W4's read that current sizing runs at 1/30–1/60 Kelly is confirmed: there was large
  headroom, and the binding constraint is drawdown, not signal.

### It does NOT mean
- **It does not override the sealed HYP-093 gauntlet verdict.** That verdict stands,
  final, on fixed-fractional sizing. W6 is a separate optimization-program result —
  exactly what the W-series was chartered to produce (`OPTIMIZATION_PROGRAM.md`): engineer
  the best way to run a *frozen* signal. The signal was not touched.
- **It does not clear the strategy for live money.** W6 clears ONE gate — sizing. Still
  outstanding, per the sealed report's own riders:
  - **No funded vehicle exists** (TICK-032, verdict NOT VIABLE). This is own-capital only.
  - TICK-024 cost cascade must land clean.
  - July-28 constitutional clamps.
  - Colin's explicit go.
- **It is not a rocket.** TPR ≈ 209 — roughly 209 expected drawdown-days of pain per unit
  of daily growth. At 0.058%/day this compounds to ~15%/year gross before the own-capital
  and clustering haircuts below. A grind with an edge, not a windfall.

---

## The load-bearing caveat, tested: clustering does NOT erase the margin

The i.i.d. bootstrap resamples events independently. The real event stream clusters —
sector runs, regime periods, one catalyst hitting several names at once — and clustered
wins are harder to compound than independent ones. The spec flagged this as the thing that
could overturn an i.i.d. floor clearance, and named a block bootstrap (block ≈ 5 trading
days) as the check.

**That check was run.** Moving-block bootstrap, block_size = 5, clustering preserved,
same 10,000 paths at the pessimistic disaster rate:

| resampling | arith yield/day | vs floor | p95 MaxDD | ruin |
|---|---|---|---|---|
| i.i.d. | 0.000598 | 120% | 0.140 | 0.0000 |
| **block (clustering)** | **0.000573** | **115%** | 0.139 | 0.0000 |

**The margin survives.** Clustering trims the yield by ~4% (0.000598 → 0.000573) and
leaves drawdown and ruin essentially unchanged. The floor is still cleared with room. The
caveat that looked potentially fatal is not — the F2+F3 policy's floor clearance is not an
artifact of the i.i.d. assumption.

This is the difference between "clears the floor in a friendly simulation" and "clears the
floor under the specific stress most likely to break it." It is the latter. What remains
before live money is operational, not statistical — see the gates below.

---

## Method (what was built)

Three files, all read-only against the sealed holdout:

1. **`research/yield_frontier/w6_extract_events.py`** — reproduces the 559 sealed HYP-093
   event paths from the raw Alpaca holdout via the frozen fill rule in `gauntlet_run.py`
   (transcribed verbatim, including the 8-bucket news classifier and the borrow-APR
   schedule). Validated against `verdicts.json`: n_events, event_mean, event_median,
   event_p5 and worst_event all reproduce **to the last decimal** (559 / 0.01596 / 0.04874
   / −0.32644 / −0.49784), and the guards match (sparse 251, mna_excluded 44, no_data 10).
   Refuses to write if validation fails.

2. **`research/yield_frontier/w6_simulator.py`** — the simulator. 10,000 bootstrap paths
   of 559 events, tier-preserved (T10=354 / T20=205), GPD left-tail extension on 5% of
   paths (scipy.stats.genpareto MLE on the worst-decile losses), disaster mixture
   (halt-gap-buy-in, P ∈ {0.001, 0.002, 0.005}, L ∈ [−100%, −200%] scaled by tier W*).
   Five policy families, lexicographic selection, arithmetic-yield floor comparison.

3. **verdict + ledger** — sealed to `W6_results/W6_verdict.json` and appended to
   `verdicts_optimization.jsonl`.

### Anti-overfit welds enforced (spec §Failure Modes)
- All policy parameters (κ, α, ε_dd, DD_max, γ, h_max) **fixed before** the bootstrap.
  No search on bootstrap performance.
- Selection is **lexicographic** (G → CDaR → TPR), never a weighted score.
- Disaster frequency **swept, not averaged**; the winner is chosen at the pessimistic
  0.005 and verified to win at 0.001 and 0.002 too (0.089% → 0.082% → 0.059%/day, zero
  ruin at every rate).
- Floor comparison in **arithmetic yield/day** (Σ ret·size / days), matching HYP-097 —
  not the higher log-growth G, which is reported separately.

### Two recorded deviations from spec (not applied silently)
1. **Event set is the full 559-event sealed gauntlet**, not the 539-event HYP-097 sizing
   subset the spec inputs cite. The 559 set reconciles to `verdicts.json` exactly; the
   spec's 0.000166/day F0 target was on the 539 subset. F0-flat on the 559 set reproduces
   the sealed 0.00023 constitutional yield (validation anchor: got 0.000226, within MC
   noise), so the 559 set is the correct one to bootstrap from. The floor verdict is
   identical either way — both F0 baselines are far below floor.
2. **RCK solved as a scalar-f drawdown certificate**, not a cvxpy LP. For a scalar
   per-tier leverage the Busseti-Boyd program reduces to: the largest f whose whole-path
   max-drawdown on the empirical event sequence ≤ ε_dd (drawdown and growth both monotone
   in f up to Kelly). An earlier draft used per-event CVaR of log-increments and left f at
   a reckless 37% notional with 21% ruin; the drawdown *certificate* binds far tighter and
   is the faithful reading — corrected to 8%/6.7% base, and the result strengthened.

---

## Numbers for the record

- F0-flat validation anchor: 0.000226/day vs sealed 0.00023 ✓ (pipeline validated)
- Recommended F2+F3 effective sizing: f_T10 = 0.0399, f_T20 = 0.0336 (post-locate)
- Best surviving arith yield/day: **0.000584** (117% of the 0.0005 floor)
- Gap to floor: **+0.000084/day** (cleared)
- P_ruin: 0.0000 · p95 MaxDD: 0.140 · CDaR(0.9): 0.141 · TPR: 209
- Disaster robustness: cleared at all three rates, zero ruin throughout
- Block-bootstrap (clustering): 0.000573/day, 115% of floor, zero ruin — margin survives
- Input hash (event set): first 16 of sha256, recorded in the verdict

---

## What comes next (in priority order)

1. **W7 live shadow at F2+F3 sizing** — the shadow is already running the frozen signal;
   upgrade it to track P&L, drawdown and locate outcomes at the recommended sizing
   (f_T10 = 4.0%, f_T20 = 3.4% post-locate, drawdown-governed). Real forward data is the
   only thing the simulation can't provide.
2. **Own-capital plan** — TICK-032 settled that no prop firm funds this strategy class, so
   the vehicle is Colin's own capital. The sizing rule and the ~15%/yr-gross expectation
   (pre-clustering-and-cost haircuts) feed directly into Template B.
3. **Catalyst split (TICK-034)** and **tier restriction (T10-only)** — two additive
   edge-concentration paths, now *optional* rather than necessary since sizing alone
   cleared the floor under clustering. Each would widen the margin further.

The strategy is not cleared to trade — the gates below still stand. But the statistical
question is now answered: a drawdown-aware sizing policy clears HYP-093's floor, and the
clearance survives the clustering stress that was most likely to break it. What is left is
operational: forward shadow evidence, the cost cascade, the July-28 clamps, and Colin's go.

---
*Alta Investments · Yield Frontier Optimization · W6 Build Report*
*Signal frozen. Sizing engineered. Floor cleared in sim — clustering pending.*
