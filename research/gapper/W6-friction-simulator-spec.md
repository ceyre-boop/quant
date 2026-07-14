# W6 — Friction Simulator Spec (SPEC-FIRST, pre-code)

**Status:** SPEC ONLY. No simulator code exists yet, and none may be written until this document
is committed. Pre-registration law (RISK_CONSTITUTION Art. 6): the measurement apparatus is
specified before it is built, so the build cannot be tuned to a desired answer.

**Ticket:** TICK-033 (W-series, gapper fade short). **Date:** 2026-07-13.
**Edge under study:** HYP-093 parabolic gapper fade short — CONFIRMED on holdout (p = 0.031,
n = 559 events, net median +4.9%). This spec is the friction layer between that raw edge and any
capital decision.

**Cross-reference / path note:** `research/yield_frontier/OPTIMIZATION_PROGRAM.md` (W6 row, and
line 53–58) earmarks a `optimization/W6_SPEC.md` for the *sizing-policy* simulator (policy grid ×
resampled years → %/day, drawdown, ruin per policy). This document is the *friction* simulator —
the return-generating engine that produces the per-event net-return distribution that the sizing
grid then resamples. They compose; they are not duplicates. **W6-friction (this doc) → net return
per event → W4 sizing grid → multi-year path distribution.** If the two specs are ever unified,
this note is the seam to reconcile. Do not let the two drift.

---

## 1. Objective

Produce a **per-trade friction-adjusted net-return distribution** over the 559 confirmed HYP-093
fade events — **not a single friction number, and not a flat cost haircut.**

The core thesis of W6 is inherited from the W3 load-bearing flag: **frictions on this edge are
adversely selected.** Locate scarcity, SSR passive-entry constraints, borrow cost, and halt risk
all bind *hardest on the best events* — the low-float day-2 runners whose deepest reversals carry
the fade edge's entire right tail are exactly the names that are hardest to borrow, most often
already on SSR at the open, and most exposed to limit-up gap-through. A simulator that models
frictions as i.i.d. costs would over-credit the strategy precisely where it is most fragile.

**Output object:** for each event *i*, a distribution over net return `r_i` (Monte Carlo draws),
plus a realized-availability flag (was the trade takeable at all). Aggregated across the 559
events and their friction draws, this yields the friction-adjusted return distribution the sizing
layer consumes.

**What "net" means here:** gross fade return (10:30 entry → exit) minus locate fee, minus borrow
accrual (if held), minus entry slippage, adjusted for SSR fill probability, and re-marked to the
actual exit (stop gap-through, halt reopen, or forced early cover) rather than the clean close.

---

## 2. Friction model components (each modeled as ADVERSELY SELECTED)

Every component below is a function of **signal quality**, proxied primarily by **gap magnitude**
(and, where noted, prior-day return, float/ADV, and prev-close price tier). None is a flat cost.
All constants are drawn from `W3_plumbing_brief.md` (primary-source-verified July 2026),
`W4_sizing_brief.md`, and `W5_data_procurement.md`.

### 2.1 Locate availability — `P(locate | gap)`

- **Adverse selection:** `P(locate)` **decreases** in gap magnitude and **decreases** in the
  depth of the expected reversal. The best fade events (deepest reversals — per W3's flag, the
  hardest-to-borrow low-float squeezers) draw the lowest locate probability. The `P(locate)`
  curve must be **monotone decreasing in gap size**, calibrated so the top reversal decile carries
  the lowest availability.
- **Realization:** Bernoulli draw. On failure → event marked **UNAVAILABLE** (contributes to the
  "% unavailable entirely" metric; no return, not a zero return).
- **W3 constants:** availability can be **0**; HTB names gate at 10:30; IBKR model has no
  per-share locate but a hard availability gate at 10:30. `P(locate)` is the gate, not a cost.
- **Data note:** historical per-name borrow *availability* is the market's true data gap (W5:
  only Ortex sells a clean 2015–2026 backfill, ≥$499/mo). Shoestring calibration = conservative
  hard-coded HTB availability curve keyed to gap/float, disclosed as an assumption (current prereg
  approach). The curve is a **modeling input to be pinned in the validation step**, not a measured
  series — flag it as such.

### 2.2 SSR trigger — DETERMINISTIC from tape

- **Rule (from W3, SEC Rule 201 FAQ):**
  `SSR_active(D) = (min RTH trade(D-1) ≤ 0.9 × close(D-2)) OR (min RTH trade(D) ≤ 0.9 × close(D-1))`.
  Day-2 runners are **often already on SSR at the open** — no free parameter, computed per event
  from the price path.
- **Adverse selection (intrinsic):** SSR effect is **passive-only entry** — post above the NBB,
  fill-on-uptick, `P(fill) < 1`, adverse delay: **you get filled on the bounces and miss the
  flushes.** On the strongest down-moves (the best fade fills), SSR is most likely active and most
  likely to deny the aggressive entry — the constraint is worst exactly where the edge is best.
- **Realization:** when SSR active, entry is a passive fill with `P(fill) < 1` and an entry-price
  penalty (filled higher than the flush low). Missed passive fills → event either skipped or
  entered at a degraded price, per the entry sub-model.

### 2.3 Borrow cost — regime draw correlated with gap size

- **Regimes:** {easy, hard, unavailable}. `unavailable` is the §2.1 gate. `easy`/`hard` set the
  per-share locate fee.
- **W3 constants:** per-share upfront, **~$0.01–0.05/sh typical HTB**, **fat tail ≥$0.10–0.30/sh**.
  IBKR: same-day round trip = **$0 borrow** (borrow accrues only at settlement), so an intraday
  cover-by-close pays locate but not overnight borrow; T+1 hold accrues next morning; past Thu = 3
  nights.
- **Adverse selection:** the regime mixture is **conditioned on gap magnitude** — larger gaps draw
  more mass on `hard`/`unavailable` and the fat-tail fee. Fee expressed in **$/share → % of entry
  price** so it scales correctly against per-event return.

### 2.4 Entry slippage — market-impact model

- **Adverse selection:** parabolic gappers have **wide spreads at 10:30** and thin displayed size;
  spread and impact **widen with gap magnitude** and shrink with ADV/float. Model:
  `slippage = half_spread(gap, ADV) + impact(order_size / ADV)`.
- Interacts with §2.2: under SSR the effective entry is worse still (passive, on a bounce).
- Calibration inputs deferred to validation (spread/ADV proxies from minute data, W5 stack).

### 2.5 LULD halt risk — limit gap-through against the short

- **Directional clarification (important):** for a fade **short**, the *adverse* halt is
  **limit-UP** (continuation) — a limit-up cascade runs the price *through* the stop. Limit-**down**
  halts are directionally favorable but can **trap the position** (cannot cover into a halt; reopen
  collar is unbounded). The simulator models **both**, but the tail-risk driver is the limit-up
  gap-through. (Where W5 says "two cycles reach the +30% stop on 10% names," that +30% is the
  short's stop being *run over* by consecutive limit-up cycles.)
- **W3/W5 constants:** LULD bands fixed **all day by previous close** — Tier-2 **>$3 → 10%**,
  **$0.75–3 → 20%** (40% after 15:35), **<$0.75 → min($0.15, 75%)**. **Gap-through bound:**
  +30% stop reachable via **two consecutive limit-up cycles on 10% bands**, or **one halt on 20%
  bands** (prev close $0.75–3). Reopen collar = `band × (1 + 0.05k)` per 5-min extension, imbalance
  side only, **unbounded in k**. No daily cap on cascades; post-reopen reversion is the norm,
  cascades are the tail (GME ~20 halts documented).
- **Adverse selection:** halt/gap-through probability **rises with gap magnitude and lower price
  tier** (tighter effective bands, more cascade-prone). The stop is **not a truncation** — see
  §2's loss model: loss-given-stop is a mixture, not a clean `-stop_distance`.
- **Realization:** per event, draw halt occurrence and, if halted, a gap-through exceedance
  (below). The +30% two-cycle scenario is the modeled worst reachable stop-run on 10% names.

### 2.6 Early-cover forced exit — recall timer

- **W3 constant:** on HTB/threshold names a **fresh locate is required to re-short after a cover**;
  Rule 204(b) penalty-box can flip a symbol to pre-borrow-only mid-day. IB HTB terms can carry a
  recall timer.
- **Adverse selection:** recall risk is **higher on the hardest-to-borrow (best) names** —
  correlated with §2.1. Model: if the event's borrow is on a recall timer (probability rising with
  borrow difficulty), force an exit at a **random time in [0.5h, 2h]** after entry; realize the
  return **at that point on the intraday path**, not at the close. This can cut off a working fade
  before it completes — a cost that lands hardest on the best borrows.

### Loss model (spans §2.5/§2.6, mandated by W4)

- **Disaster mixture (MANDATORY):** `P ≈ 0.1–0.5% per event` of a **−100% to −200%** outcome
  (halt + gap + forced buy-in; shorts unbounded). Policies that differ only in this cell **cannot
  be ranked by history** — the mixture must be an explicit modeled input, not an empirical
  frequency (the 559-event sample is too small to populate a 0.1–0.5% tail).
- **Stops do not truncate under jumps:** `loss_given_stop = mixture(prob q: fill near stop s;
  else s + GPD exceedance)`. Size on `E[L]` and the tail of `L`, not on stop distance.

---

## 3. Simulation architecture

**Event-bootstrap Monte Carlo** over the 559 confirmed HYP-093 events.

For each Monte Carlo trial, and each event *i*:

1. **Compute deterministic state** from the event's own tape: SSR boolean (§2.2), LULD band tier
   from prev close (§2.5), gap magnitude, prior-day return, float/ADV.
2. **Draw locate availability** (§2.1). If UNAVAILABLE → record flag, skip to next event (no
   return contributed).
3. **Draw entry realization:** SSR passive-fill outcome (§2.2) + slippage (§2.4) → effective entry
   price (or a missed entry).
4. **Draw borrow regime & fee** (§2.3), conditioned on gap.
5. **Draw path outcome:** halt / gap-through (§2.5) and/or forced early cover (§2.6); otherwise
   exit at the event's realized close. Apply the loss model (§2 mixture + GPD) for stop/halt cells.
6. **Compute net return** `r_i` = gross fade return at the realized exit − locate fee − borrow
   accrual (if held past close) − slippage.
7. **Accumulate** `r_i` into the per-event distribution and the trial's P&L path.

**Invariants (no look-ahead):** every friction draw for event *i* is conditioned **only** on
information available at or before the 10:30 ET entry (gap, prev close, prior-day return, SSR
state, band tier). No draw may use any bar after the entry origin, nor any cross-event future
information. (Mirrors the HYP-092/093 prereg discipline: entry origin shares no bar with the
read/friction inputs.)

**Correlation across same-day events (W4 §F4):** 2.3 events/day on average. In the **tail regime**
frictions co-move (ρ → 1: locate dries up market-wide, halts cluster) — the simulator must draw
day-level friction states, not independent per-event draws, so per-**day** worst-case heat is
represented. Normal regime uses the `1/(1+(k−1)ρ)` haircut; tail regime sets ρ = 1.

---

## 4. Output metrics

Over the full friction-adjusted distribution:

1. **Median net return** per event (compare to the +4.9% raw-edge median — the friction drag is
   `4.9% − median_net`).
2. **5th / 95th percentile** net return.
3. **% of events where frictions flip a WIN → LOSS** (gross `r > 0` but net `r < 0`).
4. **% of events UNAVAILABLE entirely** (locate gate failed — the edge exists but is untakeable).
5. **Expected annual P&L at constitutional sizing (1.25% position)** — mean net return per event ×
   events/year × 1.25% notional, reported with its 5/95 band. (1.25% = 0.75% risk / 60% worst-case
   per NN#3 / RISK_CONSTITUTION; sizing policy grid itself is W4's job, not W6's.)

Reported **as a distribution with bands**, never a point estimate. Secondary (feed W4): per-event
net-return array exported for the sizing-grid resampler.

---

## 5. Data inputs required

Per event, the simulator needs:

| Input | Use | Source (W5) |
|---|---|---|
| **Gap magnitude** (% up at 10:30 vs prev close) | conditions every friction curve | Polygon/Massive minute aggs; already in the HYP-092/093 candidate build |
| **Prior-day return / intraday lows (D-1, D)** | SSR boolean (deterministic) | same minute/daily tape |
| **Prev close price tier** | LULD band (>$3→10%, $0.75–3→20%, <$0.75→special) | daily bars |
| **Intraday price path (entry→close, 1-min)** | stop/halt detection, forced-cover mark, slippage | Polygon minute aggs |
| **Float / ADV** | borrow regime, slippage impact, recall probability | Norgate / DilutionTracker (or proxy) |
| **Halt/status records** | validate modeled halt rate | Databento status schema 2018+ (W5); pre-2018 tape-gap reconstruction |
| **Borrow fee/availability (if available)** | calibrate §2.1/§2.3 vs modeled curves | Ortex backfill (serious stack) OR conservative hard-coded HTB curve (shoestring) |

**Modeled-not-measured inputs (flag explicitly):** the `P(locate|gap)` curve and borrow-regime
mixture are **assumptions** on the shoestring stack (no clean retail borrow history pre-Ortex).
They are pinned in validation (§6) and disclosed as the load-bearing modeling choice.

---

## 6. Validation plan

Run the simulator on the **559 holdout events** and compare its output to the **W3 empirical /
primary-source friction estimates**:

- Modeled **median locate fee** vs W3's $0.01–0.05/sh typical (fat tail ≥$0.10–0.30).
- Modeled **SSR-active rate** vs the deterministic per-event SSR computation (this one is exact —
  the simulator's SSR draw must reproduce the tape boolean, tolerance ≈ 0%).
- Modeled **halt/gap-through rate** vs Databento status records where available (2018+).
- Modeled **unavailable rate** vs the HTB/threshold-name incidence in the event pool.

**Pass criterion:** simulator median friction estimate is **within 20%** of the W3 empirical
estimate on each measurable component → spec validated, build may proceed to policy consumption.
The SSR component is held to a tighter bar (near-exact) because it is deterministic. Components
that are modeled-not-measured (locate/borrow curves) are validated for *shape and monotonicity*
(adverse-selection direction correct) rather than absolute level, and the residual is disclosed.

---

## 7. What the simulator does NOT do

- **No parameter optimization.** No fitting friction parameters to maximize (or minimize) net
  return. Curves are set from W3 primary sources and pinned by §6, then frozen.
- **No regime selection / no event selection.** All 559 events run; the simulator does not pick
  favorable windows, pairs, or sub-samples.
- **No forward-looking inputs.** Every draw conditions only on ≤10:30-entry information (§3
  invariant). No future bars, no cross-event lookahead.
- **No sizing decision.** W6 outputs per-event net returns; the sizing policy grid (W4: F0–F4,
  RCK × per-day CVaR heat) is a *separate* consumer. W6 does not choose `f`.
- **No live capital, no order routing.** Simulation only.

---

## References (all primary-source, from W-series briefs)

- `data/research/yield_frontier/optimization/W3_plumbing_brief.md` — SSR/LULD/locate constants (SEC
  Rule 201 FAQ upd. 2026-06-26; Nasdaq LULD FAQ; Amdt 12; FINRA/SEC FAQ 4.4; IBKR docs; Rule 204).
- `data/research/yield_frontier/optimization/W4_sizing_brief.md` — loss model (disaster mixture,
  GPD stop-gap), sizing policy families, scoring metrics (Busseti-Ryu-Boyd 1603.06183 RCK).
- `data/research/yield_frontier/optimization/W5_data_procurement.md` — data stacks, halt/borrow
  sources, the borrow-history market gap.
- `data/research/gapper/report.md` — HYP-092 continuation-read null (the mechanized-checklist test
  that framed the fade base rate); HYP-093 fade edge is the confirmed short population studied here.
- `research/yield_frontier/OPTIMIZATION_PROGRAM.md` — W-series program map; W6 sizing-policy
  simulator seam (this friction sim feeds it).

**This spec is the only output of TICK-033/W6 at this stage. No simulator code until it is
committed.**
