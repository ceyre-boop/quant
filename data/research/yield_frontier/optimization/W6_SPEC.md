# W6 — Sizing Policy Simulator: Specification
**Date:** 2026-07-21 · **Status:** SPEC (pre-code) · **Ticket:** TICK-W6 (pending)
**Inputs:** W1–W5 briefs, HYP-097 results, HYP-093 holdout paths
**Output:** Recommended sizing policy for HYP-093 (The Undertow) + pass/fail on floor clearance
**Blocker for:** W7 (live shadow sizing protocol)

---

## Purpose

HYP-097 established that fixed-fractional sizing (F0) at constitutional risk budget (0.75%
notional / W*_tier) produces a holdout-calendar yield of **0.000166/day** — below the 0.0005/day
constitutional floor by 3×. W6 tests whether a different sizing policy can clear the floor,
and which one to recommend.

The question is **not** "can we show higher numbers." The question is: given the actual HYP-093
event paths, does the choice of sizing policy materially change realized compounded growth, and
if so, which policy family wins subject to hard drawdown and ruin constraints?

W6 is **spec-only** — no code is written until this document is approved. The build pass follows.

---

## Scope

W6 covers five policy families (F0–F4), each applied to the same event-path dataset. It
produces a policy ranking under a non-scalarized selection rule, a recommended policy, and a
verdict on floor clearance. It does NOT modify the sealed HYP-093 verdict or touch the
holdout significance test.

**In scope:**
- Policy families F0–F4 (defined below)
- Synthetic path generation from holdout event distribution + GPD tail extension
- Disaster mixture (halt-gap-buy-in) overlay
- Scoring on G, CVaR, MaxDD, CDaR, ruin probability
- Non-scalarized selection rule
- Append-only trials ledger entry

**Out of scope:**
- Options overlays (W2, closed)
- Data procurement (W5, closed)
- Live shadow sizing (W7, depends on W6 output)
- Catalyst-split subgroup analysis (TICK-034, separate hypothesis)
- Tier restriction (T10-only universe lift) — separate HYP, may be pre-registered after W6

---

## Inputs

### From HYP-097 (frozen)

```
W*_T10  = 0.6269   (worst-case loss fraction, Tier 10 events)
W*_T20  = 0.7975   (worst-case loss fraction, Tier 20 events)
T10_n   = 345      (holdout events, prev_close >= $3)
T20_n   = 194      (holdout events, $0.75 <= prev_close < $3)
Total_n = 539      (holdout events total)
```

Risk budget baseline: **b = 0.0075** (0.75% notional, per Art. 1 constitutional sizing).

The sensitivity run at b = 0.0082 raised yield to 0.000181/day — still well below floor.
Budget expansion is not a productive knob. The problem is policy, not budget.

### From HYP-093 holdout (frozen, sealed)

The event-path dataset is the sealed HYP-093 holdout sequence: 539 events, ordered by calendar
date, each with:
- `entry_px`, `stop_px`, `exit_px` (per sealed fill rule)
- `tier` (T10 or T20)
- `date`, `ticker`
- `ret_event` = (exit_px - entry_px) / entry_px (signed, negative = profit for short)

These paths are **read-only**. The simulator samples from them; it does not modify them.

### From W3 (plumbing constants, frozen)

- SSR adverse selection: correlated with best events (not modeled in F0; partially modeled in F4)
- Locate fill rate: 50% baseline (frozen in HYP-097)
- Overnight borrow: T+1 per W3 rules (not applicable — intraday strategy)
- LULD halt mechanics: embedded in W* via HYP-097; W6 does not re-derive

### From W4 (policy families, theory)

W6 implements the five families W4 identified. W4 recommendations carry forward:
- **F2 (RCK) is the recommended base** — theoretical derivation is strongest
- **F4 (day heat)** is the highest-priority add-on
- **F3 (Grossman-Zhou)** is an optional HWM governor, computationally heavier

---

## Synthetic Path Generator

### Purpose

539 holdout events is adequate for significance testing (HYP-093 p=0.031) but thin for
stress-testing sizing policies. The simulator generates **10,000 synthetic paths** of 539 events
each by bootstrapping with replacement from the holdout event distribution, then overlays the
disaster mixture.

### Bootstrap

```
For each synthetic path s in 1..10000:
    Sample 539 events with replacement from holdout event set
    Apply tier weights: 64% T10 (=345/539), 36% T20 (=194/539)
    Preserve calendar ordering within each bootstrap sample
```

Serial correlation in actual events (same ticker re-appearing, regime clustering) is NOT
modeled explicitly. The bootstrap breaks serial structure by resampling. This is conservative:
real serial wins are harder to compound than i.i.d. wins. If W6 shows floor clearance under
i.i.d. bootstrap, it may not survive real-world clustering. Flag this in the verdict.

### GPD Tail Extension

The empirical distribution has thin tails (n=539). Extreme loss events (beyond historical max)
are possible but unobserved. Extend the left tail using a **Generalized Pareto Distribution
(GPD)** fit to the worst-decile losses:

```
Fit GPD to losses exceeding 30th-percentile adverse quantile
Parameters: ξ (shape), β (scale) — estimated by MLE on the tail subsample
Replace events below the 2nd-percentile empirical loss with GPD samples in 5% of all paths
```

This gives W6 a stress-test that accounts for draws worse than any seen in the holdout. It
does not change the central estimate — only the tail behavior of the policy comparison.

### Disaster Mixture

The highest-loss scenario is not a large intraday move — it is a trading halt followed by a
gap reopening outside the stop, plus a forced buy-in. W3 documents this as an existential tail
event. Parameters from W4:

```
P_disaster   ~ Uniform(0.001, 0.005) per event  [0.1% to 0.5%]
L_disaster   ~ Uniform(-1.00, -2.00)             [100% to 200% of position notional]
```

The range reflects model uncertainty. A 200% loss is possible when: halt + reopen gap-through
below stop (50-80% position loss) + share-recall/buy-in charge (additional 50-120% of position
cost). This is rare but not theoretical — it has happened to short sellers in halted micro-caps.

**Implementation:**

```python
# For each event in each synthetic path:
if random() < p_disaster:
    event.ret_event = uniform(-2.0, -1.0) / w_star_tier  # scaled by tier worst-case
    event.is_disaster = True
```

Policies must survive disaster events. Any policy that shows ruin probability > 1% across the
10,000 synthetic paths is **excluded** regardless of its growth metrics.

---

## Policy Families

### F0 — Fixed Fractional (Control)

The baseline established by HYP-097. Size is constant as a fraction of current account value,
calibrated by worst-case loss per tier:

```
f_T10(t) = b / W*_T10  =  0.0075 / 0.6269  =  0.01197 (1.197% notional)
f_T20(t) = b / W*_T20  =  0.0075 / 0.7975  =  0.00940 (0.940% notional)
```

Size does not change based on account trajectory, recent performance, or volatility. This is
the reference case.

**HYP-097 result at F0:** yield = 0.000166/day (NOT_CLEARED). Reproduced by the simulator to
validate the pipeline before running F1–F4.

### F1 — Fractional Kelly + Stressed No-Ruin Floor

Full Kelly on a short with binary outcome (win/lose at empirical rates) overfits the tails.
Fractional Kelly at κ = 0.25 (quarter-Kelly) is the standard conservative starting point.
The stressed floor adds a ruin constraint evaluated by the disaster mixture:

```
Full Kelly:   f* = (p·b_win - (1-p)·b_loss) / (b_win · b_loss)
  where p  = empirical win rate (64.0% from HYP-093 holdout)
        b_win  = avg profit / position_size (on winners)
        b_loss = avg loss / position_size (on losers)

Fractional:   f_1(t) = min( κ · f*, f_ruin_floor(t) )
  where κ = 0.25 (fixed)
        f_ruin_floor(t) = max position size s.t. P(ruin over next N_events) < 0.005

N_events = 50  (approximately one quarter of annualized signal rate)
```

F1 is computationally simple and directly interpretable. It serves as a sanity check on whether
any Kelly-guided sizing beats F0 before committing to the more complex F2.

### F2 — Risk-Constrained Kelly (Recommended Base)

Reference: Busseti, Ryu, Boyd (2016) arXiv:1603.06183 — "Risk-Constrained Kelly Gambling."

RCK solves:

```
maximize    E[ln(1 + f · R)]
subject to  VaR_α(1 + f · R) >= (1 - ε_dd)    ∀ paths in the planning horizon
```

Where R is the per-event return random variable (estimated from holdout empirical distribution
+ GPD tail), α is the confidence level (95% in W4 specification), and ε_dd is the maximum
acceptable drawdown per rolling window.

The key difference from F1: RCK explicitly maximizes **log-wealth growth** (geometric mean)
while enforcing a whole-path drawdown certificate. F1 maximizes Kelly-Kelly approximation.
RCK maximizes the actual objective.

**Implementation parameters:**

```
alpha       = 0.95     (CVaR confidence level — tail 5% of paths)
epsilon_dd  = 0.15     (max 15% drawdown on rolling 50-event window)
planning_N  = 50       (rolling window for constraint evaluation)
solver      = CVXPY (convex optimization)
```

The optimization runs once per synthetic path at the start (offline sizing), not per-event. A
future W7 variant could re-optimize every 50 events on live shadow results — but that is W7,
not W6.

**Per-tier sizing:** RCK is run separately for T10 and T20 event sub-distributions. The
resulting f_T10* and f_T20* replace the fixed-fractional sizes.

### F3 — Drawdown-Modulated (Grossman-Zhou Governor)

Reference: Grossman and Zhou (1993) — "Optimal Investment Strategies for Controlling Drawdowns."

F3 is a governor that scales any base policy (F0 or F2) by current drawdown from high-water
mark:

```
f_F3(t) = f_base(t) · (1 - DD(t) / DD_max)^γ

where DD(t)   = (HWM(t) - W(t)) / HWM(t)   [current drawdown fraction]
      DD_max  = 0.15                          [hard drawdown ceiling]
      γ       = 1.0                           [scaling exponent; 1.0 = linear]
```

When account is at HWM (DD=0), F3 is identical to the base policy. As drawdown approaches
DD_max, F3 scales toward zero — preventing the policy from "reaching for recovery" by taking
larger sizes when the account is underwater.

F3 is tested as a governor on top of F2: **F2+F3 combined.** The question is whether the
drawdown protection materially reduces ruin probability relative to F2 alone, and at what cost
to growth rate.

### F4 — Per-Day CVaR Heat Budget + Correlation Penalty

F4 operates at a different granularity from F1–F3. Rather than sizing per-event, it caps the
total risk exposure on any given calendar day by measuring **expected tail loss across all
open/entering positions that day**:

```
Daily heat:   H(t) = Σ_i f_i(t) · CVaR_95(R_i)  <= h_max = 0.005  (0.5% per day)

Per-event size under heat constraint:
  f_i(t) = max feasible f s.t. H(t) + f · CVaR_95(R_i) <= h_max

Correlation penalty:
  If two events fire on the same day in tickers with |corr| > 0.4 (trailing 20d):
    Apply sizing haircut of 0.7 to the second event
```

The HYP-093 strategy is intraday with at most a few simultaneous positions per day (median is
0.5 events/day). The correlation penalty rarely fires. But on days where 3+ events appear
(cluster days, often when a sector runs), the heat budget prevents over-concentration.

F4 is tested both standalone and as an overlay on F2: **F2+F4 combined** (the W4 recommended
configuration).

---

## Scoring Metrics

W4 specified the complete metric set. Definitions below are binding for the build pass:

### G — Log-Growth Rate

```
G = (1 / T_calendar) · Σ_t ln(1 + f(t) · R(t))
```

Where T_calendar is the number of trading days in the synthetic path. G is the primary
optimization target. Reported as G/day and G/year (×252).

### CVaR_95 and CVaR_99

Conditional Value at Risk at 95th and 99th percentile, computed over the per-event P&L
distribution across all 10,000 synthetic paths:

```
CVaR_α = E[ loss | loss > VaR_α ]
```

Reported as a fraction of starting account value. CVaR_99 is the primary tail-loss metric
for policy exclusion (must be < 10% of account per single event).

### MaxDD Distribution

Histogram of maximum drawdown across all 10,000 paths. Reported as:
- p50 MaxDD (median worst drawdown)
- p95 MaxDD (tail worst drawdown)
- p99 MaxDD (extreme worst drawdown)

Constraint: p95 MaxDD < 0.25 (25% of account). Any policy where p95 MaxDD >= 0.25 is excluded.

### CDaR(0.9) — Conditional Drawdown at Risk

```
CDaR(α) = E[ MaxDD_path | MaxDD_path > CDaR_VaR_α ]
```

At α=0.90: expected worst drawdown among the 10% most-drawn-down paths. Measures "how bad are
the bad outcomes" beyond the median.

### Triple Penance Ratio

Introduced in W4 as a practical summarizer of drawdown pain relative to growth:

```
TPR = E[MaxDD] / (G/day)
```

Lower is better. Measures how many expected-recovery-days of pain the policy imposes per unit
of daily growth. A policy with G=0.0003/day and E[MaxDD]=0.15 has TPR = 500 days of expected
penance. That is a lot. The TPR makes this intuitive.

### Ruin Probability

```
P_ruin = (number of paths where W drops below 0.5 × W_0) / 10000
```

The ruin threshold is 50% of starting capital, not zero. A 50% drawdown in this strategy class
is operationally terminal (prop firm rules, psychological fragility, forced liquidation). Any
policy where P_ruin > 0.01 (1% of paths) is **excluded** from ranking.

---

## Non-Scalarized Selection Rule

W4 is explicit: **do not scalarize.** A weighted sum of metrics papers over the trade-offs that
actually matter. The selection rule is lexicographic:

**Step 1 — Hard exclusion.** Eliminate any policy where:
- P_ruin > 0.01, OR
- p95 MaxDD >= 0.25, OR
- CVaR_99 > 0.10

**Step 2 — Floor check.** Among surviving policies, compute mean G/day across all 10,000 paths.
Record which policies produce G/day equivalent to:
- `0.0005/day` (constitutional yield floor) when compounded over 252 days from $10,000 base
- Note: G/day (log-wealth) ≠ arithmetic yield/day. The bridge is G ≈ μ - σ²/2. W6 must
  compute both and be explicit about which is which.

**Step 3 — Rank survivors.** Among policies that pass Step 1, rank by:
1. G (primary — maximize)
2. CDaR(0.9) (secondary — minimize)
3. TPR (tertiary — minimize)

**Step 4 — Select.** The top-ranked survivor is the recommended policy. If two policies are
within 5% on G and the lower-G policy has meaningfully better tail (CDaR < 0.8× the leader),
prefer the tail-safer policy. Document the reasoning.

The final selection must be **one policy**, not a committee. The build pass must produce a
single recommended (f_T10, f_T20) sizing rule for W7.

---

## Verdict Structure

W6 produces a verdict entry appended to `data/research/preregister/verdicts_optimization.jsonl`.
The verdict schema:

```json
{
  "id": "W6",
  "date": "YYYY-MM-DD",
  "hypothesis": "Optimal sizing policy for HYP-093 (The Undertow)",
  "input_hash": "<hash of holdout event paths and HYP-097 params>",
  "policies_tested": ["F0", "F1", "F2", "F2+F3", "F2+F4", "F2+F3+F4"],
  "floor": 0.0005,
  "recommended_policy": "<policy_id>",
  "recommended_f_T10": <float>,
  "recommended_f_T20": <float>,
  "G_day_mean": <float>,
  "G_day_p10": <float>,
  "G_day_p50": <float>,
  "G_day_p90": <float>,
  "CVaR_99": <float>,
  "p95_MaxDD": <float>,
  "CDaR_90": <float>,
  "P_ruin": <float>,
  "TPR": <float>,
  "floor_cleared": <true|false>,
  "n_bootstrap_paths": 10000,
  "disaster_p_range": [0.001, 0.005],
  "disaster_L_range": [-1.0, -2.0],
  "notes": "<any deviations from spec, flags, or caveats>"
}
```

This entry is **append-only and sealed** once written. No revision to the verdict after it is
committed to the ledger. Corrections go in a separate note (per gauntlet/CORRECTION convention).

---

## Pass/Fail Definition

The pre-registered verdict rule for W6:

```
FLOOR_CLEARED  iff  mean G/day (across 10,000 paths, best surviving policy) >= 0.0005/day
                    AND P_ruin <= 0.01
                    AND p95 MaxDD < 0.25

NOT_CLEARED    otherwise
```

If NOT_CLEARED: record the **highest achievable G/day** under the surviving policy and the
**gap to floor** (floor - G_best). This is the number that determines whether the three additive
remaining paths (tier restriction TICK-034, catalyst split, TICK-034) are plausibly sufficient
to close the gap.

The floor threshold (0.0005/day) is the same as HYP-097. It was set constitutionally and is
not negotiable in W6.

---

## Failure Modes and Safeguards

### The Scalarization Trap

Do not combine G, CDaR, and TPR into a single weighted score. The weights are arbitrary and
different researchers will choose different weights to get the answer they want. Use the
lexicographic rule above.

### The Overfit-to-Bootstrap Trap

Optimizing policy parameters (κ in F1, α and ε_dd in F2, DD_max and γ in F3, h_max in F4)
against the bootstrap sample and then evaluating against the same bootstrap constitutes
in-sample optimization. The build pass must:
- Fix all policy parameters before bootstrapping
- Do not search over parameter values using bootstrap performance as the criterion
- If parameter sensitivity is needed, run a separate robustness check with ±20% perturbations
  of each parameter and verify the ranking is stable

### The Disaster-Frequency Trap

The disaster mixture uses a wide range (0.1% to 0.5%/event). The simulator must not pick the
midpoint — it must test the full range. Run the simulation at p_disaster ∈ {0.001, 0.002, 0.005}
separately and report whether the policy ranking changes. If it does, the recommended policy
must be the one that wins at p=0.005 (pessimistic scenario), not p=0.001.

### The "G Looks Higher" Trap

If the simulator shows G/day > 0.0005 for any policy, verify by computing the **arithmetic
yield path** directly (not via the log approximation) and checking that the implied account
trajectory at that G rate actually crosses the floor. G is log-additive; arithmetic yield is
not. Be explicit about which one is being compared to the 0.0005/day constitutional floor.

The correct comparison: the HYP-097 verdict uses arithmetic yield (Σ event_return × position_size
/ calendar_days). W6 must match this definition for the floor comparison, even if G (log-wealth)
is higher.

---

## Build Pass Instructions (for the coding agent)

These instructions are for the agent that implements W6, not for the spec itself. Included here
so the build pass has its spec co-located.

1. **Read this spec in full before writing any code.** The spec is the authority. If anything
   is ambiguous, stop and ask — do not improvise.

2. **Validate F0 first.** The first output of the simulator must be the F0 (fixed-fractional)
   arithmetic yield, which should match HYP-097's 0.000166/day within 5% (Monte Carlo noise).
   If it doesn't, the simulator has a bug. Fix it before running F1–F4.

3. **No holdout data re-use.** The event paths are sealed. The simulator reads them but does
   not filter, sort, or subset them in any way that wasn't specified in HYP-093.

4. **Commit the spec first.** Push this W6_SPEC.md before writing any simulator code, so the
   commit history shows spec-before-build. Tag: `[RESEARCH] Add W6 sizing policy simulator spec`.

5. **Output location:** Results go to `data/research/yield_frontier/optimization/W6_results/`.
   The verdict entry appends to `data/research/preregister/verdicts_optimization.jsonl`. Do not
   write results anywhere else.

6. **Dependencies:** numpy, scipy (GPD fitting via scipy.stats.genpareto), cvxpy (F2 RCK),
   pandas. All are available in the repo environment.

---

## Open Questions (non-blocking for the spec; must be answered in the build report)

1. **Serial correlation structure.** The bootstrap destroys the serial clustering present in
   real events (sector runs, regime periods). If the build agent finds a way to preserve block
   structure (e.g., block bootstrap with block_size=5 trading days), that is preferred over
   pure i.i.d. resampling — but must not introduce look-ahead.

2. **F2 solver stability.** CVXPY with a non-parametric empirical distribution may have
   convergence issues on pathological bootstrap samples. The build agent should catch solver
   failures and fall back to F1 for that path rather than crashing.

3. **Tier mixing.** HYP-097 sizes T10 and T20 separately. F2+F4 may recommend a single sizing
   rule that implicitly mixes tiers via the day-heat constraint. The build agent should clarify
   whether the recommended output is (f_T10, f_T20) or a single f with tier-based W* correction.

---

## Cross-References

| Document | Relationship |
|---|---|
| `gauntlet/verdicts.json` | HYP-093 sealed holdout — event paths for bootstrap |
| `data/research/preregister/HYP-097_gapthrough_sizing.json` | F0 baseline; W* derivation |
| `data/research/preregister/verdicts_optimization.jsonl` | W6 verdict destination (append-only) |
| `optimization/W1_mechanism_brief.md` | Catalyst-split as highest-value next test post-W6 |
| `optimization/W3_plumbing_brief.md` | SSR/LULD/halt mechanics; locate fill assumptions |
| `optimization/W4_sizing_brief.md` | Policy family theory; metric definitions; F2 reference |
| `optimization/W5_data_procurement.md` | Data stack (no change for W6 — uses existing holdout) |
| `optimization/W7_live_shadow.md` | Receives recommended policy from W6; W7 not yet written |
| `gauntlet/CORRECTION_2026-07-21.md` | Decimal correction for gauntlet/report.md line 19 |

---

*Alta Investments · Yield Frontier Optimization · W6 Spec v1.0*
*Spec authors: Colin + Claude · Build agent: Claude Code (sovereign-v2 branch)*
*"Spec-first. The build can be cheap once the thinking is done."*
