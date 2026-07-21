# The Pattern Framework — A Mathematical Theory of Tradeable Structure
### Alta Investments · Research Document · 2026-07-21

> *"The market is not random. It is a superposition of patterns at different frequencies.
> Our job is not to predict the future. It is to identify where energy concentrates,
> measure how reliably it resolves, and extract a fraction of each resolution."*

---

## I. The Core Claim

Every tradeable pattern is a statement of conditional probability:

```
P( outcome ∈ {X, Y, Z} | pattern P observed ) = p
```

When `p` is high enough, and the pattern occurs often enough, and the
cost of being wrong is bounded — you have an edge.

This document builds the mathematical scaffolding to find, score, and
rank every pattern in that form.

---

## II. Price as Energy and Frequency

Before patterns, we need a model of what price *is*.

Price `P(t)` is not a random walk. It is a **superposition of waves**
at different frequencies, each carrying a different amount of energy:

```
P(t) = P₀ + Σₙ Aₙ · sin(ωₙt + φₙ) + ε(t)
```

Where:
- `Aₙ` = amplitude of the nth cycle (energy in that frequency band)
- `ωₙ` = angular frequency (how fast it oscillates — seconds, minutes, days, months)
- `φₙ` = phase (where in the cycle we currently are)
- `ε(t)` = true noise (the part with no structure)

A **candlestick pattern** is a local fingerprint of this wave structure —
a brief window where amplitude, direction, and phase interact in a
recognizable way.

The reason patterns repeat is that the *same forces* (human psychology,
institutional flow, liquidity mechanics) generate the *same waves* at
similar frequencies. Energy concentrates and releases in similar ways
because the agents causing it have not changed.

### The Implication

If you observe a candlestick configuration that reflects a known phase
in a known frequency band, you can make a probabilistic statement about
what comes next — not because of mysticism, but because the underlying
energy dynamics are recurring.

The pattern is a **phase detector**. The trade is a **phase bet**.

---

## III. Formal Definition of a Pattern

Let `t` be the time of bar close. Define the **pattern window** as the
preceding `k` bars:

```
W(t, k) = { bar(t-k+1), bar(t-k+2), ..., bar(t) }
```

Each bar has four measurements: Open, High, Low, Close — and
optionally Volume. A pattern `P` is a **predicate** on `W`:

```
P : W(t,k) → {TRUE, FALSE}
```

It fires TRUE when the configuration matches the template.

**Example (Bullish Engulfing):**
```
P_engulf(t) = TRUE iff:
  close(t-1) < open(t-1)          [prior bar is red]
  AND open(t) < close(t-1)        [today opens below prior close]
  AND close(t) > open(t-1)        [today closes above prior open]
  AND volume(t) > volume(t-1)     [volume confirms]
```

Every pattern in existence can be expressed this way.
The form is `X does Y when Z`, operationalized as a predicate.

---

## IV. The Outcome Vector

When pattern `P` fires at time `t`, we measure what happens next.
Define the **outcome vector** `O`:

```
O(t) = (r_1h, r_4h, r_1d, r_max, r_min, t_max, dd_max)
```

Where:
- `r_1h`   = return 1 hour after signal
- `r_4h`   = return 4 hours after signal
- `r_1d`   = return 1 day after signal
- `r_max`  = maximum favorable excursion (MFE) before a 50% pullback
- `r_min`  = maximum adverse excursion (MAE) before recovery
- `t_max`  = time (in bars) to reach MFE
- `dd_max` = maximum drawdown during the hold period

For any pattern `P`, collected across all `n` occurrences, we get a
**distribution** over `O`:

```
{ O(t₁), O(t₂), ..., O(tₙ) }
```

The **expected outcome** is:
```
E[O | P] = (1/n) · Σᵢ O(tᵢ)
```

---

## V. The Perfect Trade

Define the **Perfect Trade** as an ideal outcome vector `O*`:

```
O* = (r_max_possible, r_max_possible, r_max_possible, +∞, 0, 1, 0)
```

In plain terms: the price moves immediately and directly in your
direction with zero drawdown, infinite MFE, and achieves maximum return
in the first bar.

No real trade achieves `O*`. But we can measure **how close** any
pattern's expected outcome is to the ideal.

### The Perfect Trade Score (PTS)

Define the **Perfect Trade Score** as the cosine similarity between
the observed mean outcome and the ideal:

```
PTS(P) = cos( E[O|P], O* ) = ( E[O|P] · O* ) / ( ||E[O|P]|| · ||O*|| )
```

A PTS of 1.0 = perfect alignment with the ideal.
A PTS of 0.0 = no relationship.
A PTS < 0 = the pattern predicts the opposite of what you want.

For practical use, normalize the components of `O` to comparable
scales (z-score each dimension) and weight by importance:

```
PTS(P) = w₁·P(r_1d > 0) + w₂·E[r_max] - w₃·E[r_min] - w₄·E[dd_max]
```

Suggested weights for intraday momentum: `w = (0.3, 0.4, 0.2, 0.1)`

---

## VI. The Three Conditions for a Tradeable Pattern

A pattern `P` is **tradeable** if and only if all three hold:

### Condition 1 — Directional Reliability (the 80% rule)
```
P( r_1d > threshold | P fires ) ≥ 0.80
```
At least 80% of occurrences move in the expected direction by a
meaningful amount (threshold = 1× typical bid-ask spread minimum).

### Condition 2 — Favorable Asymmetry
```
E[r_max | P] / E[|r_min| | P] ≥ 1.5
```
The average best-case upside is at least 1.5× the average worst-case
downside. Without this, even an 80% win rate can have negative
expectancy if the losers are large enough.

### Condition 3 — Statistical Significance
```
p_permutation < 0.05,  n ≥ 30
```
The win rate and asymmetry must not be explainable by chance alone.
We test this by randomly permuting trade labels and checking whether
the observed score appears in the top 5% of random scores.

**The formula for expectancy (E):**
```
E = (WR × avg_win) - ((1 - WR) × avg_loss)
```
Where WR = win rate, avg_win = E[r | win], avg_loss = E[|r| | loss].

A pattern is tradeable iff E > cost_of_trade (spread + slippage + borrow).

---

## VII. Pattern Frequency and the Opportunity Rate

A pattern that fires once a year with 95% reliability is less useful
than one that fires daily with 82% reliability, if the expected returns
per event are comparable.

Define **Opportunity Rate** `λ`:
```
λ(P) = (number of pattern occurrences) / (total trading days observed)
```

Define **Yield** as the total extractable return per calendar day:
```
Yield(P) = λ(P) × E[r_per_event | P]
```

This is the metric on the Yield Board. It answers: *if I traded this
pattern every time it appeared, how much would I make per day?*

The ranking function is:
```
Score(P) = PTS(P) × Yield(P) × (1 - Ruin_Probability(P))
```

Where:
```
Ruin_Probability(P) = P( max_consecutive_loss > Kelly_fraction × account )
```

---

## VIII. The X Does Y When Z Framework

Every pattern discovered by this system can be expressed as:

```
X  [the instrument / market condition]
does Y  [the directional outcome and magnitude]
when Z = %  [the pattern fires with probability %]
```

**Concrete example from confirmed research:**

```
X  = any US equity that has gapped ≥100% from previous close by 10:30 ET
does Y  = fade (decline) by median 12.5% from the 10:30 high by market close
when Z  = 65.9% of the time (n=234, permutation p < 0.001, HYP-093 confirmed)
```

This is The Undertow. It satisfies all three conditions:
- WR = 65.9% (≥ the relevant threshold for a short with stops)
- Asymmetry = tail 4.4:1 (rare runners cost less than frequent fades pay)
- p < 0.001, n = 234

**The three outcomes {X, Y, Z} in the abstract:**

For any pattern, define the **three canonical outcomes** to measure:

```
X = primary move    (did the price go where the pattern predicted?)
Y = timing          (did it get there within the hold window?)
Z = cost-adjusted   (did it get there with drawdown < the stop level?)
```

A pattern scores on all three simultaneously. The PTS weights these:
- X alone = directional accuracy
- X + Y = directional accuracy × timing efficiency
- X + Y + Z = tradeable win (the only one that matters for actual P&L)

---

## IX. The Search Algorithm

To find patterns systematically:

### Step 1 — Define the Pattern Space

Let `K` = max lookback bars (suggest K ≤ 5 for robustness).
The theoretical space of all patterns on `K` bars is vast (~2^(4K)
binary features). Restrict to **meaningful subsets**:

```
Pattern families:
F1 = single-bar configurations (doji, hammer, engulfing, etc.)
F2 = two-bar configurations (engulf, harami, gap patterns)
F3 = three-bar configurations (morning star, three soldiers, etc.)
F4 = volume-confirmed variants of F1-F3
F5 = threshold-triggered (price ≥ N% move, volume ≥ M× average)
```

The gapper research lives in F5. ICT patterns live in F2-F3.
The highest-yield confirmed edges tend to come from F5 (threshold
triggers) because the threshold itself acts as a **natural
pre-filter** — only the most energetically significant moves qualify.

### Step 2 — Scan for Occurrences

For each pattern `P` in the search space, scan every bar in the
historical universe and record `(timestamp, instrument, outcome_vector)`.

Minimum sample: **n ≥ 30** occurrences before any scoring.

### Step 3 — Compute the Score

For each pattern with n ≥ 30:
```
1. Compute WR = P(r_1d > 0)
2. Compute E[r_max], E[|r_min|]
3. Compute Expectancy E
4. Run permutation test (1000 shuffles, record p-value)
5. Compute Yield = λ × E
6. Compute PTS
7. Compute Score = PTS × Yield × (1 - Ruin_P)
```

### Step 4 — Rank and Select

Sort all patterns by Score descending. The top of the list is the
yield board. Select ≤ 3 patterns for pre-registration.

**Why ≤ 3?** Because each pattern consumes one holdout test. Holdout
data is finite and non-renewable. Spending it on patterns ranked 4-20
before patterns 1-3 are confirmed is a statistical waste.

### Step 5 — Pre-Register and Test

Before touching holdout data, lock:
- Exact pattern definition (predicate)
- Entry trigger (bar, time, price)
- Exit rule (target, stop, time-based)
- Sample universe (which instruments qualify)
- Expected WR and asymmetry (from the mining phase)
- Pass/fail threshold for the holdout

Then run the holdout once. The result is permanent.

---

## X. Robustness Criteria

A pattern is **robust** if:

### 1. It survives parameter perturbation
```
Score(P_perturbed) / Score(P_exact) ≥ 0.70
```
Perturb the threshold by ±10%. If the score collapses, the pattern
is fragile — it depends on having found the exact magic number.

### 2. It works across instruments
```
WR(P) ≥ threshold  across ≥ 2 independent instruments
```
A pattern that only works on one ticker is probably overfitted to
that ticker's specific history.

### 3. It works across time periods
Walk-forward test: train on first 60%, validate on next 20%, test
on final 20%. The WR and asymmetry should not degrade by more than:
```
WR_holdout ≥ WR_insample - 10pp
Asymmetry_holdout ≥ Asymmetry_insample × 0.70
```

### 4. The Deflated Sharpe Ratio (DSR) clears the bar
```
DSR = SR × sqrt((1 - (γ₃/6)·SR/√T + ((γ₄-3)/24)·SR²/T))
      × Φ⁻¹(1 - (1/number_of_trials))
```
Where `γ₃` = skewness, `γ₄` = kurtosis, `T` = bars in sample.
This penalizes for the number of patterns tested. A pattern tested
among 809 configurations carries a much heavier burden than one
tested in isolation.

**The practical rule:** if DSR < 0.5 after family-wise correction,
the edge is likely data-mined noise, not signal.

---

## XI. The Simplicity Principle

The best patterns are the simplest ones.

**Why:** A pattern with 7 conditions requires all 7 to align. The
probability of all 7 aligning by chance is (1/2)^7 = 0.78%. You
will see very few occurrences. A pattern with 2 conditions appears
25× more often, giving you 25× more data to validate.

The mathematical trade-off is:
```
Specificity × Frequency = constant (roughly)
```

More conditions → higher WR per occurrence but fewer occurrences → lower Yield.
Fewer conditions → lower WR per occurrence but more occurrences → higher Yield.

**The optimal zone** for intraday momentum patterns is typically:
- 2–3 conditions
- WR 60–75% (higher is suspicious; suggests overfitting)
- n ≥ 100 over the sample period

An 80% win rate on 30 occurrences is worth less than a 65% win rate
on 300 occurrences, because the confidence interval on 30 samples
spans [63%, 91%] — the true rate could be below 65% or as high as
92%. You don't actually know. On 300 samples the interval is
[59%, 71%] — you know something real.

---

## XII. The Complete Decision Rule

Given all of the above, the decision to trade a pattern reduces to:

```
TRADE iff:
  (1) WR ≥ threshold_WR(asset_class)
  (2) E[r_max] / E[|r_min|] ≥ 1.5
  (3) Expectancy E > total_cost_per_event
  (4) p_permutation < 0.05
  (5) n_occurrences ≥ 30
  (6) DSR(pattern, n_trials_in_family) > 0.5
  (7) Score survives ±10% perturbation of key threshold
  (8) Holdout result passes the pre-registered pass/fail criterion
```

All eight conditions must be met. Any one failure = pattern is NOT
tradeable, regardless of how compelling the other numbers look.

The entry is:
```
Entry_price = close(t) + ε    [at pattern confirmation bar close]
Stop = close(t) - MAE_p10     [10th percentile historical MAE]
Target = close(t) + MFE_p50   [50th percentile historical MFE]
Size = Kelly_fraction × Account / (Entry - Stop)
```

Where `Kelly_fraction` is capped at 0.25 (quarter-Kelly) for
new/unscaled patterns and raised toward 0.50 only after 100+
live-shadow events confirm the in-sample statistics.

---

## XIII. Summary Table

| Concept | Mathematical Form | Plain English |
|---|---|---|
| Pattern | P(W(t,k)) → {T,F} | A predicate on recent candles |
| Outcome | O = (r_1h, r_4h, r_1d, MFE, MAE, t_max, dd) | What happens after |
| Perfect trade | O* = (max, max, max, ∞, 0, 1, 0) | Best possible outcome |
| PTS | cos(E[O\|P], O*) | How close to perfect |
| Win rate | P(r_1d > 0 \| P) | Directional accuracy |
| Asymmetry | E[r_max] / E[\|r_min\|] | Reward vs risk |
| Expectancy | WR·avg_win - (1-WR)·avg_loss | Net per event |
| Yield | λ × Expectancy | Return per calendar day |
| Score | PTS × Yield × (1 - Ruin_P) | Total ranking metric |
| Robustness | DSR > 0.5 across perturbations | It's not an accident |

---

## XIV. What This Means for Alta

The system already implements most of this implicitly. The yield board
(`data/research/yield_frontier/yield_board.md`) is Score-ranked. The
5-step research protocol (mine → pre-reg → holdout → gauntlet → live
shadow) implements Steps 4-5 of the search algorithm above. The
permutation test is the significance gate.

What this document adds:

1. **The PTS (Perfect Trade Score)** — a single number that ranks
   patterns not just by win rate but by how *well-shaped* the wins
   are (immediate, large MFE, small MAE, no time decay).

2. **The energy/frequency framing** — every pattern is a phase
   detector in a specific frequency band. When you see the same
   pattern appearing reliably, you're seeing the same energy dynamic
   repeating at the same frequency. This is why patterns work and
   why they stop working (regime change = frequency shift).

3. **The simplicity principle as math** — fewer conditions = more
   occurrences = tighter confidence intervals = faster confirmation.
   The best patterns are the bluntest instruments.

4. **The X does Y when Z format** — the universal template for
   expressing any confirmed pattern in auditable form.

---

*Alta Investments — Research Framework · Pattern Framework v1.0*
*Filed: 2026-07-21 · Status: FRAMEWORK (not evidence)*
*Next: apply PTS scoring to the existing yield board rows*
