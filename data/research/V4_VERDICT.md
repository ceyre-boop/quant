# V4 VERDICT — NOT_SIGNIFICANT
**Alta Investments | 2026-07-30 | Prereg: `data/research/preregister/V4_PREREGISTRATION.md`**

## Verdict

**V4 does not supersede the incumbent. No config is shipped.**
Additionally: **the V3 report issued earlier today contained a 10× annualization
error and a selection-bias artifact. Both are corrected below.**

---

## 1. The annualization error (correct this first)

`give-me-the-numbers` resamples `n = len(returns)` events per path and labels the
result an *annual* return. I fed it all 411 trades — a full decade — so it priced
ten years of trading as one year.

| Scenario | Reported earlier | **Correct (41 trades/yr)** |
|---|---|---|
| v015 baseline | +44.8% | **+3.5%** |
| Pessimistic V3 | +66.7% | **+5.0%** |
| Breakeven V3 | +86.6% | **+6.2%** |
| Conservative V3 | +115.3% | **+7.8%** |

Median dollars per year, Kovner 0.25% risk, 20,000 block-bootstrap year-paths:

| Account | v015 base | Pess V3 | Beven V3 | Cons V3 |
|---|---|---|---|---|
| $10,000 | $354 | $501 | $618 | $778 |
| $25,000 | $884 | $1,253 | $1,545 | $1,945 |
| $50,000 | $1,769 | $2,506 | $3,091 | $3,889 |
| $100,000 | $3,538 | $5,011 | $6,181 | $7,778 |

P(profitable year): 93.4% baseline → 100% conservative. p5 baseline year is −0.3%.
These are believable numbers for a 4–14 trade/year edge at quarter-percent risk.
The 44–115% figures were never real.

---

## 2. V3's proposed fix is refuted by real re-simulation

V3's mechanism — *the trailing stop fires mid-drift and costs −63.4R* — is
**CONFIRMED** (HYP-059) and stands.

V3's proposed **fix** — replace trailing-stop exits with the historical mean
time-exit return — does not survive contact with a real price-path re-simulation.
RQ-REST-013's sealed arms:

| Arm | Sharpe | rescaled sumR |
|---|---|---|
| trail_wide_2.0 | 0.126 | 167.0 |
| trail_wide_1.5 | 0.119 | 138.5 |
| **trail_immediate_1.25 (incumbent)** | **0.098** | **104.2** |
| time5 (≈ V3's fix) | 0.097 | 85.3 |
| time8 | 0.089 | 95.3 |

**Pure time exits score *below* the incumbent.** V3's +312R came from crediting
71 stopped-out trades with the average time-exit outcome — but those trades were
stopped *because* price moved against them. Assuming they'd earn the average
assumes away the reason they lost. Classic selection bias.

---

## 3. What V4 tested, and what happened

Preregistered family: delay trail activation and/or widen it. 27 configs,
IS = 2015–2022, OOS = 2023–2024 touched once.

IS selection, after applying the preregistered drawdown constraint (only 2 of 27
configs passed): **k_stop 2.0 · k_trail 1.25 · delay 10 · max_hold 21**.

OOS results looked excellent:

| Segment | Incumbent | V4 | Δ meanR |
|---|---|---|---|
| **OOS 2024 (1yr, n=43)** | +0.044 R, Sharpe 0.23 | +0.634 R, Sharpe 1.75 | **+0.590** |
| OOS 2023–24 (n=92) | +0.166 R, Sharpe 0.82 | +0.626 R, Sharpe 1.72 | +0.460 |
| IS 2015–22 (n=319) | +0.130 R, Sharpe 0.48 | +0.304 R, Sharpe 0.71 | +0.174 |

Against the five preregistered success criteria:

| # | Criterion | Result |
|---|---|---|
| 1 | OOS meanR > incumbent | ✅ |
| 2 | OOS Sharpe > incumbent | ✅ |
| 3 | Permutation p < Šidák (0.0019, 27 configs) | ❌ **p = 0.0175** |
| 4 | OOS maxDD ≤ 1.25× incumbent | ❌ **2.03×** (15.8R vs 7.8R) |
| 5 | IS→OOS Sharpe decay > 0.5 | ✅ (2.42) |

**Fails 2 of 5. Verdict: NOT_SIGNIFICANT.**

### The confirming test

If the delay mechanism were real, it should show up without cherry-picking a
config. Testing the whole delayed family vs the whole no-delay family — one
hypothesis, one degree of freedom, no selection:

- no-delay family: +0.4364 R/trade
- delayed family: +0.4638 R/trade
- delta: **+0.0275 R/trade, p = 0.2988**

The effect collapses from +0.46R to +0.03R. **The headline V4 number was a
config-selection artifact**, not a mechanism. Per-year deltas confirm it: 7/10
years positive but all near zero (−0.14 to +0.10).

---

## 4. What actually stands

| Claim | Status |
|---|---|
| v015 4-pair carry edge is real | ✅ CONFIRMED (p<0.001, BH-survives, OOS Sharpe 1.25) |
| Trailing stop costs the system ~63R | ✅ CONFIRMED (HYP-059) |
| Edge is regime-fragile | ✅ CONFIRMED (walk-forward 2/4 yrs negative) |
| Replace trail with time exit (V3 fix) | ❌ REFUTED by RQ-REST-013 re-sim |
| Delay trail activation (V4) | ❌ NOT_SIGNIFICANT (p=0.30 family-level) |
| Widen trail to 2.0× ATR | ⚠️ UNTESTED here — RQ-REST-013 ranks it best; my engine did not reproduce that. Two engines disagree. Genuinely open. |

**Live config unchanged. Nothing ships.**

---

## 5. The one honest lead left

RQ-REST-013 says wider trail (2.0×) is the best arm. My independent re-simulation
says delay matters and width doesn't. **Two engines, two answers — that
disagreement is the finding worth chasing**, and it is a reconciliation problem,
not a new hypothesis.

Next step is not a V5. It is: reconcile the two exit engines on a shared trade,
bar by bar, and find out which one is wrong. Until they agree, neither's exit
ranking should be trusted enough to trade.

---

*"The elements of good trading are: (1) cutting losses, (2) cutting losses,
and (3) cutting losses." — Seykota*
*Sometimes the loss you cut is a research direction.*
