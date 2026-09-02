# HYP-109 — post-shock abstention

**VERDICT: NULL — KILL_DIRECTIONAL.** Dead. Not re-run.

Sealed `a7f32774c2ebb8b8` before any data was read (commit `1ac598b`); hash
verified at gate zero and again after the run. Ledger: `ADJUDICATED NULL`.
Prior expectation was NOT_SIGNIFICANT; the predicted failure mode was (b), and
(b) is what failed.

## The incumbent's absolute R

Buy-and-hold, equal-weight, ten ETFs, 2015-01-02 → 2026-07-16:
**+131.7% total · Sharpe 0.496 · max DD −33.2% · +0.0290 %/day.**
This is what a $2k holder gets for free. Every number below is judged against it.

## What the test found

| | statistic | result | ladder |
|---|---|---|---|
| (a) magnitude | median RV shock / non-shock | **1.360**, p ≈ 0.0000 | PASS |
| (b) direction-null | mean 5d cumr shock − non-shock | **−0.237%**, 95% CI [−0.502%, −0.006%] | **FAILS** |
| (c) tradeability | ΔSharpe on 15 purged folds | **8/15** positive; full-sample +0.120 (0.616 vs 0.496) | INCONCLUSIVE |
| (d) raw return | Δ %/day | **−0.0092** (+0.0198 vs +0.0290) | below floor |

3,153 pooled shock windows against 25,787 non-shock; every instrument well above
the 30-event abort line (302–337 each).

## What it means, honestly

**The magnitude half of the surviving finding is confirmed hard.** After a
top-decile |return| day, next-week realized vol is 36% higher, and that is not a
fluke at any conventional threshold. Vol clusters. That part of "magnitude is
conditionable" survives contact with 12.5 years across ten instruments.

**The direction-null half does NOT hold at this window.** After a shock the next
five sessions carry a small **negative** drift — −0.24% per week — and the 95% CI
excludes zero. Post-shock is not directionless; it is mildly directional, and
that makes this candidate a directional effect in disguise rather than an
instance of the finding it was meant to express. That is exactly the kill
condition that was frozen in advance, and it is applied.

**Abstention does something real, but not what was claimed.** Sitting out after
shocks halved the max drawdown (−33.2% → −17.3%) and raised full-sample Sharpe
(0.496 → 0.616) — but by giving up a third of the return (+131.7% → +77.4%) and
only 63% time in market, and it improved Sharpe in only 8 of 15 purged folds.
That is a risk-reduction overlay that is not robust across time, not an edge.

**The companion asked "what instrument pays for magnitude without direction" and
the answer was: not UVXY.** After a SPY shock its 5-session return was −2.12% vs a
−2.73% unconditional baseline; the +0.60% difference has a CI of [−5.80%, +2.38%].
Roll decay dominates even when the magnitude forecast is right. Descriptive
only, no verdict weight — a lead, and a discouraging one.

## The lead (for a SEPARATE prereg, not for this one)

Post-shock 5-session drift is **negative, small (−0.24%/wk), and fragile** — the CI
edge is −0.006%, and the block bootstrap resamples pooled events that share
dates across instruments, which under-models cross-sectional correlation and
therefore makes the CI *narrower* than it should be. So the rejection is real
under the pre-registered procedure but is not to be leaned on. Mean sign of the
post-shock week was +0.108 (more up-weeks than down) against a negative mean:
the distribution is left-skewed — most weeks mildly up, a few badly down. If
anyone registers this as a directional candidate they should expect it to be
tail risk, not drift, and they pay a 1545-trial DSR hurdle for it.

## Two things I would state against my own test

1. The pre-registered "full-sample DSR prob" was applied to the abstain series'
   Sharpe (0.616), which returns 1.000 and is nearly meaningless — any long-ETF
   series over this window clears it. It should have been specified on the
   *delta*. It did not affect the verdict, because (b) failed first, but it is a
   flaw in the spec and is recorded so the next prereg does not repeat it.
2. The block bootstrap for (b) is conservative in the wrong direction for a kill
   test, as noted above. Under a cross-sectionally-aware null the CI might well
   include zero. That would have made the verdict NULL on (c) instead of
   KILL_DIRECTIONAL — the same verdict by a different door.

## Constraints honoured

One hypothesis. One test. No parameter changed after the result. No second run.
No search for the parameterisation that passes. Reported as a null.
