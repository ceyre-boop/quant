# HYP-116 — shock-deferred contributions vs DCA into the EW-10 basket, 2007-06 → 2026-07

**VERDICT: POLICY_FAILS.** Sealed `b76837ace4234dfb`; verified before and after; ledger `ADJUDICATED`. One run.

| | DCA | DEFERRED (buy the session after a SPY down-shock, cash at 0%) |
|---|---|---|
| terminal wealth, 230 monthly units | 650.9 | 646.3 → **ratio 0.993**, block-bootstrap CI [0.981, 1.001] |
| 60/40 vehicle | 563.3 | 558.3 → 0.991 |
| rolling 5-year windows where DEFERRED wins | | **12%** of 170 |
| deployments / cash-days per contribution | | 120 / 13.4 |

287 SPY down-shocks in 4,811 sessions; the policy waited ~13 sessions per contribution on average.

## What it means

Waiting for the crowd to sell before adding costs more than the dip gives back. The −0.7% is small
and the CI touches 1.0 from below, but the rolling share is decisive: the policy loses in 88% of
five-year windows. Same conclusion with 60/40 as the vehicle. The one-session lag, the 0% cash and the
126-session cap were all declared before the run, and none of them is what killed it — 13 days of
average drag is cheap; the dip simply has no extra return to pay for it (HYP-114, 2016–2019).

This closes the shock signal at its last horizon: session (111/114), week (109), contribution (116).

## Constraints honoured
One run. No percentile, lag, cap or vehicle changed after the result.
