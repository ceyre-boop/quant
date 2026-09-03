# HYP-117 — cross-sectional G10 selection, sealed holdout 1990-01 → 2005-12

**VERDICT: IC_ONLY.** Sealed `8ce9df8d86f7da4b` with the 2006–2026 development results declared inside
the prereg; the holdout was unread (data guard) until the run; hash verified before and after; ledger
`ADJUDICATED`. One run. 192 months, 45 pairs, 8,640 pair-months. Trial count declared 1,639.

| model | holdout IC (CI) | years IC>0 | top-5 ann / Sharpe | vs carry factor (Sh 0.82) | perm p (BH) | verdict |
|---|---|---|---|---|---|---|
| **M1 baseline** z(carry)+z(mom12)+z(value), no parameters | **+0.151 [+0.09, +0.21]** | **13/16** | +6.8% / **0.80** | ΔSh CI [−0.54, +0.42] | **0.000** | IC_ONLY |
| M2 ridge (trained 2006–26) | −0.044 [−0.09, +0.01] | 3/16 | −3.4% / −0.44 | worse, CI excludes 0 | 1.0 | NULL |
| M3 ridge + carry×state (HYP-117 form) | −0.061 [−0.13, +0.00] | 6/16 | −4.1% / −0.51 | worse | 1.0 | NULL |
| M4 boosting, monotone | −0.031 [−0.08, +0.01] | 5/16 | +0.8% / 0.12 | worse | 0.64 | NULL |
| M5 SHOT: stat(ridge+derivatives) × vol | −0.007 [−0.05, +0.04] | 7/16 | −1.2% / −0.16 | worse | 1.0 | NULL |
| M5b SHOT-pure: sign(M1) × vol | +0.107 [+0.06, +0.16] | 13/16 | +1.6% / 0.22 | worse, CI excludes 0 | 0.18 | IC_ONLY |

Carry factor on the holdout: **+6.1%/yr, Sharpe 0.82** (vs +1.6–1.9% on 2006–2026). Oracle +73%.
1992 ERM and 1998 LTCM months are in `result.json`; M1 top-5 and the factor took the same hits.

## What it means

1. **The parameter-free characteristic model is real.** Carry + 12-month momentum + value ranks pairs
   with IC 0.15 across 16 years the model never saw, positive in 13 of them, permutation p < 0.001.
   That is the one cross-sectional FX result in this repo that survives a sealed holdout.
2. **It does not beat plain carry.** Top-5 Sharpe 0.80 vs the factor's 0.82 — the extra features
   re-rank the same currencies. Skill in *ranking* did not become skill in *return*.
3. **Every fitted model failed, and most were negative.** Trained on 2006–2026, ridge/boosting/the
   SHOT all had IC ≤ 0 on 1990–2005 and lost money. Fitting to one regime transferred *worse than
   nothing*. The operator's statistics × calculus shot: IC −0.007, top-5 −1.2%/yr. Its unfitted
   cousin (sign × vol) kept the IC and lost the return — magnitude scaling adds variance, not edge.
4. **The premium itself decayed.** +6%/yr and IC 0.15 in 1990–2005 → +1.7%/yr and IC 0.04 in 2006–2026.
   Whatever a retail carry book expects today should be anchored on the second number, not the first.

## Constraints honoured
One run. Models, features, k, haircut and windows unchanged after the holdout was read. Dev results
were sealed inside the prereg before the holdout.
