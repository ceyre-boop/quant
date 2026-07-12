# HYP-091 — TSMOM diversification of the v015 carry book

**VERDICT: NOT_SIGNIFICANT**  ·  prior_expectation: NOT_SIGNIFICANT  ·  prereg `data/research/preregister/HYP-091_tsmom_carry_diversification.json`  ·  TICK-027

Corrected pre-registration vs the parallel HYP-089 quick-look (proxy correlation + no financing + daily rebalance). Here: **monthly** rebalance (Moskowitz), correlation vs the **actual v015 returns**, and **correct rate-differential-derived financing** (NOT the Colin-gated SWAP_RATES_ANNUAL; TICK-024).

## Why the null triggered

- OOS Sharpe -0.349 <= 0 (null condition a)

## 1. Standalone Sharpe by financing model

| Financing | Full | IS (…2022) | OOS (2023-24) | mean/mo | n |
|-----------|------|-----------|---------------|---------|---|
| ratediff (PRIMARY, correct) | +0.206 | +0.333 | -0.349 | +0.00118 | 119 |
| broken SWAP_RATES (robust.) | +0.310 | +0.458 | -0.336 | +0.00177 | 119 |
| price-only (robust.) | +0.325 | +0.473 | -0.324 | +0.00186 | 119 |

Correct financing makes the strategy **worse** than price-only (it pays the real carry costs the broken model understated ~10x) — the OOS Sharpe is negative either way.

## 2. Per-calendar-year Sharpe (ratediff primary — DESCRIPTIVE, ~12 obs/yr)

| Year | Sharpe | Positive? | n |
|------|--------|-----------|---|
| 2015 | +0.332 | ✅ | 11 |
| 2016 | +0.392 | ✅ | 12 |
| 2017 | -0.416 | ❌ | 12 |
| 2018 | -0.022 | ❌ | 12 |
| 2019 | +0.245 | ✅ | 12 |
| 2020 | +0.614 | ✅ | 12 |
| 2021 | -1.279 | ❌ | 12 |
| 2022 | +1.265 | ✅ | 12 |
| 2023 | +0.033 | ✅ | 12 |
| 2024 | -0.589 | ❌ | 12 |

**Positive years: 6/10.** 2022 (the rate-trending regime) dominates; the OOS years 2023/2024 are ~flat/negative — the concentration the per-year table was designed to expose.

## 3. Correlation vs the ACTUAL v015 carry (monthly)

- Primary (correct-financing TSMOM vs v015): **ρ = -0.12797066517534111** over 101 months (SE≈0.0995); 2022-window ρ = -0.21303306260757277.
- Robustness (broken-model TSMOM vs v015, the mismatch-free apples-to-apples): ρ = -0.1362162224868953.
- Correlation is LOW (well below the 0.5 null bar) — so TSMOM would diversify IF it had positive OOS Sharpe. It does not.
- Confirmatory 50/50 equal-vol blend Sharpe = 1.064 vs max(tsmom 0.239, v015 1.166); diversification lift = -0.10199999999999987.

## 4. Gauntlet

| Gate | Result | Pass |
|------|--------|------|
| pre-reg null (OOS Sharpe>0 AND |corr|≤0.5) | OOS -0.349, null_triggered=True | ❌ |
| directional permutation p<0.05 | p=0.13959 (N=10000) | ❌ |
| deflated-Sharpe prob>0.95 | 0.7531 | ❌ |
| BH survives | False | ❌ |
| holdout OOS Sharpe>0 | -0.349 | ❌ |

Family of one → DSR/BH are near-vacuous; the real guards are the permutation timing test and the Phase-0 hash-lock.

## 5. Honest limitations

- ~119 monthly obs / ~24 OOS / ~12 per year → LOW power; NOT_SIGNIFICANT is about as consistent with low power as with no edge (hence the prior).
- Per-year Sharpes are descriptive (SE ≈ ±0.5), not inferential.
- yfinance FX spot ≠ tradable forward; financing is the rate-differential-anchored model (reproduces the 2026 OANDA snapshot + trade-227), not live broker fills.
- Primary ρ correlates correct-financing TSMOM against the v015 CSV (costed with the broken swap) — the broken-model robustness leg is the mismatch-free cross-check and agrees.
- **Research pass only.** VALID_EDGE/NOT_SIGNIFICANT — no deployment, no live capital (RISK_CONSTITUTION Art. 6; promotion is a separate human step).

