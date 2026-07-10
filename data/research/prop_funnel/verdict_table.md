# Prop-Funnel Verdict Table (TICK-022)

> **CAVEAT (applies to every row):** Assumes i.i.d. attempts/months — FALSE under regime shift. p^100 in particular treats 100 attempts as independent draws of the same edge; a regime that kills the edge kills all remaining attempts at once.

| Strategy | Evidence | Firm | P(funded) | P(pass 100/100) | Med cal-days P1 | E[attempts] | Fees→funded $ | P(surv 12mo) | E[payout/mo] $ | P(mo ≥$10k) | P($10k every mo ×12) | Program EV/mo $ | Pricing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ict_window_B | UNPROVEN | MFF_100K | 0.6595 | 0.0 | 24.6 | 1.52 | 569.0 | 0.827 | 3144.0 | 0.0795 | 0.0 | 2728.0 | UNVERIFIED_PRICING |
| carry_oos | PROVEN_REGIME_FRAGILE | MFF_100K | 0.6802 | 0.0 | 14.5 | 1.47 | 551.0 | 0.9952 | 2552.0 | 0.026 | 0.0 | 2138.0 | UNVERIFIED_PRICING |
| ict_window_B | UNPROVEN | FTMO_100K_SWING | 0.63684 | 0.0 | 39.1 | 1.57 | 974.0 | 0.8485 | 2838.0 | 0.0622 | 0.0 | 2108.0 | UNVERIFIED_PRICING |
| carry_oos | PROVEN_REGIME_FRAGILE | FTMO_100K_SWING | 0.9961 | 0.676785 | 23.2 | 1.0 | 622.0 | 1.0 | 2293.0 | 0.0164 | 0.0 | 1753.0 | UNVERIFIED_PRICING |
| carry_fwd_S1.25 | SCENARIO | MFF_100K | 0.4121 | 0.0 | 15.9 | 2.43 | 910.0 | 0.9833 | 1741.0 | 0.0115 | 0.0 | 1255.0 | UNVERIFIED_PRICING |
| carry_fwd_S1.25 | SCENARIO | FTMO_100K_SWING | 0.84835 | 0.0 | 37.7 | 1.18 | 731.0 | 0.9952 | 1596.0 | 0.0066 | 0.0 | 977.0 | UNVERIFIED_PRICING |
| carry_fwd_S0.69 | SCENARIO | MFF_100K | 0.3081 | 0.0 | 14.5 | 3.25 | 1217.0 | 0.9646 | 1498.0 | 0.0075 | 0.0 | 966.0 | UNVERIFIED_PRICING |
| carry_decade | PROVEN_REGIME_FRAGILE | MFF_100K | 0.2549 | 0.0 | 14.5 | 3.92 | 1471.0 | 1.0 | 1290.0 | 0.0033 | 0.0 | 746.0 | UNVERIFIED_PRICING |
| carry_decade | PROVEN_REGIME_FRAGILE | FTMO_100K_SWING | 0.89506 | 1.5e-05 | 39.1 | 1.12 | 693.0 | 1.0 | 1128.0 | 0.0013 | 0.0 | 630.0 | UNVERIFIED_PRICING |
| carry_fwd_S0.69 | SCENARIO | FTMO_100K_SWING | 0.60782 | 0.0 | 42.0 | 1.65 | 1020.0 | 0.9928 | 1289.0 | 0.0036 | 0.0 | 614.0 | UNVERIFIED_PRICING |
| carry_fwd_S0 | SCENARIO | MFF_100K | 0.1968 | 0.0 | 15.9 | 5.08 | 1905.0 | 0.9473 | 1158.0 | 0.0046 | 0.0 | 596.0 | UNVERIFIED_PRICING |
| carry_fwd_S0 | SCENARIO | FTMO_100K_SWING | 0.26379 | 0.0 | 43.5 | 3.79 | 2350.0 | 0.9952 | 1035.0 | 0.0025 | 0.0 | 264.0 | UNVERIFIED_PRICING |
| ict_london_a | UNPROVEN | FTMO_100K_SWING | INSUFFICIENT_DATA |  |  |  |  |  |  |  |  |  |  |
| ict_london_a | UNPROVEN | MFF_100K | INSUFFICIENT_DATA |  |  |  |  |  |  |  |  |  |  |
| futures_orb | UNVALIDATED | TOPSTEP_50K | INSUFFICIENT_DATA |  |  |  |  |  |  |  |  |  |  |
| futures_orb | UNVALIDATED | APEX_50K | INSUFFICIENT_DATA |  |  |  |  |  |  |  |  |  |  |

## Row caveats

- **ict_window_B × MFF_100K** — ICT edge UNPROVEN: permutation p=0.52, fails BH (2026-06-30 audit). Backtest pools only.
- **carry_oos × MFF_100K** — CRITICAL: this bootstraps the 2023-2024 OOS window only — a FAVORABLE, rate-trending regime. The forex edge is REGIME-FRAGILE: rolling walk-
- **ict_window_B × FTMO_100K_SWING** — ICT edge UNPROVEN: permutation p=0.52, fails BH (2026-06-30 audit). Backtest pools only.
- **carry_oos × FTMO_100K_SWING** — CRITICAL: this bootstraps the 2023-2024 OOS window only — a FAVORABLE, rate-trending regime. The forex edge is REGIME-FRAGILE: rolling walk-
- **carry_fwd_S1.25 × MFF_100K** — SCENARIO: carry_oos mean-shifted from Sharpe 2.54 to 1.25 ('if carry forward-Sharpe were 1.25'). CRITICAL: this bootstraps the 2023-2024 OOS
- **carry_fwd_S1.25 × FTMO_100K_SWING** — SCENARIO: carry_oos mean-shifted from Sharpe 2.54 to 1.25 ('if carry forward-Sharpe were 1.25'). CRITICAL: this bootstraps the 2023-2024 OOS
- **carry_fwd_S0.69 × MFF_100K** — SCENARIO: carry_oos mean-shifted from Sharpe 2.54 to 0.69 ('if carry forward-Sharpe were 0.69'). CRITICAL: this bootstraps the 2023-2024 OOS
- **carry_decade × MFF_100K** — Full-decade pool (Sharpe ~0.69): includes both paying and flat regimes.
- **carry_decade × FTMO_100K_SWING** — Full-decade pool (Sharpe ~0.69): includes both paying and flat regimes.
- **carry_fwd_S0.69 × FTMO_100K_SWING** — SCENARIO: carry_oos mean-shifted from Sharpe 2.54 to 0.69 ('if carry forward-Sharpe were 0.69'). CRITICAL: this bootstraps the 2023-2024 OOS
- **carry_fwd_S0 × MFF_100K** — SCENARIO: carry_oos mean-shifted from Sharpe 2.54 to 0 ('if carry forward-Sharpe were 0'). CRITICAL: this bootstraps the 2023-2024 OOS windo
- **carry_fwd_S0 × FTMO_100K_SWING** — SCENARIO: carry_oos mean-shifted from Sharpe 2.54 to 0 ('if carry forward-Sharpe were 0'). CRITICAL: this bootstraps the 2023-2024 OOS windo
- **ict_london_a × FTMO_100K_SWING** — pool n=28 < 30 — refusing to Monte-Carlo; ICT edge UNPROVEN: permutation p=0.52, fails BH (2026-06-30 audit). Backtest pools only.
- **ict_london_a × MFF_100K** — pool n=28 < 30 — refusing to Monte-Carlo; ICT edge UNPROVEN: permutation p=0.52, fails BH (2026-06-30 audit). Backtest pools only.
- **futures_orb × TOPSTEP_50K** — pool n=2 < 30 — refusing to Monte-Carlo; UNVALIDATED: n_real=0 live (futures_validation.json passed:false); replay pool is in-sample and tiny. Regenerate via Phase R (operator-gated) before trusting.
- **futures_orb × APEX_50K** — pool n=2 < 30 — refusing to Monte-Carlo; UNVALIDATED: n_real=0 live (futures_validation.json passed:false); replay pool is in-sample and tiny. Regenerate via Phase R (operator-gated) before trusting.