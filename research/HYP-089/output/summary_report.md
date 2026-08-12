# HYP-089 — TSMOM on 4 v015 FX Pairs: Backtest Report

**Verdict: NOT_SIGNIFICANT**  ·  run 2026-07-12T17:00:39.219662+00:00  ·  pre-reg `HYP-089-TSMOM-Prereg-2026-07-12.md`

Standalone TSMOM backtest, fully isolated from the v015 carry system. 12-month (252d) momentum, sign signal, inverse-vol scaling (10% target, 60d vol), 3x cap, 4 pairs equal-weighted, daily rebalance. All parameters fixed per the locked pre-reg — no optimization, no lookback search.

## 1. Portfolio equity curve (gross, from $1)

- Peak: **1.2766**  ·  Trough: **0.9244**  ·  End-2024: **1.1825**
- Max drawdown: **-15.20%**  ·  trading days: 2606
- Annualized gross return 1.84%  ·  annualized vol 6.64%

## 2. Sharpe (annualized)

- **Gross Sharpe: 0.2773**  (decision metric)
- Net Sharpe (0.3 pip round-trip): 0.2724
- Sortino (gross): 0.4213
- ⚠️ The gross Sharpe sits within ~0.02 of the 0.30 bar. It is **boundary-close**, and the economic magnitude is negligible either way (1.8%/yr gross before costs). The data vintage is pinned (`prices_cache.parquet`) so this number is reproducible; per the pre-reg it is NOT re-searched.

## 3. Annual subperiod Sharpe (10 rows)

| Year | Sharpe | Positive? | Days |
|------|--------|-----------|------|
| 2015 | 0.914 | ✅ | 261 |
| 2016 | -1.237 | ❌ | 261 |
| 2017 | 0.287 | ✅ | 258 |
| 2018 | -1.009 | ❌ | 261 |
| 2019 | 0.510 | ✅ | 260 |
| 2020 | 1.272 | ✅ | 262 |
| 2021 | -0.288 | ❌ | 261 |
| 2022 | 1.499 | ✅ | 260 |
| 2023 | -0.040 | ❌ | 260 |
| 2024 | 0.001 | ✅ | 262 |

**Positive years: 6/10** (gate requires ≥6).
- ⚠️ Marginal: 2024 count as positive only on a near-zero Sharpe (|·| < 0.05, effectively flat). The 6/10 count barely clears the gate and would tip to 5/10 under a trivially different data vintage — the subperiod support is weak.

## 4. Carry correlation

- Method: FRED policy-rate differential sign (long higher-yielder) (FRED available: True).
- **Pearson r = -0.1556**  ·  p-value = 1.34e-15
- Portfolio-level daily signals: TSMOM = Σ inverse-vol-scaled directional positions; carry = Σ of the 4 direction signs.
- Per-pair signal correlations: {'EURUSD=X': None, 'GBPUSD=X': -0.4071, 'USDJPY=X': None, 'AUDUSD=X': 0.254}
- ['EURUSD=X', 'USDJPY=X'] show a `null` per-pair correlation because their carry **direction never flipped** across 2015–2024 (the US out-yielded the EUR and JPY legs the entire decade), so the carry series is constant and Pearson r is undefined for that pair. The portfolio-level r is well-defined because the GBP and AUD carry legs do vary. Either way r ≈ −0.16 is far below the 0.7 bar — the diversification gate passes decisively.

## 5. Hutchinson decay check (reference only)

- Backtest window is 2015-2024 — entirely POST-2012 and post-publication. A true pre-2012 vs post-2012 split is not possible with this data. Early(2015-19) vs late(2020-24) in-sample Sharpes are shown as a directional decay proxy only.
- Early (2015–2019) gross Sharpe: -0.0789  ·  Late (2020–2024) gross Sharpe: 0.605
- Within our sample the late half is *stronger* than the early half — the opposite of a decay slope. This does NOT contradict Hutchinson: their decay is measured pre-2012 vs post-2012, whereas both of our halves are post-publication. The late-period strength is regime-driven (2020–2022 vol + rate-trending FX), not evidence the edge revived.

## 6. Verdict

| Gate | Threshold | Result | Pass |
|------|-----------|--------|------|
| Sharpe | > 0.3 | 0.2773 | ❌ |
| Carry r | < 0.7 | -0.1556 | ✅ |
| Positive years | ≥ 6/10 | 6/10 | ✅ |

### **VERDICT: NOT_SIGNIFICANT**

## 7. Which gate(s) failed

- Sharpe gate: gross Sharpe 0.2773 <= 0.3

Per the pre-reg conjunction rule, any single failure seals the null. **Sealed permanently — do not re-test with different parameters** (that would be data mining). The Hutchinson et al. (2022) post-publication FX-momentum decay is the sufficient explanation; no alternative parameter search is warranted.

## Data quality

- ✅ No DEGRADED pairs — all 4 pairs fetched clean yfinance history over the full window.
