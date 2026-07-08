# HYP-065 — Carry Regime Conditionality — VERDICT: REJECTED

> pre-reg hash `111854c0b912b5ef` · 457 trades (2015-2026, 4-pair v015) · READ-ONLY, no live changes. Sharpe = annualised (√n by trades/yr).

## Per-regime (pooled)

| regime | Sharpe | win | n |
|---|---|---|---|
| HIKING | 0.749 | 0.541 | 148 |
| PEAK_HOLD | 0.25 | 0.447 | 217 |
| CUTTING | 0.254 | 0.483 | 58 |
| BOTTOM | 0.093 | 0.5 | 34 |
| NEUTRAL | 0.0 | None | 0 |

## Per cycle — edge-ON (hiking+peak) vs edge-OFF (cutting+bottom)

| cycle | Sharpe ON | Sharpe OFF | n_on | n_off |
|---|---|---|---|---|
| C1 | 0.517 | 0.0 | 176 | 0 |
| C2 | 0.204 | 1.196 | 33 | 40 |
| C3 | 1.163 | 0.106 | 142 | 52 |

**Bar:** edge-ON>0.50 in **2/3** cycles (need ≥2); edge-OFF<0.20 in **1/3** (need ≥2); permutation p **0.0919** (need <0.05); robust across sweeps: **False**.

**Robustness sweep** (cycles meeting on>0.5 / off<0.2): {"delta0.1": {"cyc_on>0.5": 2, "cyc_off<0.2": 1}, "delta0.5": {"cyc_on>0.5": 2, "cyc_off<0.2": 1}, "lookback2": {"cyc_on>0.5": 2, "cyc_off<0.2": 1}, "lookback6": {"cyc_on>0.5": 2, "cyc_off<0.2": 1}}

## Load-bearing: 2024-26 fresh (Fed cutting → should be edge-OFF)

- 89 trades, regimes {'PEAK_HOLD': 37, 'BOTTOM': 31, 'CUTTING': 21}, Sharpe 0.448 → matches-expected-OFF: **False**

## Sharpe by cycle × pair

| cycle | EURUSD | GBPUSD | USDJPY | AUDUSD |
|---|---|---|---|---|
| C1 | -0.11 | 0.5 | 0.96 | 0.73 |
| C2 | 0.97 | 1.24 | -0.09 | 0.32 |
| C3 | 1.02 | 1.02 | 0.72 | 0.7 |

## Verdict

No regime pattern → carry is dead unconditionally. Pivot edge families.

