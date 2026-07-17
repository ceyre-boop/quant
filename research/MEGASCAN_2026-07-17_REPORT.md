# Megascan 2026-07-17 — largest strategy search in the repo

**Result: zero strategies beat the gapper-fade benchmark (+18%/yr, Sharpe 2,
DD<15%, ≥50 trades/yr) out of sample.** The single best dirty-scan candidate was
pre-registered and collapsed on the untouched holdout.

## Scale
- **77,016 distinct signal hypotheses** evaluated in **89 seconds** (bias-free
  daily engine, 12 cores). 75,916 per-asset + 1,100 pooled-across-universe.
- Universe: 111 liquid ETFs/large-caps, daily bars 2014→2026 (yfinance).
- Families: RSI mean-reversion, gap fade/follow, N-day dip reversion, channel
  breakout. (Intraday minute families A/B were out of data reach — Alpaca
  minute quota — so the scan ran on the daily-bar families where a real 12-month
  holdout exists. Flagged, not faked.)
- Honest FWER denominator = distinct signal rules (77,016). Sizing/exit
  permutations that don't change the hypothesis were NOT multiplied in — that
  would be statistical theater.

## Step 1–2: what came closest
6 configs beat the benchmark on RAW dirty metrics; **0 survived Bonferroni**
over 77k. Two clusters:
1. **NVDA 20-day breakout long** (Sharpe 2.99 dirty). Rejected as a single-name
   survivorship artifact — "NVDA went up 2014-2025" is not a strategy.
2. **Down-gap continuation short** (5 correlated variants, Sharpe 2.1–2.5 dirty).
   Coherent and mechanically sensible; survived removing leveraged/vol ETFs
   (Sharpe 2.3 ex-leveraged, so genuine gap-momentum, not vol-decay). This was
   the one worth testing → HYP-104.

## Step 3–4: HYP-104 pre-registered and holdout-tested
Locked rule: gap down ≥5%, short next open, hold 2 days, 15% stop, liquid names
ex-leveraged. Prereg hash a9843721…, committed 045bb03 before holdout touch.

| | Dirty 2014→2025 | Holdout 2025-07→2026-07 |
|---|---|---|
| Annual | +21.0% | **+2.5%** |
| Sharpe | 2.32 | **0.36** |
| Max DD | 7.7% | 5.5% |
| Win | 54.7% | 53.3% |
| n | 1238 | 105 |
| perm p | — | 0.35 |

**Verdict: NOT_CONFIRMED.** The edge evaporated out of sample — the dirty Sharpe
of 2.3 was overfit noise, exactly as the 0/77k Bonferroni result warned.

## Conclusion
The megascan is the honest negative the method is built to produce: across 77k
hypotheses and four families on a decade of liquid-universe data, **nothing beats
the gapper fade out of sample.** The gapper fade — now honestly ~+10–18% /
Sharpe ~2 after the bias-correction rebuild — remains the only edge in this repo
that has ever survived a real holdout. The next genuine gains are more likely to
come from the gapper's *execution* (borrow/locate, TICK-037) than from a new
signal at daily resolution.

Reusable: `backtester/daily_engine.py`, `backtester/megascan.py`,
`data/scan_results/megascan_20260717.parquet` (all 77k rows).
