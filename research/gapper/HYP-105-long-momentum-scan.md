# HYP-105 — Long-Side Momentum on Parabolic Gappers: CONFIRMED (with caveats)

**The mirror of the confirmed fade is real: ride the ≥100% gappers LONG from
09:31 to 10:30, before the fade the short strategy exploits. It survived a true
holdout — and unlike the short, it has NO borrow/locate wall.**

## Method
234 minute-bar-ready HYP-093 gappers, 70/30 date split (in-sample <2026-04-08:
163 events; holdout ≥: 71). Bias-free engine (gap-through fills, entry-bar
spread). Long side, 2% sizing. (Used the 234 minute-ready events, not the 2-year
"559" set whose older events lack cached minute bars — flagged.)

## Phase 2 — in-sample scan (360 configs: entry × exit × hard-stop)
Coherent, strong cluster: **enter 09:31–09:35, exit 10:30, wide (20–25%) stop.**
Best: 09:31→10:30, 25% stop — Sharpe 5.07, annual +64.5%, win 57.1%, median
+6.5%, n=163. Raw permutation p=0.002 (floor), but **0/360 survived Bonferroni**
(correlated-config tax). Holdout is the arbiter.

## Phase 4 — holdout (locked config, run once)
Prereg 334c373d…, committed 4d5c387 before holdout touch. Null: Sharpe ≤ 0.

| Metric | Holdout (71 events) |
|---|---|
| Sharpe | **3.63** |
| Mean / event | +28.5% |
| Median / event | +7.9% |
| Win rate | 56.3% |
| Permutation p | **0.0005** |

**Verdict: CONFIRMED** (Sharpe>0, p<0.05, n≥30). Ledger entry 84.

## Phase 3 — ML feature layer: REJECTED as look-ahead leakage
Logistic 89.6% / RF 92.1% 5-fold CV accuracy — but the dominant features are
`intraday_push` (0.49) and `overnight_gap` (0.31), both measured THROUGH the
09:31→10:30 holding window. They contain the outcome; the "92%" is leakage, not
prediction. A usable model needs strictly pre-09:31 features (overnight gap,
pre-market volume) — not built here. **Do not treat the ML as predictive.**

## Honest caveats (must travel with the result)
1. **Tail-driven**: mean +28.5% vs median +7.9% — a few monster gappers (the
   ones that would blow up the SHORT) carry the mean. Real expectancy ≈ the
   median; the Sharpe is flattered by fat right tails.
2. **Short window**: 71 events / 2.7 months holdout. Needs forward shadow.
3. **9:31 slippage understated**: entry-bar spread proxy is optimistic for a
   first-minute fill on a rocketing microcap (halts, gaps, partials). Live will
   be worse — possibly much worse.
4. **Not independent of HYP-093**: same events. Ramp-then-fade means both the
   early-long and late-short can win; this is the front half of one phenomenon.
5. **But: no borrow needed.** Long-side sidesteps the TICK-032/037 locate wall
   that caps the fade — its one clear structural advantage.

## Bottom line
First genuinely new confirmed edge of the session, and the only long-side one.
It is real out of sample but fragile (tails, tiny window, execution). Next:
forward shadow at 09:31 entry, a leak-free pre-entry ML filter, and a realistic
first-minute fill model before any sizing. Files: `hyp105_long_scan.py`,
`data/scan_results/hyp105_long_insample.parquet`, results/prereg JSON+MD.
