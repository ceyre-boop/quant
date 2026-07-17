# Backtest Bias Audit — gapper fade harness (2026-07-17)

Answer to "is the backtester foolproof / general-purpose bias-free?": **No.**
What exists is a set of study-specific scripts (sim_annual_hyp093_forward.py,
ev_scan_hyp103.py, the MC files), each hard-coding the gapper fill model. It is
not a general engine and it carries at least one large optimistic bias. Findings
ranked by materiality.

## 1. CORRECTED 2026-07-17 — stop-fill bias is SMALL; cost + IID were the real biases
Initial estimate here (−19 pts, +5.6%) was WRONG — it used a crude proxy that
penalised all 81 stops. The bias-free engine on **1-minute bars** shows only
**5 of 79 stops truly gap through** the trigger; the other 74 fill at −25%, so
the exact-trigger assumption is worth ~1–2 pts, not 19. See
`HYP093_corrected_results.md` for the real decomposition. The genuine biases:
- **Transaction cost omitted** (the big one): charging a realistic micro-cap
  entry spread drops annual +25.5% → +9.8% (full) / +18.2% (~1% spread), Sharpe
  3.5 → 1.5–2.6.
- **IID bootstrap** (finding #3): block bootstrap roughly halves prop P(PASS)
  (78.5% → 64.7% at 90d) and multiplies bust risk (1.6% → 12.7%).
Net honest read: **~+10–18% / Sharpe ~2 / ~10–13% bust**, not +24.4%/3.4/~0.

## 2. HIGH — locate/borrow availability not modeled
Backtest fills a short on 100% of signals. Measured reality (ThetaData parity):
only 6.8% of events had a measurable options market; IB EASY-borrow was 79% of
the *universe* but not measured *at 10:30 on gap day*, when borrow evaporates.
Direction: optimistic (phantom fills on unborrowable names).

## 3. MEDIUM — IID bootstrap destroys clustering
The MC draws daily P&L independently with replacement. Gapper days cluster
(hot IPO/biotech regimes) and losses co-occur. IID understates tail risk and
drawdown duration. Direction: optimistic on P(BUST).

## 4. MEDIUM — single-regime, single-year sample
The "forward year" is one 2025-07→2026-06 regime. No 2022–24 data exists to
test other regimes (entitlement limit). HYP-093's own walk-forward showed the
macro-carry edge was regime-fragile; the gapper edge has not been regime-tested
at all. Direction: unknown, likely optimistic.

## 5. LOW / clean — these are actually OK
- Universe: Polygon grouped-daily includes delisted tickers → **no survivorship
  bias** at candidate selection. Good.
- Entry/exit prices from real minute bars (10:30 open, EOD close) — no
  look-ahead in the entry itself.
- Prereg discipline + time-split holdouts handle multiple-comparison / mining
  bias at the *study* level (not the backtester's job).
- M&A exclusion + price/volume floors are applied point-in-time.

## What "foolproof, any-strategy, any-asset" would actually require
A real engine, not scripts: (a) fill model with slippage/gap-through as a
function of liquidity, (b) borrow/locate feed gating shortable names, (c) block
bootstrap or regime-stratified resampling, (d) per-asset cost/calendar configs,
(e) a golden-set reconciliation test. Est. a few days of focused build. Until
then, treat every headline as OPTIMISTIC and lead with the realistic-stop
number: **the gapper fade forward year is ~+5.6%, not +24.4%.**
