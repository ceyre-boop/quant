# Backtest Bias Audit — gapper fade harness (2026-07-17)

Answer to "is the backtester foolproof / general-purpose bias-free?": **No.**
What exists is a set of study-specific scripts (sim_annual_hyp093_forward.py,
ev_scan_hyp103.py, the MC files), each hard-coding the gapper fill model. It is
not a general engine and it carries at least one large optimistic bias. Findings
ranked by materiality.

## 1. CRITICAL — stop fills are assumed exact (optimistic ~19 pts/yr)
Every stopped short is booked at exactly `entry × 1.25`. But of the 81/234
stopped events, **65% had an intraday high >40% above entry; the worst spiked
+2382%.** A real stop on a parabolic micro-cap gaps THROUGH the trigger — you
fill far worse than −25%.
Re-pricing stops at just +10% past the trigger (still generous):
- Annual return: **+24.4% → +5.6%** (−18.8 pts)
- Max drawdown: 4.0% → 7.6%
This single assumption is the difference between "elite Sharpe" and "marginal
edge." Every downstream number (MC P(PASS), EV scan, prop tables) inherits it.

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
