# HYP-106 Addendum — Realistic Fill Model + the 1/5/10-Year Question

## What was built
`backtester/realistic_fills.py` — models the three frictions that dominate
≥100% microcap gapper execution, all detected from the minute tape:
- **LULD halts**: timestamp gaps (missing RTH minutes) + single-minute moves
  beyond the 10% band → halt-resume bars; entry/exit on a resume bar pays a 2%
  adverse slip, and events halted *through* the entry window are treated as
  UN-ENTERABLE (skipped, not filled).
- **Quoted spread**: charged ROUND-TRIP (entry + exit), scaled to bar
  volatility + inverse to minute $-volume, capped per scenario (opt 3% / base 8%
  / pess 15%).
- **Size/impact**: fills capped at 10% of the entry minute's volume; excess
  walks the book.

## HYP-106 re-tested (filtered long, 09:31→10:30, 25% stop)
| Cut | Scenario | n | median | tail ratio | win | Sharpe |
|---|---|---|---|---|---|---|
| Holdout | optimistic | 22 | +66.8% | 13.5 | 86% | 4.6 |
| Holdout | base | 22 | +61.8% | 8.1 | 86% | 4.3 |
| Holdout | pessimistic | 22 | +55.1% | 6.7 | 82% | 4.0 |
| Holdout | base, **skip entry-halted** | 17 | **+60.7%** | 8.6 | 88% | 5.3 |
| Full year | base | 70 | +51.8% | 8.1 | 89% | 8.1 |
| Full year | base, $100k size | 70 | +44.2% | 7.6 | 87% | 7.6 |
| Full year | base, **skip entry-halted** | 48 | +50.8% | 8.9 | 88% | 8.6 |

**The edge survives every friction backtestable on OHLC data.** Spread and halt
slips barely dent it because a +50% move dwarfs an 8% round-trip cost; 23–31% of
filtered events are un-enterable (halted at the open) and skipping them leaves
the rest intact.

## The honest ceiling (why this is NOT yet a green light)
Printed-OHLC backtesting cannot resolve the one risk that matters most: **can you
actually buy a screaming, thin, repeatedly-halted microcap at the 09:31 print in
real size?** The bar's "open" is a print that already happened; a live market
order arrives later, into a moving book, possibly into a halt. No amount of
historical bar analysis answers this — only a **live/paper forward shadow with
real marketable orders** does. Treat the +50% as the backtest ceiling, not the
expected live number.

## 1 / 5 / 10-year runs — data reality
- **1 year: DONE** — the 234-event forward year (2025-07→2026-06) is the full
  minute-bar dataset; results above.
- **5 year / 10 year: NOT POSSIBLE with current data access.** Cached minute
  bars reach back only to **2024-01-03** (Alpaca free tier = 2 years of minute
  history). There is no 2015–2023 intraday data, and the gapper event universe
  for those years isn't built (Polygon free grouped = 1 year).
- **What a 5–10yr intraday test would require**: (1) paid minute history
  (Polygon/ThetaData) back to 2015; (2) a rebuilt ≥100%-gapper event universe
  per year from daily grouped bars; (3) re-run this exact pipeline. The
  `megascan`/`daily_engine` universe build + `realistic_fills` model are ready
  to consume it the moment the data exists. A daily-bar proxy would NOT be the
  same strategy (it can't model the 09:31 entry or halts), so it's not offered
  as a substitute.

## Verdict
HYP-106 remains CONFIRMED and is now stress-tested through every friction the
data can express — it survives. The signal is real and robust. The remaining
uncertainty is purely execution realism on thin halted names, which is a
FORWARD-SHADOW question, not a backtest one. Next action is a live 09:31 shadow
with the locked filter, logging real fills vs the model — not more history.
