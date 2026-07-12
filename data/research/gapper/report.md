# HYP-092 — Gapper Continuation Read: One-Year Backtest Report

**Verdict: NOT_SIGNIFICANT** (pre-registered MWU one-tailed CONT > EX: primary p = 0.594 on n = 558/391; run-deduped robustness p = 0.634 on n = 347/255). Well-powered — both buckets exceed the card's n ≥ 30 floor by an order of magnitude, at the unique-ticker level too (439/326).

Prereg: `data/research/preregister/HYP-092_gapper_continuation.json` (sha256 `3e07c6a4…`, committed before any outcome data). Ticket TICK-029. Window 2025-07-01 → 2026-06-30, 251 trading days.

## What was tested

The vault decision card's scan filter (≥30% up at 10:30 ET vs prev close, ≥$2, ≥500K shares by 10:30) replayed on a survivorship-free universe (Polygon grouped daily incl. delisted; 11,396 buffered candidates → 1,475 passing the card filter at 10:30), then the card's CONT/EX/UNC checklist **frozen into deterministic rules before any outcome was observed**: VWAP position, higher-lows, up-vs-down volume, range position, lower-highs, climax-fade, rejection wick — voted CONT (≥3 cont signals, ≤1 ex), EX (mirror), else UNC. Outcome = % move from the 10:30 bar's open to the last regular-session close. No input uses any bar after 10:25 ET; entry origin shares no bar with read inputs.

## Results

| bucket | n | median | mean | >+3% | <−3% | ±3% |
|---|---|---|---|---|---|---|
| CONT | 558 | **−2.34%** | +1.15% | 33.2% | 47.5% | 19.4% |
| EX | 391 | **−1.81%** | −1.09% | 32.7% | 46.3% | 21.0% |
| UNC | 526 | −2.41% | −2.23% | 28.3% | 48.7% | 23.0% |
| ALL | 1,475 | −2.21% | −0.65% | 31.3% | 47.6% | 21.1% |

1. **The read does not separate.** CONT's median is *worse* than EX's; the distributions are statistically indistinguishable (p = 0.59). The mechanized checklist carries no information about the next 5½ hours.
2. **The filter's base rate is a downhill slope.** The median qualifying gapper bleeds −2.2% from 10:30 to close; 48% reverse more than 3%, only 31% continue more than 3%. The market's free list is mostly a fade list after the read window.
3. **The money is in the tail, and the read can't see it.** CONT's mean (+1.15%) sits far above its median (−2.34%) — a handful of monster continuations pay for many small fades. That skew is real, but it lands in CONT and EX alike (rank test flat): the structural read does not time the tail.
4. **Halted names are the elephant.** 344 candidates were excluded as unreadable (fewer than 8 bars by 10:25 or none after 10:15 — mostly LULD halt cascades). Their descriptive median post-read move: **+16.5%** (n = 331). The most violent continuations were precisely the ones you couldn't read or safely enter at 10:30. Conservative exclusion, disclosed per prereg.
5. Watchlist density: 82–162 qualifying names/month (~6 per trading day) — stable across the year.

## Post-hoc observations (NOT evidence — the sealed test is the MWU above)

- A mean-difference framing (CONT +1.15% vs EX −1.09%) would have looked interesting; it is tail-driven and was not the registered test. If tail capture matters, that is a NEW hypothesis (stop-loss + skew-riding mechanics), requiring its own prereg.
- UNC — the "do not force a read" bucket — was the weakest of all (median −2.41%, lowest continuation rate). The mixed-picture days fade hardest.
- Vote marginals expose the checklist's blind spots at this resolution: higher-lows fired on 80% of everything, climax-fade on 74% — near-constant features of any morning gapper, so they carry little discrimination. Rejection-wick (18%) was the only rare signal.

## What this does and does not say about the live logging study

This test kills the **mechanizable skeleton** of the card: VWAP/structure/volume checklist items, read at 10:30, at 5-minute resolution, do not predict the close. It does **not** test what Colin's eyes add — catalyst quality, float, level context, SPY tape, halt behavior, the feel that doesn't compress into seven booleans. That residual is exactly what the 2-3 week live logging period isolates: if his live CONT/EX calls separate where this mechanization didn't, the edge is the discretion itself, not the checklist. If they don't, this null said so first, a year cheaper.

## Caveats (from prereg, all binding)

Raw % moves — no commissions, slippage, borrow cost/availability (short-EX numbers especially optimistic on HTB names); mechanized proxy of a discretionary process; ticker-days not independent (runners repeat — hence the deduped robustness pass, same answer); Polygon-basis discovery buffered at +20% to absorb cross-source noise; Alpaca SIP coverage effectively total (1 of 11,396 candidates missing, and it was an active ticker — the delisted graveyard is in-sample).
