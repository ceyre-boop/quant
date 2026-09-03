# HYP-111a — post-shock intraday retrace-then-continuation (free-tier pilot, 2023-06 → 2026-07)

**VERDICT: VALID_BUT_BELOW_FLOOR.** Confluence: **INCONCLUSIVE** (≤1 bucket has 3 trades).
Sealed `41b7b6a15a9f2f1e` before any event-session minute bar was read; hash verified
before and after. Ledger `ADJUDICATED`. One run. (A first invocation crashed on a
zero-priced ThetaData bar before any statistic was computed or printed; zero-price bars —
147 across the cache — are dropped as data hygiene, recorded here.)

Pilot, not the scoped test: 798 events / 329 dates on the window the STOCK.FREE tier
serves. No 2020, no 2022.

## Numbers

| component | result | bar |
|---|---|---|
| (p) triggers | **102/798 = 12.8%** (target 55 · time 27 · stop 20) | ≥100 ✓ |
| incumbent — naive continuation | **−0.128%/event-day**, CI [−0.24, −0.03] | |
| structure (all events, flat when no trigger) | +0.012%/event-day, CI [−0.002, +0.027] | |
| per executed trade | +0.175%, win 68%, mean R +0.29 | |
| (b1) delta Sharpe | **+1.59**, date-block CI [+0.49, +2.66] | PASS |
| (b2) DSR @1547 | 1.000 | PASS |
| (c) folds | 12/15 | PASS |
| (g) instruments | **10/10** | PASS |
| (x) ex-2025 | +0.71 | PASS |
| (d) floor | +0.012%/event-day vs 0.05% | **BELOW** |
| break-even cost | 16.8 bp | descriptive |

## What it means, honestly

**The pass is the incumbent's failure, not the structure's success.** Every delta
component passes because *naive next-day continuation after a shock loses money* —
−0.13%/event-day with a CI that excludes zero, worse after down-shocks (−0.24%) than
up-shocks (−0.05%). That is next-day mean reversion, the same sign HYP-109 found at
five sessions. The structure trade itself is +0.012%/event-day with a CI that spans
zero: on its own, against cash, it is not distinguishable from nothing. It "beats the
incumbent" the way not-trading beats a losing trade.

Per executed trade the structure looks respectable (+0.175%, 68% win, R +0.29 on 102
trades) — but it fires on one shock in eight, so the strategy is flat 87% of the time
and its yield per event-day is a quarter of the floor. And the sample is thin: ~100
trades on ~95 dates.

**Confluence is untestable here** — 69 of 102 trades already carry ≥3 conditions; the
≤1 bucket has three trades. The conditions are so common after a shock that they don't
partition. Not a story, not monotonic: no information.

## What it says about the scoped question

Nothing about 2020–2022. It does say the path forms rarely (13%), that when it forms it
resolves to target more often than stop (55 vs 20), and that the naive momentum trade
is the wrong side the next day. The lead is the *reversal*, again — and it is a
directional lead that HYP-109 has already paid a kill for at five sessions. Whether the
one-session version survives its own prereg is a separate question with a 1548+ trial
count.

## Constraints honoured

One hypothesis, one run, no parameter changed after the result, reported as sealed.
Structure-vs-cash and up/down split are descriptive, computed after, and carry no weight.


---
**CORRECTION (2026-09-03, found by HYP-114):** every fade figure above was computed as `−naive_net`, which added the 3 bp cost back as a gain — overstated by 0.06%/event-day. Corrected HYP-111 fade: +0.087%/event-day, CI [+0.004, +0.174], 8/10, E = +0.098%/trade. HYP-114 then showed the fade absent on 2016–2019 and on 20 new ETFs. See `data/research/hyp114/VERDICT.md` and `research/EDGE_LEDGER.md`.
