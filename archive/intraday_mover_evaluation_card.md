# Intraday Mover — Current-State Evaluation Card

*Alta Investments — discretionary sleeve · Week of July 13, 2026*
*Purpose: operationalize the continuation-vs-exhaustion read so it's consistent, loggable, and
Oracle-ready. This week is PAPER. The goal is the sample, not the P&L.*

> Core idea: we don't predict which stock moves. The daily top-movers list hands us names that
> already moved. Our only job is current-state evaluation — **is it still going, or is it done?** —
> and to take a slice of the ones with intact structure. This is constraint, not prediction.

---

## Daily routine (≈15 min, then monitor)

1. Pull the **daily top % movers** (Robinhood top movers, or Finviz/most-active gainers).
2. Take the **top ~10–15**. Note each name's **% move so far** and the **catalyst** (earnings,
   guidance, FDA, news, squeeze — if you can't find one in 60 sec, mark "unknown").
3. Run the **evaluation rubric** below on each. Score it. Get a call: **CONTINUE / DONE / SKIP.**
4. Only the **CONTINUE ≥ +3** names are tradeable. Everything else you log but don't touch.
5. Log the read *before* the outcome is known (this is what makes it honest data).
6. Mark the outcome at close. That's the whole loop.

---

## The rubric — v2 (rebuilt on the 1,475-day null)

The post-hoc scan killed the original structure-based rubric: VWAP, higher-lows, swing and candle
PA came back **flat** — no information on continuation. The signal lives in two places: **how far
it's already gone (extension)** and **what the catalyst is.** Score those; log the structure
signals but don't trade off them yet.

### Hard gates — apply first, before scoring
- **M&A / deal-price gap → EXCLUDE.** 16.7% continue, pinned at the deal price by arbs. Not a
  continuation name in either direction. Off the watchlist.
- **Already ≥100% by ~10:30 → NOT A LONG.** 65.9% reversed, median −12.5%. Never chase a double.
  (This is HYP-093, testing on the holdout now — treat as fade-watch only, no live short.)

### Primary score — this is the edge
| Signal | +2 | +1 | 0 | −2 |
|--------|----|----|---|----|
| **Extension by read-time** | 30–50% (only band with + median) | 15–30% | 50–100% | ≥100% (gated out above) |
| **Catalyst** | no headline before 10:30 (quiet runner) | ordinary single-name news | — | heavily-covered pop / FDA (sold into) |
| **Relative volume** | moderate | — | elevated | extreme (loud crowd fades hardest) |

**Call:** **+3 or higher → CONTINUE** (tradeable) · +1–2 → MIXED (skip) · ≤0 → DONE.

### Secondary — LOG ONLY, do not score
VWAP hold · higher-lows intact · prior-swing · candle PA. The scan says these are noise. Record
them anyway this week to test whether your discretionary read adds anything *beyond* extension +
catalyst. If it doesn't, we drop them for good.

---

## Execution rules (paper this week)

- **Only trade CONTINUE ≥ +3.**
- **Entry:** on a pullback that holds structure (into VWAP / FVG / discount) and *resumes* — or a
  clean reclaim. **Do not chase the vertical candle.** If it won't give you a pullback, you missed
  it; there's another mover tomorrow.
- **Stop:** just below your invalidation level (the swing low / VWAP you're keying off). If it
  breaks, your read was wrong — out, no averaging down.
- **Target — take a slice:** scale the first piece at **+1R** or once you've captured a preset
  chunk of the day's move; trail the rest by structure. **Flat by the close** — no overnight hold
  (gappers reverse, and not much new happens between open and close).
- **Max 2–3 names a day.** More than that and the reads get sloppy and the log gets noisy.

---

## Decision card — copy one per name

```
DATE:            TICKER:            TIME OF READ:
% MOVE AT SCAN:            CATALYST:
DIRECTION EVALUATED:  up / down

GATES:  M&A? (y→exclude): __    ≥100% by 10:30? (y→not a long): __
PRIMARY SCORE:
  extension band: __    catalyst: __    rel-volume: __
  TOTAL: ____     CALL:  CONTINUE / MIXED / DONE
SECONDARY (log only, don't score):  vwap: __  higher-lows: __  swing: __  PA: __

CONVICTION (1-5): __     ONE-LINE REASON:

--- if traded (paper) ---
ENTRY:        STOP:        FIRST TARGET:
EXIT PRICE / TIME:        EXIT REASON: (target / stop / time / structure break)
OUTCOME:  R = ____   |   % of the move captured = ____

--- post-mortem (fill at close) ---
DID IT CONTINUE AFTER YOUR READ?  yes / no / chopped
WHAT TELL WAS RIGHT OR WRONG:
```

---

## Week log — one row per read (traded or not)

| Date | Ticker | %move | Catalyst | Call | Traded? | Outcome (R / %) | Read correct? | Note |
|------|--------|-------|----------|------|---------|-----------------|---------------|------|
| 7/13 | | | | | | | | |
| 7/13 | | | | | | | | |
| 7/14 | | | | | | | | |
| 7/14 | | | | | | | | |
| 7/15 | | | | | | | | |
| 7/15 | | | | | | | | |
| 7/16 | | | | | | | | |
| 7/16 | | | | | | | | |
| 7/17 | | | | | | | | |
| 7/17 | | | | | | | | |

---

## What we're actually measuring

By Friday you'll have ~20–40 logged reads. Three questions the sample answers:

1. **Does the read work?** Of the CONTINUE calls, what fraction actually continued and paid?
2. **Which tells carry it?** Which of the six signals separates winners from traps? (That's what
   we'll weight when we automate.)
3. **Where do you get faked?** The DONE-that-kept-going and CONTINUE-that-died rows are the gold —
   they're what the Oracle learns from.

If the CONTINUE bucket is materially better than a coin flip across 20+ reads, we have something
real to automate and wire to `decision_logger` → `update_outcome` → Oracle. If it's a coin flip,
we found that out for the price of paper. Either way it's a win.
