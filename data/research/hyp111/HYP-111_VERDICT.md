# HYP-111 — post-shock intraday retrace-then-continuation + next-session fade (2020-01 → 2026-07, as scoped)

**Primary (the path): INCONCLUSIVE** — sealed abort, 65 triggered trades < 100. The path fires on
**3.6%** of 1,788 events across the full window (12.8% on 2023+ was the regime, not the rule).
**Secondary (the fade): FADE_HOLDS.** Confluence: INCONCLUSIVE (≤1 bucket n=4).

Sealed `d3d5258285a74cf9`; hash verified before and after; ledger `ADJUDICATED`. Source: Alpaca
SIP (existing key, verified to 2016 live before sealing — no purchase). 1,788 events, 669 dates,
all ten instruments ≥ 99.4% complete sessions. One run.

## The fade — every pre-declared component

| | result | bar |
|---|---|---|
| mean fade, all events | **+0.147%/event-day**, date-block CI [+0.062, +0.233] | > 0 ✓ |
| instruments with mean > 0 | **10/10** | ≥ 7 ✓ |
| ex-2020 | +0.111%/event-day | > 0 ✓ |
| 2020 alone | **+0.367%**, CI [+0.050, +0.697], 93 dates | not refuted ✓ (it *helps*) |
| 2020-03 alone | **+0.619%**/event-day on 167 events; worst −5.58% (TLT 03-23) | descriptive |
| per year | 2020 .37 · 2021 .13 · 2022 .03 · 2023 .18 · 2024 .02 · 2025 .32 · 2026 .04 | every year ≥ 0 |
| expectancy | n=1,788 · W 0.540 · avg win +1.064% · avg loss −0.904% · **E = +0.158%/trade** · payoff 1.18 | |
| down-shocks / up-shocks | +0.173% (W .56, n 960) / +0.142% (W .51, n 828) | |
| worst 5 | −5.6, −5.5, −5.4, −5.1, −5.1% (four of five in March 2020) | |

The primary's delta components also passed on the full window (Sharpe +1.37 CI [+0.65, +2.07],
**15/15 folds**, 10/10, ex-2020 +1.18) — but they are the same fact (the incumbent loses) and the
primary verdict is the abort, as sealed.

## What it means

**My prior was wrong on the part that mattered.** I sealed "FADE_FAILS on the 2020 component —
liquidity provision gets paid in drawdown in March 2020." March 2020 was the *best* month in
the sample for the fade: +0.62% per event-day, and the year's CI clears zero on its own. The
crash days that continued (the worst five) are real and cost 5% each, but the reversal days
around them paid more. The operator's stated prior was CONFIRMED for the path; the path is
too rare to adjudicate, but the fade it sits on is the thing he was describing.

**It is a small, real, six-year, ten-instrument effect that survives the one regime it had not
seen.** It is not improvable by size (HYP-113), by path (this), or by confluence. It is not
priced in the vol surface in a way a retail trader can buy (HYP-112). It is the golden-rule
shape: no speed, no information, no leverage, and it *is* the crowd's wrong side.

**What it is not:** a living at $2k. +0.158%/trade on ~2.7 instruments per event-date, ~107
event-dates a year. On deployed capital that is ~15%/yr gross of 3 bp, with a −5% single-day
tail that arrives in clusters. The thin years (2022 +0.03, 2024 +0.02) are real too.

## What this closes and what it opens

Closed: post-shock at daily resolution (HYP-109), overnight partition (HYP-110), the retrace
path as a *selection* (too rare), magnitude as an instrument (HYP-112), size as a filter
(HYP-113). Open, and now the only thread: **how to deploy the unconditional fade** — universe
width, per-deployed-capital sizing, and an exit that isn't "15:55" (HYP-059/060 said time exits
beat stops; nothing here tested an exit). Each of those is a new prereg with the floor
denominator sealed in writing first (HYP-113 `floor_note_standing`).

## Constraints honoured

One run. HYP-111a's sub-window result was declared known before sealing and shaped only the
secondary claim. No parameter moved. The forward log (`fade_forward_log.py`) continues on the
same rule.
