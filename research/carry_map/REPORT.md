# The 20-year G10 carry map — where the premium was paid, and when

**Status: MINING (Step 1). In-sample, 76 cells compared, no verdict.** This is the map Colin asked
for — "of the FX universe, what order and sequence of events is most profitable, so I know where to
put my risk" — built to see structure, so that ONE claim can be sealed and tested afterwards.
Script `research/carry_map/map.py`; data `research/carry_map/data.py` (yfinance spot 2006-06 →
2026-04, OECD short rates via FRED 2000+); outputs `data/research/carry_map/`.

## 1. The premium itself, unconditional, 20 years

Long the 3 highest-rate G10 currencies, short the 3 lowest, equal weight, monthly, USD included:

| | ann. return | Sharpe | hit | worst month | max DD |
|---|---|---|---|---|---|
| gross | **+1.9%** | **0.23** | 59% | −15.7% (2008-10) | **−33.5%** |
| net 5 bp/mo | +1.3% | 0.16 | | | −34.1% |
| long-only top-3 vs USD cash | +0.5% | 0.05 | 50% | | −34.9% |

By year: 2008 **−30%**, 2009 **+25%**, then mostly single digits either way; 2015 −8, 2016 +8, 2024 +7.

**Honest read first:** the naive 20-year G10 carry premium is small, crash-prone, and roughly zero if
you can only go long. "The market pays anyone willing to hold the risk" is true in the sense that the
premium exists; it is *not* true that holding it passively pays well — over 20 years it paid 1.9%/yr
for a 33% drawdown. The v015 book's 1.25 Sharpe (4 pairs, real rates + IRP, gates, 2015–2024) is a
much better number than this — which is either the value of its construction or the shortness of its
window; HYP-063 says it survives FDR, the rolling walk-forward says FRAGILE (avg OOS Sharpe 0.39; the
2021 −0.13 / 2022 +0.51 / 2023 +1.26 / 2024 −0.09 figures in CLAUDE.md are Sharpes, not returns).

## 2. Who pays and who doesn't (20-year contribution to the factor)

| currency | avg rate | side | contribution | read |
|---|---|---|---|---|
| **JPY** | 0.10% | short 162 mo | **+16.7%** | the funding currency; pays, with −45% pair drawdowns |
| **AUD** | 3.07% | long 170 mo | **+13.7%** | the investment currency |
| EUR | 0.95% | short 139 mo | +9.3% | funding leg that pays |
| SEK | 1.24% | short 111 mo | +8.0% | |
| NZD | 3.17% | long 215 mo | +7.2% | held longest, paid least per month |
| GBP / CAD | ~1.8% | long 48–58 mo | −1.6 / −2.0% | mid-rate longs lose |
| CHF | −0.03% | short 239 mo | −2.8% | **the funding currency that never paid** — safe-haven bid |
| **NOK** | 2.15% | long 126 mo | **−10.3%** | the rate trap: high yield, oil-beta, lost money |

Best pairs by Sharpe (all ≈0.3): AUD/CAD, AUD/SEK, USD/JPY, NZD/NOK, JPY/AUD (+4.5%/yr, −46% DD).
Worst: anything long CHF's counterpart, and USD/GBP (−0.33).

## 3. WHEN it pays — states known at the prior month-end (in-sample)

| state | pays | doesn't | n |
|---|---|---|---|
| **S5 factor drawdown** | **>10% below peak: +9.8%/yr, Sh 0.75, hit 70%** · at peak: +7.6%, Sh 1.29 (n=24) | 0–10% off peak: −0.8% | 43 / 24 / 171 |
| **S4 trailing-12m factor** | **negative: +4.5%, Sh 0.39, hit 67%** | positive: −1.7% | 54 / 105 |
| **S2 rate-spread 12m change** | **narrowing: +8.9%, Sh 1.11** | widening: −3.4% | 77 / 71 |
| S7 Fed 6-month direction | hold: +3.9%, Sh 0.50 | **cutting: −3.5%** | 139 / 49 |
| S6 USD 12m momentum | flat: +5.6%, Sh 0.83 | USD trending either way: −1% | 76 / 151 |
| S1 rate dispersion | low/mid: +2–4% | **high: −0.3%, worst −15.7%** | 80 each |
| S3 VIX | >25: +3.8% (the recovery months) | <15: +1.1% | 46 / 73 |

Sequences (mean cumulative factor return after the event):

| event | n | +3m | +6m | +12m | +24m |
|---|---|---|---|---|---|
| **5 worst carry months** (2008-09/10, 2010-05, 2012-05, 2020-03) | 5 | −3.8% | +1.4% | **+11.3%** | +11.2% |
| VIX crosses 30 | 11 | +0.5% | +2.0% | +5.1% | +7.5% |
| Fed first hike | 3 | −0.6% | −2.4% | −1.0% | +3.7% |
| Fed first cut | 4 | −3.7% | −5.9% | **−16.0%** | −3.8% |

Holding frequency is irrelevant (re-rank monthly vs yearly: Sharpe 0.23 vs 0.20).

## 4. The sequence, as the data draws it

S5, S4, S2, S3 and the worst-months path are **one fact seen from five sides**: carry is a
crash-and-recover premium. The money is made in the 12–24 months *after* the crash (2009 +25%), when
the factor is >10% off its peak, its trailing year is negative, high-rate central banks have cut
(spread narrowing), and VIX is still elevated. It is lost in the crash itself, and it bleeds when the
Fed is cutting into a slowdown, when the USD is trending, and when rate dispersion is at its widest —
i.e. at the *top* of the cycle, exactly when carry looks most attractive.

So "where to put your risk," as the 20 years actually draw it:
- **when:** after the crash, not before it — the entry is the drawdown, the exit is the Fed cutting;
- **what:** AUD/NZD funded in JPY/EUR/SEK; never NOK for yield, never CHF as funding, never mid-rate GBP/CAD;
- **how:** monthly is fine, yearly is fine — the hold does not matter, the *state* does;
- **what it costs:** a 30% year, roughly once a decade, and years of single digits in between.

## 5. What can be sealed — and what can't

Everything above is in-sample over 76 cells. The one candidate with a mechanism, a literature
(carry crashes and recoveries, Brunnermeier–Nagel–Pedersen), and the same signal from five cells is:

> **HYP-117 candidate: G10 carry conditioned on its own drawdown — hold the factor only when it is
> >10% below its trailing peak (or trailing-12m negative); flat otherwise.**

The honest problem: this map already used all 20 years. Options for an out-of-sample test, in order
of strength: (a) pair-level — the 45 pairs' own drawdown states, CPCV over months with a date-block
bootstrap, multiplicity declared at 1558 + 76; (b) a different universe — EM carry (free data is thin);
(c) forward only. Prior to declare before sealing: the effect is real but is mostly 2009 (n=43 months,
one crash cycle carries it) — most likely failure is "one event."

Nothing in this file is a result. It is the map Colin asked for, and the one place on it worth a seal.

## 6. The lookback ceiling (`ceiling.py`, `data/research/carry_map/ceiling.log`)

What the 20 years could have paid *with hindsight*, three tiers, all in-sample:

| tier | best line | ann | CAGR | Sharpe | maxDD |
|---|---|---|---|---|---|
| T0 always-on factor | — | +1.9% | +1.6% | 0.23 | −33.5% |
| T1 map rules as if known | rate spread narrowing (77 mo on) | +2.9% | +2.8% | **0.61** | −10.2% |
| T1 | off when USD trending (88 mo on) | +2.5% | +2.5% | 0.61 | −8.4% |
| T1 | COMBO (DD>10% or narrowing) & Fed not cutting, **2× notional when on** | **+5.4%** | +5.1% | 0.58 | −19.6% |
| T2 static hindsight | best 5 pairs EW | +3.3% | +3.2% | 0.54 | −20.9% |
| T2 | short JPY vs USD, 20 years | +3.4% | +3.0% | 0.37 | −37% |
| T3 oracle | perfect sign on the factor every month | +21% | +23% | 3.7 | 0 |
| T3 oracle | best single pair every month | +69% | +95% | 7.2 | 0 |

**Read:** even with every rule the map found applied in hindsight, unlevered G10 carry tops out
around **+3%/yr at Sharpe ~0.6**; 2× margin in the best state gets ~5%. The rules buy *drawdown*
(−33% → −10%) far more than they buy return. The gap to the oracle (Sharpe 3.7 for the factor's sign
alone) is the size of the prediction problem nobody here has solved. Stacking all five states
(31 months on) drops the return back to +0.9% — over-conditioning kills the premium.
