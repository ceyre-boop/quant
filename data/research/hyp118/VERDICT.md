# HYP-118 — M6 (ranking + mechanical crash management) on the unseen EM universe, 1997–2026

**VERDICT: IC_ONLY.** Sealed `b185b5890d2f5120` before any EM row was read (data guard); verified before
and after; ledger `ADJUDICATED`. One run. 354 months, 15 positions (MXN ZAR BRL KRW INR × USD JPY EUR).

| line | ann | Sharpe | maxDD | worst month |
|---|---|---|---|---|
| plain EM carry, top-5 EW | +5.1% | 0.37 | −33.8% | −29.3% |
| M6 raw — ranking + risk parity, no leverage | +5.6% | **0.63** | **−24.4%** | −23.1% |
| **M6 managed** (sealed line) — + vol target 10%, cap 1.5×, VIX brake | +6.0% | 0.55 | **−34.6%** | −34.6% |
| M6b managed + drawdown boost | +8.0% | 0.64 | −34.6% | −34.6% |

| claim | result |
|---|---|
| c1 ranking IC | **+0.104, CI [+0.056, +0.150], 25/30 years positive** — PASS |
| c2 Sharpe vs plain carry | +0.18, CI [−0.16, +0.63] — FAIL |
| c3 drawdown ≤ ⅔ of plain | −34.6% vs −33.8% — FAIL |
| c4 permutation | p = 0.000 — PASS |

## What it means

1. **The ranking is real on a universe nobody here had seen.** Carry + momentum + value − vol ranks EM
   positions with IC 0.10, positive in 25 of 30 years, p < 0.001. Together with HYP-117 (G10, 1990–2005,
   IC 0.15) that is now two sealed, unseen samples. Cross-sectional characteristic ranking in FX is the
   one predictive result in this repo that has survived everything.
2. **It still does not turn into a Sharpe that beats plain carry** with a CI that clears zero — the
   same finding as G10. Stronger *ranking* is not stronger *return*; the premium is the carry.
3. **The crash management, as sealed, failed — and made it worse.** Vol-targeting with a 1.5× cap
   levered up after calm periods and was at full exposure into 1999 (−29% year) and the 2011/2017
   drawdowns; the "managed" line's worst month (−34.6%) is *larger* than the raw line's (−23.1%).
   Risk parity alone (M6 raw) cut the drawdown from −34% to −24% and lifted Sharpe to 0.63 — but that
   was not the sealed claim and is reported as descriptive. Lesson, written down: **volatility targeting
   in carry is leverage into the calm before the crash.** The map already said the top of the cycle is
   where carry looks safest.
4. **The drawdown question is answered by the universe, not the model.** EM carry pays +5–6%/yr with
   a −24 to −35% drawdown; G10 pays +1.7% with −33%. The premium scales with the crash you are
   underwriting. There is no construction here where the drawdown is small relative to the year.

## Compared with the numbers before it

| | G10 factor 2006–26 | G10 M1 top-5, holdout 1990–2005 | EM plain | **EM M6 raw** |
|---|---|---|---|---|
| ann | +1.9% | +6.8% | +5.1% | +5.6% |
| Sharpe | 0.23 | 0.80 | 0.37 | 0.63 |
| maxDD | −33.5% | −16.2% | −33.8% | −24.4% |
| IC | 0.04 | 0.15 | — | 0.10 |

## Constraints honoured
One run. The vol-target/cap/VIX rule was not revised after seeing that risk parity alone did better.
M6 raw is descriptive; a sealed test of "risk parity, unlevered" would be a new prereg (n_trials 1643+).
