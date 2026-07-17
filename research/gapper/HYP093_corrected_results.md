# HYP-093 Corrected Results — bias-free engine (2026-07-17)

Rerun of the 234-event forward year through `backtester/` on **1-minute bars**,
with gap-through stop fills, entry-bar spread, and the locate gate. This
supersedes both the biased +24.4% AND my earlier −19pt audit estimate (which
used a bad all-stops slippage proxy).

## Audit (auto)
- PASS: True · stops 79 · gap-through 5
  · trigger-fills 74 (94%)
- look-ahead violations: 0 · locate unknown-rate:
  100% (no IB snapshot exists for 2025-26 dates yet)
- regime Sharpe: {'H2-2025': 1.903, 'H1-2026': 1.123} · fragile: False

## The honest number is a BAND, set by the spread assumption
The stop-fill bias turns out SMALL at 1-min resolution: only
5 of 79 stops truly gap through the trigger;
the rest fill at −25%. The real correction is **transaction cost**, which the
original sim omitted entirely:

| Entry-bar spread charged | Annual | Sharpe |
|---|---|---|
| none (≈ old model) | +25.5% | 3.534 |
| ~1% median (0.45×) | +18.2% | 2.624 |
| full entry-bar range·0.5 (~2.2% median) | +9.8% | 1.494 |

Biased headline was **+24.4% / Sharpe 3.4**. Realistic is **≈+10–18% / Sharpe
1.5–2.5** depending on how much of the 1-min entry-bar range is true spread vs
momentum. Even the optimistic end is materially below the biased number, and
Sharpe roughly halves.

## Corrected prop MC (block bootstrap, 3% sizing, full spread)
- 90d ±8%: **PASS 64.7% / BUST 12.7%**
  (IID biased model said 78.5% / 1.6%)
- unlimited −10% DD: **PASS 88.9% / BUST 11.1%**
  (IID said 99.2%)
Block bootstrap + realistic edge roughly **halve** pass probability and
**multiply bust risk several-fold** — loss clustering was hidden by IID draws.

## Corrected EV grid
0 of 240 configs
survive family-wise correction (permutation date-shuffle, Bonferroni). Best raw
annual +53.8%, best Sharpe 2.87
— but none clears FWER. (Date-shuffle is a weak permutation at ~1 event/day, so
this is conservative; the point stands that no config is a discovered edge.)

## Bottom line
The strategy is still positive and still tradeable, but it is a **~+10–18%,
Sharpe ~2, ~10–13%-bust** strategy — not the +24%/3.4/near-zero-bust the biased
harness showed. Every prior prop/EV number this week was built on the optimistic
engine and should be read down accordingly.
