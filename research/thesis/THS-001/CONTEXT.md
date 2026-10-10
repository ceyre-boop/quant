# THS-001 — AI infrastructure: sellers of the compute bottleneck (margin account)

**Status: LIVE (retro-registered).** Sealed 2026-10-10, hash `868d510b…` (`data/thesis/preregister/THS-001.json`).
Spec `spec.json` here is the sealed body. Source account Public `5OR41765` (margin).

## The thesis (sealed)
Frontier-model demand is compute-bound; the money is paid to whoever sells the compute, power
and memory the hyperscalers must buy regardless of which model wins. Hold the sellers of the
bottleneck, not the users of it. Horizon 12 months. Benchmark SPY.

| leg | weight | entry 2026-08-28 |
|---|---|---|
| NVDA | 0.301 | 218.95 |
| NBIS | 0.215 | 207.07 |
| VRT | 0.162 | 259.20 |
| PLTR | 0.129 | 185.95 |
| MU | 0.107 | 925.33 |
| AMD | 0.085 | 471.46 |

**Invalidation:** basket drawdown ≥ 30% from running peak (mechanical), OR two consecutive
quarters in which ≥2 of MSFT/GOOGL/AMZN cut forward capex guidance (manual check, asked at
every scoring run). Nothing else is an exit.

## Scorecard (the sealed basket, held) — `output/scorecard_2026-10-09.json`
2026-08-28 → 2026-10-09 (42 d): basket **+7.3%**, SPY +1.4%, excess +5.8%, maxDD −9.2%. Holding.
1-month mark: basket +7.2% vs SPY −0.2%. Next mark: 3 months (2026-11-28).

## Reconciliation (what the account actually did) — `output/reconciliation_5OR41765.json`
Account since 2025-02-20: TWR **+54.0%** vs SPY +30.1%; net gain $1,009 on $6,583 net
contributed. The 2026-08-28 six-name basket: money-weighted **+22.7%** vs SPY +1.6% — NBIS, VRT,
MU, AMD were sold 2026-09-08 after 11 days (+7–13% each). That sale is the one deviation from
the hold contract; the scorecard above shows the basket would be +7.3% held, the account took
roughly the same gain earlier and sits in NVDA + PLTR + MCD (MCD is not part of any thesis).

Caveats recorded, not hidden: (1) a **+$2,003.73 external flow** is in the broker cash but
absent from the API history (`UNIDENTIFIED` row, 2026-10-01); (2) MWR +155% is inflated by a
near-zero opening base (one $93 NNE trade for the first 17 months) — the TWR is the honest
account number; (3) sub-basket TWR is undefined when the basket was fully liquidated mid-window,
so MWR is reported for baskets.

## What closes this folder
Score at 3/6/12 months (`alta thesis score THS-001`). At 12 months or invalidation: verdict
here, one line in `research/EDGE_LEDGER.md`, status CLOSED.
