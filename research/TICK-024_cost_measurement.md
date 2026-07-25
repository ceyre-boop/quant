# TICK-024 — Swap/Carry Cost Model Measurement

**Status:** MEASUREMENT COMPLETE. Fix STAGED, NOT applied (execution-path gate, unlock 2026-07-28).
**Scope:** research/measurement only. No live file touched.

## Method

Pulled actual per-trade financing directly from OANDA for all 24 trades in
`data/ledger/oanda_fills.json` via `GET /v3/accounts/{id}/trades/{tradeID}`
(read-only, practice account) — the `financing` field is OANDA's realized,
cumulative financing charge for that trade in USD, ground truth, not a rate
snapshot. Compared each trade's actual financing to what
`sovereign/forex/forex_backtester.py::_apply_costs` would have modeled for the
same pair/side/units/hold-days using the live `SWAP_RATES_ANNUAL` table.

Notional was converted to USD per OANDA's base/quote convention (`USD_XXX`:
units already USD; `XXX_USD`: units × price; cross pairs approximated — see
caveat below). Script: ad hoc, not checked in (uses only the existing
read-only `research/swap_calibration.py` pattern + the `/trades/{id}` OANDA
endpoint); reproducible from `data/ledger/oanda_fills.json` trade IDs.

## Result — position by position, the 4 live v015 pairs

| Pair/side | n trades | Actual $ (OANDA) | Model $ (SWAP_RATES_ANNUAL) | ratio actual/model |
|---|---|---|---|---|
| EUR_USD LONG | 1 | -0.0432 | -0.0043 | **10.0x** |
| EUR_USD SHORT | 3 | **+6.2362** | -0.5650 | **SIGN FLIP** (model predicts a charge; broker actually paid a credit on 2/3 trades) |
| GBP_USD LONG | 2 | -2.4881 | -0.3515 | **7.1x** |
| AUD_USD LONG | 2 | -0.3696 | -0.0310 | **11.9x** |
| USD_JPY LONG | 3 | +11.7451 | +1.3151 | **8.9x** |
| USD_JPY SHORT | 1 | -1.0548 | -0.1918 | **5.5x** |

**Median understatement across the 4 live pairs: ~9x** (range 5.5x–11.9x) —
confirms the "~10x" figure in memory/ticket text is not a rough guess, it's
what the ledger shows.

**EUR_USD SHORT sign flip CONFIRMED**: the model books every short-EUR trade
as a financing cost (-0.10%/yr in the table). OANDA actually credits it
(+6.24 USD net across 3 trades, two of which individually flipped sign vs
the model's prediction). This matches the trade-227 anchor already on record
(+1.1122 USD/8 days ≈ +0.42%/yr credit) and is independently reproduced here
across the full trade history, not just that one anchor trade.

AUD_NZD (n=4, not a live v015 pair — excluded under HYP-045) showed much
larger ratios (23x–114x), but its USD-notional conversion here is a rough
AUD_USD≈0.66 approximation, not exact — treat as directional only, it does
not change the live-pair conclusion above and AUD_NZD is out of scope for
any live-table fix.

USD_CAD (n=8, all sub-hour test-probe trades from 2026-06-30/07-02 per
[[project_invariant_guard]] triage) had zero financing on both sides — no
signal either way, excluded from the aggregate.

## Cross-check against the current OANDA rate snapshot

`data/research/swap_calibration.json` (2026-07-12, `research/swap_calibration.py`,
still read-only/unapplied) independently shows the same pattern comparing
today's quoted OANDA financing rates to the model table:

| Pair | Side | OANDA annual | Model annual | Sign mismatch |
|---|---|---|---|---|
| EURUSD | SHORT | +0.0042 | -0.0010 | **YES** |
| EURUSD | LONG | -0.0245 | -0.0015 | 16.3x |
| GBPUSD | LONG | -0.0099 | -0.0012 | 8.3x |
| GBPUSD | SHORT | -0.0104 | -0.0008 | 13x |
| USDJPY | LONG | +0.0179 | +0.0020 | 9.0x |
| USDJPY | SHORT | -0.0382 | -0.0035 | 10.9x |
| AUDUSD | LONG | -0.0047 | -0.0008 | 5.9x |
| AUDUSD | SHORT | -0.0160 | -0.0012 | 13.3x |

Two independent measurements (realized ledger financing vs. a live rate
snapshot) agree: understatement is consistently in the 6x-16x band (center
~9-10x), and the EURUSD SHORT sign flip is real, not a one-trade anomaly.

## The cascade — every downstream figure that is cost-sensitive

`SWAP_RATES_ANNUAL` → `_apply_costs` (forex_backtester.py) is read by every
backtest run through that class. Correcting it will re-baseline:

- **v015 canonical reconcile anchor, Sharpe 0.6886** (the number every study
  gates against)
- **OOS costed Sharpe 1.25** (2026-06-07 headline, decay 2.17 ROBUST) and the
  **full-decade Sharpe 0.69** — both computed with the broken table
- **HYP-045 CONFIRMED verdict** (AUDNZD exclusion, OOS Sharpe 1.08, CI [0.84,
  1.32]) — the exclusion logic itself is probably still sound (AUDNZD's
  problem is RBA/RBNZ correlation, not swap cost), but the reported Sharpe
  numbers will shift
- Any exit-research study gated against the 0.6886 anchor: HYP-066 (exit
  regime conditioning), HYP-067 (exit policy evolution GA), the exit config
  sweep (180 configs) — all NOT_SIGNIFICANT/KILLED verdicts were reached
  under the broken cost model; re-running with corrected costs could in
  principle change a verdict, though the effect sizes there were large
  enough (holdout -0.401 vs in-sample 1.102 for HYP-067) that a ~9x swap
  correction is unlikely to flip them
- `sovereign/discovery/data_adapter.py` and `sovereign/layer1/meta_label_builder.py`
  — discovery-pipeline consumers of the same table
- `scripts/run_hypothesis.py`, `scripts/audit_hypothesis_ledger.py`,
  `scripts/validate_v007_hold.py` — hypothesis-ledger infra that reads costed
  backtests
- **NOT affected**: `research/tsmom_hyp091/*` (HYP-091, already sealed
  NOT_SIGNIFICANT) — it already uses a correct rate-differential-derived
  financing model, sidestepping the broken table by design (see
  `research/tsmom_hyp091/financing.py`)

**Direction is not obvious without re-running.** EUR_USD short financing
flips from a cost to a credit (helps), while GBP/AUD/JPY costs get ~7-12x
larger (hurts, more realistic drag). Net effect on the live 4-pair blended
Sharpe is unknown until the impact study runs — that is explicitly why
TICK-024's acceptance criteria require an impact study *before* any table
change, and why this ticket does not attempt to predict the after-Sharpe.

## What still needs live data (stop conditions hit)

None — the OANDA trades endpoint served every trade ID in
`data/ledger/oanda_fills.json` (24/24, 0 errors). No missing credentials, no
silent mocking needed.
