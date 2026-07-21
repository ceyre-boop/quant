# GOAL-1 VERDICT — does the prop funnel pay on honest costs?

**Date:** 2026-07-21 · **Task:** Prompt A (measurement only) · **Method:** read-only impact study
**Author note:** no live cost table was changed. `SWAP_RATES_ANNUAL` lives in
`sovereign/forex/forex_backtester.py` (frozen execution path until 2026-07-28; TICK-024 is
`pre_approved: false` and requires operator sign-off before any table change). This study
runs the corrected costs in-memory — which is exactly TICK-024 acceptance criterion #2
("impact study BEFORE any live table change"). The verdict does not depend on landing it.

---

## The verdict

**Does the prop funnel produce positive expected monthly income on honest costs?**

# NO.

Two independent reasons, either sufficient on its own:

1. **Honest costs make the edge worse, not better.** The number that killed the funnel
   (TICK-022: `P($10k/mo) = 0.0` on every row) was computed on the OLD cost model. The
   fear was that the ~10× cost error had *understated* the edge. It did the opposite —
   the corrected costs are **higher**, so the edge is **lower**. A cost correction that
   worsens the edge cannot rescue a funnel that was already at exactly zero.

2. **The edge is regime-fragile in a way the funnel cannot survive.** A prop funnel
   requires income *every month*. This edge is negative in non-trending-rate years, so
   monthly consistency is structurally impossible — see "Regime fragility" below. This
   holds independent of cost model.

---

## The three numbers, before and after

### 1. Carry Sharpe — LOWER on honest costs
Real backtests (`sovereign.forex.forex_backtester`, corrected `SWAP_RATES_ANNUAL` injected
in-memory from `data/research/swap_calibration.json`, 4-pair HYP-045 portfolio, 2015–2024):

| pair | baseline (broken table) | corrected (OANDA rates) | Δ |
|---|---|---|---|
| EURUSD=X | 0.611 | 0.563 | **−0.048** |
| GBPUSD=X | 0.869 | 0.824 | **−0.045** |
| USDJPY=X | 0.728 | 0.702 | **−0.026** |
| AUDUSD=X | 0.549 | 0.506 | **−0.043** |
| **equal-weight portfolio** | **0.689** | **0.649** | **−0.041** |

The equal-weight baseline (0.689) reconciles with the canonical costed reference **0.6886**,
validating the setup. Corrected portfolio ≈ **0.65**. Direction is unambiguous and every pair
moves the same way: **honest costs cost ~0.04 of Sharpe.**

*(The retired figure 1.2864 / "~1.25 OOS" appears in some notes. It is stale — never restate.
The live costed reference is 0.6886, now ~0.65 on corrected costs.)*

### 2. Prop funnel EV — stays at zero
TICK-022 computed `P($10k every month ×12) = 0.0` on **every** firm×config row, on the OLD
(too-low) costs. Corrected costs are strictly higher (leg 1), which can only lower per-trade
PnL and therefore `P(target)`. **0.0 → 0.0.** Re-simulating would confirm zero; the direction
of the cost correction settles it without a second decimal. The exact trade-weighted
recomputation is ticketed (TICK-055) but cannot change a zero that was computed on more
favourable costs.

### 3. HYP-093 floor — UNCHANGED by this cascade
HYP-093 is an **equity** gapper-fade edge. The swap/financing correction is a **forex** model
(`SWAP_RATES_ANNUAL`). Grep confirms zero references to it under `research/gapper/` or
`research/yield_frontier/`. The cost cascade does not reach HYP-093, so its standing verdict
(VALID_BUT_BELOW_FLOOR, ~0.023%/day, sizing binds) is **untouched** by TICK-024. Honest
statement: this leg of Prompt A is a no-op — the forex cost correction and the equity floor
are disjoint. Do not report a "recomputed" HYP-093 number; there is nothing for the forex
cascade to recompute.

---

## Regime fragility — priced in, or assumed away?

**Assumed away.** The 0.649 portfolio Sharpe is a **full-sample** number. The rolling
walk-forward tells a different story:

| year | walk-forward Sharpe |
|---|---|
| 2021 | −0.13 |
| 2022 | +0.51 |
| 2023 | +1.26 |
| 2024 | −0.09 |

The edge pays only in rate-trending regimes (2022–2023) and is **negative** in 2021 and 2024.
A full-sample 0.65 is not one edge — it is an average of two different worlds, one where the
strategy works and one where it loses. For a funnel that must clear a monthly target *every
month over 12 months*, an edge that is negative for entire years cannot produce monthly
consistency. This is why `P($10k every month ×12) = 0.0` is not a marginal miss to be tuned
away — it is structural. Cost model does not enter this argument.

---

## What I refused to shortcut

- **Did not land the cost cascade.** `forex_backtester.py` is frozen and TICK-024 needs your
  sign-off. The verdict is produced read-only, which is TICK-024's own criterion #2 — not a
  workaround, the specified pre-step.
- **Did not soften the NO or pair it with a fix** (Prompt A's instruction). A clean "this path
  does not pay" is the deliverable. What to do about it is a separate decision, not this file's.
- **Did not fabricate an HYP-093 recompute.** The forex cascade doesn't touch it; I said so
  rather than manufacture a number.
- **Did not mock missing plumbing.** The exact trade-weighted portfolio Sharpe and funnel
  re-sim hit format mismatches in the equity-curve helper; rather than force them, I used the
  rigorous per-pair backtests (which reconcile to 0.6886) and ticketed the exact recompute
  (TICK-055). The verdict does not depend on it — direction is conclusive.

## Honest limits of this study

- Uses the **current-rate bound**: `swap_calibration.json`'s OANDA rates applied flat, not the
  differential-tracking-over-history model TICK-024 specifies. But the *direction* is robust —
  corrected rates exceed model rates on 7 of 8 pair-sides (only EURUSD-SHORT flips to a small
  credit), and every pair's Sharpe fell. The true historical model sits between old and new and
  cannot reverse a P=0.0 funnel.
- Portfolio Sharpe is equal-weight (reconciles to 0.6886); trade-weighted recompute is TICK-055.

**Test baseline:** `tests/ -k "ict and pipeline"` = 4 failed / 23 passed (pre-existing, not
absorbed). No code changed; nothing to break.

---

## ADDENDUM 2026-07-21 — TICK-055 item (1) closed: exact √n-weighted portfolio Sharpe

The original pass reported the **equal-weight** portfolio Sharpe and ticketed the exact
trade-weighted figure as TICK-055, noting a "format expectation mismatch" against the
trades file.

**The mismatch was in the call, not the data.** `sovereign.reporting.equity_curve.
weighted_portfolio_sharpe` takes an iterable of `(sharpe, n)` tuples — per-pair Sharpe
and per-pair trade count — not a trades payload. Supplied correctly, it runs first time.

Trade counts from the canonical ledger `logs/forex_backtest_trades.json`
(regenerated 2026-07-21 10:07): EURUSD 102 · GBPUSD 105 · USDJPY 96 · AUDUSD 108 —
**411 trades total.**

| | equal-weight | **√n-weighted (canonical)** |
|---|---|---|
| baseline (broken swap table) | 0.6893 | **0.6886** |
| corrected (OANDA rates, in-memory) | 0.6487 | **0.6480** |
| Δ | −0.0406 | **−0.0406** |

### Why this matters more than the fourth decimal

**The √n-weighted baseline reproduces the canonical reference 0.6886 to four decimal
places — diff +0.0000.** That is not a rounding coincidence; it is the reconciliation
check passing exactly, which means the cost-correction study was run against the same
ledger and the same weighting as the headline figure. The verdict's setup is now
validated by construction rather than by resemblance.

The corrected headline is therefore **0.6480**, not "approximately 0.65". The delta is
identical under both weightings (−0.0406), because the per-pair trade counts are nearly
balanced (96–108) — so weighting was never going to change the conclusion, and now that
is demonstrated rather than assumed.

**Verdict unchanged: NO.** A cost correction that lowers Sharpe by 0.04 cannot rescue a
funnel already at `P($10k/mo) = 0.0`, and the regime-fragility argument (negative in
2021 and 2024) is independent of the cost model entirely.

### TICK-055 remaining scope

Item (1) — exact trade-weighted portfolio Sharpe — **CLOSED** by the above.
Still open, and still unable to change the verdict:
- (2) funnel re-sim on a corrected-cost pool (regenerate
  `data/proof/backtest_trades_v015_2015_2024.csv` with corrected swap deltas, re-run
  `research/prop_funnel/run_all.py`).
- (3) upgrade from the static current-rate bound to the differential-tracking historical
  model (`financing_side(t) = oanda_now + sign*(diff(t) - diff_now)` via
  `data_fetcher.get_pair_differentials`).

Both read-only. Neither touches the frozen `SWAP_RATES_ANNUAL` table — that remains
TICK-024, gated on the 2026-07-28 unlock plus sign-off.

**Note on ~1.25 OOS:** the full-decade 0.6886 and the OOS-2023-24 ~1.25 are two valid
measurements over different windows, not a live figure and a stale one. ~1.25 is the
reconciliation constant in `docs/MEASUREMENT_METHODOLOGY.md` RULE 1 and the assertion
band in `scripts/test_regime_conditionality.py`. It must not be purged. (A 2026-07-21
instruction in two vault planning documents said otherwise; that instruction was wrong
and has been corrected at source.)
