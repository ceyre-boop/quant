# TICK-024 Impact Study — Honest Swap Costs (2026-07-28)

**Ask:** does correcting `SWAP_RATES_ANNUAL` (~9x understated, EURUSD SHORT sign flip —
`research/TICK-024_cost_measurement.md`) to the rate-differential-derived model
(`sovereign/forex/swap_model.py`, new file, committed) survive contact with the numbers?

**Method:** measured BEFORE on unmodified code, applied `research/TICK-024_staged_patch.diff`
in the working tree only, measured AFTER, then reverted. Nothing shipped — swap_model.py is
the only new committed file; the backtester is back to HEAD.

## Before → After

| Metric | Before (broken table) | After (honest costs) | Delta |
|---|---|---|---|
| Portfolio Sharpe (`prove.py`, 2015-2024, the 0.6886 anchor) | **0.6886** | **0.6452** | −0.043 |
| In-sample Sharpe (2015-2022) | 0.5765 | 0.5305 | −0.046 |
| OOS Sharpe (2023-2024, the 1.25 headline) | **1.2504** | **1.1919** | −0.059 |
| OOS 95% CI | [1.001, 1.500] | [0.948, 1.436] | lower bound drops below 1.0 |
| Decay ratio (OOS/IS) | 2.169 ROBUST | 2.247 ROBUST | still robust |
| Per-pair OOS: EURUSD | +1.278 | +1.148 | −0.130 (biggest single-pair mover — expected, sign-flip pair) |
| Per-pair OOS: GBPUSD | +1.437 | +1.355 | −0.082 |
| Per-pair OOS: USDJPY | +1.497 | +1.512 | +0.015 |
| Per-pair OOS: AUDUSD | +0.782 | +0.753 | −0.029 |
| Win rate (portfolio) | 48.4% | 48.7% | +0.3pt |
| Profit factor (portfolio) | 1.923 | 1.841 | −0.082 |
| Verdict (`prove.py`) | WEAK/REVIEW (target ≥1.5, viable ≥0.8) | WEAK/REVIEW | unchanged |
| Verdict (`holdout_validation_v014.py`) | ROBUST, deployable | ROBUST, deployable | unchanged |

Permutation p-value: not reproduced here — `prove.py` / `holdout_validation_v014.py` don't
compute it; the 2026-06-07 p<0.001 figure lives in a separate study not re-run for this task.

## Headline

**Edge SURVIVES honest costs — YES.** Every headline number moves down by roughly 4-8%
(the 9x-larger swap charges on GBP/USD/JPY/AUD costing more than the EURUSD SHORT credit
saves), but nothing flips sign or crosses a verdict threshold: portfolio Sharpe stays at
0.65 (still WEAK/REVIEW, same as before — this was never above the 1.5 target), OOS Sharpe
stays at 1.19 (still ROBUST, still "viable edge," per-pair signs unchanged), and decay ratio
actually improves slightly (2.25 vs 2.17). The one caveat: the OOS 95% CI lower bound drops
from 1.001 to 0.948 — it no longer clears 1.0, so "genuinely strong after costs" softens to
"still positive and significant" rather than a clean pass on that specific check.

## Discipline confirmed

- `sovereign/forex/swap_model.py` — new file, committed, imports clean standalone.
- `research/TICK-024_staged_patch.diff` — applied, measured, reverted via
  `git checkout -- sovereign/forex/forex_backtester.py`. Working tree diff on that file
  is empty; `import sovereign.forex.forex_backtester` succeeds post-revert.
- No commit of the applied patch. No push of a live cost change. Go/no-go is Colin's call.

## Recommendation

Numbers support applying TICK-024 — nothing regime-breaks, verdicts hold, and it fixes a
known ~9x measurement error plus a sign flip that was silently mispricing every EURUSD short.
Sign-off is still Colin's: this doc is the before/after he asked for, not an authorization.
