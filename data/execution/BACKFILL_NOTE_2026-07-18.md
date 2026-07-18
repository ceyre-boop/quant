# Execution-cost backfill — 2026-07-18

**This is an EXECUTION-COST MEASUREMENT, not a signal verdict.**
It re-prices already-adjudicated events with real NBBO quotes instead of a
bar-range proxy. It does not test the signal, does not adjudicate HYP-107, and
adds no multiplicity. Do not cite it as a verdict on the edge.

## What was run

`python -m execution.harness --backfill data/execution/backfill_sample_2026-07-18_events.json`

Event list built by `execution/build_backfill_events.py` (seed 20260718): 181
archived candidates from `data/research/gapper/per_candidate_enriched.csv` were
checked against the frozen HYP-107 filter (`og_max=0.577`, `logvol_max=5.854`,
`gap_floor=0.30`); 60 passed. 59 priced, 1 `SKIP_HALT`.

## Result (n=59)

| metric | real quotes | backtest model (same events, same module) |
|---|---|---|
| median gross | +1.5504% | — |
| **median spread cost** | **0.6113%** | model assumes 1–15% (docstring) and charges to an 8% round-trip cap |
| median net | +1.0172% | −3.4936% |
| win rate | 55.9% | — |
| `vs_backtest_delta` | **+4.5108 pp** | |

Wide quotes flagged: 0. All fills used real bid/ask; entry at ask, exit at bid.

## The one robust conclusion

**Quoted spread on these names is ~0.61% median, roughly an order of magnitude
below the 1–15% assumed in `backtester/realistic_fills.py`.** That is a property
of the tape, not of which events you sample, and it is the finding this harness
was built to produce. On a single hand-checked example (TGHL 2026-07-16 09:30:59)
the real quote was bid 1.37 / ask 1.38 = 0.73%.

Consequence: the fill model's `_half_spread()` — `k_range=0.30` on bar range,
clipped to `cap=0.08` — saturates at its cap on gapper opens and charges ~8%
round-trip against trades whose real cost is under 1%. Every gapper backtest
carrying that charge is biased pessimistic by several percentage points per trade.
That is far more consequential than the LULD halt bug corrected in the same commit
(~7bp; see `data/agent/param_change_log.jsonl`).

## What this does NOT establish — read before quoting the numbers

1. **This is NOT the sealed holdout.** These 59 events were drawn at random from
   the full archive (2025-07 → 2026-06), which OVERLAPS the HYP-107 mining window.
   The sealed holdout is n=57 with gross median +5.4% and win 70%. This sample
   shows gross +1.55% and win 55.9% — materially weaker, on a different and
   partly-mined sample. The two are not comparable and neither refutes the other.
2. **Nothing here adjudicates HYP-107.** Its ledger status remains
   `REAL_BUT_MARGINAL_EXECUTION_UNRESOLVED`. Resolving it requires the frozen
   holdout event list, not a random archive draw.
3. **`vs_backtest_delta` here measures MODEL ERROR, not edge.** The +4.51pp gap
   is mostly the spread overcharge above; it is a statement about
   `realistic_fills`, not about whether the strategy makes money.
4. **No funding implication.** TICK-022 measured prop-funnel EV at 0.0 on every
   strategy×firm row; HYP-093 sits below half its constitutional floor. Nothing
   in this file changes either.

## Suggested follow-up

Recalibrate `SCENARIOS`/`_half_spread` in `backtester/realistic_fills.py` against
measured spreads rather than assumed ones — with a `param_change_log` entry, and
noting that the correction is UPWARD and therefore the direction where motivated
reasoning enters. Then re-run the sealed holdout with the corrected cost model
before any claim about HYP-107's viability.
