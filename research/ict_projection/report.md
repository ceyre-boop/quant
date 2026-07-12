# ICT 90-Day Taken-Trade Projection (TICK-028)

> READ-ONLY research. Reads only `data/ledger/` veto shards and `data/decision_logs/` decisions. Nothing on the execution/exit path is imported or modified (shadow-freeze respected).

## Verdict

**ICT GENERATES enough setups (~94 logged/committed over 90 days, above the 20-45 band), so signal/veto frequency is NOT the bottleneck. But ~98.0% of committed setups EXPIRE unfilled: only ~2.1 trades would actually FILL over 90 days. As a source of ~30 EXECUTED prop-challenge trades, ICT is FAR BELOW unless the fill/expiry problem is fixed.**

- 90-day **logged/committed setups**: point **94** (95% bootstrap [72, 118], 80% [80, 110])
- 90-day **actually-filled trades**: point **2.1** (the number that would count toward a prop challenge)
- `near_30` (logged-setup basis): **ABOVE_RANGE** vs the 20-45 band  |  (filled basis: **BELOW_RANGE**)

## Window

- Analysis anchored to latest observed data date: **2026-07-12** (no wall-clock; deterministic).
- Trailing window: `2026-05-28` (exclusive) -> `2026-07-12` = 45 calendar days = **31 trading days**.

## 1. Dedup (the load-bearing step)

The scanner re-emits a full universe sweep every cycle, so a standing condition (e.g. "ADR exhausted") re-vetoes the same pair all day. Records are collapsed to unique `(date, pair, direction, veto_class)`, first-per-day.

- Trailing-window vetoes: **4051 raw -> 296 unique** = **13.7x** dedup.
- All-shards cross-check: 4408 raw -> 324 unique = 13.6x.
- Unique vetoed setups per trading day: **9.548**.
- Collapsed on (date, pair, direction, veto_class). Dedup factor ~13x confirms the scanner re-emits a full universe sweep every cycle.

## 2. Live veto-rate breakdown (trailing 45d, deduped)

- **ADR share: 45.3%**  |  **weekly-trend share: 6.8%**
- Recomputed live from deduped trailing-45d setups. Prior memory (55% ADR / 31% weekly) is STALE.

| veto class | unique setups | share |
|---|---:|---:|
| ADR | 134 | 45.3% |
| DISP_GATE | 64 | 21.6% |
| SCORE | 36 | 12.2% |
| OTHER | 29 | 9.8% |
| WEEKLY_TREND | 20 | 6.8% |
| TIMING | 8 | 2.7% |
| SESSION | 5 | 1.7% |

## 3. Daily taken base rate

- Raw ICT decision records in window: 222 -> unique `(date,pair,direction)` setups: **50** (with entry_level granularity: 66).
- **Mean daily taken (committed) setups: 1.4516** per trading day.
- Estimate A (direct): 1.4516/day.  Estimate B (veto-implied): 1.6129/day.  P(vetoed) at setup level: 85.5%.
- Estimate B = (total unique setups/day) x (1 - P(vetoed)), total = vetoed + taken. A and B nearly coincide (~1.5/day). They differ only because B's taken numerator counts all 50 unique setups while A's trading-day base rate uses the 45 on business days (the 5 weekend/non-business setups are excluded), both over 31 trading days. Since vetoed and taken setups partition the observed universe, B cannot diverge from A beyond this day-basis difference -- the real independent uncertainty is the FILL gap, not the veto split.

## 4. Fill gap (why committed != executed)

- Outcome distribution (window): {'EXPIRED': 221, 'LOSS': 1}
- Unique filled setups in window: **1**; fill rate: **2.0%**.
- EXPIRED = limit order placed at entry_level but never filled (r_realized=0). Nearly all committed ICT setups expire unfilled; only filled trades count toward a prop challenge.

## 5. 90-day projection

- Calendar->trading-day factor: 0.722222 (90 cal days -> **65 trading days**).
- Logged setups: point **94.4**; bootstrap over days (N=10000, seed 42): median 94.0, 80% [80.0, 110.0], 95% [72.0, 118.0].
- Sensitivity (+/-30% rate): [66.0, 122.7].
- Filled trades: point **2.1** -- Near-zero given the observed fill rate; the binding prop-challenge constraint.

## Caveats

- Only ~3 months of data (veto ledger starts 2026-05-24); a 45-day base is short and regime-specific.
- Assumes the pipeline stays frozen at current signal/veto/order config -- valid under the active shadow-freeze, but any gate change invalidates the projection.
- Selection gap #1: not every setup that passes vetoes becomes a logged decision.
- Selection gap #2 (dominant): only ~2.0% of committed ICT setups FILL in this window (the rest EXPIRE unfilled) -- 'committed setup' massively overstates 'executed trade'.
- FX weekend/non-business setups (5 in this window) are excluded from the trading-day base rate; they inflate Estimate B slightly vs Estimate A.
