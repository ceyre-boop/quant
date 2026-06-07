# Sovereign Core (Clawd Trading) — Validation Gauntlet Verdict

**Date:** 2026-06-07
**Branch:** `master` (`master-ml-archive`) — archived, divergent ML line. Read-only worktree.
**Question:** Do the Sovereign Core's models have real, structural edge — the same gauntlet
the forex carry edge passed?

## VERDICT: NOT PROVEN — NO DEMONSTRATED EDGE. Gauntlet stops at the kill-gate.

The system fails the cheapest, most decisive test (costed permutation). Do not revive,
merge, or live-trade it on this evidence. Two independent findings, either of which is
disqualifying on its own:

---

### Finding 1 — The models have no demonstrable edge (the kill-gate)

Costed permutation test, clean frozen yfinance data, 7 liquid ETFs, in-sample 2015–2022,
1000 permutations, round-trip cost 30 bps:

| Metric | Value |
|--------|-------|
| REAL portfolio Sharpe (√n-weighted) | **−0.183** |
| Null mean Sharpe | −0.410 |
| Null 95th percentile | −0.053 |
| **p-value** | **0.164** |
| Total trades | 512 |

The real signal's timing + direction is **statistically indistinguishable from random
entries** fired at the same frequency through the same costed engine (p=0.164, needs <0.05).
In absolute terms it is **Sharpe-negative** — it loses money after costs. No single symbol
shows convincing edge (best DIA +0.31, worst XLF −1.03; the spread is noise, not skill).

This was run in **"signal mode"** — the model signal (router regime + specialist direction)
isolated from the sizing throttle, which is the most *generous* possible test of the models.
It still failed.

Method: `scripts/permutation_test_sovereign.py --mode signal --perms 1000`. Reproducible
from frozen dataset hash `ce8d80d3…`; zero live network calls.

### Finding 2 — The system structurally cannot trade cold (live-readiness blocker)

Run faithfully through the five blocking gates the README claims (`--mode live`), the system
trained on clean data emits **zero trades**. Gate tally over in-sample:

- Router/FLAT: 7400 · Specialist/NEUTRAL: 2666 · **Risk/KELLY_NEGATIVE_EV: 3313** ·
  Risk/ATR_GATE_BLOCKED: 89 · **WOULD_ENTER: 0**

Cause: `SovereignRiskEngine.compute()` applies a Hoeffding *lower-bound* on win-rate with a
cold `n_trades=20`, crushing the 0.55 prior to ~0.28 → every signal is negative-EV → sized to
zero. The "200 paper trades to go live" bar in the README is **unreachable**: the EV gate
blocks the very first trade. The system's own `SovereignBacktest` hits the identical wall.

### Finding 3 — The "91% Accuracy Regime Router" is a tautology (confirmed)

Measured router OOS accuracy: **0.998**. It is near-perfect because `_label_regime()` derives
training labels from `hurst_short`, which is also an input feature — and `csd_score`,
`hmm_state`, `adx` are hardcoded constants in the serving path (`_build_records_from_df`,
`_check_for_signals`). The router predicts a bucketed function of `hurst_short` from
`hurst_short`. The accuracy number measures self-consistency, not edge.

---

## Why the gauntlet stopped here

Staging was permutation-first (the delete-parts ethos): the permutation kill-gate is the
cheapest test that can falsify edge. It falsified it (p=0.164). Per the approved plan, we do
**not** build the walk-forward or holdout stages — there is no edge to characterize across
regimes. Benjamini-Hochberg correction is moot (a single hypothesis that already failed
uncorrected). The router-relabel contingency does **not** trigger: it required the specialist
signal to look non-random, and it does not.

## Data-source note (separate live-readiness blocker)

The system trades on Alpaca's free **IEX** feed, which was empirically shown to return gappy,
truncated history (SPY only from 2018-11 with ~28% of trading days missing; QQQ only from
mid-2020). The gauntlet used clean yfinance data precisely so this would not corrupt the edge
test — but the IEX gap problem is itself a blocker for any live deployment and would need a
paid SIP feed or a different data source to even be backtestable on its own terms.

## Bottom line

The Sovereign Core is a well-architected scaffold with **no validated edge inside it**, that
**cannot place a trade cold**, whose headline accuracy number is a measurement artifact, and
whose live data feed is unfit. It is not a candidate to turn on. If the validated overnight-QQQ
effect is ever confirmed (and shown orthogonal to carry), this MoE risk/execution scaffolding
*could* be repurposed to wrap it — but the wrapper would need: (1) the router relabeled to
forward returns or removed, (2) the cold-start EV throttle fixed, (3) a real data feed. That is
a rebuild justified by a proven edge, not a revival of this system.

## Artifacts
- `scripts/freeze_sovereign_dataset.py` → `data/cache/equity/*.parquet` + `manifest.json`
- `scripts/permutation_test_sovereign.py` → `data/research/permutation_test_sovereign.json`
- `scripts/_diag_gates.py` (gate-tally diagnostic, evidence for Findings 2 & 3)
