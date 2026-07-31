# V4 PREREGISTRATION — Exit Geometry Family
**Locked: 2026-07-30, BEFORE any OOS data was examined.**

## Motivating mechanism (not mined)

Two independent prior findings point at the same lever:

1. **HYP-059 / V3 forensics (CONFIRMED):** trailing-stop exits firing on days 1–7
   have 0–9.3% win rate and −1.1 to −1.6R mean. Mechanism: the 1.25× ATR trail is
   calibrated for intraday momentum, but the macro-carry drift takes 5–20 days to
   develop. The trail fires on entry noise, before signal.

2. **RQ-REST-013 re-simulation (sealed):** among tested exit arms, wider trails
   monotonically outperformed — `trail_wide_2.0` (Sharpe 0.126) > `trail_wide_1.5`
   (0.119) > `trail_immediate_1.25` (0.098). Pure time exits were *worse* than the
   incumbent (`time5` 0.097, `time8` 0.089).

Both say: **the trail is firing too early and too tight.** V4 tests exactly that
family — delay activation, widen distance. Nothing else changes.

## What V4 explicitly does NOT do

V3's proposed fix — replace trailing-stop exits with the historical mean time-exit
return — is **not tradeable and not tested here.** It is a counterfactual accounting
substitution: those 71 trades were stopped *because* price moved against them, so
crediting them the average time-exit outcome assumes away the reason they lost.
RQ-REST-013's real re-simulation already contradicts it (time5 < trail_immediate).

## Preregistered grid (27 configs, fixed before OOS)

| Parameter | Values | Justification |
|-----------|--------|---------------|
| `k_stop`  | 2.0 (fixed) | disaster stop untouched — not the hypothesis |
| `k_trail` | 1.25, 1.75, 2.5 | 1.25 = incumbent; wider per RQ-REST-013 |
| `delay`   | 0, 5, 10 | 0 = incumbent; 5/10 = let drift establish per HYP-059 |
| `max_hold`| 8, 15, 21 | capture the 8–14d and 15–30d high-value buckets |

## Protocol

- **IS = entries 2015–2022** (n≈319). Policy selected here, by IS annualized Sharpe.
- **OOS = entries 2023–2024** (n≈92). Touched ONCE, after the config is locked.
- **Primary OOS = 2024 alone** (1 year, n≈43) per the stated requirement;
  2023–2024 reported as the extended check.
- **Selection cost:** 27 configs searched → Šidák-corrected significance threshold
  and a deflated-Sharpe note are mandatory. A nominal p<0.05 that does not survive
  correction is reported as NOT SIGNIFICANT.
- **Permutation test:** 10,000 shuffles of the exit-policy label across trades.

## Success criteria (declared in advance)

V4 supersedes the incumbent only if ALL hold:

1. OOS mean R > incumbent OOS mean R
2. OOS Sharpe > incumbent OOS Sharpe
3. Permutation p < 0.05 **after** multiple-testing correction
4. No degradation in max drawdown (R space) beyond 1.25× incumbent
5. IS→OOS Sharpe decay ratio > 0.5 (not a collapse)

**Expected verdict if the effect is not real: NOT_SIGNIFICANT.** Recording this
in advance so a null result is a valid outcome, not a failure to be re-run away.

## Known limitations (stated before results)

- Re-simulation engine reproduces recorded per-trade returns at r≈0.69, not 1.0.
  It does not model the 19% "reversal" exits (signal-based, not price-geometry).
  Arm-vs-arm comparison is valid; absolute levels are NOT comparable to the live log.
- Entries are frozen from the sealed v015 log. This tests exit policy only —
  it says nothing about entry quality.
- Carry accrual is included from real rate differentials, so longer holds are
  fairly credited. Round-trip cost 2bp, constant across arms.
