# Political-Alpha Summary Report — HYP-085

Generated 2026-07-08T21:44:29Z · pre-registration
`data/research/preregister/HYP-085_political_alpha_trump_events.json` (hash-locked
BEFORE data collection) · governing spec: vault `Political-Alpha-Claude-Code-Spec.md`.

## Verdict

**H0 NOT REJECTED** (p = 0.3637 ≥ 0.05) — the pre-registered prior (NOT_SIGNIFICANT). A null result is a real, publishable result.

## The three pre-registered tests (and no others)

**1. Normality / directional skew (pre-announcement returns).**
Pooled standardized pre-window returns (2 trading days before each event, n=446):
Shapiro–Wilk W=0.89247, p=0.0 — normality not rejected.
Direction-aligned skew = 1.544638 (positive = pre-drift toward the eventual
announcement direction). See `normality_plot.png` (left panel).

**2. SD exceedance (primary decision statistic).**
Observed: 25/223 evaluable event rows (11.21%) exceeded ±2σ of the
trailing 60-day SD on the event day or the day after. References: normal-theory two-day
baseline ≈ 8.89%; placebo-null mean = 10.30% (σ = 2.08%, p95 = 13.90%).

**3. Bootstrap null (decides the hypothesis).**
10,000 statement-level placebo sets (seed 42), identical mapping and big-move
rule, excluding ±5 trading days around real events and scheduled FOMC/CPI/NFP dates:
**p = 0.3637** (one-sided, `(n_ge+1)/(N+1)`). See `normality_plot.png` (right panel).

## Catalog

168 qualifying statements → 223 event×instrument rows
(223 evaluable). By source: {'whitehouse': 104, 'federal_register': 62, 'truth_social': 57}.
Statements in the null: 168 (skipped, no eligible placebo dates: 0).

## Positioning overlay (descriptive — carries no p-value)

manipulation_signal = post big-move AND pre-announcement rr25/put-call-volume moved
directionally (T-48h→T-0, FXE proxy for forex rows): **13** of 179 rows with
positioning data available.

## Data gaps (recorded, never fabricated)

Phase 2: none
Phase 3: {'rr25_missing_pre_window': 44}

## Method notes

- Hourly bars NOT used — daily T0/T+1 mapping per spec §6 (the stated default).
- Estimation window T-252→T-10, mean-adjusted model; big-move yardstick = trailing 60d SD
  (shifted; never includes the tested day). These are distinct by design (spec §6).
- Power: with ~223 rows against a ~10% null rate, only large effects are
  detectable; a null here does not prove absence — it bounds the effect size at this N.
- No BH/permutation/CAR were run (spec §10 — deliberately excluded from this build).
