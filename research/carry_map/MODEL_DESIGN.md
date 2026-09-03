# How to model "best single pair every month" — design, not a build

Target line from `ceiling.py`: **oracle, best single pair each month: +69%/yr, Sharpe 7.2, maxDD 0.**
This document says exactly what that number is, what a model of it can and cannot be, and how it
would be built and tested honestly if it is attempted. Written 2026-09-03 before any model is fit.

---

## 1. What the oracle line actually is

Each month the oracle chooses one of **90 positions** (45 G10 pairs × long/short) with perfect
knowledge of the coming month, always taking the largest absolute excess return.

Its monthly return is therefore the **expected maximum of 90 roughly-normal draws**:

    E[max_k |r_k|]  ≈  σ_cs · E[max of 90 half-normals]  ≈  σ_cs · 2.7

With cross-sectional pair volatility σ_cs ≈ 2.2%/month, that gives ≈ 5.7%/month ≈ 69%/yr — which is
what was measured. **The oracle is a statistic of dispersion, not of predictability.** Its Sharpe of
7.2 is the Sharpe of a max-statistic. It is the wall, and it moves with vol, not with skill.

## 2. The one equation that governs any model of it

Any real model produces a forecast score f_k for each of the 90 positions and picks the top. Its
skill is the **cross-sectional rank correlation** between forecast and realized return:

    IC_t = Spearman( f_k , r_k )   across the 90 positions in month t

For a Gaussian cross-section, the expected return of "pick the top-1 by forecast" is

    E[r_top1] ≈ IC · σ_cs · E[max of 90 normals]  ≈  IC · σ_cs · 2.5

and for the top-k, replace 2.5 with the mean of the top-k order statistics (≈1.7 for k=5, ≈1.4 for k=10).

| IC (monthly, sustained) | top-1 return/yr | who achieves it |
|---|---|---|
| 1.00 | ~69% | the oracle |
| 0.10 | ~6.6% | best published multi-factor FX models, in-sample |
| 0.05 | ~3.3% | published carry + momentum + value, out of sample |
| 0.03 | ~2.0% | the carry factor alone, which is what the map measured |
| 0.00 | 0 − costs | 116 hypotheses on daily direction |

**So the model is not "how do I reach 69%." It is "what IC can I sustain out of sample, and is it
above ~0.03."** Everything below is how to measure that honestly. Top-1 selection is also the
*worst* way to spend a small IC — the top-1 slot is dominated by noise; top-5 signal-weighted captures
~70% of the same IC at a third of the variance. A model of this line should never actually trade one pair.

## 3. Data — and there is more of it than the map used

FRED serves, free, on the existing key (verified 2026-09-03):

| series | what | from |
|---|---|---|
| DEXJPUS DEXUSUK DEXUSAL DEXUSNZ DEXCAUS DEXSZUS DEXSDUS DEXNOUS | daily spot vs USD | **1985** |
| DEXUSEU | EUR | 1999 (splice DEM via IRSTCI01DEM156N + DEXUSDM-style pre-1999 DEM spot, or start EUR 1999) |
| IRSTCI01{JP,AU,GB,NZ,CH,SE,NO,CA,DE,US}M156N | OECD immediate short rates, monthly | 1985–1990 |
| CPIAUCSL + OECD CPI (CPALTT01*) | inflation for real-rate and value features | 1985+ |
| VIXCLS | VIX | 1990 |

That is **~430 months (1990–2026) × 45 pairs ≈ 19,000 pair-months**, four carry crashes (1992 ERM,
1998 LTCM/JPY, 2008, 2020) instead of one. The map's single-crash problem is solved by data that
already exists.

**Sealed holdout, decided now: 1990-01 → 2005-12 is never read until the final test.** Development
happens on 2006–2026 (the modern regime, where the map already looked). Reverse-chronological
holdout is deliberate: the holdout contains the two crashes the model has never seen and cannot have
been tuned toward.

Retail reality goes in the return definition, not in a footnote:

    excess_k(t) = Δlog spot_k(t) + (i_high − i_low)/12 · (1 − h) − c

with h = broker swap haircut (TICK-024 measured OANDA swaps far from interbank; freeze h = 0.30 as a
conservative retail haircut and report h = 0) and c = 3 bp spread per leg per month.

## 4. Features — all computable from data through month t−1, standardized cross-sectionally

For each pair k = (a, b), each month, using only data ≤ t−1:

| block | feature | definition |
|---|---|---|
| carry | `carry` | (i_a − i_b) at t−1, % p.a. |
| carry dynamics | `carry_chg12` | carry(t−1) − carry(t−13) |
| momentum | `mom1`, `mom3`, `mom12` | trailing 1/3/12-month excess return of the pair |
| value | `value` | −(5-year change in real exchange rate a/b, CPI-adjusted) — Asness/Moskowitz/Pedersen |
| risk | `rv3` | trailing 3-month realized vol of the pair |
| risk | `dollar_beta` | 36-month beta of the pair to the USD index |
| state (common) | `factor_dd` | carry-factor drawdown at t−1 |
| state (common) | `spread_chg12` | top3−bottom3 rate spread 12-month change |
| state (common) | `vix`, `fed6` | VIX level; 6-month change in the US rate |

Every feature is **z-scored within the month across the 45 pairs** (cross-sectional demeaning removes
the time effect — the model forecasts *which* pair, not whether carry pays this month; the common
state variables enter only as interactions, e.g. carry × factor_dd). No full-sample quantiles
anywhere — the map's terciles were a leak and are not repeated. Rolling windows only.

Signed positions: each pair appears once with sign chosen by the *forecast*; the target is the
signed excess return of the long-a/short-b convention, and the model forecasts a signed quantity.

## 5. Model class — in order of what I would actually run

1. **Characteristic-score baseline (no fitting):** f = z(carry) + z(mom12) + z(value). The published
   three-factor FX model. Zero free parameters. If nothing beats it out of sample, nothing is learned.
2. **Panel ridge with time fixed effects:** r_k,t+1 = β·x_k,t + α_t + ε, β shrunk toward zero,
   refit on an expanding window every 12 months, minimum 120 training months. ~12 coefficients on
   ~10,000 rows — the only regime where a fit is defensible.
3. **Carry × state interactions (the map's hypothesis, as a model):** add carry·factor_dd,
   carry·spread_chg12, carry·fed6 to (2). This is HYP-117 in regression form; its coefficients are the
   test of whether the map's states have out-of-sample content.
4. **Gradient boosting with monotone constraints** (carry ↑, value ↑, mom12 ↑, rv ↓), depth ≤ 3,
   ≤ 200 trees, same expanding window. Only after 1–3; and only to answer "is there nonlinear IC."
   Anything deeper is the 77,016-scan trap with extra steps.

Not run: Markov-switching / HMM on the factor (four regimes in 430 months → it will fit the crashes
by name), LSTMs/transformers (19,000 rows), anything with a hyperparameter sweep that isn't declared.

## 6. From forecast to portfolio

- **Selection:** rank the 90 signed positions by f; hold the top-k, k frozen at 5 (declare; do not
  sweep). Report top-1, top-3, top-10 as descriptives only.
- **Weights:** proportional to rank score, dollar-neutral, **vol-scaled to 10% annualized** using
  trailing 36-month covariance (Ledoit-Wolf shrinkage; HYP-035/036 said RMT cleaning on 4–5 assets
  did nothing — with 45 pairs it is worth re-checking, as a descriptive).
- **Rebalance:** monthly. The map showed the hold horizon is irrelevant for carry; for momentum it
  matters — keep monthly and do not tune it.
- **Sizing at the account level:** exposure = vol-target / realized vol, capped at 2× notional on
  margin. No Kelly on an IC of 0.05 — the estimate's standard error is larger than the estimate.

## 7. Validation — the part that decides whether this is a result

| test | what it must show |
|---|---|
| **rank IC by year**, expanding-window OOS 2006–2026 | mean IC > 0 with a block-bootstrap CI (L = 6 months) that excludes 0; positive in ≥ 60% of years |
| **top-5 portfolio** vs the plain carry factor | Sharpe difference CI > 0 on jointly resampled months |
| **cross-sectional permutation null** | shuffle realized returns *across pairs within each month* (kills skill, preserves time structure); 10,000 draws; p < 0.05 |
| **CPCV** over months, 6/2, embargo 12 months | ≥ 12/15 folds with IC > 0 |
| **sealed holdout 1990–2005**, one run | IC > 0 and top-5 Sharpe > carry factor on the same window; the 1992 and 1998 crashes reported month by month |
| **DSR** | n_trials = 1558 + 76 (map) + every model config run; declared before the holdout |
| retail haircut h = 0.30, c = 3 bp | the top-5 line still positive after costs |

Verdict ladder (sealed before the holdout is read): MODEL_HOLDS (all rows) / IC_ONLY (IC > 0 but no
portfolio improvement after costs) / NULL. A NULL here closes cross-sectional FX prediction on free
data for this desk the way HYP-114 closed the fade.

## 8. Prior, stated now

IC ≈ 0.03–0.05 out of sample; top-5 Sharpe 0.5–0.8; the interactions (HYP-117) add nothing that
survives the holdout because the 1990–2005 crashes have different signatures (ERM was a peg break, not
a vol shock). Most likely failure: the value feature is the only one with holdout content and it is
already published. Expected ceiling for a real model of "the best pair each month": **~3–4%/yr
unlevered at Sharpe ~0.6 — i.e. 5% of the oracle**, which is the honest exchange rate between
hindsight and a forecast.

## 9. What would make it worth building

Only one thing: that a NULL is as valuable as a pass. The desk has never run a cross-sectional model
on 36 years of G10 with a sealed pre-1990s-style holdout; either outcome goes on the EDGE_LEDGER as
a finding about the whole asset class, not one signal. It is ~2 sessions of work and $0 of data.
