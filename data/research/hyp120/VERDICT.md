# HYP-120 — TimesFM 3.0 zero-shot on the desk's three questions

**VERDICT: CASCADE** (c3 pass; c1 FAIL; c2 FAIL). Sealed `894aafbfbb41c9a2` through the formal stage
(2 leaves, causal); verified before and after; ledger `ADJUDICATED`. One run. 50,182 daily windows
(19 series, 2016–2026) + 15,280 minute windows (7,640 HYP-119 state events + 7,640 matched non-state).
Zero-shot, MLX backend, ~400 forecasts/s on the M4 Pro.

| claim | result | bar |
|---|---|---|
| **(1) magnitude** — next-5-day RV from the model's quantile σ | beats trailing-21 RV (ΔQLIKE −0.045, CI [−0.070, −0.017]); **ties EWMA(0.94)** (ΔQLIKE +0.011, CI [−0.009, +0.035]); Spearman with realized: TimesFM 0.681 = EWMA 0.681 > RV21 0.661; σ̂/RV median 1.18 (over-forecasts by ~18%) | needed both → **FAIL** |
| **(2) direction** — sign of step-1 forecast | hit **51.45%** CI [50.8%, 52.1%]; per-series rank correlation **−0.009** CI [−0.016, −0.003]; cross-sectional daily IC +0.024 | needed both → **FAIL** |
| **(3) cascade** — 30-min σ vs realized 30-min range on the state events | Spearman **+0.685** vs the cascade z's **+0.001**; diff CI [+0.61, +0.76]; pooled state∪non-state +0.728; model's state/non-state σ ratio 1.46 vs realized 1.43 | **PASS** |

## What it means, in the order it matters

1. **The foundation model is a very good realized-volatility estimator and nothing more, here.** On
   daily data it lands exactly on EWMA (0.681 vs 0.681) — the 1994 RiskMetrics filter, one parameter.
   On minute data it ranks the coming 30-minute range at 0.73 — and a **descriptive, post-hoc check**
   (no verdict weight) shows plain trailing-30-minute realized vol does 0.73 pooled and 0.71 within
   state events, i.e. the same. TimesFM's quantile head has learned vol persistence, which is real,
   and which the desk already had for free.
2. **Direction: the 51.45% hit rate is the equity drift, not skill.** The sign of the point forecast
   inherits the sample mean of the context; ETFs drifted up 2016–2026, so "predict up" wins 51% of
   days. The rank correlation between forecast and outcome is *negative* and its CI excludes zero:
   larger forecasts were followed by smaller returns. A billion-series pretrained model, zero-shot,
   has no directional information on daily FX/ETF returns. That is the 119th direction null and the
   first from a frontier model.
3. **c3 passed as sealed — and the seal was too generous.** The comparator was the cascade *z*, which
   is a state trigger (saturated once the state fires: within-state Spearman 0.001), not a vol
   forecaster. Against the fair baseline — trailing minute RV — TimesFM ties. The sealed verdict
   stands as written; the honest reading is "TimesFM sees the blowup exactly as well as the last 30
   minutes' realized vol does." A fair follow-up would seal RV30 as the comparator (n_trials 1649+).
4. **Best of the three, as asked:** claim 3 by the sealed ladder; **claim 1 in substance** — it is the
   only place the model equals the best simple baseline on the thing that is actually predictable,
   and it over-forecasts by 18%, which is a calibration you could correct. Nowhere does it beat what
   a two-line EWMA already gives you.

## What to do with it
Use it as a vol input if you like the packaging; it will not change a number. Do not use it for
direction. Do not ship it (non-commercial weights). The result closes "a bigger model will find
direction" the way HYP-117's fitted models closed "a fitted model will."

## Constraints honoured
One run. Context, horizons, quantile rule, baselines and panel unchanged. The RV30 comparison is
labelled descriptive and post-hoc.
