# VALIDATION — HYP-064 (meta-labeling secondary model)

**Verdict: NOT_SIGNIFICANT**  ·  pre-reg hash `d22ae88f`  ·  train n=287 / val n=121  ·  holdout 2024+ UNTOUCHED

> Labels from the SIMULATED exit machine (strict_mode=False). Mandatory retrain when L2 is live.

## Statistical bar

- val ROC-AUC (best): **0.5325** vs bar 0.55 → FAIL
- best permutation p: **0.2907** · any BH-survives: False · deflated Sharpe: 3.9085
- **bar FAIL**

## Economic bar

- mean-R delta (meta-gated − carry-alone): **-0.0105** vs bar +0.10
- retention: **0.653** vs bar 0.50 · bootstrap Sharpe-diff p: **0.5227** · τ=0.3605
- **bar FAIL**

## Ablation (UNCONDITIONAL — independence from carry)

- gated-Sharpe drop when carry-overlap features removed: **-11.7%**
- REDUNDANT with carry (<5% drop) — carry-overlap features add ~nothing independent

## Calibration

- Brier 0.248 · ECE 0.0301

## SHAP top-5 (XGBoost pred_contribs — CORRELATION, NOT CAUSATION)

- `pair_real_rate_diff` — 0.50168  ⚠️carry-overlap
- `pair_price_mom_60d` — 0.22647
- `nqes_regime_flag` — 0.18886
- `bond_equity_corr_63d` — 0.16127
- `pair_rate_diff_mom_1m` — 0.15376  ⚠️carry-overlap

## 12-config metrics

| depth | n_est | lr | val_auc | perm_p | BH |
|---|---|---|---|---|---|
| 2 | 200 | 0.03 | 0.5055 | 0.4595 | · |
| 2 | 200 | 0.1 | 0.5235 | 0.3506 | · |
| 2 | 400 | 0.03 | 0.5172 | 0.3766 | · |
| 2 | 400 | 0.1 | 0.5096 | 0.4346 | · |
| 3 | 200 | 0.03 | 0.5153 | 0.4016 | · |
| 3 | 200 | 0.1 | 0.5235 | 0.3546 | · |
| 3 | 400 | 0.03 | 0.5096 | 0.4436 | · |
| 3 | 400 | 0.1 | 0.5265 | 0.3247 | · |
| 4 | 200 | 0.03 | 0.5325 | 0.2907 | · |
| 4 | 200 | 0.1 | 0.5123 | 0.3936 | · |
| 4 | 400 | 0.03 | 0.5271 | 0.3287 | · |
| 4 | 400 | 0.1 | 0.5213 | 0.3417 | · |
