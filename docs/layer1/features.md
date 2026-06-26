# Layer 1 Directional Bias — Feature Specification (HYP-064)

> **Phase 1 (pre-registration) — documentation only. No feature is computed and no model is
> trained in this phase.** The machine-readable source of truth for windows is
> [`feature_windows.json`](./feature_windows.json), validated by
> `tests/test_feature_label_isolation.py`. This file adds the economic rationale and regime
> dependencies the JSON omits.

## The hard invariant (the fix for the tautology)

Every feature's computation window must end **strictly before** the label's forward window
begins. Offsets are in trading days relative to the decision day `t0` (0 = `t0` close).

- **Label:** `fwd_direction_5d = sign(close[t0+5] − close[t0])` → window `[+1, +5]` (robustness
  check at horizon 10).
- **Isolation:** for every feature, `computation_window.latest_offset ≤ 0 < +1` = label start.

This is the automated guard against the failure that killed Sovereign Core ML (label derived
from `hurst_short`, also a feature — `data/research/sovereign_core_verdict.md`, Finding 3) and
that is **still live in `layer1/bias_engine.py`** today (`y = rsi_14 > 50`, trained on `rsi_14`
— `bias_engine.py:57,61`). The isolation test flags exactly that case.

**Model design:** a pooled panel across `EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD` — global
regime features (shared across pairs on a day) plus pair-specific features; label is that pair's
forward direction. Not per-pair separate models. `feature_importance` from SHAP is correlation,
never causation.

---

## Interest rate structure
*Why: rate differentials and their term structure are the macro driver behind the validated
carry edge; the model's job is to recognize when that structure favors directional follow-through.*

| Feature | Source | Window | Freq | Rationale / regime dependency |
|---|---|---|---|---|
| `pair_2y_rate_diff` | `data_fetcher.py` | `[0,0]` | daily | Short-end policy differential — the carry's proximate driver. Decays when both CBs are on hold. |
| `pair_10y_rate_diff` | `data_fetcher.py` | `[0,0]` | daily | Long-end growth/inflation differential. |
| `pair_real_rate_diff` | `data_fetcher.py` | `[0,0]` | daily | Nominal − breakeven; real-yield advantage drives persistent flows. |
| `us_2s10s`, `us_2s5s` | `fetch_macro_cache.py` | `[0,0]` | daily | US curve slope — risk regime + recession signal that conditions USD direction. |
| `us_10y_mom_1m`, `us_2y_mom_1m` | `fetch_macro_cache.py` | `[-21,0]` | daily | Rate momentum — the carry edge only pays in rate-**trending** regimes (2022/2023 vs 2021/2024). |
| `us_breakeven_10y` | `fetch_macro_cache.py` | `[0,0]` | daily | Inflation expectations — separates real vs nominal moves. |
| `fed_funds_level` | `fetch_fred_economic.py` | `[0,0]` | monthly | Policy stance level. Monthly cadence → forward-filled; never interpolated across the label window. |

## Dollar regime
*Why: USD is one leg of every pair; the dollar's own trend/positioning dominates short-horizon direction.*

| Feature | Source | Window | Freq | Rationale |
|---|---|---|---|---|
| `dxy_level` | `dxy_engine.py` | `[0,0]` | daily | Broad USD level. |
| `dxy_mom_1m`, `dxy_mom_3m` | `dxy_engine.py` | `[-21,0]`,`[-63,0]` | daily | USD momentum across horizons. |
| `dxy_vs_200sma` | `dxy_engine.py` | `[-200,0]` | daily | USD trend regime. |
| `dxy_pos_52w` | `dxy_engine.py` | `[-252,0]` | daily | Position within the 1y range (mean-reversion vs breakout context). |
| `dxy_smile_regime` | `dxy_engine.py` | `[-63,0]` | daily | Dollar-smile state (risk-off USD strength vs growth USD strength) — regime-flips in crises. |

## Equity regime
*Why: risk-on/off is the second-order driver of JPY/AUD/CAD direction via carry appetite.*

| Feature | Source | Window | Freq | Rationale |
|---|---|---|---|---|
| `spy_vs_200sma` | `market_data.py` | `[-200,0]` | daily | Bull/bear regime (the HYP-027 lens). |
| `spy_mom_1m`, `spy_mom_3m` | `market_data.py` | `[-21,0]`,`[-63,0]` | daily | Equity momentum → carry appetite. |
| `vix_level` | `fetch_macro_cache.py` | `[0,0]` | daily | Risk gauge; carry unwinds when VIX spikes. |
| `vix_z_1y` | `fetch_macro_cache.py` | `[-252,0]` | daily | Normalized fear vs its own year. |
| `vix_term_9d_1m`, `vix_term_1m_3m` | `fetch_macro_cache.py` | `[0,0]` | daily | VIX term structure — backwardation flags acute stress (carry-unwind trigger). |

## Commodity regime
*Why: oil drives CAD/NOK; gold is the USD inverse; broad commodities proxy global growth.*

| Feature | Source | Window | Freq | Rationale |
|---|---|---|---|---|
| `oil_mom_1m`, `oil_mom_3m` | `harvest_daily_panel.py` | `[-21,0]`,`[-63,0]` | daily | CAD beta. |
| `gold_mom_1m` | `harvest_daily_panel.py` | `[-21,0]` | daily | USD-inverse / real-rate proxy. |
| `crb_mom_3m` | `harvest_daily_panel.py` | `[-63,0]` | daily | Broad commodity / global-growth proxy. |

## Carry environment (pair-specific)
*Why: the only validated edge in this data. These features describe the carry setup the model must agree with.*

| Feature | Source | Window | Freq | Rationale |
|---|---|---|---|---|
| `pair_rate_diff_mom_1m/3m/6m` | `data_fetcher.py` | `[-21,0]`/`[-63,0]`/`[-126,0]` | daily | Carry momentum — the validated signal's core. |
| `pair_carry_differential` | `macro_engine.py` | `[0,0]` | daily | Current carry level. |
| `pair_irp_z` | `macro_engine.py` | `[-252,0]` | daily | Interest-rate-parity deviation (FairValueModel). |
| `pair_ppp_z` | `macro_engine.py` | `[-1260,0]` | daily | PPP deviation (5y) — long-run anchor. |
| `pair_cycle_divergence` | `macro_engine.py` | `[-252,0]` | daily | Business-cycle phase divergence. |
| `pair_realized_vol_20d/60d` | `data_fetcher.py` | `[-20,0]`/`[-60,0]` | daily | Vol regime → carry risk. |
| `pair_atr_pct_14`, `pair_price_mom_20d/60d` | `discovery/features.py` | `[-14,0]`/`[-20,0]`/`[-60,0]` | daily | Trend/vol context (reused discovery features, look-ahead-free by construction). |

## Positioning (CFTC COT, pair-specific)
*Why: crowding marks the "race to the bottom" — Tenet 6. Extreme net spec positioning precedes reversals.*

| Feature | Source | Window | Freq | Rationale / dependency |
|---|---|---|---|---|
| `pair_cot_net_spec` | `cot_engine.py` | `[-5,0]` | weekly | Large-spec net position. **Weekly release (Tue data, Fri publish) lagged ≥3 trading days — window starts at −5 to respect publication lag and never peek.** |
| `pair_cot_z_156w` | `cot_engine.py` | `[-1092,0]` | weekly | 3y z-score of net position. |
| `pair_cot_percentile_1y` | `cot_engine.py` | `[-252,0]` | weekly | 1y percentile. |
| `pair_cot_extreme_flag` | `cot_engine.py` | `[-252,0]` | weekly | Top/bottom-decile crowding flag. |

## Calendar
*Why: scheduled events reprice rate expectations. Dates are publicly pre-announced — known at t0, not future market data.*

| Feature | Source | Window | Freq | Rationale |
|---|---|---|---|---|
| `day_of_week`, `month_of_year` | computed | `[0,0]` | daily | Seasonality. |
| `days_to_fomc`, `days_to_nfp`, `days_to_ecb` | `event_calendar.py` | `[0,0]` | daily | Distance to scheduled catalysts. Uses the **published schedule** (public at t0); not a look-ahead leak. FOMC dates are exact; NFP/ECB partly cadence-estimated (a data-quality caveat, not an isolation breach). |

## Cross-asset
| Feature | Source | Window | Freq | Rationale |
|---|---|---|---|---|
| `credit_hy_oas` | `fetch_macro_cache.py` | `[0,0]` | daily | Credit stress. |
| `risk_on_off_hyg_ief` | `fetch_macro_cache.py` | `[-21,0]` | daily | Junk-vs-treasury risk proxy. |
| `bond_equity_corr_63d` | `harvest_daily_panel.py` | `[-63,0]` | daily | Bond/equity correlation regime (flips in inflation vs growth shocks). |
| `nqes_regime_flag` | `lead_lag.py` | `[-5,0]` | daily | NQ/ES lead-lag regime. |

---

## Rejected features (windowing / tautology)

| Candidate | Why rejected |
|---|---|
| `hurst_regime` | **Tautology basis.** It was the label-derivation basis that made Sovereign Core ML a tautology (`sovereign_core_verdict.md` Finding 3, 0.998 self-consistency, p=0.164). Excluded as a direct input; permitted only as a downstream regime gate. |
| `rsi_gt_50_at_t0` as a label | The current `layer1/bias_engine.py` bug — the label is a function of a `t0` feature. The label must be **forward** price direction (`[+1,+N]`), never a same-bar transform. |
| any feature with `latest_offset > 0` | Reaches into the label window — forbidden by the isolation invariant; the test fails on it. |
| implied-vol features | No FX options chain in the system (VRP track is `DATA_INSUFFICIENT`). Excluded for lack of a clean source, not for windowing. |

## Summary
- **Final feature set:** 50 declared (26 global + 19 pair-specific + 5 calendar). **3 rejected** for
  windowing/tautology, 1 for missing data source.
- All 50 pass the isolation test (`computation_window.latest_offset ≤ 0 < +1`).
- Expected verdict prior (pre-registered): **NOT_SIGNIFICANT** — daily forex features were already
  found flat (discovery 28→0). This is a falsification test; the value is a clean kill, not a win.

## Phase 2 build status (2026-06-26) — `data/layer1/load_report.json` is authoritative
`scripts/build_layer1_dataset.py` → `features_v1.parquet` (11,720 rows × **41 features**,
MultiIndex [date,pair], 2015-01-01 … 2023-12-29) + `labels_v1.parquet` (fwd_direction_5d/10d,
label balance 0.504). **41/51 features built; the other 10 are reported LOUD, never zero-filled:**
- **COT (4):** `pair_cot_net_spec`, `pair_cot_z_156w`, `pair_cot_percentile_1y`,
  `pair_cot_extreme_flag` — `cot_engine` only fetches the single hardcoded **2024** file (holdout)
  and FX positioning is not in the disaggregated (commodity) report. **Needs a dedicated
  historical COT fetcher (per-year 2015-2023 legacy/TFF zips, survivorship-aware). Top follow-up.**
- **Macro internals (3):** `pair_irp_z`, `pair_ppp_z`, `pair_cycle_divergence` — `macro_engine`
  is snapshot (`score_pair`); historical series need a per-date replay (BMS cadence). Deferred.
- **`pair_10y_rate_diff`** — `get_pair_differentials` exposes one tenor; foreign 10y needs separate
  (monthly) FRED series. Deferred. **`pair_carry_differential`** — identical source to
  `pair_2y_rate_diff`; merged. **`pair_atr_pct_14`** — needs pair OHLC (only close fetched); deferred.
