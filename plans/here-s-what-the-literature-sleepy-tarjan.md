# HYP-090 TSMOM Diversification Study + ICT 90-Day Trade-Count Projection

**Tickets:** TICK-027 (TSMOM / HYP-091) · TICK-028 (ICT projection) — *file both in `tickets/backlog.md` as build step 0.*
**Concurrency note (2026-07-12):** original plan said HYP-090 / TICK-023-024; parallel sessions took all of them since the scouts ran. Corrected to the next free ids: **HYP-091** (090 = parallel "MODERN adaptive params" study), **TICK-027/028** (023–026 taken). HYP-089 still reserved for the options-footprint hunt.
**Status:** planned, not built. Plan-mode output — build is a separate pass (neither ticket is `pre_approved`).
**Branch:** sovereign-v2 · **Constraints:** read-only research; shadow/execution-path freeze stays intact.

---

## Context — why this exists

The operator received a literature summary on Time-Series Momentum (Moskowitz/Ooi/Pedersen 2012; AQR "Century of Evidence"; Koijen carry) and the well-documented fact that FX carry and momentum returns are near-uncorrelated — the property that makes *combining* them rational. The honest question is not whether TSMOM worked for 58 instruments in 1965–2009 (it did), but whether it adds a **real diversification benefit to the live v015 carry book, in our 4 pairs, in the post-2015 period, net of the costs we actually face**. Post-publication FX-specific decay is severe (Hutchinson 2022: carry/mom/value OOS Sharpe fell +0.39 → −0.32), and 2009–2019 was brutal for trend — so in-sample and pre-2010 numbers cannot be taken at face value.

This is **HYP-090**: pre-register a null, backtest TSMOM on the v015 pairs/period with inverse-vol sizing, and measure both standalone Sharpe (broken down by year, to expose whether all the return is the 2022 rate regime) and correlation with v015 carry. A pass is a *research result* (`VALID_EDGE`), never deployment authorization — promotion and live capital are a separate human step under `RISK_CONSTITUTION.md` Art. 6.

Run in parallel: **TICK-024**, a read-only 90-day ICT taken-trade projection, to sanity-check whether ICT is still the right prop-challenge vehicle (≈30 trades) at current signal/veto frequency — a timeline question, kept fully decoupled from any decision to rush TSMOM into deployment.

## Locked decisions
- **HYP-091** for TSMOM (the Phase-0 "verify unused" gate fired: HYP-090 was taken by a parallel session on 2026-07-11; HYP-089 stays reserved for the options-footprint hunt per `NEXT.md:42`).
- **Correlation frequency = MONTHLY**, locked as the primary pre-registered test (see Task 1 §Q1).
- **Financing = correct rate-differential-derived model** (operator decision 2026-07-12). See Cost netting below.
- **Deployment / live sizing / promotion-to-CONFIRMED = OUT OF SCOPE.** This pass yields `VALID_EDGE` or `NOT_SIGNIFICANT` only.
- ICT projection = committed, re-runnable script + short markdown report.

---

## TASK 1 — HYP-090 TSMOM (TICK-023)

### Package: `research/tsmom/` — mirror `research/prop_funnel/`
Reads sovereign **read-only**, writes **only** under `data/research/`. Do **not** use `research/political_alpha_v2/`'s AST-enforced full isolation — we legitimately need sovereign Sharpe/DSR utils and the v015 CSV. One git commit per phase: `[RESEARCH] tsmom Phase N: …`.

```
research/tsmom/
  preregister_tsmom.py   # Phase 0 — copy register/verify/_canonical_hash from preregister_hyp085.py
  feeds.py               # read-only loaders: yfinance prices + v015 monthly carry series
  signal.py              # 12-month sign + ex-ante EWMA vol
  backtest.py            # monthly rebalance, inverse-vol sizing, cost netting → monthly return series
  correlation.py         # monthly corr vs v015 (LOCK 0.5) + 50/50 combined-portfolio Sharpe
  gauntlet.py            # directional monthly-sign permutation + DSR + BH + holdout → verdict
  run_study.py           # orchestrates Phases 1–4
  report.py              # per-year Sharpe table + verdict summary
  verdict_to_ledger.py   # Phase 4 ledger update (backup-before-mutate)
  test_isolation.py      # asserts no writes outside data/research/tsmom; no ict/config imports
```
Sanctioned writes: `data/research/preregister/HYP-090_tsmom.json`; `data/research/tsmom/{backtest,correlation,gauntlet,results}.json`; two ledger touches in `data/agent/hypothesis_ledger.json` (Phase 0 PREREGISTERED, Phase 4 verdict — each auto-backs-up `.bak-<stamp>.json` first, per `preregister_hyp085.py:164`).

### The locked spec (exact formulas)
**Signal** (per pair, daily close `C_t` from yfinance `auto_adjust=True`):
- Rebalance = last trading day of each month.
- `sign_{i,m} = sign(C_t / C_{t−252} − 1)` — trailing 252-day spot return; ±1, held through month `m+1`.
- Lock the **raw spot-return** version (deterministic, zero external dependency; financing charged explicitly in the cost layer). An rf-subtracted variant (cumulative DGS3MO) is a **descriptive robustness check only**, not adjudicated.
- **Warm-up:** first signal needs 252 prior trading days → usable series ≈ **2016-mid → 2024** (~104 months). This shrinks OOS to ~24 months — state it explicitly.

**Inverse-vol sizing:**
- `sigma_daily = r_i.ewm(com=60).std()` at the rebalance close (no look-ahead — `r_t` excludes the forward month). Write fresh; no drop-in util (nearest convention `sovereign/layer1/feature_builder.py:175`).
- `sigma_ann = sigma_daily * sqrt(252)`; `w_{i,m} = sign_{i,m} * min(target_vol_ann / sigma_ann, LEV_CAP)`.
- Lock `target_vol_ann = 0.10`, `LEV_CAP = 2.0`. **Property to pre-register:** without the cap, Sharpe is invariant to `target_vol` (gross P&L, spread turnover, swap all scale linearly → constant cancels); the cap is the only thing making `target_vol` load-bearing. Report a with/without-cap robustness pair.
- Portfolio = equal-risk average across the 4 pairs: `R_m = (1/4) Σ_i r_{i,m}^net`.

**Cost netting** (spread from `sovereign/forex/forex_backtester.py:43-51`, imported read-only; financing modeled fresh — see below):
- Turnover: `cost = |w_m − w_{m−1}| * (SPREAD_COST[pair] + 2*SLIPPAGE_PER_SIDE)`.
- **Financing = correct rate-differential-derived model** (operator decision; TICK-024 proves `SWAP_RATES_ANNUAL` is ~10× too small + one sign flip, and it's Colin-gated so we do NOT use it for the primary). Model: **anchored differential-tracking swap** — `financing_side(t) = oanda_side_now + s·(diff(t) − diff_now)`, `s = +1` long / `−1` short, where `diff(t)` is the FRED policy-rate differential from `sovereign/forex/data_fetcher.py::get_pair_differentials` (:187) and `oanda_side_now` is the 2026 OANDA snapshot in `data/research/swap_calibration.json` (built by TICK-024's `research/swap_calibration.py`). This reproduces the OANDA snapshot AND the trade-227 anchor at t=now, varies correctly across 2015–2024 (captures the 2022 USDJPY-carry blowout the per-year table targets), and touches no Colin-gated table. Accrue daily: `swap_day = w_m · financing_side(t)/365` per calendar day held.
- `r_{i,m}^net = w_m * (C_end/C_start − 1) + Σ_days swap_day − cost`.
- **Robustness legs (descriptive):** (i) broken `SWAP_RATES_ANNUAL` model — the apples-to-apples cross-check since v015's CSV was costed with it; (ii) price-only. **Primary-correlation caveat:** correct-financing TSMOM is correlated against the v015 CSV, which was costed with the *broken* (~10× understated) swap — a financing-regime mismatch. Leg (i) is the mismatch-free comparison; report the spread between (i) and primary so the mismatch's effect on ρ is visible.

**Sharpe** — reuse `sovereign/reporting/equity_curve.py::_sharpe` (:84) for v015-comparability (annualizes by `sqrt(n/years)`):
- Full: `_sharpe(R_m, n_years≈8.5)`. Per-calendar-year: `_sharpe(R_m in year Y, n_years=1)` for 2016…2024 — **the required breakdown**.
- IS = 2016→2022; **OOS/holdout = 2023–2024** (aligns with `data/risk/oos_trades_2023_2024.json` + `run_hypothesis.py` IS/OOS convention).

### Locked null + criteria (Phase 0)
- **Diversification thesis FAILS if EITHER:** (a) standalone **OOS Sharpe ≤ 0**, OR (b) monthly **corr with v015 carry > 0.5**.
- **VALID_EDGE candidate (no live capital)** requires ALL: OOS Sharpe > 0 ∧ monthly ρ ≤ 0.5 (null not triggered) ∧ `perm_p < 0.05` ∧ `dsr_prob > 0.95` ∧ `bh_survives` ∧ `holdout_sharpe > 0`.
- `prior_expectation = NOT_SIGNIFICANT`.

### Resolved design questions
- **Q1 Correlation → MONTHLY (locked).** v015 has no native daily series; expanding per-trade `pnl_pct` over `hold_days` manufactures serial correlation and invalidates both ρ and its CI. Build the v015 monthly series by summing `risk_adjusted_pnl_pct` per **exit-month** (the field v015's own Sharpe uses) and correlate vs TSMOM `R_m`. Reuse the *shape* of `sovereign/research/vrp/validator.py::_corr` (:55) / `_analyze` (:106) — full + rolling-60d + a 2022 rate-shock window — but **adjudicate on 0.5**, not VRP's native 0.35. Report the CI honestly: ~96 overlaps → SE ≈ 0.10, so a point estimate near 0.5 is fragile. **Confirmatory (not adjudicated):** does a 50/50 monthly combined portfolio Sharpe exceed `max(carry, tsmom)`? — the direct diversification payoff, far more robust than ρ near the boundary. Daily-expansion ρ = caveated secondary only.
- **Q3 Harness → standalone backtest feeding `gate.py` *primitives*.** `scripts/run_hypothesis.py` only accepts ForexBacktester config knobs — structurally can't express a sign+inverse-vol rule. The full `Gate.evaluate()` models discrete fixed-size trades via `adapter.eval_signals` and permutes per-bar — impedance mismatch with a continuously-held per-month position. So `gauntlet.py` calls `deflated_sharpe_ratio` (:36) and `benjamini_hochberg` (:57) directly on the monthly series and implements a **directional monthly-sign permutation** modeled on `_permutation_p` (:195, `directional_perm=True`): permute the timing of monthly ±signs preserving the long/short ratio, recompute net Sharpe ≥10k times, `p=(n_ge+1)/(N+1)`. Optional CPCV cross-check (`sovereign/discovery/cpcv.py`), noting ~104 months limits folds.
- **Family-of-one caveat:** DSR `n_trials=1` ⇒ no deflation and BH ≡ raw `p<0.05`. Say so — the real guards are the permutation timing test and the Phase-0 hash-lock. Robustness variants (lookback, cap on/off, rf-subtracted) stay **descriptive**; do not inflate the BH/DSR family (reverse-p-hacking).
- **Q4 Power (state honestly):** ~104 obs full / ~24 OOS / ~12 per year. Permutation significance is exact at small n, but **power is low** — a true Sharpe ~0.5 over 24 OOS months may not reach `perm_p<0.05`. Per-year Sharpes are **descriptive only** (SE ≈ ±0.5). A `NOT_SIGNIFICANT` verdict is about as consistent with low power as with no edge — hence the honest prior.

### Phases
| Phase | Deliverable | Gate |
|---|---|---|
| **0 Pre-register** | `preregister_tsmom.py` → `data/research/preregister/HYP-090_tsmom.json` (hash-locked) + PREREGISTERED ledger entry (backup first). Encodes signal, sizing, cost model, locked null, MONTHLY corr, OOS 2023–24, DSR n_trials=1, prior=NOT_SIGNIFICANT, 4-pair universe. | `verify()` (hash intact) **must pass before any price data is touched**. Commit. |
| **1 Backtest** | `feeds.py`/`signal.py`/`backtest.py` → monthly `R_m`, full + IS/OOS Sharpe, **per-year Sharpe table** → `backtest.json`. | Commit. |
| **2 Correlation** | `correlation.py` → monthly ρ vs v015 (+ rolling-60d, 2022 window, CI) + 50/50 combined Sharpe → `correlation.json`. | Commit. |
| **3 Gauntlet** | `gauntlet.py` → directional perm p (≥10k), DSR, BH, holdout>0 → verdict → `gauntlet.json`. | Commit. |
| **4 Verdict** | `verdict_to_ledger.py` (backup first, preserve `hash_lock`) + `report.py` summary. | Commit. **Stop — no deployment.** |

### Honest limitations (surface in report)
~104/~24/~12 obs → low power (NOT_SIGNIFICANT ≠ no edge) · yfinance FX spot ≠ tradable forward, carry approximated via `SWAP_RATES_ANNUAL`, spreads modeled not live-calibrated · family-of-one ⇒ DSR/BH near-vacuous · ρ SE ~0.10 makes the 0.5 boundary fragile (combined-Sharpe is the tie-breaker) · per-year Sharpes descriptive; expect 2022 to dominate — that is the point.

---

## TASK 2 — 90-day ICT taken-trade projection (TICK-024)

**Package:** `research/ict_projection/` → writes only `data/research/ict_projection/projection_90d.json` + a short markdown report. **Read-only** — never import or touch `ict/pipeline.py`, `ict/orchestrator.py`, or the exit path (shadow freeze).

**Inputs (read-only):** veto shards `data/ledger/ict_veto_ledger_2026_{05,06,07}.jsonl` (writer `ict/ict_veto_ledger.py:79`; fields `pair`, `intended_direction`, `veto_reason`, `veto_stage`, `timestamp`) + taken decisions `data/decision_logs/decisions_2026_{05,06,07}.jsonl`.

**Steps:**
1. Load 3 veto shards + 3 decision-log months.
2. **Dedup per-scan re-emission** → unique `(date, pair, intended_direction, veto_reason)`, first-per-day. **Critical:** raw ~95 records/day → ~7 unique setups/day; skipping this inflates the count ~13×.
3. **Recompute veto rates LIVE** (memory's 55%/31% is stale — live is ~51% ADR / ~9% weekly-trend, plus new classes `HYP046_DISP_GATE` ~20%, `TIMING_GATE` ~7.5%). Classify off `veto_reason`: `.startswith("ADR exhausted")`, `.startswith("WEEKLY_TREND_CONFLICT")`.
4. **Daily taken base rate** from deduped decision logs; cross-check against `unique_setups/day × (1 − P(vetoed))`; reconcile and report any gap.
5. **Project 90 days:** `taken_90 = mean_daily_taken × ~62 trading days` (state calendar→trading-day factor). CI via **bootstrap over days** (resample daily taken counts → 90-day-sum distribution → median + 80/95% bands). Add a rate-sensitivity band for regime drift.
6. Emit `projection_90d.json`: dedup factor, live ADR/weekly rates, daily-taken mean, 90-day point + CI, caveats (3-month base is short; assumes pipeline stays frozen — valid under shadow-freeze). Report whether the projection lands near 30 (ICT still the right prop vehicle) or far below.

---

## Constraints compliance
- **Time-Horizon doctrine** (`TRADING_PHILOSOPHY.md`): TSMOM and carry are both Forex-bucket (daily data, multi-day holds) — combining them is **not** cross-system contamination. The prohibited mixing is ICT (intraday) ↔ Forex; untouched here.
- **Spec-first / pre-register before data** (CLAUDE.md, RISK_CONSTITUTION Art. 6): Phase 0 hash-lock gates Phase 1. No live param changes, no launchd, no OANDA.
- **Training gate:** research pass ends at `VALID_EDGE`/`NOT_SIGNIFICANT`; no training/ignition (`sovereign/autonomous/research_factory.py` stays `live:false`; promotion via `scripts/approve_edge.py` is a separate human step).
- **Prop-challenge guard** (`TRADING_PHILOSOPHY.md` "never"): a TSMOM research pass never authorizes a funded-challenge deployment; TICK-024 is a timeline read-out only.

## Reusable anchors (do not rewrite)
- Pre-reg/ledger: `research/political_alpha/preregister_hyp085.py:141` (`register`/`verify`/`_canonical_hash`) · ledger `data/agent/hypothesis_ledger.json`.
- Gauntlet primitives: `sovereign/discovery/gate.py` (`deflated_sharpe_ratio` :36, `benjamini_hochberg` :57, `_permutation_p` :195) · CPCV `sovereign/discovery/cpcv.py`.
- Sharpe: `sovereign/reporting/equity_curve.py::_sharpe` :84. Correlation shape: `sovereign/research/vrp/validator.py::_corr` :55 / `_analyze` :106.
- Prices + costs: `sovereign/forex/forex_backtester.py::_download_price` :616, constants :43-65 · pairs `sovereign/forex/pair_universe.py:7`.
- v015 returns: `data/proof/backtest_trades_v015_2015_2024.csv` · loader pattern `research/prop_funnel/feeds.py::load_carry_decade`.

## Verification (end-to-end, at build time)
- **Phase 0 gate:** `python3 research/tsmom/preregister_tsmom.py --verify` returns hash-intact PASS before Phase 1 runs; confirm PREREGISTERED entry appended and a `.bak-*` ledger backup exists.
- **Backtest sanity:** confirm the v015 monthly series reconstructed in Phase 2 reproduces the known v015 decade Sharpe ≈ 0.69 (`sum risk_adjusted_pnl_pct` by exit-month → `_sharpe`) — if it doesn't, the loader is wrong before trusting any correlation.
- **Signal no-look-ahead:** unit test that `sigma_daily` and the 252-day sign at rebalance `t` use only data ≤ `t` (shift-by-one assertion in `test_isolation.py`).
- **Permutation self-check:** feed a known-null random-sign series → `perm_p` should be ~uniform (not systematically <0.05).
- **Isolation:** `python3 -m pytest research/tsmom/test_isolation.py` green (no writes outside `data/research/tsmom`, no ict/config imports); full suite `python3 -m pytest tests/ -q` unaffected; ICT isolation test `-k test_pipeline_does_not_import_sovereign` still green.
- **Task 2:** re-run `research/ict_projection/run.py` twice → identical `projection_90d.json` (determinism); manually spot-check the dedup factor (~13×) and that ADR+weekly rates match a hand grep of one veto shard.
- **NEXT.md:** log both tickets, push confirmation, verdicts, and that the shadow/exit path was untouched (no unlock).
