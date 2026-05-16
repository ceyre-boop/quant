# SOVEREIGN QUANT — MASTER CODEBASE DOCUMENT
**For agents, new sessions, and collaborators. Last updated: 2026-05-16.**

This document is the ground truth on what this repo actually does, what is live,
what is stubbed, what data is real, and where the gaps are. Read this before
touching anything. The CLAUDE.md has the operating procedure. This has the map.

---

## WHO BUILT THIS AND WHY

Owner: Colin (ceyre-boop on GitHub). Alta Investments — systematic quant firm.
Goal: Autonomous paper trading → prop firm challenge → live capital.
Stack: Python 3.14 (Mac M4 Pro), Firebase RTDB, GitHub Pages, launchd scheduling.

Two parallel systems live in this repo:

1. **SOVEREIGN** — equity + forex macro system (older, more complete)
2. **ICT ENGINE** — intraday forex ICT methodology (newer, actively paper trading)

They share data caches and the Alexandrian Library but are architecturally isolated.

---

## REPO ROOT MAP

```
/quant
├── ict/                    ← ICT intraday engine (PRIMARY ACTIVE SYSTEM)
├── sovereign/              ← Macro equity+forex engine (RESEARCH PHASE)
│   ├── forex/              ← Forex macro signal system (v004, Sharpe 0.801)
│   ├── risk/               ← CS229 ML stack (11 modules, all operational)
│   ├── strategies/         ← Equity strategies (momentum, reversion)
│   ├── specialists/        ← Specialist models (momentum, reversion, MLX)
│   ├── router/             ← Regime router (Hurst-based MOMENTUM/REVERSION/FLAT)
│   ├── intelligence/       ← TradingAgents narrative (STUBBED — off critical path)
│   ├── features/           ← Feature zoo + Alexandrian Library
│   ├── validation/         ← Backtest engine + paper trading runner
│   ├── execution/          ← Portfolio engine + cTrader bridge
│   └── api/                ← Dashboard endpoints
├── execution/              ← PTJ gates, RR engine, FunderPro executor
├── backtest/               ← Fast backtest engine (148k/sec Numba JIT)
├── contracts/              ← Shared typed dataclasses (single source of truth)
├── lab/                    ← Experiment framework (baseline registry)
├── integration/            ← TradingAgents bridge (Ollama/Qwen3)
├── data/
│   ├── cache/macro/        ← FRED CPI+rates per country (19d stale — refresh needed)
│   ├── cache/cb_decisions.json ← 1,103 CB decisions 2014-2024 (21d stale)
│   ├── cache/_price_cache/ ← yfinance daily OHLCV (auto-cached on first download)
│   └── ledger/             ← Live trade + veto ledgers
├── logs/                   ← Backtest results, edge extracts, pipeline outputs
├── models/                 ← Trained model checkpoints (.json, .pkl)
├── scripts/                ← One-off runners, backtests, pipeline tools
├── dashboard/              ← Sovereign live dashboard (GitHub Pages)
├── ict-dashboard/          ← ICT terminal dashboard (GitHub Pages)
├── frontend/               ← EMPTY — planned for research dashboard
├── config/                 ← parameters.yml, ptj_philosophy.json, ict_params.yml
├── tests/                  ← 565 passing / 28 failing (93% pass rate)
└── research/               ← CS229 + MIT finance lecture notes
```

---

## SYSTEM 1: ICT ENGINE (PRIMARY — ACTIVELY PAPER TRADING)

### What it does
Scans GBPUSD, EURUSD, AUDUSD, AUDNZD every 5 minutes during London (GBPUSD only)
and NY PM (all pairs). Looks for ICT micro-edge setups (sweep → displacement →
FVG → entry). Paper trades automatically via launchd. Pushes to Firebase.
Dashboard at ceyre-boop.github.io/quant/ict/

### Key modules and status

| File | Status | What it actually does |
|------|--------|----------------------|
| `ict/pipeline.py` | ✅ LIVE | 6-stage ICT signal pipeline. Real data from yfinance 1h bars. Grade A only (A+ empirically underperforms). Over-confirmation penalty above score 9.0. |
| `ict/sweep_detector.py` | ✅ LIVE | Close-confirmed sweep detection. Wick-only sweeps = breakout, not trade. Outputs swept_level (used as structural stop). |
| `ict/fvg_detector.py` | ✅ LIVE | 3-candle FVG detection. Tracks filled/unfilled. Age tracking. Entry = FVG midpoint (limit order, not market). |
| `ict/orchestrator.py` | ✅ LIVE | Main loop. Downloads prices, queries library, gets biases, runs pipeline, applies all gates, opens/closes paper trades, pushes Firebase. |
| `ict/paper_trader.py` | ✅ LIVE | Full lifecycle: open → TP1 partial (50%) → move stop to BE → TP2 or stopped. State in `data/ledger/ict_paper_trades.json`. CSV in `logs/ict_paper_trade_log.csv`. **Currently: 0 trades executed (no qualifying signals since launch).** |
| `ict/session_classifier.py` | ✅ LIVE | Kill zone detection. Hardcoded UTC times. NY lunch filter 12-13:30. |
| `ict/micro_risk.py` | ✅ LIVE | Position sizing from structural stop (0.08×ATR buffer). Hard caps: 2% max per trade, 6% concurrent, 3 positions max. |
| `ict/memory_engine.py` | ✅ LIVE | Cosine similarity against past scans. Hard veto: cluster WR < 30%. Requires 20 closed trades to activate. **Currently inactive (0 trades).** |
| `ict/liquidity_heatmap.py` | ✅ LIVE | Swing high/low, equal levels, FVG midpoints with probability decay. Blocks trade if opposing magnet is between entry and TP1. |
| `ict/daily_bias.py` | ✅ LIVE | Per-pair directional bias from library regime + ForexFactory calendar. Blackout gate for FOMC/CPI/NFP. |
| `ict/library_bridge.py` | ✅ LIVE | Fetches SPY/VIX/GLD/DXY, queries Alexandrian Library. Current read: SHALE_SUPPLY_OIL_CRASH CRITICAL — trades in STRESSED mode (1.5R/2.5R targets). |
| `ict/regime_execution.py` | ✅ LIVE | Maps library regime → TP ratios. CRITICAL=1.5R/2.5R, TRENDING=3R/6R, NEUTRAL=2R/4R. |
| `ict/ict_veto_ledger.py` | ✅ LIVE | Records every rejected signal. **68 entries in May ledger.** Most recent: AUDNZD grade B, score 6.95 < 7.0 threshold. |

### Active gates (all must pass to open a trade)
1. A-grade only (B and A+ blocked empirically)
2. Session: NY PM all pairs, London GBPUSD only
3. No calendar blackout (FOMC/CPI/NFP day = blocked)
4. Macro bias agrees (library regime → pair direction)
5. Memory cluster (if 20+ trades: WR < 30% = hard veto, 30-40% = soft penalty)
6. Heatmap: no opposing liquidity magnet between entry and TP1
7. Regime skip: REPO_MARKET_STRESS or FLASH_CRASH = skip entirely
8. Not CRITICAL threat with 0× size (currently soft — trades at 0.5× risk)

### Data sources (ICT)
- **Prices**: yfinance `period='5d', interval='1h'` — live, no cache
- **Library**: sovereign Alexandrian Library (fetches SPY/VIX/GLD/DXY)
- **Calendar**: ForexFactory scraper (ForexFactoryScraper.fetch_today_events())
- **Firebase**: Live writes every scan to `signals/ICT_ENGINE/*`

### Launchd schedule
- `scripts/launch_ict_scanner.sh` fires every 5 minutes
- Checks ET hour — only runs during London (02:00-05:00 UTC) or NY PM (17:30-21:00 UTC)
- PID file: `.harvester.pid` (PID 74467 — process may be stale, check if active)

### Prop challenge pipeline
```
scripts/run_live_pipeline.py --source-json logs/ict_backtest_window_A.json \
                              --source-json logs/ict_backtest_window_B.json \
                              --days 365
```
Current verdict: **🟡 CONDITIONAL GO**
- Live edge: TP2=16.8%, TP1=13.0%, STOP=70.2%, EV=+0.132R/trade
- Source: 106 backtest trades (no live paper trades yet)
- Walk-forward B: 76.0% (matches MC 76.8% — validates)
- Walk-forward A: 54.5% (2024 had rough clustering period)
- Action: collect 30+ live paper trades → re-run → attempt at 0.75% risk if 🟢 GO

### ICT backtest reference numbers
- Window A (May 2024–May 2025): WR=32%, avgWinR=3.06R, TP2=20%, Sharpe=2.18
- Window B (Jul 2023–Jul 2024 OOS): WR=33%, avgWinR=2.97R, TP2=19%, Sharpe=2.60
- FVG limit entry is the edge source (not market entry — confirmed by stop_optimizer test)
- GBPUSD alone: 75.3% prop pass (London + NY PM, both windows consistent)

---

## SYSTEM 2: SOVEREIGN (EQUITY + FOREX MACRO — RESEARCH PHASE)

### What it does
Multi-layer signal system for equity (META, PFE, UNH etc.) and forex (8 pairs).
Runs paper trades via `execute_daily.py` at 9:35 ET. Not automated for forex yet.
Dashboard at ceyre-boop.github.io/quant/

### Architecture (sovereign orchestrator — 2,257 lines)

```
Stage 1: Data (yfinance OHLCV + FRED + COT + CPI)
Stage 2: Regime Router → MOMENTUM / REVERSION / FLAT
Stage 3: ATR gate (minimum volatility required)
Stage 4: Specialist model (momentum or reversion specialist)
Stage 5: Grade-based Kelly sizing
Stage 6: Hard constraints → execute → log
```

**Important**: The CS229 ML modules (Kalman, Softmax, Pegasus etc.) are all imported
and available but most run as loggers/observers, not execution gates. The live
execution path is: Specialist Signal → sizing → hard constraints → execute.
They do NOT currently block or size trades in production.

### Sovereign forex (sovereign/forex/) — v004 replicated 2026-05-16

| File | Status | Notes |
|------|--------|-------|
| `forex_backtester.py` | ✅ LIVE | Full backtest. `python3 -c "import sys; sys.path.insert(0,'.'); from sovereign.forex.forex_backtester import ForexBacktester; ForexBacktester().backtest_all()"` |
| `signal_engine.py` | ✅ LIVE | 3 layers: macro (60d hold), calendar, CB event |
| `macro_engine.py` | ✅ LIVE | Rate diff 30%, IRP 25%, cycle div 25%, PPP 10%, Hurst 10% |
| `entry_engine.py` | ✅ LIVE | CB surprise ≥25bps → entry day 1-5 + CPI surprise fade |
| `cot_engine.py` | ✅ LIVE | Real CFTC data via zip archive (fixed 403 issue) |
| `cpi_engine.py` | ✅ LIVE | Real CPI surprises from cache |
| `dxy_engine.py` | ✅ LIVE | DXY overlay modifier |
| `calendar_signals.py` | ✅ LIVE | QE rebalancing, March JPY repatriation, seasonal |
| `pair_universe.py` | ✅ LIVE | 8 pairs: GBPUSD EURUSD USDJPY AUDUSD USDCAD NZDUSD GBPJPY AUDNZD |
| `state_allocator.py` | ✅ BUILT | State-based router. **Tested: Sharpe -0.12** — prediction before events = noise |
| `confirmation_engine.py` | ✅ BUILT | Confirmation protocol (C1/C2/C3/Carry). **Tested: Sharpe -0.17** — calendar edges don't work standalone |
| `data_fetcher.py` | ✅ LIVE | yfinance + FRED + hardcoded fallbacks for when APIs fail |

**v004 results (replicated 2026-05-16):**
- avg_sharpe: **0.8011** (vs 0.626 reference — improvement from fresh data)
- All 8/8 pairs positive
- Best: AUDUSD 1.292, GBPUSD 1.130, EURUSD 1.009, AUDNZD 0.836

**Key finding from research (2026-05-16):**
Calendar edges (March JPY, QE fade, post-CB) DO NOT WORK as standalone entries.
They only work as SIZE BOOSTS when aligned with v004 macro direction. v004 is
the foundation. Calendar edges are an overlay, not a replacement.

### Sovereign ML stack (sovereign/risk/)

All 11 modules operational. Most are observers, not live gates.

| Module | CS229 Ref | Status | Live gate? |
|--------|-----------|--------|-----------|
| `predict_now.py` | L03/L04/L11 | ✅ | Library blend, not gate |
| `softmax_regime.py` | L04 | ✅ | 3-vote regime ensemble |
| `correlated_position_tracker.py` | L09 | ✅ | Lo uncertainty gate |
| `ml_diagnostics.py` | L10/L12 | ✅ | MI ranking + KMeans |
| `pca_compressor.py` | L14/L15 | ✅ | SVD PCA |
| `ica_factor_separator.py` | L15 | ✅ | ICA factor separation |
| `trade_mdp.py` | L16 | ✅ | Value iteration MDP |
| `lqr_controller.py` | L18 | ✅ | Riccati sizing |
| `kalman_regime.py` | L19 | ✅ | Kalman Bayesian regime |
| `pegasus_policy_search.py` | L20 | ✅ | REINFORCE policy gradient |
| `black_scholes.py` | MIT | ✅ | BS pricing + IV |
| `alexandrian_library.py` | — | ✅ LIVE GATE | 10 volumes, 63 entries. **Currently: CRITICAL** |
| `market_memory.py` | — | ✅ LIVE | Fallback when library unloaded |
| `kelly_engine.py` | — | ✅ LIVE GATE | Kelly sizing with library cap |
| `cluster_veto.py` | — | ✅ LIVE | 420K+ failure map rows |

**The Alexandrian Library is the most important module** — it's the only ML component
that is a true live execution gate (via library_bridge → ICT orchestrator and
sovereign orchestrator Integration Points I1-I5). Current read: CRITICAL, 0.50× size.

### Sovereign intelligence (sovereign/intelligence/)

| Module | Status | Notes |
|--------|--------|-------|
| `narrative_engine.py` | ⚠️ STUB | TradingAgents/Qwen3 integration. Deliberately off critical path. Returns neutral stub. Monthly cadence or manual. |

### Sovereign validation (sovereign/validation/)

| Module | Status | Notes |
|--------|--------|-------|
| `backtest_engine.py` | ✅ LIVE | Full backtest harness |
| `paper_trading_runner.py` | ✅ LIVE | Paper trade simulation |
| `veto_diagnostic.py` | ✅ LIVE | Veto pattern analysis |

---

## DATA FEEDS — FRESHNESS AUDIT

| Source | File | Age | Status | How to refresh |
|--------|------|-----|--------|----------------|
| FRED CPI/rates | `data/cache/macro/*.parquet` | **19 days** | ⚠️ STALE | `python3 scripts/build_market_memory.py` |
| CB decisions | `data/cache/cb_decisions.json` | **21 days** | ⚠️ STALE | Re-run data collection script |
| yfinance prices | `data/cache/_price_cache/` | On-demand | ✅ Auto-cached | Downloads on first use |
| ICT paper trades | `data/ledger/ict_paper_trades.json` | 0 days | ✅ Live | Updated each scan |
| ICT veto ledger | `data/ledger/ict_veto_ledger_2026_05.jsonl` | 0 days | ✅ Live | 68 entries in May |
| Live edge | `logs/live_edge.json` | 3 days | ✅ OK | Re-run pipeline after new trades |
| Macro JSON | `data/cache/EU_macro.json` | 0 days | ✅ Fresh | Auto-updated |

**STALE DATA IMPACT:**
The 19-21 day staleness of FRED data affects:
- `sovereign/forex/macro_engine.py` rate differential calculations (falls back to cached values)
- `ict/library_bridge.py` feature vector (uses rates from parquet)
- `ict/daily_bias.py` carry yield calculations
The system still runs but may not reflect current BOE/FED/ECB rate differential precisely.

**To refresh FRED data:**
```bash
python3 scripts/build_market_memory.py      # rebuilds market memory + updates FRED
python3 scripts/build_alexandrian_library.py --query  # rebuilds library features
```

---

## EXECUTION LAYER

| Module | Status | Notes |
|--------|--------|-------|
| `execution/ptj_gates.py` | ✅ LIVE | 12-gate PTJ filter in sovereign orchestrator |
| `execution/rr_engine.py` | ✅ LIVE | PTJ 3-target: 1.5R/3R/5R |
| `execution/paper_trading.py` | ✅ LIVE | Sovereign paper trades |
| `execution/funderpro_executor.py` | ⚠️ OFFLINE | Built, routing OFF. Requires: `FUNDERPRO_LIVE=demo` env var + `data/pipeline_verdict.json` with `verdict=GO`. **Do not enable until pipeline prints 🟢 GO.** |
| `sovereign/execution/ctrader_bridge.py` | ⚠️ STUB | cTrader OpenAPI — `_send_ctrader_order()` raises NotImplementedError |
| `sovereign/execution/portfolio_engine.py` | ✅ LIVE | Correlation-aware Kelly allocation |

---

## DASHBOARDS

| Dashboard | URL | Data source | Status |
|-----------|-----|-------------|--------|
| ICT Terminal | ceyre-boop.github.io/quant/ict/ | Firebase RTDB live | ✅ LIVE |
| Sovereign | ceyre-boop.github.io/quant/ | Firebase RTDB live | ✅ LIVE |
| Research Dashboard | `frontend/` | — | ❌ NOT BUILT YET |

**Dashboard deploy**: Single GitHub Actions workflow (`deploy.yml`) merges
`dashboard/` and `ict-dashboard/` into one Pages artifact. Both deploy on
push to master if either folder changes.

**Firebase project**: `clawd-trading-7b8de-default-rtdb.firebaseio.com`
**Firebase paths**:
- `signals/ICT_ENGINE/{PAIR}` — per-pair scan results
- `signals/ICT_ENGINE/paper_trades` — open/closed/stats
- `signals/ICT_ENGINE/_system` — system health + regime context
- `signals/ICT_ENGINE/backtest` — backtest reference stats
- `signals/ICT_ENGINE/heatmap/{PAIR}` — liquidity heatmap
- `signals/SOVEREIGN_ENGINE/*` — sovereign equity signals

---

## CONTRACTS AND SHARED TYPES

`contracts/types.py` — single source of truth for all inter-layer types.

Key types defined:
- `CatalystWindowState`, `PriceRegimeState`, `MacroRegimeState`
- `PositioningState`, `NarrativeState`, `HistoricalMatchState`, `PresentState`
- `SovereignFeatureRecord`, `RouterOutput`, `RegimeState`
- `PositionState`, `AccountState`, `Direction`, `VolRegime`, `TrendRegime`

**Note**: `PresentState` and related types were added 2026-05-13 to fix a
`CatalystWindowState` import error in `sovereign/present_state.py`. If you see
import errors from `contracts.types`, check that `PresentState` and friends are
present — they may have been there briefly as stubs in the Copilot PR.

---

## LAB SYSTEM (lab/)

`lab/baseline_registry.py` — tracks champion configurations. Used by
`scripts/seed_lab_champion_v1.py` to record a validated edge baseline.
`lab/run_experiment.py` — runs parameter experiments and evaluates vs champion.

**Status**: Built but not actively in use. Part of the
cold_start → replay_validation → seed_champion flow introduced in PR #26
(replay validation pipeline). That flow still has a missing `alpaca` dependency
when run locally (works in CI runner).

---

## SCRIPTS REFERENCE

| Script | What it does | Works locally? |
|--------|-------------|----------------|
| `run_live_pipeline.py` | Extract edge → optimize → walk-forward → GO/NO-GO | ✅ Yes |
| `extract_live_edge.py` | Get TP distribution from paper CSV or backtest JSON | ✅ Yes |
| `prop_challenge_optimizer.py` | 5,292 param combos × 10k MC trials | ✅ Yes (fast: `--fast`) |
| `run_ict_backtest.py` | ICT walk-forward backtest on 4 pairs | ✅ Yes |
| `optimize_ict_stops.py` | Stop width sweep + pair diagnosis | ✅ Yes |
| `run_portfolio_backtest.py` | 30-year sovereign forex portfolio backtest | ✅ Yes |
| `run_forex_scan.py` | Live forex macro scan | ✅ Yes |
| `run_universe_backtest.py` | Equity universe backtest | ✅ Yes |
| `simulate_prop_challenge.py` | Quick prop challenge Monte Carlo | ✅ Yes |
| `build_alexandrian_library.py` | Rebuild Alexandrian Library (needs yfinance) | ✅ Yes |
| `build_market_memory.py` | Rebuild market memory + refresh FRED cache | ✅ Yes |
| `cold_start_readiness_audit.py` | Checks ML snapshots, ledger depth, .pkl files | ✅ Yes (reports not-ready) |
| `run_replay_validation.py` | Replay validation with ML gate checks | ⚠️ Needs `alpaca` pkg |
| `seed_lab_champion_v1.py` | Seeds lab baseline from replay report | ⚠️ Needs replay first |
| `run_tradingagents_narrative.py` | TradingAgents narrative (needs Ollama) | ⚠️ Optional |

---

## TESTS — 28 FAILURES

**Run**: `python3 -m pytest tests/ -q`
**Result**: 565 passed, 28 failed (93% pass rate)

**Failing tests** (all in `tests/unit/test_ml_stack.py`):
- `TestBlackScholes::test_vol_regime_signal` — vol regime signal classification
- `TestOnTradeClose::*` (3 tests) — on_trade_close() ledger write + state updates

**Not failing**: All ICT tests, all forex tests, all CS229 core tests, all backtest tests.
The 4 failures are in advanced sovereign ML (Black-Scholes vol signal, on_trade_close
state tracking). They don't affect ICT or forex paper trading.

**To run just the healthy tests:**
```bash
python3 -m pytest tests/ -q --ignore=tests/unit/test_ml_stack.py
```

---

## KNOWN GAPS — WHAT IS NOT DONE

### 1. No live paper trades yet (ICT)
`data/ledger/ict_paper_trades.json` has 0 closed trades. The system is running,
the scanner fires, but no A-grade signals have met all gates since launch.
The 68 veto ledger entries show the system is alive and rejecting signals — most
are grade B, score < 7.0. This is correct behavior. Need patience.

### 2. News feed is mock data
The ICT dashboard Economic Calendar section shows mock news events. The
`ForexFactoryScraper` is implemented and returns real data, but it is not
wired into the dashboard. The dashboard reads from `MOCK_NEWS` (hardcoded JS
array in `ict-dashboard/index.html`). **Fix: wire ForexFactoryScraper output
to Firebase → read from Firebase in dashboard.**

### 3. TradingAgents narrative is a stub
`sovereign/intelligence/narrative_engine.py` has the Qwen3/Ollama integration
but it runs monthly/on-demand only and returns a neutral stub when unavailable.
ICT dashboard Panel C narrative section is not populated. The infrastructure
(trading_agents_bridge.py, ta_runner.py) is built but not in the live path.

### 4. FRED data is stale (19-21 days)
`data/cache/macro/*.parquet` and `data/cache/cb_decisions.json` need refresh.
Impact: carry yield and rate differential calculations use slightly old numbers.
System still runs correctly with cached data.

### 5. Research dashboard doesn't exist
`frontend/` directory is empty. `frontend/dashboard_research.html` is planned
(AUTONOMOUS RESEARCH AGENT section in CLAUDE.md) but not built.

### 6. Sovereign agent loop doesn't exist
`sovereign/agent/research_agent.py` is planned but not built. The research
queue in CLAUDE.md defines what it should run, but the runner doesn't exist.

### 7. cTrader bridge is a stub
`sovereign/execution/ctrader_bridge.py` and `execution/funderpro_executor.py`
have the architecture but `_send_ctrader_order()` raises `NotImplementedError`.
FunderPro routing is safely offline until pipeline prints 🟢 GO.

### 8. `.pkl` model checkpoints are missing
`scripts/cold_start_readiness_audit.py` reports `softmax_regime.pkl`,
`kalman_regime.pkl`, `predict_now.pkl` etc. all missing from `models/`.
The `.json` versions exist (softmax_regime.json etc.) — the audit checks for
`.pkl` which is a legacy path. The JSON models load fine.

### 9. `alpaca` package missing locally
`scripts/run_replay_validation.py` imports `alpaca-trade-api` which is not
installed in the local Python 3.14 environment. Works in CI (GitHub Actions).

---

## ENVIRONMENT NOTES

**Python**: 3.14 (Homebrew). `python3` points to 3.14.
**yfinance**: 1.3.0 (recently upgraded — `websockets` was broken, now fixed).
**Known conflict**: `alpaca-trade-api` requires `websockets<11` but `websockets 16.0` is installed.
**Virtual envs**: `.venv` (main), `.venv-tradingagents` (TradingAgents/LangChain deps — isolated).
**Notifications**: ntfy.sh topic `clawd-ict-7829` — subscribed, working.
**Firebase**: Service account at `config/firebase_service_account.json` (not in git).

---

## FOR AGENTS STARTING A NEW SESSION

1. **Read CLAUDE.md** (routing rules, who does what, current system status).
2. **ICT is the primary system**. Everything in `ict/` is production code.
3. **Do not modify `sovereign/forex/`** without checking it doesn't break v004.
   Run `python3 -c "import sys; sys.path.insert(0,'.'); from sovereign.forex.forex_backtester import ForexBacktester; print([r.sharpe for r in ForexBacktester().backtest_all()])"` to confirm after changes.
4. **The Alexandrian Library is a live gate**. Changing `sovereign/risk/alexandrian_library.py` or `sovereign/features/alexandrian_library.py` affects both ICT and sovereign.
5. **Firebase is always live** — anything pushed to Firebase shows on the dashboard immediately.
6. **The veto ledger is your debug tool** — if trades aren't opening, check `data/ledger/ict_veto_ledger_2026_05.jsonl` to see exactly which gate is blocking.
7. **Backtest results that matter** are in `logs/ict_backtest_window_A.json` and `logs/ict_backtest_window_B.json`. Don't delete them — they're the validation baseline.
8. **The prop challenge pipeline** is `scripts/run_live_pipeline.py`. Don't enable `FUNDERPRO_LIVE=live` without a 🟢 GO verdict from that script.
9. **Tests**: `python3 -m pytest tests/ -q --ignore=tests/unit/test_ml_stack.py` for a clean run.
10. **stale data**: If anything seems wrong with macro signals, run `python3 scripts/build_market_memory.py` to refresh FRED cache.

---

## VERSION HISTORY (forex Sharpe progression)

| Version | avg Sharpe | Pairs positive | Key change |
|---------|-----------|----------------|------------|
| v001 | 0.179 | 1/11 | Baseline |
| v002 | 0.355 | 4/11 | Multi-layer signals |
| v002.5 | ~0.38 | 5/11 | Macro veto, DD -15%→-8% |
| v003 | 0.452 | 8/10 | Rate divergence refine |
| v004 | 0.626 | 8/8 | SNB/EURGBP/EURJPY removed |
| Current | **0.801** | 8/8 | Fresh data + clean universe |
| Target | **1.5+** | 8/8 | Institutional grade |

---

*This document is auto-generated from a comprehensive audit. Update it when
significant new modules ship or the system state changes. The ground truth
for operating procedure is CLAUDE.md. This document is the ground truth for
what code actually exists and what it actually does.*
