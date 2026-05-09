# SOVEREIGN / QUANT — AGENT OPERATING PROCEDURE
## Claude Code + Codex Autonomous Collaboration Protocol

Before making any architectural decision, read TRADING_PHILOSOPHY.md. Every component must serve one of the six tenets or it should not exist.

---

## WHO DOES WHAT

### Claude Code handles:
- Architecture decisions and system design
- Debugging complex failures (reading full stack traces, 
  tracing across multiple files)
- Any task requiring understanding of the full codebase context
- Tasks where output must be reviewed before committing
- Decisions that affect live trading logic
- Anything touching: orchestrator.py, risk engines, 
  veto logic, XGBoost models, forex macro engine

### Codex handles (via /codex:run or codex exec):
- Unit test generation for new modules
- Boilerplate file creation from a clear spec
- Central bank calendar data entry
- Backtest runs that only need a terminal command
- Documentation generation
- Simple find-and-replace refactors across many files
- Tasks where the spec is fully defined and output 
  is independently verifiable (tests pass / numbers match)

### Neither agent decides alone:
- Changes to live trading parameters (config/parameters.yml)
- Changes to risk limits or position sizing
- Anything touching real money execution paths
- These require human confirmation before applying

---

## RATE LIMIT AWARE ROUTING

Claude Code has a context window and session rate limit.
Codex has a separate rate limit on its own account.
Use this to your advantage:

When Claude Code context is getting full (>80k tokens):
  → Delegate the next well-defined task to Codex
  → Keep Claude Code context for high-judgment work
  → Use /clear only after delegating pending tasks

When a task is parallelizable:
  → Fire it to Codex immediately, don't queue it
  → Continue working on something else in Claude Code
  → Review Codex output when it finishes

Pattern: Claude Code thinks, Codex executes in parallel.

---

## HOW TO DELEGATE TO CODEX

From inside Claude Code, use the plugin:
/codex:run "clear task description with file paths"

Or for longer tasks, write to scripts/codex_tasks.sh and run:
bash scripts/codex_tasks.sh

Codex sandbox CANNOT access external networks (yfinance, FRED).
Any task needing live data must run in the normal terminal, 
not via codex exec.

Good Codex tasks (self-contained, no network):
  "Write pytest tests for sovereign/forex/macro_engine.py"
  "Generate cb_calendar.py with these 8 banks and dates: ..."
  "Refactor all float64 to float32 in backtest/fast_engine.py"
  "Add docstrings to every function in sovereign/forex/*.py"

Bad Codex tasks (needs network or judgment):
  "Run the forex backtest" → do this in terminal directly
  "Debug why the HMM is slow" → needs full context, Claude Code
  "Decide if we should change the conviction threshold" → human

---

## TASK DECISION TREE

When a new task arrives, ask in order:

1. Does it require understanding the full system state?
   YES → Claude Code handles it
   NO  → continue

2. Is the spec fully defined (I know exactly what to build)?
   NO  → Claude Code defines the spec first, then delegates
   YES → continue

3. Does it need external network access?
   YES → run in terminal directly (python3 or bash)
   NO  → continue

4. Is Claude Code context above 80k tokens?
   YES → delegate to Codex, /clear after
   NO  → either agent can handle it

5. Can it run in parallel with current Claude Code work?
   YES → delegate to Codex immediately, don't wait
   NO  → Claude Code handles sequentially

---

## REVIEWING CODEX OUTPUT

Never accept Codex output without verification:

For code changes:
  - Read the generated file
  - Run existing tests: python3 -m pytest tests/
  - Confirm no imports break: python3 -c "import sovereign"

For backtest results:
  - Sanity check: are Sharpe values between -5 and 5?
  - Are all pairs returning different numbers? (identical = bug)
  - Does trade count make sense (>10 per pair per year = suspicious)

For test files:
  - Run them: python3 -m pytest tests/unit/test_*.py -v
  - All must pass before accepting

---

## REPO STRUCTURE REMINDER

Live trading:        sovereign/orchestrator.py → execute_daily.py
Forex system:        sovereign/forex/
Equity strategies:   sovereign/strategies/
Backtest engine:     backtest/fast_engine.py (148k/sec)
Config:              config/parameters.yml (never hardcode)
Ledger:              data/ledger/trade_ledger_*.jsonl
Logs:                logs/
Models:              models/sovereign/

Do not modify trading_strategies/ict_amd_swing_nas100.py
Do not modify layer3/game_engine.py directly — wrap it

---

## CURRENT SYSTEM STATUS (update this section each session)

Equity system:    LIVE — paper trading 9:35 ET via launchd
Forex system:     BUILT — paper scan only, not automated yet
Backtest speed:   148,193/sec (Numba JIT, 12 cores)
XGBoost models:   Both specialists trained and live
Veto pipeline:    4/5 clusters active
Forex backtest:   v004 — avg_sharpe=0.6260, 8/8 pairs positive (clean universe)
                  Best: GBPUSD +1.094 | EURUSD +0.982 | AUDUSD +0.896 | AUDNZD +0.884
                  USDCHF removed: SNB held flat 8yr, rate-diff structurally broken (-0.45)
                  EURGBP removed: ECB+BOE lockstep, no rate divergence edge (PF 1.00)
                  EURJPY removed v001: dual ECB+BOJ conflict (-0.62)
                  Active universe: GBPUSD EURUSD AUDUSD AUDNZD GBPJPY USDJPY USDCAD NZDUSD
                  Portfolio (macro-only): Sharpe 0.326 | MaxDD -14.2% | SNB survived 0.0%
                  Signal frequency gap: 0.33/pair/month vs target 2 → factor model is next unlock
Signal layers:    5 layers — Macro + CPI fade + Calendar + CB events + DXY overlay
                  + COT gate (Dalio Q3) on entry sizing

ML STACK (Stanford CS229 + MIT Finance — 11/11 operational, all loops closed):
  sovereign/risk/predict_now.py          — LOESS win rate + Newton IRLS + L2 MAP (L03/L04/L11)
  sovereign/risk/softmax_regime.py       — Softmax 3-class regime classifier (L04)
  sovereign/risk/correlated_position_tracker.py — Lo sequential info + uncertainty gate (L09)
  sovereign/risk/ml_diagnostics.py       — MI feature ranking + bias-var + K-means (L10/L12)
  sovereign/risk/pca_compressor.py       — SVD PCA + LSI LOESS kernel (L14/L15)
  sovereign/risk/ica_factor_separator.py — ICA factor separation 0.81→0.015 corr (L15)
  sovereign/risk/trade_mdp.py            — Value iteration on 72-state trade MDP (L16)
  sovereign/risk/lqr_controller.py       — Riccati equation, linear optimal sizing (L18)
  sovereign/risk/kalman_regime.py        — Kalman filter Bayesian regime estimator (L19)
  sovereign/risk/pegasus_policy_search.py — Pegasus + REINFORCE policy gradient (L20)
  sovereign/risk/black_scholes.py        — BS pricing, IV inversion, risk-neutral MC (MIT)

Research vault:
  research/stanford_cs229/lectures_01-03.md through lectures_19-20.md
  research/mit_quantitative_finance.md

PTJ Gates:        LIVE in orchestrator + execution layer
  TRADING_PHILOSOPHY_PTJ.py — source of truth for all PTJ thresholds
  execution/ptj_gates.py    — 12-gate filter: 200SMA, circuit breakers, R:R, phase tracker
  execution/rr_engine.py    — PTJ 3-target structure: TP1=1.5R(40%) TP2=3R(35%) TP3=5R/7R(25%)
  execution/paper_trading.py — circuit breaker + PTJ gates wired into execute_signal
  research/dislocation_library.py — 20-event crash registry + precursor features + rule-based scorer
  config/ptj_philosophy.json — all thresholds readable without touching code

PTJ Gate order (sovereign orchestrator):
  1-3: Weekly/monthly circuit breaker + post-loss 30min cooldown
  4:   Max 5 concurrent positions
  5:   SPY macro 200 SMA (SPY below 200 SMA → block ALL new longs)
  6:   Asset 200 SMA (individual must be on right side)
  7:   Shock candle (>2.5ATR against = exit immediately)
  4h (CS229): Kalman + VolRegime + TradeMDP + LQR + Pegasus size modulators
  10:  R:R gate (min 2:1 at TP1)
  11:  Portfolio risk cap (8% total, 1.5% per trade)

Alexandrian Library: LIVE in orchestrator (primary market intelligence)
  sovereign/risk/alexandrian_library.py — 10 volumes, 63 entries, full market circumstance coverage
  Volumes:        I Crashes | II Rate Cycles | III Bull Regimes | IV Currency Crises | V Vol Regimes
                  VI Econ Cycles | VII Liquidity | VIII Commodities | IX Geopolitical | X Sector Rotation
  Architecture:   23-feature cosine similarity per entry, multi-volume convergence amplifier
                  3+ volumes >0.60 similarity → composite score amplified ×1.25
                  Returns LibraryInsight: primary_regime, threat_score, size_modifier, advisory
  Threat levels:  NORMAL(1.0×) ELEVATED(0.75×) WARNING(0.50×) DANGER(0.25×) CRITICAL(0.0×)
  Auto-learns:    any live drawdown ≥8% → added to library + MarketMemory simultaneously
  Build library:  python3 scripts/build_alexandrian_library.py --query  (requires yfinance)
  Feeds:          stage 4h of orchestrator — runs FIRST before all other ML modulators

CS229 Stack — All Loops Closed (session 2026-05-05):
  Softmax (L04):   3-vote ensemble regime confidence | HMM+Softmax+KMeans → blended_conf
                   Online SGD update on every on_trade_close() | bootstrap from ledger
  ICA (L15):       PredictNow LOESS uses ICA-projected distances (priority over PCA)
                   Refit every 25 new trades | bootstrap from ledger on cold start
  KMeans (L12):    Third regime vote | refit every 10 new trades (≥30 obs)
                   3-way agreement → conf×1.15 | 1-way → conf×0.85
  Pegasus (L20):   All 6 params now live: entry_threshold, size_multiplier, stop_atr_mult,
                   tp_rr_ratio, hmm_conf_gate, kelly_fraction_cap
                   Trust ramp: proportional 0→1 over 30 updates (gate at 10, execute at 20)
                   Full REINFORCE: Gaussian policy gradient for continuous; sign for thresholds
  ml_diagnostics:  KMeans feeds 3-vote system | bias-var in walk-forward | MI available

Library ↔ CS229 Integration (5 points, all LIVE):
  I1  — library_adjusted_uncertainty_level() in correlated_position_tracker.py
          7+ volumes → Lo Level 3 minimum (0.50×), 5-6 → 0.75×, regardless of HMM state
  I2  — library_informed_win_rate() in predict_now.py
          Blends own trade LOESS estimate with 63 Library historical win rates by regime
          Hoeffding ramp: Library prior dominates when n_trades < 400
  I3  — Gate 5b: _library_asset_gate() in orchestrator.py (after PTJ 200 SMA)
          VALUATION_DISLOCATION: AMD/NVDA/XLK longs blocked; PFE/GLD/XLV pass
          REPO_MARKET_STRESS: all equity blocked; only GLD passes
  I4  — _grade_risk_pct(library_insight) in kelly_engine.py
          7+ volumes ×0.50 Kelly cap (A+ 4%→2%), 5-6 ×0.625, 3+ ×0.75
  I5  — ptj_dislocation_from_library() in orchestrator.py
          Severe pattern (REPO/CONTAGION/COVID) + 5+ converging → cat2 (0.50×)
          Moderate pattern (VALUATION/STAGFLATION) + 3+ converging → cat1 (0.75×)

Current read (May 2026):  10 volumes converging | ASIAN_CURRENCY_CONTAGION primary | sim=0.927
  Lo Level 3: 0.50×  |  Kelly cap: 2.0% (was 4%)  |  PTJ SEVERE: 0.50×
  Net A+ long: 1.5% × 0.50 × 0.50 = 0.38% risk per trade (defence-first in force)

Trading Memory:   LIVE in orchestrator (legacy fallback when library not loaded)
  sovereign/risk/market_memory.py — 20 historical crash events, 23-feature cosine similarity
  Build memory:   python3 scripts/build_market_memory.py  (requires yfinance, network)

Tests:            23/23 passing — python3 tests/run_ml_tests.py
                  Covers: PCA, ICA, Kalman, LQR, BS, MDP, Pegasus, ATR, on_trade_close,
                  AlexandrianLibrary (5 tests: import, no-data, with-data, volume coverage, size range)

Ultraplan (Day 1-4 BUILT):
  Day 1 — Portfolio Engine:   sovereign/execution/portfolio_engine.py  ← DONE
           PortfolioEngine: score = conviction × predicted_r × (1/atr_pct)
           Correlation penalty: EURUSD+GBPUSD LONG same USD bet → both reduced 61.5%
           AUDNZD (no USD) → 0% penalty (correctly identified as diversifying)
           Hard constraints: 6% total, 20% per pair, 40% per currency, SNB 50% cap
           Test: 4-pair simultaneous allocation — all assertions pass

  Day 2 — 30-year backtest:   scripts/run_portfolio_backtest.py  ← DONE
           python3 scripts/run_portfolio_backtest.py --start 1993-01-01
           python3 scripts/run_portfolio_backtest.py --snb-only  (2015 stress test)
           Targets: Sharpe>1.5, maxDD<15%, survive SNB, ≥2 trades/pair/month

  Day 3-4 — Factor Discovery: sovereign/ml/factor_discovery.py  ← DONE
           42 candidate factors: 13 macro, 15 technical, 5 sentiment, 3 carry, 6 calendar
           IC/ICIR validation + Benjamini-Hochberg FDR correction + 3-period robustness
           FactorModel: XGBRegressor(y=forward_R) — replaces hardcoded signal weights
           Updates monthly. Run: FactorDiscovery.validate_all_pairs()

  Day 5 — TradingAgents narrative: integration/trading_agents_bridge.py (pre-existing)
           Factor model says WHAT. TradingAgents explains WHY.

  Day 6-7 — Full 30-year simulation: pending (run after v004 completes)

Next milestone:   python3 scripts/run_portfolio_backtest.py --snb-only  ← SNB stress test
                  python3 scripts/run_universe_backtest.py              ← run v004
                  v004 target: avg_sharpe > 0.50 (USDCHF + EURGBP signal fixes applied)
                  Run FactorDiscovery.validate_all_pairs() for real factor validation

---

*This file is the agent handshake. Both Claude Code and Codex 
read it. It removes the human from routine task routing.*
*Update CURRENT SYSTEM STATUS at the end of every session.*

---

## LEARNING SYSTEM

This is the most important section. Every session must grow the vault. Do not skip this.

---

### 1. Session End Protocol (EVERY session, no exceptions)

Before your final message, write a session note using the `obsidian-vault` MCP tool (`write_file`).

**File:** `C:\Users\Admin\clawdbot-vault\Projects\clawd_trading\Sessions\YYYY-MM-DD-[topic].md`

```
# Session: [topic] — YYYY-MM-DD

## What we worked on
[Tasks completed, questions answered]

## What changed in the system
[Files modified, new modules, removed code]

## Decisions made
[Architecture choices, parameter changes, risk rule updates — and WHY]

## Backtest / research results
[Any numbers from this session — Sharpe, win rate, trade count, key findings]

## Hypotheses
[New hypotheses formed, existing ones updated, any falsified]

## System state at end of session
[Copy updated CURRENT SYSTEM STATUS here]

## Next session
[Highest priority items, open questions, what data is needed]
```

---

### 2. Update CURRENT SYSTEM STATUS

After every session, update the `## CURRENT SYSTEM STATUS` section at the top of this file to reflect the real current state. It should never be stale. This is the first thing Claude reads next session.

---

### 3. Hypothesis Tracking

Every time a hypothesis is formed OR a result comes in, write to the vault.

**File:** `C:\Users\Admin\clawdbot-vault\Projects\clawd_trading\Research\Hypotheses\HYP-NNN-[name].md`

```
# HYP-NNN: [Hypothesis Name]

**Status:** Pending | Testing | Validated | Falsified | Archived
**Formed:** YYYY-MM-DD
**Closed:** YYYY-MM-DD (if applicable)

## The Hypothesis
[Clear statement of what we believe and why]

## Test Method
[How we tested it — which script, what data, what conditions]

## Results
[Numbers, charts description, raw outcome]

## Conclusion
[What this means for the system]

## Impact
[What changed in the system as a result — or why nothing changed]
```

Also update `Reality Bridge MOC.md` → `Current Active Hypotheses` section when status changes.

---

### 4. Backtest Result Logging

After any meaningful backtest run, write results to the vault.

**File:** `C:\Users\Admin\clawdbot-vault\Projects\clawd_trading\Research\Backtest-Results\YYYY-MM-DD-[description].md`

```
# Backtest: [description] — YYYY-MM-DD

## What was tested
[System version, parameters, universe, date range]

## Results
| Metric | Value |
|--------|-------|
| Sharpe | |
| Win Rate | |
| Trade Count | |
| Max Drawdown | |
| Avg R:R | |

## Interpretation
[What these numbers mean — is this good, bad, surprising?]

## Action
[What changes, if any, this result drives]
```

---

### 5. Architecture Decisions

When making a significant design decision (changing layers, adding/removing signals, modifying risk rules, new data source), write it down.

**File:** `C:\Users\Admin\clawdbot-vault\Projects\clawd_trading\Architecture\Decisions\YYYY-MM-DD-[decision].md`

```
# Decision: [title] — YYYY-MM-DD

## What changed
[The architectural change]

## Why
[The reasoning — performance data, failure mode, new insight]

## Alternatives considered
[What else was evaluated and why it lost]

## Risk
[What could go wrong with this decision]
```

---

### 6. Reality Bridge MOC Sync

Update `C:\Users\Admin\clawdbot-vault\Reality Bridge MOC.md` whenever:
- A hypothesis moves from Pending → Testing → Validated/Falsified
- System state changes (equity live, forex automated, new circuit breaker, etc.)
- A new blocker appears or is resolved
- Stage gate passes (paper → live)

Keep the `System State` and `Current Active Hypotheses` sections in that file accurate.

---

### 7. Pre-Session Vault Check

At the START of every session, before doing anything else:
1. Read today's date
2. Check `Sessions/` for the most recent note — read it for context
3. Read `Reality Bridge MOC.md` for current system state
4. Read `CURRENT SYSTEM STATUS` in this file

This means every session starts with full context from the last one, not a cold start.
