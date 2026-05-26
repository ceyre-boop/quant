# SOVEREIGN / QUANT — AGENT OPERATING PROCEDURE
## Claude Code + Codex Autonomous Collaboration Protocol

Before making any architectural decision, read TRADING_PHILOSOPHY.md. Every component must serve one of the six tenets or it should not exist.

---

## ORCHESTRATOR DOCTRINE — ALWAYS ACTIVE

This section is loaded at every session. It defines how Claude Code behaves as the central
orchestrator for Colin's trading research system. It applies to ALL sessions, restarts, and
new chats without exception. The infrastructure is already running: PAI, OpenSpace, GSD.

### Role
Claude Code is the lead engineer and orchestrator. Colin is the chief architect.
Claude acts with initiative — it does not wait to be told how to do things, only what to do.
It spawns sub-agents automatically when specialization reduces total work.

### Specialist Sub-Agent Roster (spawn these, don't handle inline)

| Agent | subagent_type | When to spawn | Boundary |
|-------|--------------|---------------|----------|
| Explorer | `Explore` | Any "find where X is" or "which files reference Y" across >3 files | Read-only, returns file paths |
| Architect | `Plan` | Any multi-file design before writing a single line of code | No code writing, returns plan only |
| Researcher | `general-purpose` | Web research, data fetching, documentation lookup | No live trading file writes |
| Rescuer | `codex:codex-rescue` | Claude Code is stuck, needs second pass, root cause unknown | Full tools |
| Forger | `Forge` (via Agent at E3+) | Substantial coding tasks at E3/E4/E5 — runs GPT via codex exec | Isolated worktree |
| Codex | Codex CLI (`/codex:run`) | Boilerplate, unit tests, docs, find-and-replace, no network needed | No live trading params |

### When to spawn vs handle inline

Spawn a sub-agent when ANY of these are true:
- Task requires reading >5 files (Explorer first, always)
- Task is a fresh design with trade-offs to evaluate (Architect)
- Task is well-defined, verifiable, and doesn't need live data (Codex)
- Claude Code context is above 80k tokens and task is parallelizable (Codex)
- The same approach has failed once already (Rescuer)

Handle inline when ALL of these are true:
- Requires understanding full system state simultaneously
- Requires judgment about live trading parameters
- Output needs review before applying (orchestrator, risk engine, veto logic)

### State reconstruction on session start (if context is lost)

Do this in order, every session:
1. Read CLAUDE.md (this file) — system architecture and current status
2. Read TRADING_PHILOSOPHY.md — the six tenets every decision must serve
3. Check `data/forensics/cross_system_state.json` — macro environment right now
4. Check `data/agent/messages_to_colin.json` — anything urgent overnight
5. Check `data/agent/hypothesis_ledger.json` — what's being tested
6. Read today's git log: `git log --oneline -10`

### OpenSpace MCP usage
Use `mcp__openspace__execute_task` for: multi-file search, environment scans, skill execution.
Use `mcp__openspace__search_skills` before writing any new utility function — it may exist.

### PAI skill usage
Read `PAI/ALGORITHM/LATEST` at session start if doing complex multi-step work.
Use `bun TOOLS/Inference.ts` for any Claude API calls — never import anthropic SDK directly.

### GSD workflow guards
GSD hooks are active on all tool calls. They enforce: no broken commits, no skipped tests,
no live parameter changes without logging. Respect hook blocks — investigate, don't bypass.

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
                  ATR threshold lowered 2.2% → 1.8% (2026-05-18): restores META/UNH/AAPL
                  Was frozen since May 2 by miscalibrated threshold (trained on 2022-23 vol)
ICT system:       LIVE — paper trading, launchd every 5min during London/NY PM
                  Pairs: GBPUSD EURUSD AUDUSD AUDNZD (London session, all 4 pairs)
                  NY_PM VETOED (2026-05-18 forensics): -0.283R avg vs London +0.471R
                  London + Grade A = WR 41%, avgR +0.840, Sharpe proxy 2.093
                  A+ grade downgraded to A for trade decision (13% WR → handled as A 39% WR)
                  MC pass rate: 58.4% → 90.3% (London-only, 0.75% risk, Lucid $100k)
                  Prop firm: sovereign/propfirm/ engine built. Paper challenge #1 active.
                  Pipeline verdict: 🟢 GO — MC pass rate >70% confirmed
                  Dashboard: ceyre-boop.github.io/quant/ict/ (Oracle-driven, live)
                  LIVE: pd_alignment weight=0 (HYP-024 confirmed anti-edge, wired 2026-05-19)
                  LIVE: Timing gate UTC 03:xx only (2026-05-22): WR=80% avgR=+2.100 vs UTC 04:xx WR=14%
Forex system:     LIVE — v014 paper scan, 5 pairs | 🏆 INSTITUTIONAL GRADE ACHIEVED
                  v014 (2026-05-25): avg_sharpe=2.0970 | target v015 > 2.1470 (+0.05 gate)
                  HYP-044 CONFIRMED: USDJPY/AUDNZD VIX gate 15→13 (+0.242 Sharpe)
                  Universal bull+VIX gate: macro rate-diff signals degrade when fear flows compete
                  Tiered thresholds: USDJPY/AUDNZD VIX>15 | EURUSD/GBPUSD/AUDUSD VIX>20
                  Per-pair (v014): USDJPY→2.979 | AUDNZD→2.246 | GBPUSD→1.885 | AUDUSD→1.665 | EURUSD→1.710
                  All 5 pairs now individually above Sharpe 1.5 (institutional grade)
                  Signal decay detector: sovereign/research/signal_decay.py (monthly)
Backtest speed:   148,193/sec (Numba JIT, 12 cores)
XGBoost models:   Both specialists trained and live
Veto pipeline:    4/5 clusters active
New tools:        PAI (46 skills), OpenSpace MCP (4 tools), GSD hooks (8 groups)

Forex version tracker:
  v001: 0.1785 | v002: 0.3551 | v002.5: ~0.38 | v003: 0.4523 | v004: 0.6260
  v005: 0.8843 → 1.0237 (trailing 1.25x, forensics v1)
  v006: 1.0237 → 1.0547 (GBPUSD per-pair 6d hold, micro-edge sweep)
  v007: 1.0547 → 1.0713 (all-pair hold sweep: 5 pairs updated)
  v008: 1.0713 → 1.1955 (USDCAD retired, +0.124 — clears +0.05 gate by 2.5×)
  v009: 1.1955 → 1.2864 (GBPJPY retired, +0.091 — clears +0.05 gate by 1.8×)
  v010: 1.2864 → 1.4396 (USDJPY regime gate VIX>15/bull, +0.153 — largest jump since v005)
  v011: 1.4396 → 1.6476 (universal bull+VIX gate all pairs, +0.208) ← INSTITUTIONAL GRADE ✓
  v012: 1.6476 → 1.7176 (+0.070)
  v013: 1.7176 → 1.8552 (+0.138)
  v014: 1.8552 → 2.0970 (+0.242) ← CURRENT LIVE (USDJPY/AUDNZD VIX 15→13)
  Target: Sharpe > 2.1470 (v015 gate, +0.05 from v014)

Unified forensics (2026-05-18): sovereign/research/unified_forensics.py
  SHARED ROOT CAUSE: Both ICT and Forex fail from PREMATURE ENTRY
  ICT: 61% of losses stop within 3 bars (entry before market commits)
  Forex: UNEXPLAINED 59 losses — macro-aligned but requires unknown third factor
  HYP-022 → HYP-026 added to hypothesis ledger (28 total)

Intelligence Architecture (2026-05-19) — 4 layers complete:
  Layer 1 — Trade Forensic Engine:   sovereign/forensics/trade_forensic_engine.py
    6-label taxonomy (TIMING/THESIS/REGIME/EXECUTION/SIZING/COMMITMENT)
    1,160 forensic records in data/forensics/trade_forensics.jsonl
  Layer 2 — Commitment Detector:     sovereign/intelligence/commitment_detector.py
    ICT gate: mkt_struct >= 1.5 → UNCOMMITTED veto (87.5% accuracy)
    London + Grade A + committed → Sharpe 3.314, MC pass 100% in-sample
    Wired into ict/pipeline.py Stage 5.7-c
  Layer 3 — Latent Feature Search:   sovereign/forensics/latent_feature_search.py
    59 forex COMMITMENT_FAILUREs tested across 5 candidate features
    No feature clears IC > 0.1 threshold — accept structural failure rate
    Counter-momentum sizing: +0.331R vs +0.107R (3× better, same 52% WR)
    VIX slope wiring confirmed in signal_engine.py (size_mult column)
  Layer 4 — Cross-System Bridge:     sovereign/intelligence/cross_system_bridge.py
    QUANT→ICT: Library threat >= 0.95 → HALT_NEW | >= 0.85 → TIGHTEN
    ICT→QUANT: 3+ stops/commitment-fails in 24h → REDUCE_CONVICTION (0.50×, 48h)
    State: data/forensics/cross_system_state.json (6h TTL)
    Wired: ict/pipeline.py Stage 0 + agent_scheduler.py every 2h
    Current: NORMAL (threat=0.00, cleared 2026-05-24)

Oracle Research Agent (2026-05-24) — autonomous loop now live:
  sovereign/agent/research_agent.py    — CREATED (was missing). Executor for all queue tasks.
  sovereign/oracle/oracle_agent.py     — CREATED. Writes SUG-### with auto_execute; fires agent immediately.
  Wiring: suggestions.json PENDING → research_agent.check_suggestions() → run in milliseconds
  Wiring: prompt_queue.json auto_execute=true → process_prompt_queue() → run without human gate
  Safety: auto_execute=False for anything touching live trading params (parameters.yml, execute_daily)
  agent_scheduler.py: check_suggestions() + process_prompt_queue() run every cycle before queue dispatch

Oracle findings confirmed and deployed (2026-05-24):
  RQ-AUTO-001 ✅ CONFIRMED: UTC 03xx Grade A WR=56.2%, avgR=+1.078 (n=16). UTC 04xx WR=14% — timing gate LIVE
  RQ-AUTO-003 ✅ COMPLETE: market_structure IC=-0.195 (anti-edge, deployed as HYP-034, weight=0 wired)
  RQ-AUTO-005 ✅ CONFIRMED: Empirical MC 97.9% pass rate (n=2000 sims, London+GradeA config)
  RQ-004     ✅ COMPLETE: Library CRITICAL fingerprint = consecutive_down_weeks (2.22 weight)
  HYP-027    ✅ CONFIRMED: USDJPY regime gate suppress signals in bull+elevated VIX
  HYP-036    ❌ REJECTED: RMT portfolio Kelly (Sharpe delta holdout=-0.103)
  SUG-001    ✅ IMPLEMENTED: Dynamic ADR threshold 1.0→1.2 during high-vol sessions

Oracle suggestion history (6 total):
  SUG-001 IMPLEMENTED | SUG-002 VETOED (AUDNZD > GBPJPY in backtest) | SUG-003 VETOED (already done)
  SUG-004 VETOED (already live at 85% ADR) | SUG-005 VETOED (stop width Sharpe-invariant)
  SUG-006 NEW — isolate Library feature criticality ranking (pending)

Prop firm:        sovereign/propfirm/ — rules engine + MC simulator + paper tracker + checklist
  MC results (2026-05-19): london_a=100% | london_all=99.7% | both clear 70% gate
  Deployment checklist: sovereign/propfirm/deployment_checklist.py
    G1 MC pass rate:     🟢 99.7-100% (DONE)
    G2 Live trades:      🔴 0/30 (collecting London+GradeA paper trades)
    G3 WR alignment:     🟡 collecting (need 10+ closed trades)
    G4 Bridge threat:    🟢 0.00 NORMAL (cleared 2026-05-24)
    G5 Non-bust days:    🟡 collecting
    VERDICT: WAIT — buy when G2+G4 flip GREEN
  Paper challenge #1: ACTIVE (Day 0, balance=$100k, floor=$92k, target=$108k)
  CLI: python3 sovereign/propfirm/paper_challenge.py --status/--eod/--trade
       python3 scripts/agent_scheduler.py --checklist

ICT prop challenge math:
  Edge: TP2=16.8%@4.5R | TP1=13.0%@1.5R | STOP=70.2%@-1.0R | EV=+0.40R/trade
  London-only: avgR=+0.471 | WR=32% | Sharpe proxy=1.225
  Best config: London + Grade A | avgR=+0.840 | WR=41%

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

Alexandrian Library: LIVE in orchestrator + ICT engine (primary market intelligence)
  sovereign/risk/alexandrian_library.py — 10 volumes, 63 entries, full market circumstance coverage
  Volumes:        I Crashes | II Rate Cycles | III Bull Regimes | IV Currency Crises | V Vol Regimes
                  VI Econ Cycles | VII Liquidity | VIII Commodities | IX Geopolitical | X Sector Rotation
  Architecture:   23-feature cosine similarity per entry, multi-volume convergence amplifier
                  3+ volumes >0.60 similarity → composite score amplified ×1.25
                  Returns LibraryInsight: primary_regime, threat_score, size_modifier, advisory
  Threat levels:  NORMAL(1.0×) ELEVATED(0.75×) WARNING(0.50×) DANGER(0.25×) CRITICAL(0.0×)
  Auto-learns:    any live drawdown ≥8% → added to library + MarketMemory simultaneously
  Current read:   ASIAN_CURRENCY_CONTAGION | CRITICAL | threat=1.00 → ICT HALT_NEW via bridge
                  Lo Level 3: 0.50× | Kelly cap: 2.0× | Bridge blocks all new ICT entries

CS229 Stack — All Loops Closed:
  Softmax / ICA / KMeans / Pegasus — all live, all online-learning from ledger

Library ↔ CS229 Integration (5 points, all LIVE — see prior session notes)

Trading Memory:   LIVE in orchestrator (legacy fallback when library not loaded)
  sovereign/risk/market_memory.py — 20 historical crash events, 23-feature cosine similarity

Tests:            555/593 passing (38 pre-existing failures: sklearn/fredapi missing, bs_call API mismatch, kelly pegasus_params bug)
                  ICT pipeline suite: 21/21 (was 15/21 before bridge mock conftest)
                  Run: python3 -m pytest tests/ -q

Intelligence Architecture — ALL 5 LAYERS COMPLETE (2026-05-19):
  Layer 1  Trade Forensic Engine     sovereign/forensics/trade_forensic_engine.py
  Layer 2  Commitment Detector       sovereign/intelligence/commitment_detector.py
  Layer 3  Latent Feature Search     sovereign/forensics/latent_feature_search.py
  Layer 4  Cross-System Bridge       sovereign/intelligence/cross_system_bridge.py
  Layer 5  Prop Deployment Checklist sovereign/propfirm/deployment_checklist.py

HYP-024 DEPLOYED (2026-05-19): pd_alignment weight 1.5 → 0.0 in config/ict_params.yml
  Anti-edge confirmed: pd_alignment>0 = 20% WR | pd_alignment=0 = 35% WR
  Code default was already changed; YAML override now updated — weight is live

Next milestones:
  1. PROP CHALLENGE: buy when G2 (30 live trades) + G4 (threat < 0.85) both GREEN
     → checklist auto-posts URGENT when ready: python3 scripts/agent_scheduler.py --checklist
  2. ICT dashboard: add cross-system bridge panel (ict_mode, quant_signal, threat score)
  3. Forex v007: run scripts/v007_prompt.md in Claude Code — per-pair hold optimization
     AUDUSD 5d / EURUSD 5d / AUDNZD 7d / GBPJPY 5d (target Sharpe > 1.10 from 1.0547)
  4. Forex: wire calendar edges as SIZE BOOSTS on top of v006 macro signal (HYP queued)
  5. Forex Sharpe gap: 1.0547 → 1.5 target | remaining gap = 0.445
  6. Research queue: E1 at 60-day hold | March JPY full history | carry pair optimization

---

*This file is the agent handshake. Both Claude Code and Codex 
read it. It removes the human from routine task routing.*
*Update CURRENT SYSTEM STATUS at the end of every session.*

---

## AUTONOMOUS RESEARCH AGENT

Designed 2026-05-16. Three Claude Code sessions to make live.

### Philosophy

Automated = does what you told it to do.
Autonomous = figures out what needs doing, does it, tells you what it found.

The agent is not a trading bot. It is a research partner that works while
you sleep, surfaces findings when you wake up, and flags problems before
you ask. It does not make decisions. It validates hypotheses, checks health,
and writes plain-English findings to a dashboard.

### Architecture

```
LAYER 1 — RESEARCH AGENT (runs every 2-4 hours via launchd)
  sovereign/agent/research_agent.py

  Every cycle it:
    1. Runs health check (all data feeds + Ollama + paper account)
    2. Picks highest-priority QUEUED task from research_queue.json
    3. Runs it (backtest, validation, data check, prop sim)
    4. Writes finding to findings.jsonl
    5. Updates hypothesis_ledger.json (CONFIRMED/REJECTED/QUEUED)
    6. Writes any urgent messages to messages_to_colin.json
    7. Adds follow-up tasks to queue based on what it found

LAYER 2 — RESEARCH DASHBOARD (your morning briefing)
  frontend/dashboard_research.html

  Sections:
    - SYSTEM HEALTH: green/red for every data source
    - OVERNIGHT SUMMARY: 3-5 bullets of what ran and found
    - HYPOTHESIS LEDGER: every hypothesis ever tested with status
    - MESSAGES TO COLIN: URGENT/IMPORTANT/FYI sorted by priority
    - RESEARCH QUEUE: what's next (Colin can reprioritize)
    - VERSION TRACKER: Sharpe progress v001→current toward 1.5

LAYER 3 — COMMUNICATION PROTOCOL (flat files, no DB)
  data/agent/findings.jsonl          — append-only finding log
  data/agent/health.json             — current system health snapshot
  data/agent/hypothesis_ledger.json  — all hypotheses + status + results
  data/agent/messages_to_colin.json  — things needing human attention
  data/agent/research_queue.json     — prioritized task queue

LAYER 4 — SCHEDULER (usage-aware)
  scripts/agent_scheduler.py

  Health check:   every 2 hours (lightweight)
  Research task:  every 4 hours (heavy compute)
  Dashboard push: every 30 minutes (write JSON files)
  Never burns Claude Pro budget — runs local Python only
```

### Research Queue (pre-populated)

Priority order for the agent to work through:

1. E1 rate divergence at 60-day hold with multi-layer confirmation
   (state_allocator E1 failed at 20d hold; hypothesis: 60d changes sign)
2. March JPY repatriation: full history 2010-2024, not just 2015-2024
   (test if BOJ interventions are a special case or permanent override)
3. Stop width optimization: 0.08 vs 0.15 vs 0.25 ATR on all 8 forex pairs
4. Library feature transparency: what features drive CRITICAL reads
5. Carry base: optimal pair selection across all 28 G10 combinations
6. ICT walk-forward stability: re-run after every 10 new paper trades
7. Calendar edges as SIZE BOOSTS: test C1/C2 layered on v004 direction

### Hypothesis Ledger (seeded from this session)

✅ CONFIRMED: Carry base generates positive EV (59% WR, +0.311R)
✅ CONFIRMED: Quarter-end rebalancing works at small sample (60% WR, n=15)
✅ CONFIRMED: v004 macro system is the best confirmed forex edge (Sharpe 0.801)
✅ CONFIRMED: ICT FVG limit entry has positive EV (+0.40R/trade, replicated 2 windows)
✅ CONFIRMED: Walk-forward B (most recent 12mo) validates ICT edge (76% vs MC 76.8%)
❌ REJECTED:  CPI surprise fade as standalone entry (43% WR, -0.320R, n=63)
❌ REJECTED:  Post-CB drift before price confirmation (36% WR, -0.517R, n=14)
❌ REJECTED:  Rate divergence at 20-day hold (44% WR, -0.186R avg, n=312)
❌ REJECTED:  March JPY in isolation 2015-2024 (26% WR, swamped by BOJ/FED)
❌ REJECTED:  Quarter-end fade at scale (39% WR, n=366 — small sample was noise)
❌ REJECTED:  State allocator outperforms v004 (Sharpe -0.12 vs v004 0.801)
❌ REJECTED:  Confirmation protocol outperforms v004 (Sharpe -0.17)
🔄 TESTING:   E1 at 60-day hold with multi-layer confirmation
🔄 TESTING:   Calendar edges as v004 size boosts (not standalone)
📋 QUEUED:    ICT walk-forward stability with live paper trades (need 30+ trades)
📋 QUEUED:    FunderPro challenge attempt (awaiting 🟢 GO from pipeline)

### Message Format

Agent writes to messages_to_colin.json in this format:
  🔴 URGENT:    "Ollama has not responded since 14:32. Run: ollama serve"
  🟡 IMPORTANT: "Veto ledger shows 73 ATR_TOO_LOW blocks on AUDUSD (24hrs).
                 RBA meeting in 3 days — compression likely cause."
  🟢 FYI:       "E1 60-day backtest complete. Sharpe improved from -0.186 to +0.41.
                 Hypothesis partially confirmed. Phase 2 test added to queue."

### Build Order

Session 1: frontend/dashboard_research.html (reads flat JSON files, dark theme)
Session 2: sovereign/agent/research_agent.py (runs tasks, writes findings)
Session 3: scripts/agent_scheduler.py (launchd integration, usage-aware)

### Your Week Optimized

WHEN YOU'RE HERE (Claude Pro sessions, ~10/week):
  Open dashboard_research.html → read overnight summary
  Paste interesting findings here → interpret together
  Agree on next experiment → write Claude Code prompt
  Run it (uses Claude Code budget, not Pro)

WHEN YOU'RE AWAY:
  Agent runs every 2-4 hours
  Tests queue items, checks health, writes findings
  Flags anything urgent in messages_to_colin.json

NET RESULT: 24/7 research with ~10 hours of your attention per week

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
