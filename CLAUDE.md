# SOVEREIGN / QUANT — AGENT OPERATING PROCEDURE
## Claude Code + Codex Autonomous Collaboration Protocol

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
Forex backtest:   11/11 pairs at 30+ trades, 7/11 positive Sharpe
                  Best: GBPUSD +0.97 | EURUSD +0.92 | AUDNZD +0.47
Signal layers:    5 layers — Macro + CPI fade + Calendar + CB events + DXY overlay
                  + COT gate (Dalio Q3) on entry sizing
Charts:           logs/charts/ — 4 PNGs via plot_forex_results.py
Missing edges:    None from cause-effect map — all 7 edges implemented
Next milestone:   EURJPY/USDJPY SHORT when CHOCH confirms on daily

---

*This file is the agent handshake. Both Claude Code and Codex 
read it. It removes the human from routine task routing.*
*Update CURRENT SYSTEM STATUS at the end of every session.*
