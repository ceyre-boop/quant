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
