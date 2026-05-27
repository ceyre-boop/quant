# Oracle & Scheduler Infrastructure Exploration

## 1. SCHEDULING MECHANISM

**Implementation: launchd (macOS native)**

- **File**: `/Users/taboost/quant/scripts/agent_scheduler.py`
- **Launch interval**: 2 hours (7200 seconds) for health checks, 4 hours (14400 seconds) for full research cycles
- **Plist generation**: `agent_scheduler.py --gen-plist` outputs XML plist configuration
- **Key timer**: `<key>StartInterval</key><integer>7200</integer>`
- **Label**: `com.alta.agent_scheduler`

The scheduler runs via macOS launchd, NOT via the `schedule` library or cron. It is configured via XML plist files and runs periodically every 2-4 hours.

### Time Gates (ET timezone)
- **Prime time (agent preferred)**: 00:00–09:00 ET, 23:00–24:00 ET
- **Work hours (agent backs off)**: 09:00–23:00 ET

### Budget-Based Run Modes
- **FULL**: Heavy backtests, data pulls, simulations (weekly_remaining > 50k)
- **HEAVY_OK**: Run if not during work hours (20k–50k remaining)
- **LIGHT_ONLY**: Health + ledger only (< 20k remaining)
- **SKIP**: Nothing to do (exhausted budget or no eligible tasks)

---

## 2. EXISTING HEALTH CHECK PATTERN

The scheduler includes a comprehensive health check system (`run_health_check()`) with 17+ component checks:

### Pattern Used for `_check_*` Functions

Each check function returns a **tuple of (status_str, detail_str)**:
- **Status values**: `"GREEN"`, `"YELLOW"`, `"RED"`
- **Returns**: `(status, detail)` tuple
- **Examples**:

```python
def _check_duckdb() -> tuple:
    try:
        import duckdb
        db_path = ROOT / "data" / "sovereign.duckdb"
        if db_path.exists():
            con = duckdb.connect(str(db_path), read_only=True)
            con.close()
            return "GREEN", "connected to sovereign.duckdb"
        con = duckdb.connect(":memory:")
        con.close()
        return "GREEN", f"v{duckdb.__version__} ready (in-memory)"
    except ImportError:
        return "YELLOW", "duckdb not installed"
    except Exception as e:
        return "RED", str(e)

def _check_decision_logs() -> tuple:
    """Check decision logging pipeline health."""
    try:
        from sovereign.agent.research_agent import check_decision_log_health
        issues = check_decision_log_health()
        if not issues:
            return "GREEN", "all decision logs healthy"
        critical = [i for i in issues if i.startswith("SCHEMA") or i.startswith("CRITICAL")]
        if critical:
            return "RED", critical[0]
        return "YELLOW", issues[0]
    except Exception as e:
        return "YELLOW", f"health check unavailable: {e}"
```

### Pattern for Adding a New Periodic Task

1. Define a `_check_*()` function returning `(status, detail)` tuple
2. Add to `checks` dict in `run_health_check()`
3. Scheduler automatically aggregates all statuses into `health["overall"]`
4. Post messages to user if RED status detected

Example skeleton:
```python
def _check_my_system() -> tuple:
    try:
        # Check logic here
        return "GREEN", "status message"
    except SomeError:
        return "YELLOW", "warning message"
    except Exception as e:
        return "RED", str(e)
```

---

## 3. DECISION LOG & JSONL LOADING PATTERNS

### File Locations

**Decision Logs** (closed trades with reasoning + outcomes):
- Path: `/Users/taboost/quant/data/decision_logs/decisions_2026_05.jsonl`
- Format: JSONL (one JSON object per line)
- Schema: See section 3.2 below

**Trade Forensics** (enriched forensic classification):
- Path: `/Users/taboost/quant/data/forensics/trade_forensics.jsonl`
- Format: JSONL
- Schema: See section 3.3 below

**Trade Ledger** (historical trades, CSV format):
- Paths: `/Users/taboost/quant/data/ledger/trade_ledger_YYYY_MM.csv` (historical)
- ICT ledger: `/Users/taboost/quant/data/ledger/ict_veto_ledger_2026_05.jsonl`

### 3.1 Core JSONL Loading Function

From `harvest_cycle.py`, the standard pattern:

```python
def _load_recent_ledger_trades(hours: int = 24) -> list[dict]:
    """Load closed trades from trade ledger files within rolling window."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    trades = []
    for path in sorted(LEDGER_DIR.glob("trade_ledger_*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    try:
                        t = json.loads(line)
                        ts_str = t.get("closed_at") or t.get("timestamp", "")
                        if not ts_str:
                            continue
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts > cutoff:
                            trades.append(t)
                    except Exception:
                        pass
        except Exception:
            pass
    return trades
```

**Key pattern**: Read JSONL line-by-line, parse ISO timestamps, filter by rolling window (hours).

### 3.2 Decision Log Schema (entry_timestamp + outcome)

**File**: `/Users/taboost/quant/data/decision_logs/decisions_2026_05.jsonl`

Full list of fields available for pattern analysis:

```
entry_timestamp        # ISO 8601, when trade entered
exit_timestamp         # ISO 8601, when trade closed (null if open)
system                 # "ICT" | "EQUITY" | other
pair                   # "GBPUSD", "AAPL", etc.
direction              # "LONG" | "SHORT"
entry_level            # float, entry price
stop_loss              # float, SL price
grade                  # "A" | "B" | "C"
session                # "LONDON session" | null
score                  # float, ICT/strategy score
vix_at_entry           # float | null
rate_differential_zscore  # float | null (for FX macro strength)
cot_percentile         # float | null (Commitment of Traders)
library_match          # string like "2022_RATE_SHOCK at 0.61"
commitment_score       # float 0.0–1.0 (price action quality)
bars_since_signal      # int, how many bars since signal fired
adr_pct_used           # float | null
risk_pct               # float, account risk %
risk_dollars           # float, dollar risk
why_this_trade         # string, reasoning for entry
why_this_size          # string, reasoning for position size
signal_layers_active   # list of strings (signal layer names)
component_scores       # dict with scores for kill_zone, sweep, etc.
confirmations          # list of confirmation reasons
missing                # list of missing confirmations
outcome                # "COMMITMENT_FAILURE" | null (when closed, or while open: null)
r_realized             # float, realized R-multiple (null if open)
```

### 3.3 Trade Forensics Schema (for Oracle validation)

**File**: `/Users/taboost/quant/data/forensics/trade_forensics.jsonl`

Used in validation cycle for statistical testing:

```
trade_id              # unique ID
system                # "ICT" | "EQUITY"
pair                  # trading pair
direction             # "LONG" | "SHORT"
grade                 # "A" | "B" | "C"
session               # trading session label | null
score                 # float score
outcome               # "WIN" | "LOSS"
pnl_r                 # float, realized R-multiple
failure_label         # "TIMING_FAILURE", "THESIS_FAILURE", etc.
hold                  # int, bars/days held
entry_date            # ISO date
hold_days             # float, days in trade
hold_minutes          # float, minutes in trade
mfe_ratio             # float, max favorable excursion ratio
mae_ratio             # float, max adverse excursion ratio
momentum_5d           # float | null
vix_slope             # float | null
win_driver            # string (short description of why it won)
features              # dict with computed features for decision tree analysis
commitment_score      # float 0.0–1.0
```

### 3.4 Utility Functions for Loading Entries from JSONL

**In `harvest_cycle.py`**:

```python
def _load_recent_forensics(hours: int = 24) -> list[dict]:
    """Load enriched forensic records within rolling window."""
    if not FORENSICS_FILE.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    records = []
    with open(FORENSICS_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                ts_str = r.get("entry_time", "")
                if not ts_str:
                    continue
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff:
                    records.append(r)
            except Exception:
                pass
    return records
```

**In `validation_cycle.py`**:

```python
def _load_all_forensics() -> list[dict]:
    if not FORENSICS_FILE.exists():
        return []
    records = []
    with open(FORENSICS_FILE) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records
```

**In `reflect_cycle.py`**:

```python
def _load_decision_log_summary(days: int = 7, max_entries: int = 20) -> str:
    """Load recent decision log entries that have outcomes filled."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = []

    if not DECISION_LOG_DIR.exists():
        return "No decision logs yet..."

    for log_file in sorted(DECISION_LOG_DIR.glob("decisions_*.jsonl")):
        try:
            for line in log_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                # Only include closed trades (outcome filled)
                if not rec.get("outcome"):
                    continue
                try:
                    ts = datetime.fromisoformat(rec["entry_timestamp"].replace("Z", "+00:00"))
                    if ts < cutoff:
                        continue
                except Exception:
                    pass
                # Compact projection — only fields Oracle needs
                entries.append({
                    "pair": rec.get("pair"),
                    "system": rec.get("system"),
                    "commitment_score": rec.get("commitment_score"),
                    "r_realized": rec.get("r_realized"),
                    # ... more fields
                })
        except Exception:
            continue

    entries = entries[-max_entries:]  # Newest first, cap at max_entries
    return header + json.dumps(entries, indent=2)
```

---

## 4. ORACLE API CALL PATTERN

### File: `/Users/taboost/quant/sovereign/oracle/reflect_cycle.py`

**Method**: Direct Anthropic SDK (`anthropic.Anthropic`)

```python
def run_reflect(harvests: list[dict], date: Optional[str] = None) -> dict:
    """Call Oracle with compact prompt."""
    from sovereign.oracle.oracle_agent import _load_dotenv
    _load_dotenv()

    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()

    # Parse JSON from response
    try:
        reflection = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract JSON block
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            reflection = json.loads(match.group())
        else:
            reflection = {"raw_response": raw_text, "parse_error": True}

    output = {
        "date": date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "estimated_cost_usd": round(
            response.usage.input_tokens * 0.00000025 +
            response.usage.output_tokens * 0.00000125, 6
        ),
        "reflection": reflection,
    }

    out_path = REFLECTIONS_DIR / f"{date}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return output
```

### Key Details

- **Client initialization**: `anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])`
- **Model**: `claude-haiku-4-5` (cheapest for frequent cycles)
- **Max tokens**: 1024 (structured JSON response with schema)
- **Response handling**: Robustly extracts JSON even if wrapped in markdown
- **Cost tracking**: Computes estimated cost based on token counts (input: $0.00000025/token, output: $0.00000125/token)
- **Env loading**: Uses `_load_dotenv()` from `oracle_agent.py` to load .env file

### Reusable Wrapper

No explicit `call_oracle()` wrapper exists, but the pattern is:

```python
def _get_api_key():
    from sovereign.oracle.oracle_agent import _load_dotenv
    _load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return api_key

def call_oracle(prompt: str, model: str = "claude-haiku-4-5", max_tokens: int = 1024) -> dict:
    """Reusable Oracle API wrapper."""
    import anthropic
    api_key = _get_api_key()
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text": response.content[0].text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": round(
            response.usage.input_tokens * 0.00000025 +
            response.usage.output_tokens * 0.00000125, 6
        ),
    }
```

---

## 5. EXISTING DATA DIRECTORIES

### Oracle Data Structure

```
data/oracle/
├── daily_harvest_YYYY-MM-DD.json      # Phase 1 output (trade summaries)
├── reflections/YYYY-MM-DD.json        # Phase 2 output (Oracle suggestions)
├── validations/YYYY-MM-DD.json        # Phase 3 output (test results)
├── validations/
│   └── YYYY-MM-DD.json
├── pending_implementations/           # Phase 4 output (implementation prompts)
│   └── YYYY-MM-DD.md                 # Claude Code prompt for feature implementation
├── pending_corrections/               # (DOES NOT EXIST YET)
├── pulses/                            # (DOES NOT EXIST YET)
├── proven_research.json               # Lessons passed all tests
├── regime_study_2026_05_22.json       # Market regime analysis
└── reasoning_analysis/                # Monthly reasoning pattern clustering
    └── YYYY-MM.json
```

### Directories That Need to Be Created

For new oracle components:

```
data/oracle/pulses/                   # If building "pulse" collection cycle
├── YYYY-MM-DD.json                   # Daily pulse snapshot
└── ...

data/oracle/pending_corrections/      # If building correction/refinement cycle
├── YYYY-MM-DD.json
└── ...
```

Both directories **do not currently exist** and would be created on first use by the new component.

---

## 6. TIME WINDOW FILTERING IN `reflect_cycle.py`

### Pattern Used in `_load_decision_log_summary()`

```python
def _load_decision_log_summary(days: int = 7, max_entries: int = 20) -> str:
    """Load recent decision log entries that have outcomes filled."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)  # <-- filter logic
    entries = []

    if not DECISION_LOG_DIR.exists():
        return "No decision logs yet..."

    for log_file in sorted(DECISION_LOG_DIR.glob("decisions_*.jsonl")):
        try:
            for line in log_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                # Filter 1: Only include closed trades (outcome filled)
                if not rec.get("outcome"):
                    continue
                # Filter 2: Only include trades within time window
                try:
                    ts = datetime.fromisoformat(rec["entry_timestamp"].replace("Z", "+00:00"))
                    if ts < cutoff:  # <-- skip older entries
                        continue
                except Exception:
                    pass
                entries.append({...})
        except Exception:
            continue

    entries = entries[-max_entries:]  # Cap at max_entries
    return header + json.dumps(entries, indent=2)
```

### Key Reusable Logic

This time-window pattern is **identical** in:
- `harvest_cycle.py`: `_load_recent_ledger_trades(hours=24)`
- `harvest_cycle.py`: `_load_recent_forensics(hours=24)`
- `validation_cycle.py`: `run_monthly_monitor()` uses `timedelta(days=30)` for 30-day window
- `reflect_cycle.py`: `_load_decision_log_summary(days=7)`

**Generic pattern for new code**:

```python
from datetime import datetime, timezone, timedelta

def load_entries_in_window(path, timestamp_field: str, hours: int = None, days: int = None) -> list[dict]:
    """Load JSONL entries within a rolling time window."""
    if hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days or 7)
    
    entries = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                ts_str = rec.get(timestamp_field, "")
                if not ts_str:
                    continue
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff:
                    entries.append(rec)
            except Exception:
                pass
    return entries
```

---

## 7. LAST-RUN TRACKING MECHANISM

**No dedicated "last run" tracking currently exists.**

Instead, last-run state is inferred from:

1. **Last harvest file**: `data/oracle/daily_harvest_YYYY-MM-DD.json`
   - Check most recent file's `date` field
   - Pattern: `max(glob("daily_harvest_*.json"))` by filename

2. **Last reflection file**: `data/oracle/reflections/YYYY-MM-DD.json`
   - Similar pattern for last reflection

3. **Last validation file**: `data/oracle/validations/YYYY-MM-DD.json`
   - Last test run date

4. **Wisdom file timestamps**: In `I_am_a_good_trader.md`, each lesson has:
   - `**Last validated:** YYYY-MM-DD` field
   - `**Health:** 🟢 HEALTHY` field (updated by monthly monitor)

### To Implement Last-Pulse Tracking

Create a lightweight state file:

```python
PULSE_STATE_PATH = ROOT / "data" / "oracle" / ".pulse_state.json"

def get_last_pulse_time() -> datetime:
    """Get timestamp of last pulse collection."""
    if PULSE_STATE_PATH.exists():
        data = json.loads(PULSE_STATE_PATH.read_text())
        ts_str = data.get("last_pulse_time")
        if ts_str:
            return datetime.fromisoformat(ts_str)
    return datetime.now(timezone.utc) - timedelta(days=999)  # Never run

def save_pulse_state(timestamp: datetime) -> None:
    """Record that a pulse was collected at this time."""
    PULSE_STATE_PATH.write_text(json.dumps({
        "last_pulse_time": timestamp.isoformat(),
        "last_cycle_date": timestamp.strftime("%Y-%m-%d"),
    }, indent=2))
```

---

## 8. ORACLE LEARNING CYCLE ORCHESTRATION

**File**: `/Users/taboost/quant/sovereign/oracle/oracle_cycle.py`

The full learning cycle runs in this sequence:

### Daily Cycle (02:00 AM ET)

1. **Phase 1: HARVEST** (free)
   - Runs `run_harvest()` from `harvest_cycle.py`
   - Outputs: `data/oracle/daily_harvest_YYYY_MM_DD.json`
   - Processes: Last 24h of closed trades
   - Metrics: win_rate, avg_r, failure_distribution, anomalies

2. **Phase 2: REFLECT** (~2 cents)
   - Runs `run_reflect()` from `reflect_cycle.py`
   - Calls Claude Haiku API
   - Outputs: `data/oracle/reflections/YYYY_MM_DD.json`
   - Input: Last 7 days of harvests + decision logs + proven research
   - Output: Candidate lesson proposal with testable_rule

3. **Phase 3: TEST** (free)
   - Runs `run_validation()` from `validation_cycle.py`
   - Outputs: `data/oracle/validations/YYYY_MM_DD.json`
   - Tests: Three statistical tests on candidate lesson
     - Test A: Delta Sharpe > 0.05
     - Test B: Two-sample t-test p < 0.05
     - Test C: Holdout replication (80% of training effect)

4. **Phase 4: CODIFY** (free)
   - Runs `run_codify()` from `codify_cycle.py`
   - Action: If verdict == VALIDATED, add to `I_am_a_good_trader.md`
   - Outputs:
     - Updated `I_am_a_good_trader.md` (max 10 lessons)
     - `data/oracle/pending_implementations/YYYY_MM_DD.md` (Claude Code prompt)
     - Updated `data/oracle/proven_research.json`
   - Retirement: If lesson count > 10, retire weakest to `I_was_a_good_trader.md`

### Monthly Cycle (Phase 5: MONITOR)

- Re-validates all active lessons against last 30 days
- Updates health status in wisdom file
- Retires lessons with decay_ratio < 0.0 (harmful)
- Runs reasoning analysis clustering (via `sovereign/forensics/reasoning_analyzer.py`)

---

## 9. SUMMARY: WHAT EXISTS VS WHAT NEEDS CREATING

### Existing Infrastructure (Ready to Extend)

✅ **Scheduling**: launchd-based periodic task execution
✅ **Health checks**: 17+ component check functions with (status, detail) tuple pattern
✅ **JSONL loading**: Standardized utilities for reading trades, forensics, decision logs
✅ **Time-window filtering**: Generic pattern for rolling-window entry filtering
✅ **Oracle API calls**: Anthropic SDK integration with token tracking and cost estimation
✅ **Learning cycle orchestration**: 4-phase daily cycle (harvest → reflect → test → codify)
✅ **Monthly monitoring**: Lesson health re-validation and decay tracking
✅ **Data directories**: `data/oracle/` with subdirs for harvests, reflections, validations, implementations

### Missing Infrastructure (Needs Creation)

❌ **`data/oracle/pulses/`** — directory for pulse collection snapshots
❌ **`data/oracle/pending_corrections/`** — directory for correction/refinement items
❌ **Pulse collection cycle** — new orchestration phase
❌ **Correction/refinement cycle** — new orchestration phase
❌ **Last-run tracking** — `.pulse_state.json` or similar state file
❌ **Correction deduplication** — logic to avoid re-proposing same corrections

---

## 10. RECOMMENDED PATTERNS FOR NEW ORACLE COMPONENTS

### Template for Adding a New Check Function

```python
def _check_my_component() -> tuple:
    """Check status of my new component."""
    try:
        from my_module import check_health
        status = check_health()
        if status:
            return "GREEN", f"my component is working: {status}"
        return "YELLOW", "component available but no data yet"
    except ImportError:
        return "YELLOW", "my component not installed"
    except Exception as e:
        return "RED", str(e)

# Add to checks dict in run_health_check():
# checks = {
#     "my_component": _check_my_component,
#     # ... rest of checks
# }
```

### Template for New JSONL Loading Utility

```python
def load_my_entries(time_window_hours: int = 24) -> list[dict]:
    """Load entries from my JSONL file within time window."""
    MY_FILE = ROOT / "data" / "my_data" / "entries.jsonl"
    if not MY_FILE.exists():
        return []
    
    cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
    entries = []
    
    with open(MY_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
                ts_str = rec.get("timestamp_field", "")
                if not ts_str:
                    continue
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff:
                    entries.append(rec)
            except Exception:
                pass
    
    return entries
```

### Template for New Oracle API Call

```python
def run_my_cycle(dry_run: bool = False) -> dict:
    """Call Oracle for my use case."""
    from sovereign.oracle.oracle_agent import _load_dotenv
    _load_dotenv()

    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    
    if dry_run:
        return {"status": "DRY_RUN", "cost_usd": 0.0}

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": my_prompt}],
    )

    result = {
        "status": "COMPLETE",
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": round(
            response.usage.input_tokens * 0.00000025 +
            response.usage.output_tokens * 0.00000125, 6
        ),
        "response": response.content[0].text,
    }

    return result
```

