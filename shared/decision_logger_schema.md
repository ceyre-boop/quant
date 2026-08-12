# Decision Logger Schema

**Every live entry/exit is logged with full context. The Oracle reads this to learn.**

*Last updated: 2026-08-12*

---

## Purpose

The decision logger captures what the system was thinking at the moment of entry. It records:
1. Market state (pair, timeframe, price, regime)
2. Entry signal (which hypothesis triggered it)
3. Decision metrics (commitment score, rate differential, library match)
4. Sized position (conviction, dollar amount, risk)
5. Exit outcome (when the trade closes, update_outcome() records P&L and lessons)

**The Oracle reads this stream daily and reflects: which decisions predicted winners? Which predicted losers? What should we learn?**

---

## Schema (Entry)

Each entry is logged via `sovereign/intelligence/decision_logger.py::log()`:

```json
{
  "trade_id": "GBPUSD-2026-08-12-0930",
  "timestamp_utc": "2026-08-12T09:30:00Z",
  "timestamp_et": "2026-08-12T05:30:00Z",
  
  "pair": "GBPUSD",
  "system": "forex",
  "hypothesis_id": "HYP-045",
  "signal_reason": "Rate differential +145bps (BOE 5.25%, Fed 5.00%); GBPUSD in 6-month uptrend",
  
  "market_state": {
    "price_entry": 1.2845,
    "atr_14d": 0.0087,
    "recent_volatility_pct": 12.3,
    "regime": "rate_trending",
    "regime_confidence": 0.78
  },
  
  "decision_metrics": {
    "commitment_score": 0.72,
    "rate_differential_bps": 145,
    "library_match": 4,
    "bars_since_signal": 2,
    "computed_conviction": 0.75
  },
  
  "position": {
    "size_usd": 562.50,
    "leverage_multiplier": 0.75,
    "per_trade_cap_checked": true,
    "carry_heat_before": 1800.00,
    "carry_heat_after": 2362.50,
    "carry_heat_approved": true,
    "protective_stop_loss": 1.2720,
    "target_exit": 1.2950,
    "hold_days": 5
  },
  
  "context_metadata": {
    "account_equity": 100000,
    "drawdown_current_pct": 1.2,
    "throttle_factor": 1.0,
    "session": "london_am",
    "fomc_window": false,
    "margin_availability_pct": 45
  },
  
  "execution": {
    "executed": true,
    "execution_price": 1.2846,
    "execution_time_utc": "2026-08-12T09:31:02Z",
    "broker": "oanda",
    "order_id": "oanda-5847293"
  },
  
  "live_or_paper": "live",
  "logged_by": "sovereign_forex_scanner",
  "notes": "Rate differential highest in 90d. GBPUSD breaking out of 30d range. High conviction setup."
}
```

---

## Schema (Exit/Update Outcome)

When the trade closes, call `update_outcome()` to record the result:

```json
{
  "trade_id": "GBPUSD-2026-08-12-0930",
  "outcome_timestamp_utc": "2026-08-15T14:30:00Z",
  "outcome_timestamp_et": "2026-08-15T10:30:00Z",
  
  "exit_reason": "target_hit",
  "exit_price": 1.2950,
  "exit_time_utc": "2026-08-15T14:29:45Z",
  
  "pnl": {
    "pnl_pips": 105,
    "pnl_usd": 591.30,
    "pnl_pct_return": 1.05,
    "pnl_pct_of_account": 0.59
  },
  
  "hold_duration": {
    "calendar_days": 3.2,
    "trading_hours": 45,
    "bars_held": 18
  },
  
  "outcome_analysis": {
    "prediction_was_correct": true,
    "commitment_score_predictive": true,
    "rate_diff_held": true,
    "regime_stayed_consistent": true,
    "lessons_learned": [
      "Commitment 0.7+ predicts winners when rate differential >140bps",
      "Library match (4+ priors) adds 0.15 to conviction multiplier",
      "GBPUSD mean-reversion targets hit in <4 calendar days on avg"
    ]
  },
  
  "oracle_feedback": {
    "expected_sharpe_contribution": 0.15,
    "sharpe_prediction_error": 0.02,
    "regime_classification_accuracy": 0.92,
    "recommendation": "This trade archetype can be sized larger. Increasing conviction multiplier for rate_diff>140 + library_match>3 from 0.75 to 0.85."
  }
}
```

---

## Field Reference (Entry)

| Field | Type | Notes |
|-------|------|-------|
| `trade_id` | String | Unique ID (PAIR-DATE-HHMM). Used to link entry → outcome. |
| `timestamp_utc` | ISO8601 | Entry time (UTC, canonical). |
| `timestamp_et` | ISO8601 | Entry time (ET, for human readability). |
| `pair` | String | Trading pair (GBPUSD, ICT setup ID, etc.). |
| `system` | String | forex, ict, equity, yield, etc. |
| `hypothesis_id` | String | HYP-NNN that triggered this entry. |
| `signal_reason` | String | Plain-English: why did the system think this was a good trade? |
| `market_state` | Object | Price, volatility, regime, confidence. Input data for Oracle. |
| `decision_metrics` | Object | Commitment score, rate diff, library match, bars since signal, conviction. |
| `position` | Object | Size, leverage, stops, targets, hold duration. |
| `context_metadata` | Object | Account state, drawdown, throttle, session, margin. |
| `execution` | Object | Filled? At what price/time? Which broker? |
| `live_or_paper` | String | live, paper, shadow, backtest. |
| `notes` | String | Human commentary (optional). |

---

## Field Reference (Exit/Outcome Update)

| Field | Type | Notes |
|-------|------|-------|
| `trade_id` | String | Must match the entry log's trade_id. |
| `outcome_timestamp_utc` | ISO8601 | When trade closed. |
| `exit_reason` | String | Enum: target_hit, stop_hit, time_exit, manually_closed, system_abstain |
| `exit_price` | Float | Actual exit price. |
| `pnl.pnl_usd` | Float | Profit/loss in dollars. |
| `pnl.pnl_pct_return` | Float | Return as %. |
| `hold_duration` | Object | How long was the position held? |
| `outcome_analysis` | Object | Was the commitment score predictive? Did the regime hold? Lessons learned? |
| `oracle_feedback` | Object | Recommendations from Oracle (sizing adjustments, conviction tweaks). |

---

## Usage Pattern

### At Entry (Mandatory)

```python
from sovereign.intelligence.decision_logger import log as decision_log

decision_log(
    pair="GBPUSD",
    system="forex",
    hypothesis_id="HYP-045",
    signal_reason="Rate differential +145bps in uptrend",
    commitment_score=0.72,
    rate_differential_bps=145,
    library_match=4,
    conviction=0.75,
    size_usd=562.50,
    protective_stop=1.2720,
    target_exit=1.2950,
    hold_days=5,
)
```

### At Exit (Mandatory)

```python
from sovereign.intelligence.decision_logger import update_outcome

update_outcome(
    trade_id="GBPUSD-2026-08-12-0930",
    exit_reason="target_hit",
    exit_price=1.2950,
    pnl_usd=591.30,
)
```

**CRITICAL:** Missing `update_outcome()` calls are silent data loss. The Oracle cannot learn from trades that don't get recorded. This is a non-negotiable gate: every live entry must receive an outcome update when it closes.

---

## File Location

```
data/agent/decision_logs/
  ├── live/                    # Production trades
  │   ├── 2026-08.jsonl       # August 2026
  │   └── ...
  └── paper/                   # Backtest/paper trades
      ├── backtest-2026-07.jsonl
      └── ...
```

Each file is append-only JSONL (one JSON object per line).

---

## Oracle Query Examples

```bash
# Get all GBPUSD entries in the last 7 days
cat data/agent/decision_logs/live/2026-08.jsonl | jq 'select(.pair == "GBPUSD" and .timestamp_utc > "2026-08-05")'

# Get all winning trades (pnl_usd > 0)
cat data/agent/decision_logs/live/*.jsonl | jq 'select(.pnl.pnl_usd > 0)'

# Correlation: commitment_score vs outcome
cat data/agent/decision_logs/live/*.jsonl | jq '[.decision_metrics.commitment_score, .pnl.pnl_pct_return]' > commitment_vs_outcome.json
# Compute correlation in Python: scipy.stats.pearsonr(commitment, outcomes)
```

---

## Amendment History

- **2026-08-12** — Schema formalized for ICM integration. Outcome update made mandatory.
- **2026-07-01** — Initial version (decision logger shipped).

---

## Non-Negotiable Rules

1. **Every live entry must log.** No exceptions.
2. **Every live exit must update_outcome().** Trades without outcomes are invisible to the Oracle and degrade its learning.
3. **Logs are immutable.** Once written, do not edit. Corrections are new entries with "correction_to" field.
4. **Schema must stay in sync** with the decision_logger.py code. If the code changes, schema changes.

---

**Reference:** `CLAUDE.md` NON-NEGOTIABLE #2 — "Close the Oracle loop. Every trade decision wired to decision_logger must receive an update_outcome() call when the trade closes. Oracle cannot learn without closed-loop outcomes."
