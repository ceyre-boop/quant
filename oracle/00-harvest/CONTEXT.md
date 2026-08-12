# Oracle Stage 00: Harvest Decision Logs

**Layer 2: Harvest market data and decision logs for reflection**

*Last updated: 2026-08-12*

---

## Purpose

Every morning, the Oracle reads the previous day's trades and market data. This stage gathers both and prepares them for reflection.

**Runs at:** 02:30 UTC (21:30 ET previous day) — immediately after US market close

---

## Inputs (Layer 3 + Layer 4)

### Layer 3 (Reference)
- `../../shared/decision_logger_schema.md` — Schema for reading trade logs
- `_config/macro_data_catalog.md` — FRED series, central bank rates, VIX levels (data source URLs)
- `_config/market_regimes.md` — How to classify trading regimes (trending vs choppy vs crash)

### Layer 4 (Working artifacts)
- `data/agent/decision_logs/live/YYYY-MM.jsonl` — Previous day's entry/exit logs (append-only)
- `data/macro/econ_indicators.csv` — Daily economic releases (CPI, jobless claims, Fed speaker calendar)
- `data/market/market_state.json` — Previous day's close: VIX, major indices, USD index

---

## Process

### Step 1: Read Decision Logs (Automated, ~10 sec)

**Actions:**
1. Tail the latest decision_logs/live/*.jsonl file (yesterday's entries)
2. Filter for completed trades: trades with both `log()` and `update_outcome()` calls
3. Group by hypothesis_id (which HYP-* triggered each trade?)
4. Compute hypothesis-specific metrics:
   - Sharpe on this hypothesis's trades (if n≥2)
   - Win rate
   - Average hold duration
   - Largest winning and losing trades

**Output:** `00-harvest/output/decision_logs_summary.json`
```json
{
  "date": "2026-08-12",
  "harvest_timestamp_utc": "2026-08-12T06:30:00Z",
  "trades_summary": {
    "total_completed": 4,
    "total_pnl_usd": 1847.30,
    "total_pnl_pct": 1.85,
    "hypothesis_breakdown": {
      "HYP-045": {
        "trades": 2,
        "pnl": 921.50,
        "sharpe_est": 1.12,
        "win_rate": 1.0
      },
      "HYP-093": {
        "trades": 2,
        "pnl": 925.80,
        "sharpe_est": 0.89,
        "win_rate": 1.0
      }
    }
  }
}
```

### Step 2: Read Market State (Automated, ~5 sec)

**Actions:**
1. Fetch previous day's close: VIX, S&P 500, USD index, commodity indices
2. Classify regime: trending vs choppy vs crash (see `_config/market_regimes.md`)
3. Check for macro events: FOMC decision, Fed speakers, employment report, etc.
4. Read central bank rate decisions (has any major CB changed rates overnight?)

**Output:** `00-harvest/output/market_state.json`
```json
{
  "date": "2026-08-12",
  "close_utc": "2026-08-12T21:00:00Z",
  "vix_close": 13.2,
  "sp500_close": 5923.4,
  "sp500_daily_return_pct": 0.34,
  "usd_index": 104.21,
  "regime": "rate_trending",
  "regime_confidence": 0.79,
  "macro_events": [
    {
      "event": "ECB Member Speech (De Guindos)",
      "time": "08:00 UTC",
      "impact": "medium"
    }
  ],
  "central_bank_changes": []
}
```

### Step 3: Read Hypothesis Ledger (Automated, ~2 sec)

**Actions:**
1. Tail `data/hypotheses_ledger.jsonl` — get status of all active hypotheses
2. Identify which hypotheses are LIVE (trading today)
3. Identify which hypotheses are IN_RESEARCH (backtesting today?)
4. Flag any that transitioned from IN_RESEARCH → LIVE or IN_RESEARCH → GRAVEYARD

**Output:** `00-harvest/output/hypothesis_status.json`
```json
{
  "date": "2026-08-12",
  "live_hypotheses": ["HYP-045", "HYP-093"],
  "in_research_hypotheses": ["HYP-071"],
  "status_changes": []
}
```

---

## Outputs

Write to `output/`:

1. **`decision_logs_summary.json`** — Trade metadata for reflection stage
2. **`market_state.json`** — Market regime and macro context
3. **`hypothesis_status.json`** — Which hypotheses were active
4. **`harvest_complete.txt`** — Timestamp and status ("OK" or error details)

---

## Success Criteria

✅ **All three outputs present and valid JSON**  
✅ **No missing decision logs** (if a live trade was entered, it was logged)  
✅ **Market state data is fresh** (previous day's close, not stale)  
✅ **Hypothesis ledger is consistent** (no conflicts between live hypotheses)

---

## Failure Modes

❌ **Missing decision logs** → Try backfill with `backfill_decision_records.py` (idempotent)  
❌ **Stale market data** → Flag as "data_available=False", skip reflection (Oracle will retry at next cycle)  
❌ **Ledger conflict** (two hypotheses claim same trade_id) → Stop, alert Colin  

---

## Next Stage

Outputs from this stage feed into `01-reflect/CONTEXT.md`.

**Handoff:** Copy outputs to `../01-reflect/output/00-harvest-output/` (reference inputs for reflection)
