# Decision Logger Integration Plan

**Status:** READY FOR REVIEW (not yet executed)  
**Impact:** CRITICAL — touches live execution paths  
**Estimated Time:** 2-3 hours (careful integration + testing)  
**Risk Level:** HIGH — any mistake breaks live trading

---

## Objective

Wire `decision_logger.log()` and `update_outcome()` calls into live execution paths so every trade is logged with full context (commitment score, rate differential, conviction, etc.).

**Current State:** Decision logger exists (`sovereign/intelligence/decision_logger.py`) but is never called by live code. Oracle has nothing to learn from.

**End State:** Every live trade logs entry context + outcome, enabling Oracle reflection cycle.

---

## Execution Paths to Wire

### ICT (Intraday)

**Entry Point:** `ict/paper_trader.py::open_trade()` (line 130)

**Integration Point:** Line 172, after `_live_log("ENTRY", ...)` 

**Exact code location:**
```python
# Line 168: PaperTrade object created
trade = PaperTrade(
    id=f"{scan_result.pair}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    pair=scan_result.pair,
    direction=scan_result.signal,
    # ... other fields ...
)

# Line 170-171: Added to state
self._state.setdefault('open', []).append(asdict(trade))
self._save_state()

# Line 172-175: CURRENT logging
_live_log("ENTRY", "ICT", trade.pair, trade.direction, trade.entry,
          grade=trade.grade, session=trade.session,
          meta={"stop": trade.stop, "tp1": trade.tp1, "tp2": trade.tp2,
                "risk_dollars": trade.risk_dollars, "score": trade.score, "id": trade.id})

# >>> INJECT HERE <<<
# decision_logger.log_ict_decision(
#     pair=trade.pair,
#     direction=trade.direction,
#     entry_price=trade.entry,
#     stop_price=trade.stop,
#     commitment_score=scan_result.score,  # or from a commitment engine
#     grade=trade.grade,
#     session=trade.session,
#     risk_pct=RISK_PCT,
#     trade_id=trade.id,
# )
```

**Exit Point:** `ict/paper_trader.py::_log_trade()` (line 348)
- Lines 373-382 **ALREADY CALL** `causal_journal.update_outcome()` — Oracle loop closure is already wired! ✅
- No additional wiring needed for ICT exits

**Files to Modify:**
1. `ict/paper_trader.py` — import decision_logger, add `log_ict_decision()` call after line 175

**Safety Checks:**
- Logging must NEVER block trade execution (use try/except)
- Log write failure must log warning but not raise
- ICT already has `update_outcome()` wired via causal_journal (line 375)

---

### Forex (Multi-day Carry)

**Entry Point:** `scripts/forex_live_scan.py` (main execution loop)

**Integration Point:** Lines 195-234, inside the `for c in tradeable:` loop, right **before** `bridge.place_trade()` call (line 230)

**Exact location to inject:**
```python
# Line 195-230 (CURRENT)
for c in tradeable:
    s, p = c.entry_signal, c.position
    pair, direction = s.pair, s.direction
    # ... sizing logic ...
    risk_check = prop.check_trade_allowed(pair, direction, risk_pct)
    
    if not risk_check.allowed:
        rec = {**base, "verdict": "DENIED", ...}
    elif dry_run:
        rec = {**base, "verdict": "WOULD_PLACE", ...}
    else:
        # >>> INJECT HERE <<<
        # decision_logger.log_forex_decision(pair, direction, s.entry_price, 
        #                                     s.stop_price, s.macro_conviction, ...)
        
        oanda_pair = to_oanda_pair(pair)
        units = bridge.compute_units(oanda_pair, s.entry_price, s.stop_price, risk_pct)
        if units == 0:
            rec = {**base, "verdict": "NO_TRADE", ...}
        else:
            placed = bridge.place_trade(oanda_pair, direction, units, s.stop_price, s.t1)
            rec = {**base, "verdict": "PLACED", ...}
```

**Required Parameters:**
- `pair`: currency pair name
- `direction`: LONG or SHORT
- `entry_level`: entry price
- `stop_loss`: stop price
- `hold_days`: target hold duration
- `commitment_score`: from model or macro gate
- `rate_diff_z`: z-score of rate differential at entry
- `vix_at_entry`: from market snapshot
- `signal_layers`: ["Rate Differential", "Macro Gate", ...] (whatever triggered it)

**Files to Modify:**
- TBD (need to find forex entry point)

---

### Exit Handler (Critical Path)

**Entry Point:** Position close handler (wherever trades are closed — execution, EOD, stop-hit)

Current state: Unknown — need to find where positions are marked as closed.

**Integration Point:** On close, call `update_outcome(trade_id, outcome)`

**Parameters:**
- `trade_id`: from the entry log (e.g., "HYP-045-GBPUSD-2026-08-12-0930")
- `outcome`: "WIN" or "LOSS"
- `exit_price`: actual close price
- `exit_reason`: "target_hit", "stop_hit", "time_exit", etc.

**Files to Modify:**
- TBD (need to find exit handler)

---

## Work Breakdown

### Phase 1: Discovery (COMPLETE ✅)

**Findings:**
- ✅ ICT entry: `ict/paper_trader.py::open_trade()` line 130
- ✅ ICT exit: Already wired via causal_journal (line 375)
- ✅ Forex entry: `scripts/forex_live_scan.py` main loop lines 195-234
- ✅ Forex exit: Handled by OANDA bridge close (need to wire decision_logger.update_outcome() here)

### Phase 2: Implementation (1.5-2 hours)

For each entry point:
1. Import decision_logger at top of file
2. Collect all required parameters (may need to pass them through call chain)
3. Call appropriate log_*_decision() function
4. Wrap in try/except to never block execution
5. Add debug logging to confirm calls

For exit handler:
1. Import decision_logger
2. Call update_outcome() on position close
3. Handle trade_id lookup (must match entry log)

### Phase 3: Testing (1 hour)

- [ ] Paper trade mode: run live scanner, verify logs appear in `data/decision_logs/`
- [ ] Check log format: `cat data/decision_logs/decisions_YYYY_MM.jsonl | jq . | head`
- [ ] Verify trade_id matching: entry log trade_id == outcome trade_id
- [ ] Run Oracle harvest stage on fresh logs: should produce trade summary
- [ ] 1-day live monitoring before full deployment

---

## Rollback Plan

If anything breaks:
1. Comment out decision_logger imports and calls (execution continues, logs silent)
2. Delete bad log files: `rm data/decision_logs/decisions_*.jsonl`
3. Restart execution path (orchestrator, forex scanner, etc.)

**No data loss:** Bad logs can be deleted; trades continue.

---

## Open Questions (Resolved by Code Inspection)

✅ **Trade ID format:** ICT already uses `pair_YYYYMMDD_HHMMSS` format (line 155 of paper_trader.py)  
✅ **Forex entry point:** Found in `scripts/forex_live_scan.py` (lines 195-234)  
✅ **Exit handlers:** ICT already wired via causal_journal.update_outcome() (paper_trader line 375)  

**Remaining open questions for Colin:**
1. Should Forex also backfill `update_outcome()` on position close (e.g., via the OANDA bridge)?
2. What is the forex "commitment score" equivalent? (currently using `s.macro_conviction`)
3. Should we log to the same decision_logs directory for both ICT and Forex, or separate files?
4. Do we backfill historical trades from the paper trade ledger, or start fresh from this point?

---

## Success Criteria

✅ Decision logs appear in `data/decision_logs/decisions_YYYY_MM.jsonl`  
✅ Each log has trade_id, commitment_score, and market context  
✅ Logs continue appearing even if Oracle harvest fails  
✅ Oracle harvest can read logs and produce trade summary  
✅ No trading interruption (logging never blocks execution)  

---

## Recommended Next Step

1. **Colin review:** Confirm Phase 1 search results and answer open questions
2. **Phase 1 execution:** Run grep searches to find actual entry/exit points
3. **Phase 2 execution:** Wire logging in carefully, with try/except guards
4. **Phase 3 testing:** Paper trade for 1 day, verify logs + Oracle harvest works
5. **Go live:** Flip decision_logger live with monitoring

---

## Risk Mitigation

- **Never block trades:** All logging is async/best-effort with try/except
- **Versioning:** If log schema needs to change, new schema goes in a new file (decisions_v2_YYYY_MM.jsonl)
- **Monitoring:** Add a health check that looks for recent log entries (alerts if logs stop flowing)
- **Backfill:** Keep decision_logs directory gitignored but backed up daily

---

**Status:** Awaiting Colin's input on open questions + approval to proceed with Phase 1.
