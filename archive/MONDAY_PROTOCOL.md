# 🚀 MONDAY NY OPEN PROTOCOL (Phase 1)
**Projected Performance: 70.69% OOS Win Rate | 1.5x ATR Stops**

## 🗓️ The Monday Sequence (All times ET)
### 1. 08:45 AM → Premarket Safety Scan
`python check_monday_premarket.py`
- [ ] **Data Check**: Verify Alpaca/Polygon feeds for Trinity Assets (META, PFE, UNH).
- [ ] **Vol Gate**: Confirm VIX < 25 (Prevents 2025-style instability).
- [ ] **News Gate**: Check for high-impact CPI/FOMC events today.
- [ ] **ATR Validation**: Ensure ATR is within allowed range (No hemorrhage).

### 2. 09:50 AM → Kill Zone Execution
`python execute_monday_killzone.py`
- [ ] **Window Search**: Wait for Layer 2 Router to signal a Momentum Window.
- [ ] **Execution**: RR Engine calculates 1.5x ATR stops automatically.
- [ ] **Sizing**: Grade-based risk sizing applied (A+=1.5%, A=1%, B=0.5%).
- [ ] **The 1-Trade Rule**: Goal is one valid trade logged correctly. 

### 3. 04:30 PM → Post-Session Review
`python check_post_session.py`
- [ ] **P&L Check**: Compare paper execution to expected ATR targets.
- [ ] **Bayesian Update**: Update win-rate tracker with today's result.
- [ ] **Log Cleanup**: Ensure `paper_trades.jsonl` is correctly formatted.

## 🛡️ The Monday Configuration
- **Trinity Assets**: META, PFE, UNH.
- **Stop-Loss**: 1.5x ATR (20).
- **Take-Profit**: 3.0x ATR (2:1 Ratio).
- **Trailing Stop**: Activate at 1.5R (Lock in 0.5R).
- **Risk per Grade**:
  - `A+` (92%+ confidence) -> 1.50%
  - `A` (78%+ confidence) -> 1.00%
  - `B` (65%+ confidence) -> 0.50%
  - `C` (45%+ confidence) -> 0.25%

*Architecture evolves from real data, not speculation. Execute with discipline.*
