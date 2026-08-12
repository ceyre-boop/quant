# Sizing Model Reference

**All live positions use the conviction-based sizing model. No flat sizes. No exceptions.**

*Last updated: 2026-08-12*

---

## Core Principle

Position size is proportional to conviction (confidence that the entry is correct). Conviction is computed from:
1. **Commitment score** — How aligned are entry conditions with the learned pattern?
2. **Rate differential** — For carry trades, how wide is the interest rate spread?
3. **Library match** — How many historical analogs support this setup?
4. **Bars since signal** — How fresh is the signal (decay over time)?

**Result:** Each trade sized individually. Sizes vary from 0.1 to 1.0 leverage multiplier depending on conviction.

---

## Live Sizing Pipeline

All live sizes flow through:
```
Entry signal
  ↓
conviction_scorer.score(context)  [returns 0.0 to 1.0]
  ↓
risk_engine.compute_position_size(conviction)  [applies Article 1 cap: 0.75% per trade]
  ↓
carry_tracker.check_complex_heat(size)  [Article 2 cap: 2.5% total carry exposure]
  ↓
EXECUTE or ABSTAIN
```

**Machine reference:** `sovereign/intelligence/decision_logger.py::log()` captures all three inputs (commitment, rate_diff, library_match); `sovereign/risk/position_sizing.py` computes final size.

---

## Conviction Score Components

### 1. Commitment Score (0–1)

How tightly does this entry match the learned pattern?

**For Forex (carry):**
- High (0.7+): Entry occurs in confirmed rate-trend window (HYP-045 rates in momentum)
- Medium (0.4–0.7): Entry in grey zone (rates flat-ish, some carry premium)
- Low (<0.4): Entry against rate trend or in low-premium window

**For ICT (intraday):**
- High (0.7+): Commit score >0.65 from ICT classifier (high probability setup)
- Medium (0.4–0.7): Commit score 0.50–0.65 (moderate setup)
- Low (<0.4): Commit score <0.50 (low confidence; abstain)

**Reference:** `sovereign/intelligence/decision_logger.py` — each log entry records the commitment_score as-measured.

### 2. Rate Differential (Carry only, 0–1)

For pairs like GBPUSD, EURUSD: how wide is the interest rate spread?

- Wide spread (>150 bps): Conviction multiplier +0.3
- Medium spread (50–150 bps): No multiplier
- Tight spread (<50 bps): Conviction multiplier −0.2

**Data source:** Central bank rates via `data/macro/central_bank_rates.py` (FRED, ECB).
**Cadence:** Updated daily; impacts sizing the next day.

### 3. Library Match (0–1)

How many historical analogs exist for this exact entry condition?

- High match (>5 historical trades in ledger): +0.2 conviction
- Medium match (2–5 trades): No bonus
- Low match (<2 trades): −0.1 conviction

**Calculation:** Query hypothesis ledger for prior trades with same (pair, entry reason, rate regime). Count.

### 4. Bars Since Signal (decay)

Fresh signals (entered <2 bars ago) get full conviction. Signals decay over time.

- Bars 0–2 since entry signal: 1.0x conviction
- Bars 3–5: 0.8x conviction
- Bars 6+: 0.5x conviction
- Bars 20+: Signal considered stale; abstain

**Purpose:** Prevents "entering late" on a signal that's already 20 bars old (edge is gone).

---

## Position Size Calculation

```python
def compute_position_size(conviction_score, account_equity, per_trade_cap=0.0075):
    """
    Conviction-based position sizing.
    
    Args:
        conviction_score: 0.0 to 1.0 (from commitment + adjustments)
        account_equity: Current account balance
        per_trade_cap: Article 1 cap (max 0.75% per trade)
    
    Returns:
        position_size_usd: Dollar amount to risk on this trade
    """
    # Base risk per account size
    base_risk = account_equity * per_trade_cap
    
    # Scale by conviction
    risk_adjusted = base_risk * conviction_score
    
    return risk_adjusted
```

**Example:**
- Account: $100,000
- Conviction: 0.75
- Per-trade cap: 0.75% ($750)
- Sized position: $750 × 0.75 = $562.50 risk

**Article 1 enforcement:** Even if conviction is 1.0, position never exceeds 0.75% per RISK_CONSTITUTION.

---

## Carry Complex Heat Check (Article 2)

Carry pairs (`GBPUSD`, `EURUSD`, `AUDUSD`, `USDJPY`) are correlated. Check total simultaneous exposure:

```python
def check_carry_heat(new_position_usd, open_carry_positions, carry_cap=0.025):
    """
    Carry complex heat gate (Article 2).
    Prevents over-concentration in correlated positions.
    
    Args:
        new_position_usd: Size of the new position being considered
        open_carry_positions: Dict of open positions {pair: risk_usd}
        carry_cap: Article 2 cap (max 2.5% of equity)
    
    Returns:
        passes: Boolean. True if new position fits in the cap.
    """
    current_heat = sum(open_carry_positions.values())
    total_heat = current_heat + new_position_usd
    account_equity = 100000  # assumed; pass from context
    
    cap = account_equity * carry_cap
    passes = total_heat <= cap
    
    return passes
```

**Example:**
- Account: $100,000
- Carry cap: 2.5% = $2,500
- Open: GBPUSD +$1,200, EURUSD +$900 = $2,100 heat
- New AUDUSD proposal: +$600
- Total would be $2,700 → **REJECTED** (exceeds cap)
- Max new size: $400 → **ACCEPTED**

---

## Drawdown Breakers (Article 3)

Sizing gets throttled when drawdown rises:

| Drawdown | Action |
|----------|--------|
| 0–3.5% | Full sizing |
| 3.5–5% | All new sizes halved |
| 5–6.5% | No new entries; close only |
| >6.5% | Flatten all predictive-layer positions |

**Measurement:** Peak-to-trough at account level. Updated daily.

**Machine gate:** `sovereign/risk/position_sizing.py::apply_drawdown_throttle()`

---

## Amendment History

- **2026-08-12** — Formalized library_match component (was ad-hoc before).
- **2026-07-20** — Carry heat check automated (Article 2 gate wired live).
- **2026-07-07** — Initial version (Articles 1–3 ratified).

---

## Usage Pattern

In a hypothesis test stage that involves live trading:

```markdown
### Layer 3 (Reference)
- `_config/sizing_model.md` — Conviction-based sizing, carries 2.5% complex heat cap

### Process
1. Entry signal fires (commitment_score=0.72)
2. Size = 0.0075 × $100k × 0.72 = $540
3. Check carry heat: current $1,800 + $540 = $2,340 < $2,500 ✅
4. EXECUTE $540 position

### Outputs
- Every trade logged with commitment_score, rate_diff, library_match
- Ledger entry includes: sized position, cap check result
```

**Reference in decision_logger.py output:**
```json
{
  "trade_id": "GBPUSD-2026-08-12-0930",
  "commitment_score": 0.72,
  "rate_differential": 120,
  "library_match": 3,
  "bars_since_signal": 1,
  "computed_conviction": 0.75,
  "sized_position_usd": 562.50,
  "carry_heat_check": "PASS",
  "executed": true
}
```
