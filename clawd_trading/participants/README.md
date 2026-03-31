# Participant Analysis Module - Integration Guide

## Overview

Successfully ported participant detection and risk adjustment system from trading-stockfish to QUANT repo.

## What Was Ported

### 1. Participant Taxonomy (`participant_taxonomy.py`)
- **7 Participant Types:**
  - RETAIL - Small, slow, avoids volatility
  - ALGO - Fast, balanced execution
  - MARKET_MAKER - Provides liquidity, throttles in high vol
  - FUND - Block trades, mid-day bias
  - NEWS_ALGO - Ultra-fast news reaction
  - LIQUIDITY_HUNTER - Seeks hidden liquidity
  - SWEEP_BOT - Rapid sweeps, close-oriented

- **Risk Multipliers:** Each type has position sizing adjustments

### 2. Feature Extraction (`participant_features.py`)
- Extracts microstructure features from tick data
- Integrates with Layer 1 output
- Features:
  - Orderflow velocity
  - Sweep intensity
  - Absorption ratio
  - Spread pressure
  - Liquidity removal rate
  - Volatility reaction

### 3. Classification (`participant_likelihood.py`)
- Rule-based classification (no ML needed)
- Returns probability distribution over participant types
- Deterministic and explainable

### 4. Risk Envelopes (`participant_risk.py`)
- Adjusts position sizing based on detected participants
- Integrates with Gate 12 (Entry Engine)
- Key adjustments:
  - SWEEP_BOT: -30% size, tighter stops
  - LIQUIDITY_HUNTER: -15% size
  - MARKET_MAKER: +10% stop width
  - FUND: -10% size, wider targets
  - NEWS_ALGO: BLOCK TRADES

## Integration Points

### Layer 1 (Hard Constraints)
```python
from clawd_trading.participants import extract_from_layer1_context

# After Layer 1 runs
features = extract_from_layer1_context(layer1_output)
```

### Layer 2 (Bias Engine)
```python
from clawd_trading.participants import get_participant_bias_adjustment

# Get bias adjustments
adjustments = get_participant_bias_adjustment(likelihoods)
bias += adjustments['bias_shift']
confidence += adjustments['confidence_adjustment']
```

### Layer 3 / Gate 12 (Risk & Entry)
```python
from clawd_trading.participants import (
    calculate_participant_risk_limits,
    apply_participant_risk_to_gate12
)

# Calculate participant-adjusted limits
risk_limits = calculate_participant_risk_limits(likelihoods)

# Apply to Gate 12
adjusted = apply_participant_risk_to_gate12(base_limits, risk_limits)

# Check if blocked
if adjusted.get('no_trade'):
    return None  # Block entry
```

## Files Added

```
clawd_trading/participants/
├── __init__.py                    # Package exports
├── participant_taxonomy.py        # 7 participant types
├── participant_features.py        # Feature extraction
├── participant_likelihood.py      # Classification
├── participant_risk.py            # Risk envelopes
├── integration_example.py         # Usage examples
└── test_participants.py           # 9 unit tests
```

## Tests

All 9 tests passing:
- ✅ Participant types exist
- ✅ SWEEP_BOT detection
- ✅ MARKET_MAKER detection  
- ✅ NEWS_ALGO blocks trade
- ✅ SWEEP_BOT reduces size
- ✅ Risk multiplier values
- ✅ Gate 12 integration
- ✅ Gate 12 blocks on news
- ✅ Dominant participant selection

## Next Steps

1. **Integrate with Layer 1:** Call `extract_from_layer1_context()` in `layer1/feature_builder.py`

2. **Integrate with Gate 12:** Add `apply_participant_risk_to_gate12()` to `entry_engine/entry_engine.py`

3. **Add to Layer 2:** Use `get_participant_bias_adjustment()` in `layer2/bias_engine.py`

4. **Port Regime Risk:** Next phase - add regime-based risk envelopes

## Value Added

- **Smarter Risk:** Position sizing adapts to market microstructure
- **News Protection:** Blocks entries during news algo activity
- **Better Entries:** Gate 12 now considers who you're trading against
- **Explainable:** Rule-based system (no black box ML)
