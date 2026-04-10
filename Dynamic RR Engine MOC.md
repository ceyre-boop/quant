# Dynamic Risk-to-Reward Engine — Complete Specification 🎯

#risk-reward #dynamic-exits #trailing-stop #position-management #code-spec

> **Purpose:** A complete, asset-aware, trade-lifecycle dynamic R:R engine. Every decision — entry sizing, stop placement, partial exits, trailing logic, emergency exits — computed in real time. No static rules. No human discretion required. The engine watches the trade the way a perfect trader would: always ready to protect, always ready to let winners run.

> **Status:** SPECIFICATION FOR IMMEDIATE CODE BUILD
> **Target File:** `execution/rr_engine.py`
> **Build Command:** Feed this entire document to your AI build engine. Every class, method, and parameter is fully defined.

---

## 🗺️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RR_ENGINE.PY — FULL LIFECYCLE                    │
│                                                                       │
│  ENTRY PHASE          ACTIVE PHASE           EXIT PHASE              │
│  ─────────────        ─────────────          ──────────              │
│  AssetProfile    →    TradeMonitor      →    ExitExecutor            │
│  StopCalculator  →    BreakevenTrigger  →    PartialExitManager      │
│  TargetCalculator →   TrailEngine       →    EmergencyExit           │
│  PositionSizer   →    RegimeWatcher     →    PostTradeAnalyzer       │
└─────────────────────────────────────────────────────────────────────┘

DATA FLOW:
  Signal (from BiasEngine + ICT Layer)
    → AssetProfile.classify(symbol)
    → StopCalculator.compute(asset_profile, market_data)
    → TargetCalculator.compute(asset_profile, stop_distance, market_data)
    → PositionSizer.compute(account, risk_pct, stop_distance)
    → TradeRecord created and registered
    → TradeMonitor.run() — continuous loop every new bar
        → checks all 8 exit conditions
        → fires appropriate action
    → PostTradeAnalyzer.record()
```

---

## MODULE 1 — AssetProfile

> Every asset behaves differently. The engine must know what it is trading.

### Class Definition

```python
# FILE: execution/rr_engine.py
# SECTION 1: AssetProfile

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd

class AssetClass(Enum):
    LARGE_CAP_EQUITY   = "large_cap_equity"    # AAPL, META, MSFT, UNH
    MID_CAP_EQUITY     = "mid_cap_equity"       # AMD, ARM, PFE
    COMMODITY_ETF      = "commodity_etf"        # SLV, GLD, USO
    VOLATILITY_PRODUCT = "volatility_product"   # UVXY (special rules)
    BROAD_INDEX_ETF    = "broad_index_etf"      # SPY, QQQ, IWM
    SECTOR_ETF         = "sector_etf"           # XLK, XLE, XLF
    LEVERAGED_ETF      = "leveraged_etf"        # FORBIDDEN — gate rejects these

class SessionBehavior(Enum):
    MOMENTUM  = "momentum"    # Trends strongly in direction of session open
    REVERSAL  = "reversal"    # Tends to fade initial session move
    BALANCED  = "balanced"    # No strong session bias

@dataclass
class AssetProfile:
    symbol:             str
    asset_class:        AssetClass
    avg_daily_atr_pct:  float        # Average ATR as % of price (rolling 20-day)
    avg_spread_pct:     float        # Typical bid-ask spread as % of price
    avg_daily_volume:   float        # Shares/units per day (for liquidity sizing)
    beta:               float        # vs SPY
    session_behavior:   SessionBehavior

    # Stop parameters (computed from historical ATR behavior)
    stop_atr_multiplier:    float    # How many ATRs to place stop
    stop_min_pct:           float    # Minimum stop distance as % regardless of ATR
    stop_max_pct:           float    # Maximum stop distance; if wider, skip trade

    # Target parameters
    tp1_r_ratio:    float   # First partial exit R:R (e.g., 1.5)
    tp2_r_ratio:    float   # Second partial exit R:R (e.g., 3.0)
    tp3_r_ratio:    float   # Runner target R:R (e.g., 5.0+)

    # Trailing parameters
    trail_activation_r:     float   # When to start trailing (e.g., after 1.5R)
    trail_atr_multiplier:   float   # Trail stop = N × ATR behind price
    breakeven_trigger_r:    float   # Move stop to breakeven at this R level

    # Emergency exit parameters
    max_adverse_move_r:     float   # If trade moves this far against: immediate review
    shock_exit_atr_mult:    float   # If single candle > N ATR against: emergency exit

    # Regime adjustments
    avoid_in_high_vix:      bool    # Skip this asset when VIX > 25
    avoid_in_low_volume:    bool    # Skip when volume < 50% of average


# ASSET PROFILE REGISTRY
# These values are derived from your 11,090-signal dataset.
# Update quarterly via: python training/compute_asset_profiles.py

ASSET_PROFILES = {

    "META": AssetProfile(
        symbol="META", asset_class=AssetClass.LARGE_CAP_EQUITY,
        avg_daily_atr_pct=1.8, avg_spread_pct=0.01, avg_daily_volume=15_000_000,
        beta=1.3, session_behavior=SessionBehavior.MOMENTUM,
        stop_atr_multiplier=1.5, stop_min_pct=0.8, stop_max_pct=3.0,
        tp1_r_ratio=1.5, tp2_r_ratio=3.0, tp3_r_ratio=5.0,
        trail_activation_r=1.5, trail_atr_multiplier=1.2, breakeven_trigger_r=1.0,
        max_adverse_move_r=1.8, shock_exit_atr_mult=2.5,
        avoid_in_high_vix=False, avoid_in_low_volume=True
    ),

    "AAPL": AssetProfile(
        symbol="AAPL", asset_class=AssetClass.LARGE_CAP_EQUITY,
        avg_daily_atr_pct=1.4, avg_spread_pct=0.005, avg_daily_volume=60_000_000,
        beta=1.2, session_behavior=SessionBehavior.BALANCED,
        stop_atr_multiplier=1.5, stop_min_pct=0.6, stop_max_pct=2.5,
        tp1_r_ratio=1.5, tp2_r_ratio=2.5, tp3_r_ratio=4.5,
        trail_activation_r=1.5, trail_atr_multiplier=1.2, breakeven_trigger_r=1.0,
        max_adverse_move_r=1.8, shock_exit_atr_mult=2.5,
        avoid_in_high_vix=False, avoid_in_low_volume=True
    ),

    "AMD": AssetProfile(
        symbol="AMD", asset_class=AssetClass.MID_CAP_EQUITY,
        avg_daily_atr_pct=3.2, avg_spread_pct=0.02, avg_daily_volume=40_000_000,
        beta=1.8, session_behavior=SessionBehavior.MOMENTUM,
        stop_atr_multiplier=1.8, stop_min_pct=1.2, stop_max_pct=4.5,
        tp1_r_ratio=1.5, tp2_r_ratio=3.0, tp3_r_ratio=6.0,
        trail_activation_r=1.5, trail_atr_multiplier=1.5, breakeven_trigger_r=1.0,
        max_adverse_move_r=2.0, shock_exit_atr_mult=2.2,
        avoid_in_high_vix=False, avoid_in_low_volume=True
    ),

    "ARM": AssetProfile(
        symbol="ARM", asset_class=AssetClass.MID_CAP_EQUITY,
        avg_daily_atr_pct=3.8, avg_spread_pct=0.04, avg_daily_volume=8_000_000,
        beta=2.1, session_behavior=SessionBehavior.MOMENTUM,
        stop_atr_multiplier=2.0, stop_min_pct=1.5, stop_max_pct=5.5,
        tp1_r_ratio=1.5, tp2_r_ratio=3.0, tp3_r_ratio=5.5,
        trail_activation_r=1.5, trail_atr_multiplier=1.8, breakeven_trigger_r=1.0,
        max_adverse_move_r=1.8, shock_exit_atr_mult=2.0,
        avoid_in_high_vix=True, avoid_in_low_volume=True
    ),

    "PFE": AssetProfile(
        symbol="PFE", asset_class=AssetClass.LARGE_CAP_EQUITY,
        avg_daily_atr_pct=1.1, avg_spread_pct=0.01, avg_daily_volume=30_000_000,
        beta=0.6, session_behavior=SessionBehavior.BALANCED,
        stop_atr_multiplier=1.5, stop_min_pct=0.5, stop_max_pct=2.0,
        tp1_r_ratio=1.5, tp2_r_ratio=2.5, tp3_r_ratio=4.0,
        trail_activation_r=1.2, trail_atr_multiplier=1.0, breakeven_trigger_r=0.8,
        max_adverse_move_r=1.8, shock_exit_atr_mult=3.0,
        avoid_in_high_vix=False, avoid_in_low_volume=False
    ),

    "UNH": AssetProfile(
        symbol="UNH", asset_class=AssetClass.LARGE_CAP_EQUITY,
        avg_daily_atr_pct=1.3, avg_spread_pct=0.01, avg_daily_volume=3_000_000,
        beta=0.7, session_behavior=SessionBehavior.BALANCED,
        stop_atr_multiplier=1.5, stop_min_pct=0.6, stop_max_pct=2.5,
        tp1_r_ratio=1.5, tp2_r_ratio=3.0, tp3_r_ratio=5.0,
        trail_activation_r=1.2, trail_atr_multiplier=1.0, breakeven_trigger_r=0.8,
        max_adverse_move_r=1.8, shock_exit_atr_mult=3.0,
        avoid_in_high_vix=False, avoid_in_low_volume=False
    ),

    "SLV": AssetProfile(
        symbol="SLV", asset_class=AssetClass.COMMODITY_ETF,
        avg_daily_atr_pct=1.6, avg_spread_pct=0.03, avg_daily_volume=20_000_000,
        beta=0.3, session_behavior=SessionBehavior.REVERSAL,
        stop_atr_multiplier=1.8, stop_min_pct=0.8, stop_max_pct=3.5,
        tp1_r_ratio=2.0, tp2_r_ratio=4.0, tp3_r_ratio=7.0,
        trail_activation_r=2.0, trail_atr_multiplier=1.5, breakeven_trigger_r=1.2,
        max_adverse_move_r=2.0, shock_exit_atr_mult=2.5,
        avoid_in_high_vix=False, avoid_in_low_volume=True
    ),

    "UVXY": AssetProfile(
        symbol="UVXY", asset_class=AssetClass.VOLATILITY_PRODUCT,
        avg_daily_atr_pct=5.0, avg_spread_pct=0.15, avg_daily_volume=5_000_000,
        beta=-0.8, session_behavior=SessionBehavior.REVERSAL,
        stop_atr_multiplier=1.0, stop_min_pct=2.0, stop_max_pct=4.0,
        tp1_r_ratio=1.2, tp2_r_ratio=2.0, tp3_r_ratio=3.0,
        trail_activation_r=1.0, trail_atr_multiplier=0.8, breakeven_trigger_r=0.7,
        max_adverse_move_r=1.5, shock_exit_atr_mult=1.5,
        avoid_in_high_vix=False, avoid_in_low_volume=True
    ),

    "SPY": AssetProfile(
        symbol="SPY", asset_class=AssetClass.BROAD_INDEX_ETF,
        avg_daily_atr_pct=0.9, avg_spread_pct=0.001, avg_daily_volume=80_000_000,
        beta=1.0, session_behavior=SessionBehavior.BALANCED,
        stop_atr_multiplier=1.5, stop_min_pct=0.4, stop_max_pct=2.0,
        tp1_r_ratio=1.5, tp2_r_ratio=3.0, tp3_r_ratio=5.0,
        trail_activation_r=1.5, trail_atr_multiplier=1.2, breakeven_trigger_r=1.0,
        max_adverse_move_r=2.0, shock_exit_atr_mult=3.0,
        avoid_in_high_vix=False, avoid_in_low_volume=False
    ),

    # DEFAULT PROFILE — used for any symbol not explicitly listed
    "_DEFAULT": AssetProfile(
        symbol="_DEFAULT", asset_class=AssetClass.LARGE_CAP_EQUITY,
        avg_daily_atr_pct=2.0, avg_spread_pct=0.02, avg_daily_volume=5_000_000,
        beta=1.0, session_behavior=SessionBehavior.BALANCED,
        stop_atr_multiplier=1.5, stop_min_pct=0.8, stop_max_pct=3.5,
        tp1_r_ratio=1.5, tp2_r_ratio=3.0, tp3_r_ratio=5.0,
        trail_activation_r=1.5, trail_atr_multiplier=1.2, breakeven_trigger_r=1.0,
        max_adverse_move_r=1.8, shock_exit_atr_mult=2.5,
        avoid_in_high_vix=False, avoid_in_low_volume=True
    ),
}

def get_asset_profile(symbol: str) -> AssetProfile:
    return ASSET_PROFILES.get(symbol, ASSET_PROFILES["_DEFAULT"])
```

---

## MODULE 2 — MarketContext

> Real-time market state. Changes stop/target parameters dynamically.

```python
# SECTION 2: MarketContext

@dataclass
class MarketContext:
    vix_level:          float    # Current VIX
    vix_regime:         str      # 'low' (<15), 'normal' (15-20), 'elevated' (20-30), 'crisis' (>30)
    trend_regime:       str      # 'bull', 'bear', 'neutral'
    session:            str      # 'premarket', 'ny_open', 'midday', 'pm_session', 'afterhours'
    atr_pct_current:    float    # Current 14-day ATR as % of price FOR THIS ASSET
    news_risk_window:   bool     # True if FOMC/CPI/NFP/earnings within 4 hours
    spread_multiplier:  float    # Current spread vs average (1.0 = normal; 3.0 = 3x wider)

def build_market_context(symbol: str, data_feed) -> MarketContext:
    """
    Called at signal time and refreshed every bar during trade.
    data_feed: your existing market data connection.
    """
    vix = data_feed.get_vix()
    price = data_feed.get_price(symbol)
    atr14 = data_feed.get_atr(symbol, period=14)
    atr_pct = (atr14 / price) * 100

    if vix < 15:   vix_regime = 'low'
    elif vix < 20: vix_regime = 'normal'
    elif vix < 30: vix_regime = 'elevated'
    else:          vix_regime = 'crisis'

    return MarketContext(
        vix_level       = vix,
        vix_regime      = vix_regime,
        trend_regime    = data_feed.get_trend_regime(),   # from BiasEngine
        session         = data_feed.get_session(),
        atr_pct_current = atr_pct,
        news_risk_window= data_feed.is_news_window(),
        spread_multiplier = data_feed.get_spread_multiplier(symbol)
    )
```

---

## MODULE 3 — StopCalculator

> ICT-structure-first stop placement. ATR-validated. Asset-specific. Never arbitrary.

```python
# SECTION 3: StopCalculator

@dataclass
class StopResult:
    stop_price:         float
    stop_distance_pct:  float
    stop_distance_r1:   float    # = 1.0 always (this IS the 1R definition)
    method:             str      # which method was used
    atr_multiple:       float    # how many ATRs wide is this stop
    valid:              bool     # False = stop too wide; skip trade

class StopCalculator:

    @staticmethod
    def compute(
        direction:      str,          # 'long' or 'short'
        entry_price:    float,
        ict_structure:  dict,         # from ICT Layer: {'ob_low': X, 'fvg_bottom': X, 'sweep_low': X}
        profile:        AssetProfile,
        context:        MarketContext,
        atr_value:      float         # raw ATR in price units
    ) -> StopResult:

        # ── Step 1: ICT Structural Stop ──────────────────────────────
        # Priority order for long trades:
        # 1. Below sweep candle low (if liquidity sweep setup)
        # 2. Below OB low (if order block setup)
        # 3. Below FVG bottom (if FVG setup)
        # 4. ATR fallback

        if direction == 'long':
            structural_candidates = []
            if ict_structure.get('sweep_low'):
                structural_candidates.append(ict_structure['sweep_low'] * 0.999)
            if ict_structure.get('ob_low'):
                structural_candidates.append(ict_structure['ob_low'] * 0.999)
            if ict_structure.get('fvg_bottom'):
                structural_candidates.append(ict_structure['fvg_bottom'] * 0.999)

            if structural_candidates:
                # Use the HIGHEST structural stop (closest to entry = smallest loss)
                ict_stop = max(structural_candidates)
                method = 'ict_structural'
            else:
                ict_stop = entry_price - (atr_value * profile.stop_atr_multiplier)
                method = 'atr_fallback'

        else:  # short
            structural_candidates = []
            if ict_structure.get('sweep_high'):
                structural_candidates.append(ict_structure['sweep_high'] * 1.001)
            if ict_structure.get('ob_high'):
                structural_candidates.append(ict_structure['ob_high'] * 1.001)
            if ict_structure.get('fvg_top'):
                structural_candidates.append(ict_structure['fvg_top'] * 1.001)

            if structural_candidates:
                ict_stop = min(structural_candidates)
                method = 'ict_structural'
            else:
                ict_stop = entry_price + (atr_value * profile.stop_atr_multiplier)
                method = 'atr_fallback'

        # ── Step 2: ATR Validation ────────────────────────────────────
        # The structural stop must be AT LEAST 0.75x ATR (avoid noise stops)
        # and NO MORE THAN profile.stop_max_pct (avoid giving away too much)

        stop_distance = abs(entry_price - ict_stop)
        stop_pct      = (stop_distance / entry_price) * 100
        atr_multiple  = stop_distance / atr_value if atr_value > 0 else 0

        # Too tight: noise will stop you out
        if atr_multiple < 0.75:
            # Widen to 0.75 ATR minimum
            if direction == 'long':
                ict_stop = entry_price - (atr_value * 0.75)
            else:
                ict_stop = entry_price + (atr_value * 0.75)
            stop_distance = abs(entry_price - ict_stop)
            stop_pct      = (stop_distance / entry_price) * 100
            atr_multiple  = 0.75
            method += '_widened_for_noise'

        # Too wide: R:R will be unacceptable
        max_stop_pct = profile.stop_max_pct
        # In elevated VIX regime: allow wider stops
        if context.vix_regime == 'elevated': max_stop_pct *= 1.3
        if context.vix_regime == 'crisis':   max_stop_pct *= 1.6

        valid = stop_pct <= max_stop_pct

        return StopResult(
            stop_price        = round(ict_stop, 4),
            stop_distance_pct = stop_pct,
            stop_distance_r1  = 1.0,
            method            = method,
            atr_multiple      = atr_multiple,
            valid             = valid
        )
```

---

## MODULE 4 — TargetCalculator

> Multi-target system. ICT liquidity levels first. R-ratio floors as minimum. Dynamic adjustments for regime and session.

```python
# SECTION 4: TargetCalculator

@dataclass
class TargetSet:
    tp1_price:      float    # Partial exit 1 (1/3 position)
    tp2_price:      float    # Partial exit 2 (1/3 position)
    tp3_price:      float    # Runner target  (1/3 position)
    tp1_r:          float    # Actual R:R achieved at TP1
    tp2_r:          float    # Actual R:R achieved at TP2
    tp3_r:          float    # Actual R:R achieved at TP3
    min_acceptable_r: float  # Below this: skip trade (R:R gate)

class TargetCalculator:

    @staticmethod
    def compute(
        direction:      str,
        entry_price:    float,
        stop_result:    StopResult,
        ict_levels:     dict,    # {'pdh': X, 'pwh': X, 'bsl': X, 'fvg_above': X, 'daily_ob': X}
        profile:        AssetProfile,
        context:        MarketContext
    ) -> TargetSet:

        stop_dist = abs(entry_price - stop_result.stop_price)

        # ── Minimum R-ratio targets (floors) ──
        # These are the MINIMUM acceptable targets.
        # ICT levels will override upward if they're further away.
        min_tp1 = entry_price + stop_dist * profile.tp1_r_ratio if direction == 'long' \
                  else entry_price - stop_dist * profile.tp1_r_ratio
        min_tp2 = entry_price + stop_dist * profile.tp2_r_ratio if direction == 'long' \
                  else entry_price - stop_dist * profile.tp2_r_ratio
        min_tp3 = entry_price + stop_dist * profile.tp3_r_ratio if direction == 'long' \
                  else entry_price - stop_dist * profile.tp3_r_ratio

        # ── ICT Level Targets (preferred over pure R-ratio) ──
        # Use the nearest ICT level that is BEYOND the minimum R-ratio floor
        # This ensures we always get AT LEAST the minimum R:R

        ict_targets = []
        if direction == 'long':
            level_priority = ['fvg_above', 'pdh', 'daily_ob_above', 'bsl', 'pwh', 'weekly_bsl']
            for level_name in level_priority:
                level_price = ict_levels.get(level_name)
                if level_price and level_price > entry_price:
                    ict_targets.append(level_price)
            ict_targets.sort()  # ascending for longs
        else:
            level_priority = ['fvg_below', 'pdl', 'daily_ob_below', 'ssl', 'pwl', 'weekly_ssl']
            for level_name in level_priority:
                level_price = ict_levels.get(level_name)
                if level_price and level_price < entry_price:
                    ict_targets.append(level_price)
            ict_targets.sort(reverse=True)  # descending for shorts

        # Assign ICT levels to TPs if they exceed the R-ratio minimums
        def best_target(ict_list, minimum, direction):
            candidates = [t for t in ict_list if
                          (t > minimum if direction == 'long' else t < minimum)]
            if candidates:
                return candidates[0]  # nearest ICT level beyond minimum
            return minimum            # fall back to pure R-ratio minimum

        tp1 = best_target(ict_targets, min_tp1, direction)
        tp2 = best_target(ict_targets, min_tp2, direction)
        tp3 = best_target(ict_targets, min_tp3, direction)

        # Ensure TPs are strictly ordered
        if direction == 'long':
            tp1 = min(tp1, tp2 * 0.98)   # TP1 must be below TP2
            tp2 = min(tp2, tp3 * 0.97)
        else:
            tp1 = max(tp1, tp2 * 1.02)
            tp2 = max(tp2, tp3 * 1.03)

        # ── Regime Adjustment ──────────────────────────────────────
        # In crisis VIX: compress targets (market is unpredictable)
        if context.vix_regime == 'crisis':
            tp1 = entry_price + (tp1 - entry_price) * 0.7 if direction == 'long' \
                  else entry_price - (entry_price - tp1) * 0.7
            tp2 = entry_price + (tp2 - entry_price) * 0.7 if direction == 'long' \
                  else entry_price - (entry_price - tp2) * 0.7
            tp3 = entry_price + (tp3 - entry_price) * 0.7 if direction == 'long' \
                  else entry_price - (entry_price - tp3) * 0.7

        # In news risk window: compress to TP1 only (safety)
        if context.news_risk_window:
            tp3 = tp1
            tp2 = tp1

        # ── R:R Gate ──────────────────────────────────────────────
        # Minimum OVERALL R:R must be 2:1 at TP1
        # If TP1 cannot achieve 2:1 given the stop: SKIP THE TRADE
        tp1_r = abs(tp1 - entry_price) / stop_dist if stop_dist > 0 else 0
        tp2_r = abs(tp2 - entry_price) / stop_dist if stop_dist > 0 else 0
        tp3_r = abs(tp3 - entry_price) / stop_dist if stop_dist > 0 else 0

        min_acceptable_r = 2.0  # Global floor: no trade with R:R < 2:1

        return TargetSet(
            tp1_price         = round(tp1, 4),
            tp2_price         = round(tp2, 4),
            tp3_price         = round(tp3, 4),
            tp1_r             = round(tp1_r, 2),
            tp2_r             = round(tp2_r, 2),
            tp3_r             = round(tp3_r, 2),
            min_acceptable_r  = min_acceptable_r
        )
```

---

## MODULE 5 — PositionSizer

> Grade-aware. Asset-aware. Account-state-aware. Volatility-normalized.

```python
# SECTION 5: PositionSizer

@dataclass
class PositionResult:
    shares:             int
    dollar_risk:        float
    dollar_notional:    float
    risk_pct_of_acct:   float
    max_concurrent_r:   float    # Total account risk if all positions hit stop
    approved:           bool
    rejection_reason:   Optional[str]

class PositionSizer:

    # Grade-to-risk mapping (from your V2 scaling model)
    GRADE_RISK_PCT = {
        'A+': 0.015,   # 1.5% of account (max; only for confirmed dual-signal)
        'A':  0.010,   # 1.0% of account
        'B':  0.005,   # 0.5% of account
        'C':  0.0025,  # 0.25% of account (rarely used; see Reality Bridge)
    }

    @staticmethod
    def compute(
        account_balance:    float,
        signal_grade:       str,
        stop_result:        StopResult,
        entry_price:        float,
        profile:            AssetProfile,
        context:            MarketContext,
        open_positions:     list,     # list of currently open TradeRecord objects
        drawdown_pct:       float     # current drawdown from high-water mark (0.05 = 5%)
    ) -> PositionResult:

        # ── Base Risk Percentage ───────────────────────────────────────
        base_risk_pct = PositionSizer.GRADE_RISK_PCT.get(signal_grade, 0.005)

        # ── Drawdown Scaling ───────────────────────────────────────────
        # As drawdown increases, risk decreases automatically
        if drawdown_pct >= 0.15:
            rejection = f"Kill switch: {drawdown_pct:.1%} drawdown exceeds 15% limit"
            return PositionResult(0, 0, 0, 0, 0, False, rejection)
        elif drawdown_pct >= 0.10:
            base_risk_pct *= 0.50   # half size when in 10-15% drawdown
        elif drawdown_pct >= 0.05:
            base_risk_pct *= 0.75   # 3/4 size when in 5-10% drawdown

        # ── VIX Regime Scaling ─────────────────────────────────────────
        vix_scalers = {'low': 1.0, 'normal': 1.0, 'elevated': 0.75, 'crisis': 0.50}
        base_risk_pct *= vix_scalers.get(context.vix_regime, 1.0)

        # ── Concurrent Position Limit ──────────────────────────────────
        n_open = len(open_positions)
        total_open_risk = sum(p.risk_pct_of_acct for p in open_positions)

        if n_open >= 5:
            return PositionResult(0, 0, 0, 0, total_open_risk, False,
                                  "Max 5 concurrent positions reached")
        if total_open_risk + base_risk_pct > 0.08:  # 8% max total portfolio risk
            return PositionResult(0, 0, 0, 0, total_open_risk, False,
                                  f"Adding this trade would exceed 8% total portfolio risk")

        # ── Leverage/Volatility Product Override ──────────────────────
        if profile.asset_class == AssetClass.LEVERAGED_ETF:
            return PositionResult(0, 0, 0, 0, total_open_risk, False,
                                  "LEVERAGED ETF: Permanently gated out")
        if profile.asset_class == AssetClass.VOLATILITY_PRODUCT:
            base_risk_pct = min(base_risk_pct, 0.005)  # cap vol products at 0.5%

        # ── Spread/Liquidity Adjustment ────────────────────────────────
        if context.spread_multiplier > 3.0:
            base_risk_pct *= 0.5   # Half size when spreads are 3x normal

        # ── Final Calculation ──────────────────────────────────────────
        dollar_risk     = account_balance * base_risk_pct
        stop_distance   = abs(entry_price - stop_result.stop_price)
        shares          = int(dollar_risk / stop_distance) if stop_distance > 0 else 0
        dollar_notional = shares * entry_price

        # Liquidity check: order should be < 1% of average daily volume
        if shares > profile.avg_daily_volume * 0.01:
            shares = int(profile.avg_daily_volume * 0.01)
            dollar_notional = shares * entry_price

        return PositionResult(
            shares          = shares,
            dollar_risk     = dollar_risk,
            dollar_notional = dollar_notional,
            risk_pct_of_acct= base_risk_pct,
            max_concurrent_r= total_open_risk + base_risk_pct,
            approved        = shares > 0,
            rejection_reason= None if shares > 0 else "Zero shares calculated"
        )
```

---

## MODULE 6 — TradeRecord

> The live state object. Everything about the trade in one place.

```python
# SECTION 6: TradeRecord

from datetime import datetime

@dataclass
class TradeRecord:
    # Identity
    trade_id:       str
    symbol:         str
    direction:      str     # 'long' or 'short'
    signal_grade:   str
    signal_time:    datetime
    entry_time:     datetime
    entry_price:    float

    # Size
    shares:         int
    dollar_risk:    float
    risk_pct:       float

    # Levels (set at entry; some updated during trade)
    stop_price:     float           # HARD stop — broker order
    stop_method:    str
    breakeven_price: float          # Entry price (for breakeven move)
    tp1_price:      float
    tp2_price:      float
    tp3_price:      float
    tp1_r:          float
    tp2_r:          float
    tp3_r:          float

    # State (updated continuously)
    current_price:  float           = 0.0
    current_r:      float           = 0.0    # Current unrealized R
    max_favorable_r: float          = 0.0    # Best R reached (for trailing)
    max_adverse_r:  float           = 0.0    # Worst R reached (for monitoring)
    trail_stop:     Optional[float] = None   # Active trailing stop level
    trail_active:   bool            = False
    breakeven_moved: bool           = False

    # Partial exit tracking
    tp1_hit:        bool            = False
    tp2_hit:        bool            = False
    shares_remaining: int           = 0      # Updated as partials fill

    # Status
    status:         str             = 'open'  # 'open', 'closed', 'emergency_closed'
    exit_price:     Optional[float] = None
    exit_time:      Optional[datetime] = None
    exit_reason:    Optional[str]   = None
    final_r:        Optional[float] = None
    final_pnl:      Optional[float] = None

    def __post_init__(self):
        self.shares_remaining = self.shares

    def current_r(self, current_price: float) -> float:
        """Compute current unrealized R-multiple."""
        stop_dist = abs(self.entry_price - self.stop_price)
        if stop_dist == 0: return 0
        if self.direction == 'long':
            return (current_price - self.entry_price) / stop_dist
        else:
            return (self.entry_price - current_price) / stop_dist

    def is_above_breakeven(self, current_price: float) -> bool:
        r = self.current_r(current_price)
        return r >= 0.0
```

---

## MODULE 7 — TradeMonitor (The Brain of the Engine)

> Runs on every new bar. Checks every condition. Fires actions. This is the perfect trader watching every tick.

```python
# SECTION 7: TradeMonitor

from typing import List, Callable

@dataclass
class MonitorAction:
    action_type:    str      # 'move_stop', 'close_partial', 'close_all', 'move_to_breakeven'
    new_stop:       Optional[float]
    close_shares:   Optional[int]
    reason:         str
    urgency:        str      # 'immediate', 'next_bar', 'end_of_bar'

class TradeMonitor:
    """
    The continuous watchdog. Called every new bar for every open trade.
    Returns a list of actions to execute.
    """

    def __init__(self, profile: AssetProfile, context_fn: Callable):
        self.profile    = profile
        self.context_fn = context_fn   # function that returns fresh MarketContext

    def evaluate(
        self,
        trade:          TradeRecord,
        current_price:  float,
        current_bar:    dict,    # {'open': X, 'high': X, 'low': X, 'close': X, 'volume': X}
        atr_current:    float,
        ict_state:      dict     # fresh ICT structure analysis
    ) -> List[MonitorAction]:

        actions = []
        context = self.context_fn(trade.symbol)
        r       = trade.current_r(current_price)

        # Update trade state
        trade.current_price   = current_price
        trade.max_favorable_r = max(trade.max_favorable_r, r)
        trade.max_adverse_r   = min(trade.max_adverse_r, r)

        # ══════════════════════════════════════════════════════════════
        # CONDITION 1 — HARD STOP HIT
        # ══════════════════════════════════════════════════════════════
        stop_breached = (
            (trade.direction == 'long'  and current_price <= trade.stop_price) or
            (trade.direction == 'short' and current_price >= trade.stop_price)
        )
        if stop_breached:
            actions.append(MonitorAction(
                action_type  = 'close_all',
                new_stop     = None,
                close_shares = trade.shares_remaining,
                reason       = 'HARD_STOP_HIT',
                urgency      = 'immediate'
            ))
            return actions   # Stop hit = end evaluation; no other actions needed

        # ══════════════════════════════════════════════════════════════
        # CONDITION 2 — EMERGENCY: SHOCK CANDLE AGAINST POSITION
        # A single bar has moved more than N × ATR against the trade.
        # This is the "Trump announcement goes straight the other way" scenario.
        # ══════════════════════════════════════════════════════════════
        candle_adverse_move = 0
        if trade.direction == 'long':
            candle_adverse_move = (current_bar['open'] - current_bar['low']) / atr_current
        else:
            candle_adverse_move = (current_bar['high'] - current_bar['open']) / atr_current

        if candle_adverse_move > self.profile.shock_exit_atr_mult:
            # Shock candle detected. Close ALL immediately.
            actions.append(MonitorAction(
                action_type  = 'close_all',
                new_stop     = None,
                close_shares = trade.shares_remaining,
                reason       = f'SHOCK_CANDLE: {candle_adverse_move:.1f}x ATR adverse move',
                urgency      = 'immediate'
            ))
            return actions

        # ══════════════════════════════════════════════════════════════
        # CONDITION 3 — NEWS/EVENT WINDOW EMERGENCY EXIT
        # If trade is open and a major news event starts within 30 minutes:
        # Exit based on current profitability.
        # ══════════════════════════════════════════════════════════════
        if context.news_risk_window:
            if r >= 1.0:
                # In profit: close all to protect gains
                actions.append(MonitorAction(
                    action_type  = 'close_all',
                    new_stop     = None,
                    close_shares = trade.shares_remaining,
                    reason       = 'NEWS_WINDOW_PROFITABLE_EXIT',
                    urgency      = 'immediate'
                ))
            elif r >= 0.0:
                # At breakeven: close all (no point holding through news at breakeven)
                actions.append(MonitorAction(
                    action_type  = 'close_all',
                    new_stop     = None,
                    close_shares = trade.shares_remaining,
                    reason       = 'NEWS_WINDOW_BREAKEVEN_EXIT',
                    urgency      = 'immediate'
                ))
            # If negative: hold (already losing; news might help; stop protects)
            return actions if actions else []

        # ══════════════════════════════════════════════════════════════
        # CONDITION 4 — ICT STRUCTURE INVALIDATION
        # The setup that created the trade no longer exists.
        # Candle CLOSED through the OB/FVG that generated the signal.
        # ══════════════════════════════════════════════════════════════
        structure_invalidated = False
        if trade.direction == 'long':
            ob_low = ict_state.get('original_ob_low')
            if ob_low and current_bar['close'] < ob_low and r < 0:
                structure_invalidated = True
        else:
            ob_high = ict_state.get('original_ob_high')
            if ob_high and current_bar['close'] > ob_high and r < 0:
                structure_invalidated = True

        if structure_invalidated:
            actions.append(MonitorAction(
                action_type  = 'close_all',
                new_stop     = None,
                close_shares = trade.shares_remaining,
                reason       = 'ICT_STRUCTURE_INVALIDATED',
                urgency      = 'next_bar'
            ))
            return actions

        # ══════════════════════════════════════════════════════════════
        # CONDITION 5 — BREAKEVEN MOVE
        # Trade has reached breakeven trigger R. Move stop to entry.
        # ══════════════════════════════════════════════════════════════
        if r >= self.profile.breakeven_trigger_r and not trade.breakeven_moved:
            new_stop = trade.entry_price
            actions.append(MonitorAction(
                action_type  = 'move_stop',
                new_stop     = new_stop,
                close_shares = None,
                reason       = f'BREAKEVEN_MOVE at {r:.2f}R',
                urgency      = 'next_bar'
            ))
            trade.breakeven_moved = True
            trade.stop_price = new_stop  # Update immediately for downstream checks

        # ══════════════════════════════════════════════════════════════
        # CONDITION 6 — PARTIAL EXIT: TP1
        # ══════════════════════════════════════════════════════════════
        tp1_triggered = (
            not trade.tp1_hit and (
                (trade.direction == 'long'  and current_price >= trade.tp1_price) or
                (trade.direction == 'short' and current_price <= trade.tp1_price)
            )
        )
        if tp1_triggered:
            shares_to_close = trade.shares // 3   # Close 1/3
            actions.append(MonitorAction(
                action_type  = 'close_partial',
                new_stop     = None,
                close_shares = shares_to_close,
                reason       = f'TP1_HIT at {trade.tp1_r:.1f}R',
                urgency      = 'immediate'
            ))
            trade.tp1_hit = True
            trade.shares_remaining -= shares_to_close

        # ══════════════════════════════════════════════════════════════
        # CONDITION 7 — PARTIAL EXIT: TP2
        # ══════════════════════════════════════════════════════════════
        tp2_triggered = (
            trade.tp1_hit and not trade.tp2_hit and (
                (trade.direction == 'long'  and current_price >= trade.tp2_price) or
                (trade.direction == 'short' and current_price <= trade.tp2_price)
            )
        )
        if tp2_triggered:
            shares_to_close = trade.shares // 3
            actions.append(MonitorAction(
                action_type  = 'close_partial',
                new_stop     = None,
                close_shares = shares_to_close,
                reason       = f'TP2_HIT at {trade.tp2_r:.1f}R',
                urgency      = 'immediate'
            ))
            trade.tp2_hit = True
            trade.shares_remaining -= shares_to_close

        # ══════════════════════════════════════════════════════════════
        # CONDITION 8 — TRAILING STOP (THE RUNNER)
        # After TP1 hit, trail the remaining position.
        # ══════════════════════════════════════════════════════════════
        if trade.tp1_hit and r >= self.profile.trail_activation_r:
            trade.trail_active = True
            trail_distance = atr_current * self.profile.trail_atr_multiplier

            if trade.direction == 'long':
                new_trail = current_price - trail_distance
                # Trail only moves UP, never down
                if trade.trail_stop is None or new_trail > trade.trail_stop:
                    # Ensure trail stop is never below breakeven
                    new_trail = max(new_trail, trade.entry_price)
                    if new_trail != trade.trail_stop:
                        actions.append(MonitorAction(
                            action_type  = 'move_stop',
                            new_stop     = round(new_trail, 4),
                            close_shares = None,
                            reason       = f'TRAIL_MOVE: {r:.2f}R, ATR×{self.profile.trail_atr_multiplier}',
                            urgency      = 'next_bar'
                        ))
                        trade.trail_stop = new_trail
                        trade.stop_price = new_trail  # Update for downstream

            else:  # short
                new_trail = current_price + trail_distance
                if trade.trail_stop is None or new_trail < trade.trail_stop:
                    new_trail = min(new_trail, trade.entry_price)
                    if new_trail != trade.trail_stop:
                        actions.append(MonitorAction(
                            action_type  = 'move_stop',
                            new_stop     = round(new_trail, 4),
                            close_shares = None,
                            reason       = f'TRAIL_MOVE: {r:.2f}R',
                            urgency      = 'next_bar'
                        ))
                        trade.trail_stop = new_trail
                        trade.stop_price = new_trail

        # ══════════════════════════════════════════════════════════════
        # CONDITION 9 — TIME STOP
        # Trade has been open for N days without hitting TP1.
        # Dead money. Exit half; move stop to breakeven on rest.
        # ══════════════════════════════════════════════════════════════
        days_open = (datetime.now() - trade.entry_time).days

        if days_open >= 5 and not trade.tp1_hit and r < 0.5:
            shares_to_close = trade.shares_remaining // 2
            actions.append(MonitorAction(
                action_type  = 'close_partial',
                new_stop     = trade.entry_price,  # move remainder to BE
                close_shares = shares_to_close,
                reason       = f'TIME_STOP: {days_open}d open, no TP1',
                urgency      = 'next_bar'
            ))

        if days_open >= 8 and not trade.tp1_hit:
            actions.append(MonitorAction(
                action_type  = 'close_all',
                new_stop     = None,
                close_shares = trade.shares_remaining,
                reason       = f'HARD_TIME_STOP: {days_open}d without progress',
                urgency      = 'next_bar'
            ))

        # ══════════════════════════════════════════════════════════════
        # CONDITION 10 — REVERSAL SIGNAL OVERRIDE
        # A strong reversal candlestick has formed at or near a target.
        # This is the "bearish engulfing at resistance" case.
        # Close ALL remaining if reversal signal forms near TP1+.
        # ══════════════════════════════════════════════════════════════
        if r >= (self.profile.tp1_r_ratio * 0.8):  # Near TP1
            reversal_detected = self._detect_reversal_candle(
                current_bar, trade.direction, atr_current
            )
            if reversal_detected and trade.tp1_hit:
                actions.append(MonitorAction(
                    action_type  = 'close_all',
                    new_stop     = None,
                    close_shares = trade.shares_remaining,
                    reason       = f'REVERSAL_SIGNAL at {r:.2f}R',
                    urgency      = 'immediate'
                ))

        return actions

    @staticmethod
    def _detect_reversal_candle(bar: dict, direction: str, atr: float) -> bool:
        """Detect bearish engulfing / shooting star / pin bar at target."""
        body     = abs(bar['close'] - bar['open'])
        upper_wk = bar['high'] - max(bar['close'], bar['open'])
        lower_wk = min(bar['close'], bar['open']) - bar['low']

        if direction == 'long':
            # Bearish signals at top: shooting star or bearish engulfing
            shooting_star = upper_wk > body * 2 and upper_wk > atr * 0.5
            bearish_engulf = bar['close'] < bar['open'] and body > atr * 0.6
            return shooting_star or bearish_engulf
        else:
            # Bullish signals at bottom
            hammer        = lower_wk > body * 2 and lower_wk > atr * 0.5
            bullish_engulf = bar['close'] > bar['open'] and body > atr * 0.6
            return hammer or bullish_engulf
```

---

## MODULE 8 — RREngine (Master Orchestrator)

> The single entry point. Everything flows through here.

```python
# SECTION 8: RREngine — Master Orchestrator

import uuid
from typing import Optional

class RREngine:
    """
    Master orchestrator for the entire R:R lifecycle.
    Single entry point for the trading system.

    USAGE:
        engine = RREngine(account_balance=50000)

        # At signal time:
        trade = engine.evaluate_signal(
            symbol='META', direction='long', grade='A+',
            entry_price=485.20,
            ict_structure={'ob_low': 482.10, 'sweep_low': 481.50},
            ict_levels={'pdh': 492.00, 'bsl': 498.50, 'fvg_above': 487.00},
            atr_value=8.40
        )
        if trade: execute_entry(trade)

        # Every new bar:
        for trade in engine.open_trades:
            actions = engine.update_trade(
                trade_id=trade.trade_id,
                current_price=487.50,
                current_bar={...},
                atr_current=8.20,
                ict_state={...}
            )
            for action in actions:
                execute_action(trade, action)
    """

    def __init__(self, account_balance: float, data_feed=None):
        self.account_balance  = account_balance
        self.data_feed        = data_feed
        self.open_trades:     List[TradeRecord] = []
        self.closed_trades:   List[TradeRecord] = []
        self.high_water_mark: float = account_balance

    @property
    def current_drawdown(self) -> float:
        return max(0, (self.high_water_mark - self.account_balance) / self.high_water_mark)

    def evaluate_signal(
        self,
        symbol:         str,
        direction:      str,
        grade:          str,
        entry_price:    float,
        ict_structure:  dict,
        ict_levels:     dict,
        atr_value:      float,
        signal_time:    Optional[datetime] = None
    ) -> Optional[TradeRecord]:
        """
        Evaluate a new signal. Returns TradeRecord if approved, None if rejected.
        """
        profile = get_asset_profile(symbol)
        context = build_market_context(symbol, self.data_feed)

        # ── Gate 0: Leveraged ETF Block ────────────────────────────────
        if profile.asset_class == AssetClass.LEVERAGED_ETF:
            self._log(f"REJECTED {symbol}: Leveraged ETF permanently gated")
            return None

        # ── Gate 1: ATR% Volatility Gate ──────────────────────────────
        if context.atr_pct_current > 4.0:
            self._log(f"REJECTED {symbol}: ATR% {context.atr_pct_current:.1f}% > 4.0% gate")
            return None

        # ── Gate 2: Asset-Specific VIX Gate ───────────────────────────
        if profile.avoid_in_high_vix and context.vix_level > 25:
            self._log(f"REJECTED {symbol}: High VIX {context.vix_level:.0f} gate")
            return None

        # ── Compute Stop ───────────────────────────────────────────────
        stop_result = StopCalculator.compute(
            direction, entry_price, ict_structure, profile, context, atr_value
        )
        if not stop_result.valid:
            self._log(f"REJECTED {symbol}: Stop too wide ({stop_result.stop_distance_pct:.1f}%)")
            return None

        # ── Compute Targets ────────────────────────────────────────────
        target_set = TargetCalculator.compute(
            direction, entry_price, stop_result, ict_levels, profile, context
        )
        if target_set.tp1_r < target_set.min_acceptable_r:
            self._log(f"REJECTED {symbol}: R:R {target_set.tp1_r:.1f} < minimum {target_set.min_acceptable_r}")
            return None

        # ── Compute Position Size ──────────────────────────────────────
        size_result = PositionSizer.compute(
            self.account_balance, grade, stop_result, entry_price,
            profile, context, self.open_trades, self.current_drawdown
        )
        if not size_result.approved:
            self._log(f"REJECTED {symbol}: Sizing rejected — {size_result.rejection_reason}")
            return None

        # ── Create Trade Record ────────────────────────────────────────
        trade = TradeRecord(
            trade_id        = str(uuid.uuid4())[:8],
            symbol          = symbol,
            direction       = direction,
            signal_grade    = grade,
            signal_time     = signal_time or datetime.now(),
            entry_time      = datetime.now(),
            entry_price     = entry_price,
            shares          = size_result.shares,
            dollar_risk     = size_result.dollar_risk,
            risk_pct        = size_result.risk_pct_of_acct,
            stop_price      = stop_result.stop_price,
            stop_method     = stop_result.method,
            breakeven_price = entry_price,
            tp1_price       = target_set.tp1_price,
            tp2_price       = target_set.tp2_price,
            tp3_price       = target_set.tp3_price,
            tp1_r           = target_set.tp1_r,
            tp2_r           = target_set.tp2_r,
            tp3_r           = target_set.tp3_r,
        )
        trade.shares_remaining = trade.shares

        self.open_trades.append(trade)
        self._log_entry(trade, stop_result, target_set, size_result)
        return trade

    def update_trade(
        self,
        trade_id:       str,
        current_price:  float,
        current_bar:    dict,
        atr_current:    float,
        ict_state:      dict
    ) -> List[MonitorAction]:
        """Called every bar for every open trade."""
        trade = next((t for t in self.open_trades if t.trade_id == trade_id), None)
        if not trade: return []

        profile = get_asset_profile(trade.symbol)
        context_fn = lambda sym: build_market_context(sym, self.data_feed)
        monitor = TradeMonitor(profile, context_fn)

        actions = monitor.evaluate(trade, current_price, current_bar, atr_current, ict_state)

        # Process close_all actions
        for action in actions:
            if action.action_type == 'close_all':
                self._close_trade(trade, current_price, action.reason)

        return actions

    def _close_trade(self, trade: TradeRecord, exit_price: float, reason: str):
        stop_dist = abs(trade.entry_price - trade.stop_price)
        if trade.direction == 'long':
            final_r = (exit_price - trade.entry_price) / stop_dist if stop_dist else 0
        else:
            final_r = (trade.entry_price - exit_price) / stop_dist if stop_dist else 0

        trade.status       = 'closed'
        trade.exit_price   = exit_price
        trade.exit_time    = datetime.now()
        trade.exit_reason  = reason
        trade.final_r      = round(final_r, 3)
        trade.final_pnl    = (exit_price - trade.entry_price) * trade.shares_remaining \
                              * (1 if trade.direction == 'long' else -1)

        self.open_trades   = [t for t in self.open_trades if t.trade_id != trade.trade_id]
        self.closed_trades.append(trade)

        # Update account balance
        self.account_balance += trade.final_pnl
        if self.account_balance > self.high_water_mark:
            self.high_water_mark = self.account_balance

        self._log_exit(trade)

    def _log(self, message: str):
        print(f"[RR_ENGINE] {datetime.now().strftime('%H:%M:%S')} {message}")

    def _log_entry(self, trade, stop, targets, size):
        self._log(
            f"APPROVED {trade.symbol} {trade.direction.upper()} "
            f"Grade={trade.signal_grade} "
            f"Shares={trade.shares} Risk=${trade.dollar_risk:.0f} ({trade.risk_pct*100:.1f}%) "
            f"Stop={trade.stop_price:.2f} ({stop.method}) "
            f"TP1={trade.tp1_price:.2f} ({trade.tp1_r:.1f}R) "
            f"TP2={trade.tp2_price:.2f} ({trade.tp2_r:.1f}R) "
            f"TP3={trade.tp3_price:.2f} ({trade.tp3_r:.1f}R)"
        )

    def _log_exit(self, trade):
        emoji = "✅" if trade.final_r and trade.final_r > 0 else "❌"
        self._log(
            f"{emoji} CLOSED {trade.symbol} | "
            f"R={trade.final_r:.2f} | "
            f"PnL=${trade.final_pnl:.2f} | "
            f"Reason={trade.exit_reason}"
        )
```

---

## MODULE 9 — PostTradeAnalyzer

> Every trade teaches the system. This module collects the data.

```python
# SECTION 9: PostTradeAnalyzer

class PostTradeAnalyzer:
    """
    After every closed trade: compute what the optimal exit would have been.
    Feed this back to improve asset profiles over time.
    """

    @staticmethod
    def analyze(trade: TradeRecord, full_price_series: pd.Series) -> dict:
        """
        Compare actual exit to optimal exit using MAE/MFE analysis.
        MAE = Maximum Adverse Excursion (worst point reached before exit)
        MFE = Maximum Favorable Excursion (best point reached before exit)
        """
        if trade.direction == 'long':
            mae_price = full_price_series.loc[trade.entry_time:trade.exit_time].min()
            mfe_price = full_price_series.loc[trade.entry_time:trade.exit_time].max()
        else:
            mae_price = full_price_series.loc[trade.entry_time:trade.exit_time].max()
            mfe_price = full_price_series.loc[trade.entry_time:trade.exit_time].min()

        stop_dist = abs(trade.entry_price - trade.stop_price)
        mae_r = abs(mae_price - trade.entry_price) / stop_dist
        mfe_r = abs(mfe_price - trade.entry_price) / stop_dist

        # Efficiency: how much of the available move did we capture?
        if mfe_r > 0:
            capture_efficiency = trade.final_r / mfe_r
        else:
            capture_efficiency = 0

        # Was the stop placement correct?
        # If MAE < stop distance: stop was not needed (good)
        # If MAE > stop distance: stop was correctly used (good)
        stop_was_hit   = trade.exit_reason == 'HARD_STOP_HIT'
        stop_was_close = mae_r > 0.8   # Came within 80% of stop without hitting

        return {
            'trade_id':           trade.trade_id,
            'symbol':             trade.symbol,
            'direction':          trade.direction,
            'entry_price':        trade.entry_price,
            'exit_price':         trade.exit_price,
            'final_r':            trade.final_r,
            'mae_r':              round(mae_r, 3),
            'mfe_r':              round(mfe_r, 3),
            'capture_efficiency': round(capture_efficiency, 3),
            'stop_hit':           stop_was_hit,
            'stop_close_call':    stop_was_close,
            'exit_reason':        trade.exit_reason,
            # Actionable insight: should the stop multiplier be adjusted?
            'stop_too_tight':     stop_was_hit and mfe_r > 1.5,   # Hit stop but trade would have worked
            'stop_too_wide':      not stop_was_hit and mae_r > 1.8, # Came close but overrisked
            'exit_too_early':     trade.final_r < mfe_r * 0.5,    # Left half the move on table
            'exit_too_late':      trade.final_r < 0 and mfe_r > 1.0, # Was profitable; gave back
        }

    @staticmethod
    def update_asset_profile(symbol: str, analysis_batch: list):
        """
        After 50+ trades on a symbol: adjust its profile parameters based on MAE/MFE data.
        This is the self-learning loop.
        """
        if len(analysis_batch) < 50: return  # Need minimum sample

        # Optimal stop multiplier: set so stops are hit on ~15% of MAE readings
        mae_values    = [a['mae_r'] for a in analysis_batch if not a['stop_hit']]
        optimal_stop_mult = np.percentile(mae_values, 85) if mae_values else 1.5

        # Optimal TP1: where does MFE most commonly stall?
        mfe_values  = [a['mfe_r'] for a in analysis_batch]
        optimal_tp1 = np.percentile(mfe_values, 40)  # 40th percentile of best moves

        print(f"[PROFILE_UPDATE] {symbol}: "
              f"Optimal stop mult={optimal_stop_mult:.2f} "
              f"(currently {ASSET_PROFILES[symbol].stop_atr_multiplier:.2f}), "
              f"Optimal TP1 R={optimal_tp1:.2f}R "
              f"(currently {ASSET_PROFILES[symbol].tp1_r_ratio:.2f}R) "
              f"based on {len(analysis_batch)} trades")

        # NOTE: Do NOT auto-update. Log recommendation. Human reviews quarterly.
        # Auto-updating creates feedback loops that can destabilize the system.
```

---

## MODULE 10 — Integration Instructions

> Exact instructions for the AI build engine.

```
══════════════════════════════════════════════════════════════════════
AI BUILD ENGINE INSTRUCTIONS — READ BEFORE CODING
══════════════════════════════════════════════════════════════════════

TARGET FILE: execution/rr_engine.py
ALL SECTIONS ABOVE GO INTO ONE SINGLE FILE in this order:
  1. Imports (add any missing imports at top)
  2. AssetProfile + AssetClass + ASSET_PROFILES registry
  3. MarketContext + build_market_context()
  4. StopCalculator
  5. TargetCalculator
  6. PositionSizer
  7. TradeRecord
  8. MonitorAction
  9. TradeMonitor
  10. RREngine
  11. PostTradeAnalyzer

INTEGRATION POINTS (connect to existing code):

1. IN orchestrator/backtest_lifecycle.py:
   Replace any existing stop/target logic with:
   
   from execution.rr_engine import RREngine
   engine = RREngine(account_balance=config.STARTING_CAPITAL, data_feed=data_feed)
   
   At each signal:
   trade = engine.evaluate_signal(
       symbol=signal.symbol,
       direction=signal.direction,
       grade=signal.grade,
       entry_price=signal.entry_price,
       ict_structure=signal.ict_structure,   # from Layer 3 ICT output
       ict_levels=signal.ict_levels,          # PDH, PWH, FVG levels from ICT layer
       atr_value=signal.atr_14d
   )
   
   In the bar loop:
   for trade in engine.open_trades:
       actions = engine.update_trade(
           trade_id=trade.trade_id,
           current_price=bar.close,
           current_bar={'open':bar.open,'high':bar.high,'low':bar.low,'close':bar.close,'volume':bar.volume},
           atr_current=bar.atr_14d,
           ict_state=ict_layer.get_current_state(trade.symbol)
       )
       for action in actions:
           execute_action_on_broker(trade, action)

2. IN layer1/bias_engine.py:
   The RREngine is downstream of BiasEngine.
   BiasEngine outputs signal grade (A+/A/B/C).
   Pass grade directly to engine.evaluate_signal(grade=bias_result.grade)

3. IN execution/paper_trading.py:
   Replace manual stop/target tracking with RREngine.
   Treat paper_trading as a wrapper: instead of sending broker orders,
   log them to paper_trade_log.csv.
   All logic stays identical.

4. ASSET PROFILE AUTO-COMPUTATION:
   Create: training/compute_asset_profiles.py
   This script should:
   - Load master_harvest_8mo.pkl
   - For each symbol: compute rolling 20-day ATR%, avg spread, avg volume
   - Compute optimal stop multiplier from MAE analysis
   - Compute optimal TP ratios from MFE analysis  
   - Output updated ASSET_PROFILES dict
   - Run quarterly or when adding new symbols to universe

5. DATA FEED INTERFACE:
   build_market_context() calls self.data_feed.get_vix(), .get_atr(), etc.
   Implement a DataFeedInterface class that wraps your existing data source.
   Methods needed:
   - get_vix() → float
   - get_price(symbol) → float
   - get_atr(symbol, period=14) → float
   - get_trend_regime() → str ('bull'/'bear'/'neutral') from BiasEngine
   - get_session() → str
   - is_news_window() → bool (check economic calendar)
   - get_spread_multiplier(symbol) → float

6. ICT STRUCTURE OUTPUT:
   ict_structure dict keys needed from Layer 3:
   - 'ob_low': float or None      (order block low for longs)
   - 'ob_high': float or None     (order block high for shorts)
   - 'fvg_bottom': float or None  (FVG bottom for longs)
   - 'fvg_top': float or None     (FVG top for shorts)
   - 'sweep_low': float or None   (liquidity sweep low for longs)
   - 'sweep_high': float or None  (liquidity sweep high for shorts)
   - 'original_ob_low': float     (for structure invalidation check)
   - 'original_ob_high': float    (for structure invalidation check)

   ict_levels dict keys needed:
   - 'pdh': Previous Day High
   - 'pdl': Previous Day Low
   - 'pwh': Previous Week High
   - 'pwl': Previous Week Low
   - 'fvg_above': Nearest FVG above price (for longs)
   - 'fvg_below': Nearest FVG below price (for shorts)
   - 'bsl': Buy Side Liquidity level
   - 'ssl': Sell Side Liquidity level
   - 'daily_ob_above': Daily OB above price
   - 'daily_ob_below': Daily OB below price

7. MISSING SYMBOLS:
   For any symbol in your 57-asset universe not listed in ASSET_PROFILES:
   The _DEFAULT profile is used automatically.
   Add symbol-specific profiles as you accumulate 50+ trades per symbol.
   PostTradeAnalyzer.update_asset_profile() will recommend the parameters.

8. LOGGING:
   All engine decisions write to: logs/rr_engine_[DATE].log
   All trade records write to: data/live_trade_log.csv
   All post-trade analysis writes to: data/mae_mfe_analysis.csv

══════════════════════════════════════════════════════════════════════
TESTING THE ENGINE BEFORE LIVE DEPLOYMENT:
══════════════════════════════════════════════════════════════════════

Run: python execution/test_rr_engine.py

The test file should:
1. Initialize RREngine with $50,000 paper account
2. Feed in 10 synthetic signals (long and short, various grades)
3. Simulate 20 bars of price movement for each
4. Verify all 10 conditions fire correctly:
   - Hard stop fires on adverse price
   - Shock candle fires on 2.5 ATR adverse bar
   - TP1 fires and closes 1/3 of position
   - Breakeven moves after 1R
   - Trailing stop activates after TP1
   - Time stop fires at day 5 without progress
   - News window exits profitable trades
   - ICT structure invalidation exits
   - Reversal candle exits at target zone
5. Print final equity curve
6. Assert: all stops are enforced, no trade loses more than 1R without shock/gap
══════════════════════════════════════════════════════════════════════
```

---

## Decision Logic Summary (Plain English)

```
WHEN A SIGNAL ARRIVES:
  Gate check (leveraged ETF? ATR% > 4%? VIX too high?) → reject if any fail
  Compute ICT structural stop → validate with ATR → reject if too wide
  Compute ICT level targets → ensure 2:1 minimum R:R → reject if fails
  Compute position size → account for grade, drawdown, VIX, concurrent positions
  Create trade. Set hard stop order immediately.

EVERY BAR WHILE TRADE IS OPEN (in priority order):
  1. Hard stop hit? → CLOSE ALL IMMEDIATELY
  2. Shock candle (>2.5 ATR adverse)? → CLOSE ALL IMMEDIATELY
  3. News window opening? → if profitable: CLOSE ALL; if breakeven: CLOSE ALL
  4. ICT structure closed through? → CLOSE ALL NEXT BAR
  5. Reached breakeven trigger R? → MOVE STOP TO ENTRY (once only)
  6. Reached TP1? → CLOSE 1/3 IMMEDIATELY
  7. Reached TP2? → CLOSE 1/3 IMMEDIATELY
  8. Trailing stop active (after TP1)? → MOVE STOP UP WITH PRICE (never down)
  9. Day 5 without TP1? → CLOSE HALF; MOVE REST TO BREAKEVEN
  10. Day 8 without TP1? → CLOSE ALL
  11. Reversal candle at target zone? → CLOSE ALL IMMEDIATELY

THIS IS HOW A PERFECT TRADER WATCHES EVERY TRADE.
THE ENGINE NEVER SLEEPS. NEVER GETS EMOTIONAL. NEVER HESITATES.
```

---

## Related Notes

- [[ICT Swing Trade Decision Engine]]
- [[Reality Bridge MOC]]
- [[Edge Research Program MOC]]
- [[Imbalance Engine MOC]]
- [[XGBoost Bias Engine]]
- [[Asset Profile Registry]]
- [[MAE MFE Analysis]]
- [[PostTrade Analyzer Output]]
- [[Paper Trading Log]]

---

*The stop protects you from being wrong. The trailing stop rewards you for being right. The partial exits guarantee you never give back a winner. The time stop frees capital from dead trades. Together they form the only thing a trader truly controls: how much they lose when wrong, and how much they keep when right.*
