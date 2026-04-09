# CLAWD Trading — Architecture Overview

## System Overview

CLAWD Trading is a three-layer algorithmic trading system that combines machine learning, quantitative risk management, and game-theoretic execution. Each layer has a single, well-defined responsibility.

```
Market Data ──► Layer 1 (Bias) ──► Layer 2 (Risk) ──► Layer 3 (Game) ──► Execution
                    │                    │                   │
                  XGBoost             Kelly/ATR          Liquidity/Nash
                  Regime              Sizing              Detection
```

---

## Layer 1: AI Bias Engine (`layer1/`)

**Purpose:** Determine directional bias (LONG / SHORT / FLAT) and regime classification.

| Component | File | Responsibility |
|-----------|------|----------------|
| `BiasEngine` | `bias_engine.py` | XGBoost model inference |
| `BiasEngineV2` | `bias_engine_v2.py` | Enhanced model with SHAP explainability |
| `FeatureBuilder` | `feature_builder.py` | 43-feature engineering pipeline |
| `FeatureBuilderV2` | `feature_builder_v2.py` | Expanded feature set |
| `RegimeClassifier` | `regime_classifier.py` | 5-axis regime detection (trend, vol, breadth, sector, macro) |
| `HardConstraints` | `hard_constraints.py` | Non-negotiable pre-trade filters |

**Data flow:**
```
OHLCV bars → FeatureBuilder → XGBoost model → BiasOutput(direction, confidence, regime)
```

---

## Layer 2: Quant Risk Model (`layer2/`)

**Purpose:** Size positions, set stops/targets, calculate expected value.

| Component | File | Responsibility |
|-----------|------|----------------|
| `PositionSizing` | `risk_engine.py` | Kelly criterion position sizing |
| `StopCalculator` | `risk_engine.py` | ATR + structural stop levels |
| `TargetCalculator` | `risk_engine.py` | R-multiple and liquidity targets |
| `ExpectedValueCalculator` | `risk_engine.py` | EV gate (only trade if EV > threshold) |
| `RiskEngine` | `risk_engine.py` | Orchestrates all risk calculations |

**Data flow:**
```
BiasOutput + market data → RiskEngine → RiskOutput(size, stop, target, ev)
```

---

## Layer 3: Game-Theoretic Engine (`layer3/`)

**Purpose:** Detect institutional liquidity, trapped positions, and optimal entry timing.

| Component | File | Responsibility |
|-----------|------|----------------|
| `LiquidityMap` | `game_engine.py` | Map buy/sell liquidity pools |
| `TrappedPositionDetector` | `game_engine.py` | Identify trapped longs/shorts |
| `AdversarialLevelModel` | `game_engine.py` | Model where market makers will push price |
| `OrderFlowAnalyzer` | `game_engine.py` | Microstructure signal detection |
| `GameEngine` | `game_engine.py` | Orchestrates all game-theoretic analysis |

**Data flow:**
```
RiskOutput + order flow → GameEngine → GameOutput(entry_zone, liquidity_target, timing_score)
```

---

## Data Pipeline (`data/`)

```
External APIs
    │
    ├── Alpaca (primary)    → data/alpaca_client.py
    ├── Polygon (backup)    → data/polygon_client.py
    ├── TradeLocker         → data/tradelocker_client.py
    └── yfinance (fallback) → data/providers.py
    │
    ▼
DataPipeline (data/pipeline.py)
    │  - Normalises OHLCV bars to OHLCVBar schema
    │  - Validates with DataValidator
    │  - Caches to disk (DATA_DIR)
    ▼
Feature Engineering (layer1/feature_builder*.py)
```

### Schema contract

All internal data must conform to `data/schema.py`:
- `OHLCVBar` — single OHLCV bar with timestamp, symbol, timeframe
- `FeatureSchema` — 43-feature vector fed to the XGBoost model

---

## Model Lifecycle

```
1. Training
   └── scripts/full_backtest.py ──► XGBoost model (.pkl) ──► infrastructure/model_registry.py

2. Live inference
   └── layer1/bias_engine_v2.py (loads model from registry)

3. Performance monitoring
   └── clawd_trading/meta_evaluator/performance_monitor.py
       ├── Tracks Sharpe, win rate, drawdown, model drift
       └── Publishes metrics to Firebase RTDB

4. Retraining trigger
   └── clawd_trading/meta_evaluator/refit_scheduler.py
       ├── Checks for model drift or schedule
       └── Triggers retrain if criteria met
```

---

## Retraining Loop

```
PerformanceMonitor detects drift
        │
        ▼
RefitScheduler schedules retrain
        │
        ▼
run_retrain.py (or nightly-retrain.yml CI workflow)
        │
        ▼
New model trained and registered
        │
        ▼
BiasEngineV2 hot-reloads model
```

The nightly retrain workflow (`.github/workflows/nightly-retrain.yml`) runs this loop automatically at 03:00 UTC on weekdays.

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `run_backtest.py` | Run historical backtests |
| `run_live.py` | Start live / paper trading engine |
| `run_retrain.py` | Trigger model retraining |

Legacy scripts are preserved in `scripts/legacy/` for reference.

---

## Configuration

All runtime configuration lives in `config.py` (root) and `config/settings.py`.
Settings are loaded from environment variables (`.env` file) with safe defaults.
See `.env.example` for the full list of required variables.
