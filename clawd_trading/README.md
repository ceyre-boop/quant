# Clawd Trading System - Architecture Overview

Complete quant trading platform with ML-driven signals, Firebase integration, and GitHub automation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export FIREBASE_SERVICE_ACCOUNT=path/to/credentials.json
export POLYGON_API_KEY=your_key
export TRADELOCKER_API_KEY=your_key

# Run the system
python main.py --mode demo
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 4: ORCHESTRATOR                      │
│                    (Execution coordination)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Layer 1:    │  │ Layer 2:    │  │ Layer 3:    │             │
│  │ Bias Engine │→ │ Risk Engine │→ │ Game Theory │             │
│  │ (Direction) │  │ (Sizing)    │  │ (Execution) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Guardrails  │  │ Risk Manager│  │ Model       │             │
│  │ (Safety)    │  │ (Limits)    │  │ Registry    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                    META EVALUATION LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Performance │  │ Signal      │  │ Refit       │             │
│  │ Monitor     │  │ Analyzer    │  │ Scheduler   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                    INTEGRATION LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Firebase    │  │ Polygon     │  │ TradeLocker │             │
│  │ (State)     │  │ (Market)    │  │ (Broker)    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
clawd_trading/
├── contracts/           # Type definitions
│   └── types.py
├── layer1/             # Directional bias
│   ├── bias_engine.py
│   └── feature_tracker.py
├── layer2/             # Risk management
│   ├── risk_engine.py
│   └── hard_logic.py
├── layer3/             # Execution optimization
│   └── game_theory.py
├── layer4/             # Orchestration
│   └── orchestrator.py
├── infrastructure/     # Safety & management
│   ├── guardrails.py       # Execution guardrails
│   ├── risk_manager.py     # Position risk
│   └── model_registry.py   # Model versioning
├── meta_evaluator/     # Performance & analysis
│   ├── performance_monitor.py
│   ├── metrics_calculator.py
│   ├── signal_analyzer.py
│   └── refit_scheduler.py
├── integration/        # External services
│   ├── firebase_client.py
│   ├── firebase_broadcaster.py
│   ├── firebase_ui_writer.py
│   ├── polygon_client.py
│   └── tradelocker_client.py
├── data/               # Data management
│   ├── pipeline.py
│   └── signal_archive.py
├── models/             # Saved models
│   └── README.md
├── docs/               # Documentation
│   ├── EXPERIMENT_BRANCHES.md
│   ├── RESEARCH_JOURNAL.md
│   └── LIQUIDITY_HEATMAPS.md
└── .github/workflows/  # Automation
    ├── ci-cd.yml
    ├── nightly-retrain.yml
    ├── weekly-report.yml
    └── data-snapshot.yml
```

## Key Features

### 1. Execution Guardrails (`infrastructure/guardrails.py`)

Safety controls preventing catastrophic losses:

```python
from infrastructure.guardrails import ExecutionGuardrails

guardrails = ExecutionGuardrails()

# Check if trading allowed
allowed, reason = guardrails.check_trading_allowed()

# Check position limits
allowed, reason = guardrails.check_position_limits('NQ', 3)

# Trigger emergency stop
guardrails.trigger_emergency_stop("Daily loss limit reached")
```

**Firebase Controls** (`/system_controls/`):
- `trading_enabled`: Master on/off switch
- `emergency_stop`: Immediate halt
- `max_daily_loss`: Loss limit ($)
- `max_position_size`: Max contracts
- `allowed_symbols`: Whitelist

### 2. Risk Manager (`infrastructure/risk_manager.py`)

Position-level risk management:

```python
from infrastructure.risk_manager import RiskManager

risk_mgr = RiskManager(account_size=100000)

# Validate trade before execution
allowed, reason, risk_data = risk_mgr.validate_trade(
    symbol='NQ',
    direction='LONG',
    size=2,
    entry_price=18450,
    stop_price=18400
)

if allowed:
    # Execute trade
    risk_mgr.record_trade_execution(...)
```

### 3. Model Registry (`infrastructure/model_registry.py`)

Version control for ML models:

```python
from infrastructure.model_registry import ModelRegistry

registry = ModelRegistry()

# Save new model
metadata = registry.save_model(
    model=xgb_model,
    name='bias_engine',
    metrics={'sharpe': 1.42, 'win_rate': 0.58}
)

# Load specific version
model, metadata = registry.load_model('bias_engine', version=3)

# List all versions
models = registry.list_models('bias_engine')
```

### 4. Signal Archive (`data/signal_archive.py`)

Complete history of all signals:

```python
from data.signal_archive import SignalArchive

archive = SignalArchive()

# Save signal with full context
signal_id = archive.save_signal(
    signal=entry_signal,
    bias=bias_output,
    risk=risk_output,
    features=feature_dict,
    market_context={'vix': 18.5}
)

# Later, update with outcome
archive.update_outcome(signal_id, exit_price=18500, pnl=500)

# Analyze history
signals = archive.get_signals_for_date('2026-03-11')
```

### 5. Performance Monitor (`meta_evaluator/performance_monitor.py`)

Weekly performance tracking:

```python
from meta_evaluator.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

# Run weekly analysis
metrics = monitor.run_weekly_analysis(days_back=7)

print(f"Sharpe: {metrics.sharpe_ratio}")
print(f"Win Rate: {metrics.win_rate}%")
print(f"Max DD: {metrics.max_drawdown}%")
```

### 6. Signal Analyzer (`meta_evaluator/signal_analyzer.py`)

Find patterns in your trading:

```python
from meta_evaluator.signal_analyzer import SignalAnalyzer

analyzer = SignalAnalyzer()

# Best performing regime
regime_stats = analyzer.analyze_by_regime()

# Best hour of day
hourly_stats = analyzer.analyze_by_hour()

# Full report
report = analyzer.generate_full_report()
```

### 7. Refit Scheduler (`meta_evaluator/refit_scheduler.py`)

Automated model retraining:

```python
from meta_evaluator.refit_scheduler import RefitScheduler

scheduler = RefitScheduler()

# Check if retraining needed
should_refit, reason = scheduler.should_refit()

# Run scheduled retraining
results = scheduler.run_scheduled_refit(
    model_types=['bias_engine'],
    commit_results=True
)
```

### 8. Feature Tracker (`layer1/feature_tracker.py`)

Monitor what your models see:

```python
from layer1.feature_tracker import FeatureTracker

tracker = FeatureTracker()

# Capture features
tracker.capture_features(
    symbol='NQ',
    price=18450,
    layer1_features={'momentum_score': 0.65},
    layer2_features={'liquidity_score': 0.80},
    market_context={'vix': 18.5}
)
```

### 9. Enhanced Firebase Broadcasting

New broadcast methods:

```python
from integration.firebase_broadcaster import FirebaseBroadcaster

broadcaster = FirebaseBroadcaster()

# Broadcast features
broadcaster.broadcast_features('NQ', feature_dict)

# Broadcast SHAP explainability
broadcaster.broadcast_explainability(
    symbol='NQ',
    model_name='bias_engine',
    prediction=0.75,
    shap_values={'momentum': 0.3, 'volatility': -0.2},
    top_positive=['momentum', 'trend'],
    top_negative=['vix', 'spread']
)

# Broadcast market state
broadcaster.broadcast_market_state(
    symbol='NQ',
    regime=2,
    vix=18.5,
    liquidity_score=0.8,
    momentum_score=0.6
)
```

## GitHub Actions Automation

### Nightly Model Retraining

Runs daily at 2:00 AM EST:
- Fetches latest data
- Retrains models
- Evaluates performance
- Commits if improved

```yaml
.github/workflows/nightly-retrain.yml
```

### Weekly Performance Report

Runs every Monday at 9:00 AM EST:
- Calculates weekly metrics
- Pushes to Firebase
- Creates GitHub Issue if performance degrades

```yaml
.github/workflows/weekly-report.yml
```

### Data Snapshots

Runs weekly:
- Archives market features
- Archives signal history
- Archives position history
- Creates reproducible dataset

```yaml
.github/workflows/data-snapshot.yml
```

### CI/CD Pipeline

On every push:
- Runs tests
- Lints code
- Deploys Firebase functions
- Verifies deployment

```yaml
.github/workflows/ci-cd.yml
```

## Firebase Realtime Database Structure

```
/
├── live_state/
│   └── {symbol}/
│       ├── current_signal
│       ├── current_position
│       ├── bias
│       └── regime
├── features/
│   └── {symbol}/
│       ├── current
│       └── history
├── explainability/
│   └── {symbol}/
│       ├── current
│       └── history
├── market_state/
│   └── {symbol}/
├── performance/
│   ├── snapshot
│   ├── latest
│   └── weekly_metrics/
│       └── {week_id}/
├── positions/
│   └── {trade_id}/
├── system_controls/
│   ├── trading_enabled
│   ├── emergency_stop
│   ├── max_daily_loss
│   └── ...
├── session_controls/
├── connection_status/
├── regime_state/
└── account_state/
```

## Environment Variables

```bash
# Required
export FIREBASE_SERVICE_ACCOUNT=path/to/serviceAccount.json
export POLYGON_API_KEY=your_polygon_key
export TRADELOCKER_API_KEY=your_tradelocker_key

# Optional
export LOG_LEVEL=INFO
export DEFAULT_SYMBOLS=NQ,ES,BTC
export RISK_PER_TRADE=1.0
```

## Safety First

The system has multiple safety layers:

1. **Guardrails** - System-wide controls (daily loss limits, emergency stop)
2. **Risk Manager** - Per-trade validation (position sizing, stop distances)
3. **Hard Logic** - Code-level rules (no weekend trading, max spread)
4. **GitHub Actions** - Automated testing and deployment
5. **Model Registry** - Version control with performance tracking

## Documentation

- **Experiment Branches**: `docs/EXPERIMENT_BRANCHES.md`
- **Research Journal**: `docs/RESEARCH_JOURNAL.md`
- **Liquidity Heatmaps**: `docs/LIQUIDITY_HEATMAPS.md`

## CLI Commands

```bash
# Run performance analysis
python -m meta_evaluator.performance_monitor --period weekly

# Analyze signals
python -m meta_evaluator.signal_analyzer --days 30

# Check if retraining needed
python -m meta_evaluator.refit_scheduler --check

# Retrain model
python -m meta_evaluator.refit_scheduler --model bias_engine --commit
```

## Next Steps

1. Review and adjust guardrail limits in Firebase
2. Set up GitHub secrets for Actions
3. Run initial model training
4. Start paper trading
5. Monitor Firebase dashboards
