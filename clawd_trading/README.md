# Clawd Trading - Three-Layer System

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Start the system:
   ```bash
   python -m orchestrator.daily_lifecycle
   ```

## Architecture

The system consists of three intelligence layers:

### Layer 1: AI Bias Engine
- 43 engineered features
- XGBoost classification model
- 5-axis regime classification
- SHAP explainability

### Layer 2: Quant Risk Model
- Kelly criterion position sizing
- ATR and structural stop calculation
- Expected value computation
- Risk-adjusted return metrics

### Layer 3: Game-Theoretic Engine
- Liquidity pool mapping
- Trapped position detection
- Nash equilibrium zones
- Kyle lambda order flow analysis

## Data Sources
- Polygon.io (primary market data)
- TradeLocker (execution)
- Alpha Vantage (news sentiment)
- Trading Economics (economic calendar)

## Firebase Configuration
- **Project:** taboost-platform
- **Auth Domain:** taboost-platform.firebaseapp.com
- **RTDB:** taboost-platform-default-rtdb.firebaseio.com

## Repository
https://github.com/ceyre-boop/quant

## License
Proprietary - All rights reserved.
