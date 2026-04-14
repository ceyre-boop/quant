# Clawd Trading — Sovereign Core

**Stripped execution machine. No gates that don't make money.**

---

## What's Real

The system is 6 stages:

1. **Data** (Alpaca)
2. **Regime Router** (Hurst-based: MOMENTUM / REVERSION / FLAT)
3. **ATR Gate** (asset-specific thresholds)
4. **Specialist Signal** (XGBoost momentum or reversion)
5. **Grade-Based Sizing** (confidence → risk%)
6. **Hard Constraints** → Execute → Log

Everything else is optional diagnostic or logger.

---

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Train (One Time)
```bash
python -m sovereign.data.train_core
```

Or manually:
```bash
python
>>> from sovereign.orchestrator import SovereignOrchestrator
>>> orch = SovereignOrchestrator()
>>> orch.train(records)  # your historical feature records
>>> orch.save_models()
```

### Run Daily Session
```bash
python execute_daily.py --mode paper
```

### Run Specific Phase
```bash
python execute_daily.py --phase premarket
python execute_daily.py --phase killzone
python execute_daily.py --phase postsession
```

### Automate (Windows)
```bash
# Run as Administrator
python setup_task_scheduler.py
```

This creates:
- **8:45 AM** — Pre-market checklist
- **9:50 AM** — Kill zone execution
- **4:35 PM** — Post-session report

---

## Architecture (Teardown Compliant)

### ✅ Kept

| Component | Role |
|-----------|------|
| **Regime Router** | Hurst + XGBoost. Routes to momentum/reversion/flat. |
| **Specialists** | XGBoost models for each regime. |
| **Risk Engine** | Grade sizing + ATR gate. |
| **Trade Ledger** | Logs every executed trade. |
| **Veto Ledger** | Logs every rejection. |
| **Hard Constraints** | Daily loss limit, max positions, VIX. |

### ⚠️ Demoted (Logger / Advisory)

| Component | Role |
|-----------|------|
| **Petroulas Gate** | Logs macro warnings. Does NOT block. |
| **Layer 3 (Game Theory)** | Observes trades. Logs predictions. No veto. |
| **Factor Zoo** | Optional diagnostic. Run anytime. Not a gate. |

### ❌ Deleted

| Component | Reason |
|-----------|--------|
| Walk-forward backtest as gate | 20 trades in a year is not meaningful. Paper trading IS the test. |
| Veto Rate Diagnostic as phase | Ledger already shows this in real time. |
| Swing Prediction Layer | Monthly macro filter on zero live trades = zero trades forever. |
| Layer 3 in execution gate | Unvalidated component blocking validated ones. |

---

## The Only Gates That Block

1. **Router/FLAT** — Hurst dead zone (0.45-0.52)
2. **Specialist/NEUTRAL** — No directional signal
3. **Risk/ATR** — Volatility too high for asset
4. **Risk/EV** — Expected value negative
5. **Hard Constraints** — Daily loss limit, max positions

---

## Live Transition

After **200 paper trades** with positive expectancy, the system is live-ready.

---

## License

Proprietary — All rights reserved.
