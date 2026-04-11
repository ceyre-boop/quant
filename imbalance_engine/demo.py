"""Demo harness — run the full Imbalance Engine pipeline with today's macro data.

Usage:
    python imbalance_engine/demo.py

Feeds synthetic but realistic macro inputs through all six frameworks,
prints the composite report, then exercises the Petroulas Gate.

In production: plug real FRED/Quandl API data into MacroImbalanceFramework.analyze()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

from imbalance_engine.frameworks import MacroImbalanceFramework
from imbalance_engine.petroulas_gate import PetroulsasGate
from imbalance_engine.falsification import FalsificationDiscipline

def run_demo():
    print("=" * 70)
    print("IMBALANCE ENGINE — FULL PIPELINE DEMO")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1. Build synthetic macro data (replace with FRED/Alpaca reality)
    # -------------------------------------------------------------------------
    
    # Yield curve history (90 days of 2yr-10yr spread in bps)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=90, freq='B')
    
    # Simulate: was 50bps positive, inverted 60 days ago, now at -25bps
    spread_2_10 = pd.Series(
        np.linspace(50, -25, 90) + np.random.normal(0, 5, 90),
        index=dates
    )
    spread_3m_10yr = pd.Series(
        np.linspace(80, -55, 90) + np.random.normal(0, 5, 90),
        index=dates
    )
    
    # Current macro snapshot
    macro = dict(
        cape=35.8,                       # Current CAPE (historically elevated)
        bond_yield_10yr_pct=4.45,        # 10yr yield %
        vix=22.0,                        # VIX level
        spy_return_1m=-0.035,            # SPY -3.5% last month
        market_breadth=0.42,             # 42% of stocks above 200d MA (weak)
        spread_3m_10yr_bps=float(spread_3m_10yr.iloc[-1]),
        spread_velocity_bps_month=(spread_3m_10yr.iloc[-1] - spread_3m_10yr.iloc[-22]) / 1.0,
        spread_2_10_history=spread_2_10,
        spread_3m_10yr_history=spread_3m_10yr,
        symbol='SPY'
    )
    
    # -------------------------------------------------------------------------
    # 2. Run all six frameworks
    # -------------------------------------------------------------------------
    
    framework = MacroImbalanceFramework()
    result = framework.analyze(**macro)
    
    print(result.fault_summary)
    print(f"\nConsensus Fault Detected: {result.consensus_fault_detected}")
    print(f"HMM Stress Score (→ XGBoost feature): {result.hmm_regime_stress:.4f}")
    
    # -------------------------------------------------------------------------
    # 3. Petroulas Gate (Kimi not connected, framework-only path)
    # -------------------------------------------------------------------------
    
    gate = PetroulsasGate(kimi_client=None)
    
    # Simulate XGBoost output
    xgb_confidence = 0.71
    direction = 'SHORT'
    
    decision = gate.evaluate(
        symbol='SPY',
        regime_stress=result,
        xgb_confidence=xgb_confidence,
        direction=direction,
        entry_price=520.0
    )
    
    print("\n" + "=" * 70)
    print("PETROULAS GATE DECISION")
    print("=" * 70)
    print(f"Approved: {decision.approved}")
    print(f"Position Size: {decision.position_size_pct:.1f}% (normal={decision.normal_size_pct:.1f}%)")
    print(f"Reason: {decision.reason}")
    
    # -------------------------------------------------------------------------
    # 4. Falsification discipline
    # -------------------------------------------------------------------------
    
    discipline = FalsificationDiscipline()
    
    if decision.approved:
        entry = discipline.open_thesis(decision=decision, kimi_score=None)
        discipline.set_entry_details(entry.thesis_id, direction=direction, entry_price=520.0)
        print(f"\nThesis registered: {entry.thesis_id}")
        print(f"Kill test: {entry.falsification_test}")
        print(f"Deadline: {entry.deadline_date}")
    
    discipline.print_report()
    
    # -------------------------------------------------------------------------
    # 5. Show how HMM stress feeds into XGBoost features
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("XGBoost FEATURE INJECTION")
    print("=" * 70)
    print(f"features.hmm_regime_stress = {result.hmm_regime_stress:.4f}")
    print("→ Insert this into FeatureVector before BiasEngine.get_daily_bias()")
    print("→ Model trained with this feature has macro regime awareness")


if __name__ == '__main__':
    run_demo()
