"""Imbalance Engine — Mathematical fault detection weeks-months before moves.

Six frameworks:
  1. ERP (Equity Risk Premium) — precision overvaluation meter
  2. Shiller CAPE z-score — 130-year historical normalization
  3. PCA Macro Regime — stress detection before price moves
  4. Hidden Markov Model — regime transition probability (XGBoost feature)
  5. NY Fed Probit — yield-curve recession model (predicted every recession since 1960)
  6. Yield Curve Velocity — acceleration/deceleration, not just level

Petroulas Gate:
  - Kimi scores structural faults 1-10 on magnitude AND conviction
  - Dual-confirmation: XGBoost + Kimi must agree
  - Position sizing: 3-5% for Petroulas trades vs 1-2% normal
  - Falsification discipline: every thesis has a 30-day kill test
"""

from imbalance_engine.frameworks import MacroImbalanceFramework
from imbalance_engine.petroulas_gate import PetroulsasGate
from imbalance_engine.falsification import FalsificationDiscipline

__all__ = ['MacroImbalanceFramework', 'PetroulsasGate', 'FalsificationDiscipline']
