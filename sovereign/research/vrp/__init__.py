"""VRP research module — volatility risk premium validation (research only).

Isolation (NN#1): this package imports NOTHING from sovereign.forex or sovereign.ict.
data_loader may READ their output files, but no module imports their code. Enforced by
tests/unit/test_vrp_isolation.py.

__init__ stays side-effect-free: it re-exports the PURE calculator/validator/simulator
surface only. data_loader (the impure, yfinance-touching module) is imported explicitly
by callers, never at package import.
"""
from sovereign.research.vrp.strategy_simulator import iron_condor_simulate
from sovereign.research.vrp.validator import (
    CRISES,
    build_verdict,
    stage1_existence,
    stage2_orthogonality,
)
from sovereign.research.vrp.vrp_calculator import (
    btz_vrp_gap,
    harvest_return_causal,
    realized_variance,
    regime_split,
)

__all__ = [
    "realized_variance", "btz_vrp_gap", "harvest_return_causal", "regime_split",
    "stage1_existence", "stage2_orthogonality", "build_verdict", "CRISES",
    "iron_condor_simulate",
]
