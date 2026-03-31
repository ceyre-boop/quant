"""
Swing Prediction Layer — Monthly Macro Scoring Engine

Sits on top of existing three-layer system. Does not modify any existing code.
Produces SwingBias objects that gate intraday trading.
"""

from .swing_engine import SwingEngine, SwingBias, SwingDirection, get_swing_engine
from .layer_fv import FairValueLayer, FairValueResult
from .layer_positioning import PositioningLayer, PositioningResult
from .layer_regime import RegimeLayer, RegimeResult
from .layer_options import OptionsLayer, OptionsResult
from .layer_timing import TimingLayer, TimingResult
from .scorer import CompositeScorer
from .backtest_base_rates import BaseRateCalculator
from .firebase_writer import FirebaseWriter

__all__ = [
    'SwingEngine',
    'SwingBias',
    'SwingDirection',
    'get_swing_engine',
    'FairValueLayer',
    'PositioningLayer',
    'RegimeLayer',
    'OptionsLayer',
    'TimingLayer',
    'CompositeScorer',
    'BaseRateCalculator',
    'FirebaseWriter',
]
