"""Regime Classifier - 5-axis regime classification."""

import logging
from typing import Dict, Any
from dataclasses import dataclass

from contracts.types import (
    RegimeState,
    VolRegime,
    TrendRegime,
    RiskAppetite,
    MomentumRegime,
    EventRisk,
)
from layer1.feature_builder import FeatureVector

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """5-axis regime classification."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def classify(
        self,
        features: FeatureVector,
        vix_value: float,
        event_risk: str = "CLEAR",
        market_breadth: float = 1.0,
    ) -> RegimeState:
        """Classify market regime across 5 axes.

        Args:
            features: Feature vector
            vix_value: Current VIX level
            event_risk: Event risk level
            market_breadth: Market breadth ratio

        Returns:
            RegimeState with 5-axis classification
        """
        volatility = self._classify_volatility(features, vix_value)
        trend = self._classify_trend(features)
        risk_appetite = self._classify_risk_appetite(features, market_breadth)
        momentum = self._classify_momentum(features)
        event = self._classify_event_risk(event_risk)

        # Composite score (0.0 to 1.0)
        composite = self._calculate_composite(
            volatility, trend, risk_appetite, momentum, event
        )

        return RegimeState(
            volatility=volatility,
            trend=trend,
            risk_appetite=risk_appetite,
            momentum=momentum,
            event_risk=event,
            composite_score=composite,
        )

    def _classify_volatility(
        self, features: FeatureVector, vix_value: float
    ) -> VolRegime:
        """Classify volatility regime."""
        # Use VIX and historical volatility
        if vix_value < 15 and features.volatility_regime == 1:
            return VolRegime.LOW
        elif vix_value < 25 and features.volatility_regime <= 2:
            return VolRegime.NORMAL
        elif vix_value < 35 or features.volatility_regime == 3:
            return VolRegime.ELEVATED
        else:
            return VolRegime.EXTREME

    def _classify_trend(self, features: FeatureVector) -> TrendRegime:
        """Classify trend regime."""
        # Use ADX, moving average alignment, and price vs MAs
        adx = features.adx_14
        ma_alignment = (
            (1 if features.price_vs_sma_20 > 0 else 0)
            + (1 if features.price_vs_sma_50 > 0 else 0)
            + (1 if features.price_vs_sma_200 > 0 else 0)
        ) / 3

        if adx > 25 and ma_alignment > 0.6:
            return TrendRegime.STRONG_TREND
        elif adx > 25 and ma_alignment < 0.4:
            return TrendRegime.STRONG_TREND  # Strong downtrend
        elif adx > 20:
            return TrendRegime.WAK_TREND
        elif adx < 15:
            return TrendRegime.CHOPPY
        else:
            return TrendRegime.RANGING

    def _classify_risk_appetite(
        self, features: FeatureVector, market_breadth: float
    ) -> RiskAppetite:
        """Classify risk appetite."""
        # Use VIX, breadth, and momentum
        vix_regime = features.vix_regime
        rsi = features.rsi_14

        if vix_regime <= 2 and market_breadth > 1.2 and rsi > 50:
            return RiskAppetite.RISK_ON
        elif vix_regime >= 3 or market_breadth < 0.8 or rsi < 40:
            return RiskAppetite.RISK_OFF
        else:
            return RiskAppetite.NEUTRAL

    def _classify_momentum(self, features: FeatureVector) -> MomentumRegime:
        """Classify momentum regime."""
        # Use RSI slope, MACD, and price momentum
        rsi_slope = features.rsi_slope_5
        macd_hist = features.macd_histogram

        if rsi_slope > 2 and macd_hist > 0:
            return MomentumRegime.ACCELERATING
        elif rsi_slope < -2 and macd_hist < 0:
            return MomentumRegime.REVERSING
        elif abs(rsi_slope) < 1:
            return MomentumRegime.STEADY
        else:
            return MomentumRegime.DECELERATING

    def _classify_event_risk(self, event_risk: str) -> EventRisk:
        """Classify event risk."""
        mapping = {
            "CLEAR": EventRisk.CLEAR,
            "ELEVATED": EventRisk.ELEVATED,
            "HIGH": EventRisk.HIGH,
            "EXTREME": EventRisk.EXTREME,
        }
        return mapping.get(event_risk, EventRisk.CLEAR)

    def _calculate_composite(
        self,
        volatility: VolRegime,
        trend: TrendRegime,
        risk_appetite: RiskAppetite,
        momentum: MomentumRegime,
        event: EventRisk,
    ) -> float:
        """Calculate composite regime score."""
        # Score components
        vol_score = {"LOW": 1.0, "NORMAL": 0.75, "ELEVATED": 0.5, "EXTREME": 0.25}
        trend_score = {
            "STRONG_TREND": 1.0,
            "WEAK_TREND": 0.7,
            "RANGING": 0.4,
            "CHOPPY": 0.2,
        }
        risk_score = {"RISK_ON": 1.0, "NEUTRAL": 0.6, "RISK_OFF": 0.2}
        mom_score = {
            "ACCELERATING": 1.0,
            "STEADY": 0.7,
            "DECELERATING": 0.4,
            "REVERSING": 0.2,
        }
        event_score = {"CLEAR": 1.0, "ELEVATED": 0.7, "HIGH": 0.4, "EXTREME": 0.1}

        scores = [
            vol_score.get(volatility.value, 0.5),
            trend_score.get(trend.value, 0.5),
            risk_score.get(risk_appetite.value, 0.5),
            mom_score.get(momentum.value, 0.5),
            event_score.get(event.value, 0.5),
        ]

        return sum(scores) / len(scores)
