"""Layer 1: AI Bias Engine - XGBoost Implementation

Production-grade XGBoost model with SHAP explainability.
Replaces the heuristic model with a properly trained classifier.
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Optional XGBoost import with fallback
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Optional SHAP import with fallback
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from contracts.types import Direction, Magnitude, BiasOutput, FeatureGroup, RegimeState

logger = logging.getLogger(__name__)


class XGBoostBiasModel:
    """XGBoost-based bias prediction model."""

    # Feature names in exact order expected by model
    FEATURE_NAMES = [
        "returns_1h",
        "returns_4h",
        "returns_daily",
        "returns_5d",
        "price_vs_sma_20",
        "price_vs_sma_50",
        "price_vs_sma_200",
        "price_vs_ema_12",
        "price_vs_ema_26",
        "price_position_daily_range",
        "bollinger_position",
        "bollinger_width",
        "keltner_position",
        "atr_14",
        "atr_percent_14",
        "historical_volatility_20d",
        "realized_volatility_5d",
        "volatility_regime",
        "adx_14",
        "dmi_plus",
        "dmi_minus",
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "rsi_14",
        "rsi_slope_5",
        "stochastic_k",
        "stochastic_d",
        "cci_20",
        "volume_sma_ratio",
        "obv_slope",
        "vwap_deviation",
        "volume_profile_poc_dist",
        "volume_trend_5d",
        "swing_high_20",
        "swing_low_20",
        "distance_to_resistance",
        "distance_to_support",
        "higher_highs_5d",
        "higher_lows_5d",
        "vix_level",
        "vix_regime",
        "spy_correlation_20d",
        "market_breadth_ratio",
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_version = "v1.0"
        self.feature_importance: Dict[str, float] = {}

        # Model paths
        self.model_dir = Path("layer1/bias_model")
        self.model_path = model_path or self.model_dir / "model_v1.pkl"
        self.registry_path = self.model_dir / "model_registry.json"

        # Load or initialize model
        self._load_model()

    def _load_model(self) -> None:
        """Load model from disk or create new."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, using fallback heuristics")
            return

        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded XGBoost model from {self.model_path}")

                # Load version from registry
                if self.registry_path.exists():
                    with open(self.registry_path, "r") as f:
                        registry = json.load(f)
                        self.model_version = registry.get("current_version", "v1.0")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
        else:
            logger.warning(f"Model file not found at {self.model_path}")
            self.model = None

    def predict(self, features: Dict[str, float]) -> Tuple[Direction, float]:
        """Predict direction and confidence.

        Returns:
            Tuple of (Direction, confidence)
        """
        if self.model is None or not XGBOOST_AVAILABLE:
            return self._fallback_prediction(features)

        try:
            # Extract features in correct order
            feature_vector = self._extract_features(features)
            X = np.array([feature_vector])

            # Get prediction probabilities
            proba = self.model.predict_proba(X)[0]

            # Class mapping: 0=SHORT, 1=NEUTRAL, 2=LONG
            short_prob, neutral_prob, long_prob = proba

            # Determine direction
            if long_prob > short_prob and long_prob > neutral_prob:
                direction = Direction.LONG
                confidence = long_prob
            elif short_prob > long_prob and short_prob > neutral_prob:
                direction = Direction.SHORT
                confidence = short_prob
            else:
                direction = Direction.NEUTRAL
                confidence = neutral_prob

            return direction, float(confidence)

        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return self._fallback_prediction(features)

    def _extract_features(self, features: Dict[str, float]) -> List[float]:
        """Extract features in model's expected order."""
        return [features.get(name, 0.0) for name in self.FEATURE_NAMES]

    def _fallback_prediction(
        self, features: Dict[str, float]
    ) -> Tuple[Direction, float]:
        """Fallback heuristic prediction when XGBoost unavailable."""
        # Simple trend + momentum heuristic
        trend_score = (
            features.get("price_vs_sma_20", 0) * 0.3
            + features.get("price_vs_sma_50", 0) * 0.2
            + features.get("macd_histogram", 0) / 100 * 0.3
        )

        momentum_score = (features.get("rsi_14", 50) - 50) / 50 * 0.3 + features.get(
            "rsi_slope_5", 0
        ) * 0.1

        total_score = trend_score + momentum_score

        if total_score > 0.1:
            return Direction.LONG, min(0.5 + abs(total_score), 0.95)
        elif total_score < -0.1:
            return Direction.SHORT, min(0.5 + abs(total_score), 0.95)
        else:
            return Direction.NEUTRAL, 0.5

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model."""
        if self.model is None:
            return {}

        try:
            importance = self.model.feature_importances_
            return {
                name: float(imp) for name, imp in zip(self.FEATURE_NAMES, importance)
            }
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}


class SHAPExplainer:
    """SHAP-based feature importance explainer."""

    def __init__(self, model: Optional[Any] = None):
        self.model = model
        self.explainer = None

        if SHAP_AVAILABLE and model is not None:
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception as e:
                logger.warning(f"Failed to create SHAP explainer: {e}")

    def explain(self, features: Dict[str, float]) -> Dict[str, float]:
        """Generate SHAP explanation for features."""
        if self.explainer is None or not SHAP_AVAILABLE:
            return self._fallback_explanation(features)

        try:
            feature_vector = [
                features.get(name, 0.0) for name in XGBoostBiasModel.FEATURE_NAMES
            ]
            X = np.array([feature_vector])

            shap_values = self.explainer.shap_values(X)

            # Get SHAP values for predicted class
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Binary classification case

            # Map to feature names and group by category
            feature_shap = {
                name: float(val)
                for name, val in zip(XGBoostBiasModel.FEATURE_NAMES, shap_values[0])
            }

            # Group into feature groups
            return self._group_by_category(feature_shap)

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(features)

    def _fallback_explanation(self, features: Dict[str, float]) -> Dict[str, float]:
        """Fallback explanation using feature values."""
        groups = {
            "VOLATILITY_SPIKE": ["atr_14", "atr_percent_14", "bollinger_width"],
            "TREND_STRENGTH": ["adx_14", "price_vs_sma_20", "price_vs_sma_50"],
            "MOMENTUM_SHIFT": ["rsi_14", "rsi_slope_5", "macd_histogram"],
            "SUPPORT_RESISTANCE": ["distance_to_support", "distance_to_resistance"],
            "MARKET_BREADTH": ["market_breadth_ratio"],
            "SENTIMENT_EXTREME": ["vix_level", "vix_regime"],
        }

        importances = {}
        for group, feature_names in groups.items():
            group_importance = sum(
                abs(features.get(f, 0)) for f in feature_names
            ) / len(feature_names)
            importances[group] = group_importance

        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    def _group_by_category(self, feature_shap: Dict[str, float]) -> Dict[str, float]:
        """Group SHAP values by feature category."""
        groups = {
            "VOLATILITY_SPIKE": [
                "atr_14",
                "atr_percent_14",
                "bollinger_width",
                "volatility_regime",
            ],
            "TREND_STRENGTH": [
                "adx_14",
                "price_vs_sma_20",
                "price_vs_sma_50",
                "price_vs_sma_200",
            ],
            "MOMENTUM_SHIFT": [
                "rsi_14",
                "rsi_slope_5",
                "macd_histogram",
                "macd_line",
                "macd_signal",
            ],
            "SUPPORT_RESISTANCE": [
                "distance_to_support",
                "distance_to_resistance",
                "swing_high_20",
                "swing_low_20",
            ],
            "MARKET_BREADTH": ["market_breadth_ratio", "spy_correlation_20d"],
            "SENTIMENT_EXTREME": ["vix_level", "vix_regime"],
        }

        group_importance = {}
        for group_name, features in groups.items():
            total_shap = sum(abs(feature_shap.get(f, 0)) for f in features)
            group_importance[group_name] = total_shap

        # Normalize to probabilities
        total = sum(group_importance.values())
        if total > 0:
            group_importance = {k: v / total for k, v in group_importance.items()}

        return group_importance


class BiasEngineV2:
    """Enhanced Bias Engine with XGBoost and SHAP."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = XGBoostBiasModel(model_path)
        self.explainer = SHAPExplainer(self.model.model)
        self.model_version = self.model.model_version
        self.logger = logging.getLogger(__name__)

    def get_daily_bias(
        self, symbol: str, features: Dict[str, float], regime: RegimeState
    ) -> BiasOutput:
        """Generate daily bias prediction using XGBoost.

        Args:
            symbol: Trading symbol
            features: Feature vector (43 features)
            regime: Current regime state

        Returns:
            BiasOutput with direction, confidence, and SHAP rationale
        """
        # Get prediction from model
        direction, confidence = self.model.predict(features)

        # Determine magnitude
        magnitude = self._determine_magnitude(confidence, features)

        # Check for regime override
        regime_override = self._check_regime_override(regime)

        # Generate SHAP explanation
        shap_importance = self.explainer.explain(features)

        # Convert to canonical feature group names
        rationale = self._generate_rationale(shap_importance, features)

        # Build feature snapshot
        feature_snapshot = {
            "raw_features": features,
            "feature_group_tags": self._tag_feature_groups(features),
            "regime_at_inference": regime.to_dict(),
            "inference_timestamp": datetime.utcnow().isoformat(),
        }

        return BiasOutput(
            direction=direction,
            magnitude=magnitude,
            confidence=confidence,
            regime_override=regime_override,
            rationale=rationale,
            model_version=self.model_version,
            feature_snapshot=feature_snapshot,
        )

    def _determine_magnitude(
        self, confidence: float, features: Dict[str, float]
    ) -> Magnitude:
        """Determine bias magnitude."""
        adx = features.get("adx_14", 0)

        if confidence > 0.8 and adx > 30:
            return Magnitude.LARGE
        elif confidence > 0.65:
            return Magnitude.NORMAL
        else:
            return Magnitude.SMALL

    def _check_regime_override(self, regime: RegimeState) -> bool:
        """Check if regime requires model override."""
        if regime.event_risk.value in ["HIGH", "EXTREME"]:
            return True
        if regime.volatility.value == "EXTREME":
            return True
        return False

    def _generate_rationale(
        self, shap_importance: Dict[str, float], features: Dict[str, float]
    ) -> List[str]:
        """Generate rationale from SHAP importance."""
        rationale = []

        # Sort by importance
        sorted_groups = sorted(
            shap_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Include top 3 feature groups
        for group_name, importance in sorted_groups[:3]:
            if importance > 0.1:  # Threshold for inclusion
                rationale.append(group_name)

        # Ensure at least one rationale
        if not rationale:
            rationale.append("REGIME_ALIGNMENT")

        return rationale

    def _tag_feature_groups(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Tag features with their group memberships."""
        return {
            "VOLATILITY_SPIKE": features.get("atr_percent_14", 0) > 0.02,
            "TREND_STRENGTH": features.get("adx_14", 0) > 25,
            "MOMENTUM_SHIFT": abs(features.get("rsi_slope_5", 0)) > 3,
            "SUPPORT_RESISTANCE": (
                features.get("distance_to_support", 999) < features.get("atr_14", 1)
                or features.get("distance_to_resistance", 999)
                < features.get("atr_14", 1)
            ),
            "MARKET_BREADTH": features.get("market_breadth_ratio", 1) > 1.5
            or features.get("market_breadth_ratio", 1) < 0.7,
            "SENTIMENT_EXTREME": features.get("vix_level", 20) > 25,
        }


def create_bias_engine(model_path: Optional[str] = None) -> BiasEngineV2:
    """Factory function to create bias engine."""
    return BiasEngineV2(model_path)
