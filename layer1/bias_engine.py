"""Bias Engine - Core AI function for Layer 1.

XGBoost model with SHAP explainability for directional bias.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import pickle
import pandas as pd

from contracts.types import (
    Direction, Magnitude, BiasOutput, FeatureGroup, RegimeState
)
from layer1.feature_builder import FeatureVector
from layer1.regime_classifier import RegimeClassifier

logger = logging.getLogger(__name__)


class BiasEngine:
    """AI Bias Engine for directional prediction."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_version: str = 'v1.0'
    ):
        self.model_version = model_version
        self.regime_classifier = RegimeClassifier()
        
        # Load or create model
        self.model = None
        self.feature_names = None
        self.model_path = model_path or 'training/xgb_model.pkl'
        
        self._load_model()
        
        # Feature group mapping for rationale
        self.feature_to_group = self._build_feature_group_mapping()
    
    def _load_model(self):
        """Load XGBoost model and metadata if they exist."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Handle dictionary format (from training/train_xgb.py)
                if isinstance(data, dict) and 'model' in data:
                    self.model = data['model']
                    self.feature_names = data.get('features')
                    logger.info(f"Loaded XGBoost model dictionary from {self.model_path}")
                else:
                    # Handle raw model format
                    self.model = data
                    
                    # Try to load metadata if separate
                    meta_path = self.model_path.replace('.pkl', '_metadata.json')
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            self.feature_names = meta.get('feature_names')
                    
                    logger.info(f"Loaded raw XGBoost model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load XGBoost model: {e}")
                self.model = None
    
    def get_daily_bias(
        self,
        symbol: str,
        features: FeatureVector,
        regime: RegimeState
    ) -> BiasOutput:
        """Generate daily bias prediction.
        
        Args:
            symbol: Trading symbol
            features: Feature vector
            regime: Current regime state
        
        Returns:
            BiasOutput with direction, confidence, and rationale
        """
        # Get model prediction
        direction, confidence = self._predict_direction(features)
        
        # Determine magnitude
        magnitude = self._determine_magnitude(confidence, features)
        
        # Check for regime override
        regime_override = self._check_regime_override(regime, features)
        
        # Generate rationale (feature group names)
        rationale = self._generate_rationale(features)
        
        # Build feature snapshot
        feature_snapshot = {
            'raw_features': features.to_dict(),
            'feature_group_tags': self._tag_feature_groups(features),
            'regime_at_inference': regime.to_dict(),
            'inference_timestamp': datetime.utcnow().isoformat()
        }
        
        return BiasOutput(
            direction=direction,
            magnitude=magnitude,
            confidence=confidence,
            regime_override=regime_override,
            rationale=rationale,
            model_version=self.model_version,
            feature_snapshot=feature_snapshot
        )
    
    def _predict_direction(
        self,
        features: FeatureVector
    ) -> tuple:
        """Predict direction using model or heuristics.
        
        Returns:
            (Direction, confidence)
        """
        feature_dict = features.to_dict()
        
        # Use XGBoost model if available
        if self.model is not None and self.feature_names is not None:
            try:
                # Prepare input dataframe
                X = pd.DataFrame([feature_dict])[self.feature_names].fillna(0)
                
                # Predict probability of win (target=1)
                prob = self.model.predict_proba(X)[0, 1]
                
                # Map prob to direction and confidence
                if prob > 0.55:
                    direction = Direction.LONG
                    confidence = prob
                elif prob < 0.45:
                    direction = Direction.SHORT
                    confidence = 1 - prob
                else:
                    direction = Direction.NEUTRAL
                    confidence = 0.5
                    
                return direction, confidence
            except Exception as e:
                logger.error(f"Inference error: {e}")
        
        # Fallback to heuristic model
        trend_score = (
            feature_dict.get('price_vs_sma_20', 0) * 0.3 +
            feature_dict.get('price_vs_sma_50', 0) * 0.2 +
            feature_dict.get('macd_histogram', 0) / 100 * 0.3
        )
        
        momentum_score = (
            (feature_dict.get('rsi_14', 50) - 50) / 50 * 0.3 +
            feature_dict.get('rsi_slope_5', 0) * 0.1
        )
        
        # Combine scores
        total_score = trend_score + momentum_score
        
        # Convert to direction and confidence
        if total_score > 0.1:
            direction = Direction.LONG
            confidence = min(0.5 + abs(total_score), 0.95)
        elif total_score < -0.1:
            direction = Direction.SHORT
            confidence = min(0.5 + abs(total_score), 0.95)
        else:
            direction = Direction.NEUTRAL
            confidence = 0.5
        
        return direction, confidence
    
    def _determine_magnitude(
        self,
        confidence: float,
        features: FeatureVector
    ) -> Magnitude:
        """Determine bias magnitude."""
        adx = features.adx_14
        
        if confidence > 0.8 and adx > 30:
            return Magnitude.LARGE
        elif confidence > 0.65:
            return Magnitude.NORMAL
        else:
            return Magnitude.SMALL
    
    def _check_regime_override(self, regime: RegimeState, features: FeatureVector) -> bool:
        """Check if regime requires model override.
        
        Override conditions:
        - ATR% > 4.0% (Structural Noise Gate)
        - EXTREME volatility + trend uncertainty
        - HIGH/EXTREME event risk
        """
        if features.atr_percent_14 > 0.04:  # 4% daily ATR is often structural chaos
            return True
        if regime.event_risk.value in ['HIGH', 'EXTREME']:
            return True
        if regime.volatility.value == 'EXTREME':
            return True
        return False
    
    def _generate_rationale(self, features: FeatureVector) -> List[str]:
        """Generate rationale from feature group contributions."""
        rationale = []
        
        # Check which feature groups are most significant
        if abs(features.atr_percent_14) > 0.02:
            rationale.append(FeatureGroup.VOLATILITY_SPIKE.value)
        
        if abs(features.adx_14) > 25:
            rationale.append(FeatureGroup.TREND_STRENGTH.value)
        
        if abs(features.rsi_slope_5) > 3:
            rationale.append(FeatureGroup.MOMENTUM_SHIFT.value)
        
        if abs(features.distance_to_support) < features.atr_14 or \
           abs(features.distance_to_resistance) < features.atr_14:
            rationale.append(FeatureGroup.SUPPORT_RESISTANCE.value)
        
        if features.market_breadth_ratio > 1.5 or features.market_breadth_ratio < 0.7:
            rationale.append(FeatureGroup.MARKET_BREADTH.value)
        
        if len(rationale) == 0:
            rationale.append(FeatureGroup.REGIME_ALIGNMENT.value)
        
        return rationale
    
    def _tag_feature_groups(self, features: FeatureVector) -> Dict[str, Any]:
        """Tag features with their group memberships."""
        return {
            'VOLATILITY_SPIKE': features.atr_percent_14 > 0.02,
            'TREND_STRENGTH': features.adx_14 > 25,
            'MOMENTUM_SHIFT': abs(features.rsi_slope_5) > 3,
            'SUPPORT_RESISTANCE': features.distance_to_support < features.atr_14
        }
    
    def _build_feature_group_mapping(self) -> Dict[str, FeatureGroup]:
        """Map individual features to their groups."""
        return {
            # Volatility
            'atr_14': FeatureGroup.VOLATILITY_SPIKE,
            'atr_percent_14': FeatureGroup.VOLATILITY_SPIKE,
            'bollinger_width': FeatureGroup.VOLATILITY_SPIKE,
            
            # Trend
            'adx_14': FeatureGroup.TREND_STRENGTH,
            'price_vs_sma_20': FeatureGroup.TREND_STRENGTH,
            'price_vs_sma_50': FeatureGroup.TREND_STRENGTH,
            
            # Momentum
            'rsi_14': FeatureGroup.MOMENTUM_SHIFT,
            'rsi_slope_5': FeatureGroup.MOMENTUM_SHIFT,
            'macd_histogram': FeatureGroup.MOMENTUM_SHIFT,
            
            # Structure
            'distance_to_support': FeatureGroup.SUPPORT_RESISTANCE,
            'distance_to_resistance': FeatureGroup.SUPPORT_RESISTANCE,
            
            # Breadth
            'market_breadth_ratio': FeatureGroup.MARKET_BREADTH,
            
            # Sentiment
            'vix_level': FeatureGroup.SENTIMENT_EXTREME,
            'vix_regime': FeatureGroup.SENTIMENT_EXTREME
        }


class SHAPExplainer:
    """SHAP-based feature importance explainer."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def explain_prediction(
        self,
        features: FeatureVector,
        prediction: Direction
    ) -> Dict[str, float]:
        """Generate SHAP-like feature importance.
        
        Returns:
            Dict mapping feature group to importance score
        """
        # Simplified SHAP approximation
        importances = {}
        
        feature_dict = features.to_dict()
        
        # Calculate contribution for each group
        groups = {
            'VOLATILITY_SPIKE': ['atr_14', 'atr_percent_14', 'bollinger_width'],
            'TREND_STRENGTH': ['adx_14', 'price_vs_sma_20', 'price_vs_sma_50'],
            'MOMENTUM_SHIFT': ['rsi_14', 'rsi_slope_5', 'macd_histogram'],
            'SUPPORT_RESISTANCE': ['distance_to_support', 'distance_to_resistance'],
            'MARKET_BREADTH': ['market_breadth_ratio'],
            'SENTIMENT_EXTREME': ['vix_level', 'vix_regime']
        }
        
        for group, feature_names in groups.items():
            group_importance = sum(
                abs(feature_dict.get(f, 0))
                for f in feature_names
            ) / len(feature_names)
            importances[group] = group_importance
        
        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}
        
        return importances
