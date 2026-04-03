"""
XGBoost Brain for Live Trading

Pre-trained model loaded and ready to trade via AI Bridge.
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ai_trading_bridge import AIBrain, Signal
from training.feature_generator import FeatureGenerator


class XGBoostBrain(AIBrain):
    """
    XGBoost model for live trading.
    
    Usage:
        brain = XGBoostBrain()
        brain.load_model('training/xgb_model.pkl')
        
        bridge = AITradingBridge(brain)
        bridge.run_cycle()
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.55):
        self.model_path = model_path or Path(__file__).parent / 'training' / 'xgb_model.pkl'
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.feature_cols = None
        self.feature_gen = FeatureGenerator()
        self._model_name = "XGBoostBrain-v1"
        
    @property
    def name(self) -> str:
        return self._model_name
    
    def load_model(self, path: str = None):
        """Load trained XGBoost model"""
        path = path or self.model_path
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}. Run training/train_xgb.py first.")
        
        with open(path, 'rb') as f:
            saved = pickle.load(f)
            self.model = saved['model']
            self.feature_cols = saved['features']
            
        print(f"[XGBoostBrain] Loaded model from {path}")
        print(f"[XGBoostBrain] Features: {len(self.feature_cols)}")
        return self
    
    def train_if_needed(self, timeframe: str = '1D', days: int = 365):
        """Train model if not found"""
        if not Path(self.model_path).exists():
            print("[XGBoostBrain] Model not found, training now...")
            from training.train_xgb import train_model
            train_model(timeframe=timeframe, days=days)
        return self.load_model()
    
    def predict(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate signals using XGBoost predictions.
        
        Args:
            data: Dict of symbol -> OHLCV DataFrame
            
        Returns:
            List of Signal objects with confidence scores
        """
        if self.model is None:
            self.train_if_needed()
        
        signals = []
        
        for symbol, df in data.items():
            if len(df) < 30:
                continue
            
            # Generate features
            features_df = self.feature_gen.generate_features(df)
            
            if features_df.empty:
                continue
            
            # Get latest features
            latest = features_df.iloc[-1]
            
            # Extract feature values in correct order
            try:
                feature_values = []
                for col in self.feature_cols:
                    if col in latest.index:
                        val = latest[col]
                        # Handle NaN
                        if pd.isna(val):
                            val = 0.0
                        feature_values.append(val)
                    else:
                        feature_values.append(0.0)  # Missing feature
                
                X = np.array(feature_values).reshape(1, -1)
                
                # Predict probability
                prob = self.model.predict_proba(X)[0]
                
                # prob[1] = probability of UP move
                up_prob = prob[1]
                
                # Only trade if confidence above threshold
                if up_prob > self.confidence_threshold:
                    signals.append(Signal(
                        symbol=symbol,
                        direction="LONG",
                        confidence=float(up_prob),
                        size=10,
                        metadata={
                            'model': self.name,
                            'up_probability': float(up_prob),
                            'down_probability': float(prob[0]),
                            'features_used': len(self.feature_cols)
                        }
                    ))
                elif (1 - up_prob) > self.confidence_threshold:
                    # Strong DOWN signal
                    signals.append(Signal(
                        symbol=symbol,
                        direction="SHORT",
                        confidence=float(1 - up_prob),
                        size=10,
                        metadata={
                            'model': self.name,
                            'up_probability': float(up_prob),
                            'down_probability': float(prob[0]),
                            'features_used': len(self.feature_cols)
                        }
                    ))
                    
            except Exception as e:
                print(f"[XGBoostBrain] Error predicting {symbol}: {e}")
                continue
        
        return signals


def test_xgboost_brain():
    """Test XGBoost brain with live data"""
    from ai_trading_bridge import AITradingBridge
    
    print("="*60)
    print("TESTING XGBOOST BRAIN")
    print("="*60)
    
    # Create brain
    brain = XGBoostBrain(confidence_threshold=0.55)
    
    # Load or train model
    try:
        brain.load_model()
    except FileNotFoundError:
        print("Model not found, training...")
        brain.train_if_needed()
    
    # Create bridge
    bridge = AITradingBridge(
        brain=brain,
        symbols=["SPY", "QQQ", "IWM", "NVDA", "AAPL"],
        timeframe="1D",
        lookback_days=60,
        min_confidence=0.55,
        paper=True
    )
    
    # Run cycle
    result = bridge.run_cycle()
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Brain: {result['brain']}")
    print(f"Signals: {result['signals']}")
    print(f"Executed: {result['executed']}")
    
    return result


if __name__ == '__main__':
    test_xgboost_brain()
