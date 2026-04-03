"""
ClawdBrain - Full 3-Layer AI Trading System

Architecture:
  Layer 1: XGBoost Bias Prediction (direction + confidence)
  Layer 2: Kelly Position Sizing + Game Theory Validation
  Layer 3: Entry Gates + Risk Checks
  Output: Signal with calculated position size
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent))

from ai_trading_bridge import AIBrain, Signal, AITradingBridge
from training.feature_generator import FeatureGenerator
from data.alpaca_client import AlpacaDataClient


@dataclass
class Layer1Output:
    """Output from XGBoost bias layer"""
    symbol: str
    direction: str  # LONG or SHORT
    bias: float  # -1.0 to 1.0 (prediction strength)
    confidence: float  # 0.0 to 1.0 (model confidence)
    win_prob: float  # Probability of winning
    features: Dict[str, float]  # Feature values used


@dataclass  
class Layer2Output:
    """Output from Kelly + Game Theory layer"""
    symbol: str
    direction: str
    kelly_fraction: float  # Optimal bet size (0 to 1)
    game_theory_score: float  # Adversarial analysis score
    edge: float  # Expected edge vs market
    max_position_size: float  # Dollar amount
    recommended_shares: int  # Based on Kelly


@dataclass
class Layer3Output:
    """Output from Entry Gates"""
    passed: bool
    reject_reason: Optional[str]
    final_shares: int
    final_confidence: float
    metadata: Dict


class KellyCalculator:
    """
    Kelly Criterion position sizing with fractional Kelly for safety.
    """
    
    def __init__(self, fraction: float = 0.25, max_position_pct: float = 0.10):
        """
        Args:
            fraction: Fractional Kelly (0.25 = Quarter Kelly for safety)
            max_position_pct: Max position as % of equity
        """
        self.fraction = fraction
        self.max_position_pct = max_position_pct
    
    def calculate(
        self,
        win_prob: float,
        avg_win: float,
        avg_loss: float,
        equity: float,
        current_price: float
    ) -> Tuple[float, int]:
        """
        Calculate Kelly-optimal position size.
        
        Args:
            win_prob: Probability of winning (0-1)
            avg_win: Average win amount (as decimal, e.g., 0.02 = 2%)
            avg_loss: Average loss amount (as decimal, positive)
            equity: Current account equity
            current_price: Current asset price
            
        Returns:
            (kelly_fraction, recommended_shares)
        """
        # Edge calculation: b*p - q where b = win/loss ratio, p = win prob, q = loss prob
        if avg_loss == 0:
            return 0, 0
            
        b = avg_win / avg_loss  # Win/loss ratio
        q = 1 - win_prob  # Loss probability
        
        # Full Kelly fraction
        kelly_f = (b * win_prob - q) / b if b > 0 else 0
        
        # Apply fractional Kelly for safety
        kelly_f *= self.fraction
        
        # Clamp to max position size
        kelly_f = min(kelly_f, self.max_position_pct)
        kelly_f = max(kelly_f, 0)  # No negative Kelly (would be short)
        
        # Calculate dollar amount and shares
        position_dollars = equity * kelly_f
        shares = int(position_dollars / current_price)
        
        return kelly_f, shares


class GameTheoryValidator:
    """
    Game Theory Layer - adversarial analysis and market microstructure.
    """
    
    def __init__(self):
        self.min_adversarial_score = 0.3
        
    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        direction: str,
        bias: float
    ) -> Tuple[float, float, str]:
        """
        Run game theory analysis on signal.
        
        Returns:
            (game_theory_score, edge_estimate, reasoning)
        """
        if len(df) < 20:
            return 0, 0, "Insufficient data"
        
        scores = []
        reasons = []
        
        # 1. Momentum alignment check
        returns_5d = (df['close'].iloc[-1] / df['close'].iloc[-5]) - 1
        momentum_aligned = (direction == "LONG" and returns_5d > 0) or \
                          (direction == "SHORT" and returns_5d < 0)
        scores.append(1.0 if momentum_aligned else 0.3)
        reasons.append(f"5d momentum: {returns_5d:.2%}")
        
        # 2. Volatility regime check
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
        if volatility < 0.02:  # Low vol regime
            scores.append(1.0)
            reasons.append(f"Low vol regime: {volatility:.2%}")
        elif volatility < 0.04:  # Normal
            scores.append(0.7)
            reasons.append(f"Normal vol: {volatility:.2%}")
        else:  # High vol
            scores.append(0.4)
            reasons.append(f"High vol: {volatility:.2%}")
        
        # 3. Volume confirmation
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        current_vol = df['volume'].iloc[-1]
        volume_spike = current_vol > vol_sma * 1.2
        scores.append(1.0 if volume_spike else 0.6)
        reasons.append(f"Volume: {current_vol/vol_sma:.1f}x avg")
        
        # 4. Price distance from recent extremes (avoid buying tops)
        high_20 = df['high'].rolling(20).max().iloc[-1]
        low_20 = df['low'].rolling(20).min().iloc[-1]
        current = df['close'].iloc[-1]
        
        if direction == "LONG":
            distance_from_high = (high_20 - current) / high_20
            scores.append(min(1.0, distance_from_high * 5))  # Prefer lower prices
            reasons.append(f"Distance from 20d high: {distance_from_high:.1%}")
        else:
            distance_from_low = (current - low_20) / low_20
            scores.append(min(1.0, distance_from_low * 5))
            reasons.append(f"Distance from 20d low: {distance_from_low:.1%}")
        
        # Aggregate score
        avg_score = np.mean(scores)
        
        # Calculate edge estimate based on bias and game theory
        edge = bias * avg_score * 0.02  # Max 2% edge assumption
        
        reasoning = " | ".join(reasons)
        
        return avg_score, edge, reasoning


class EntryGates:
    """
    Layer 3 - Entry validation gates.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.55,
        min_kelly_fraction: float = 0.01,
        max_positions: int = 5,
        daily_loss_limit: float = 0.03
    ):
        self.min_confidence = min_confidence
        self.min_kelly_fraction = min_kelly_fraction
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit
        self.current_positions = 0
        self.daily_pnl = 0
        
    def check(
        self,
        layer1: Layer1Output,
        layer2: Layer2Output,
        current_equity: float
    ) -> Layer3Output:
        """
        Run all entry gates on the signal.
        """
        # Gate 1: Confidence check
        if layer1.confidence < self.min_confidence:
            return Layer3Output(
                passed=False,
                reject_reason=f"Confidence {layer1.confidence:.2f} < {self.min_confidence}",
                final_shares=0,
                final_confidence=layer1.confidence,
                metadata={}
            )
        
        # Gate 2: Kelly fraction check
        if layer2.kelly_fraction < self.min_kelly_fraction:
            return Layer3Output(
                passed=False,
                reject_reason=f"Kelly fraction {layer2.kelly_fraction:.3f} too small",
                final_shares=0,
                final_confidence=layer1.confidence,
                metadata={}
            )
        
        # Gate 3: Game theory check
        if layer2.game_theory_score < 0.3:
            return Layer3Output(
                passed=False,
                reject_reason=f"Game theory score {layer2.game_theory_score:.2f} too low",
                final_shares=0,
                final_confidence=layer1.confidence,
                metadata={}
            )
        
        # Gate 4: Position limit
        if self.current_positions >= self.max_positions:
            return Layer3Output(
                passed=False,
                reject_reason=f"Max positions ({self.max_positions}) reached",
                final_shares=0,
                final_confidence=layer1.confidence,
                metadata={}
            )
        
        # Gate 5: Daily loss limit
        if self.daily_pnl < -current_equity * self.daily_loss_limit:
            return Layer3Output(
                passed=False,
                reject_reason="Daily loss limit hit",
                final_shares=0,
                final_confidence=layer1.confidence,
                metadata={}
            )
        
        # All gates passed
        return Layer3Output(
            passed=True,
            reject_reason=None,
            final_shares=layer2.recommended_shares,
            final_confidence=layer1.confidence * layer2.game_theory_score,
            metadata={
                'layer1': layer1,
                'layer2': layer2,
                'gates_passed': 5
            }
        )


class ClawdBrain(AIBrain):
    """
    Full 3-Layer AI Trading Brain.
    
    Implements the complete Clawd trading pipeline:
    1. XGBoost Bias Layer - Predict direction and confidence
    2. Kelly + Game Theory Layer - Size positions, validate against market
    3. Entry Gates Layer - Risk management and final validation
    """
    
    def __init__(
        self,
        model_path: str = None,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.10,
        min_confidence: float = 0.55,
        avg_win_pct: float = 0.02,  # Assumed 2% avg win
        avg_loss_pct: float = 0.015,  # Assumed 1.5% avg loss
        paper: bool = True
    ):
        """
        Initialize ClawdBrain with 3-layer architecture.
        
        Args:
            model_path: Path to XGBoost model
            kelly_fraction: Fractional Kelly (0.25 = Quarter Kelly)
            max_position_pct: Max position as % of equity
            min_confidence: Minimum confidence to trade
            avg_win_pct: Assumed average win percentage
            avg_loss_pct: Assumed average loss percentage
        """
        self.model_path = model_path or Path(__file__).parent / 'training' / 'xgb_model.pkl'
        self.paper = paper
        
        # Initialize layers
        self.kelly = KellyCalculator(fraction=kelly_fraction, max_position_pct=max_position_pct)
        self.game_theory = GameTheoryValidator()
        self.gates = EntryGates(min_confidence=min_confidence)
        
        # Assumptions for Kelly calculation
        self.avg_win = avg_win_pct
        self.avg_loss = avg_loss_pct
        
        # Components
        self.feature_gen = FeatureGenerator()
        self.model = None
        self.feature_cols = None
        self.alpaca_client = AlpacaDataClient() if paper else None
        
        # State
        self._name = "ClawdBrain-v3.0"
        
    @property
    def name(self) -> str:
        return self._name
    
    def load_model(self) -> 'ClawdBrain':
        """Load pre-trained XGBoost model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Run: python training/train_xgb.py"
            )
        
        with open(self.model_path, 'rb') as f:
            saved = pickle.load(f)
            self.model = saved['model']
            self.feature_cols = saved['features']
            
        print(f"[ClawdBrain] Loaded model: {len(self.feature_cols)} features")
        return self
    
    def _layer1_predict(self, symbol: str, df: pd.DataFrame) -> Optional[Layer1Output]:
        """
        Layer 1: XGBoost Bias Prediction.
        
        Returns prediction with confidence and bias score.
        """
        if self.model is None:
            self.load_model()
        
        if len(df) < 30:
            return None
        
        # Generate features
        features_df = self.feature_gen.generate_features(df)
        if features_df.empty:
            return None
        
        # Get latest features
        latest = features_df.iloc[-1]
        
        # Build feature vector
        feature_values = []
        for col in self.feature_cols:
            val = latest.get(col, 0)
            if pd.isna(val):
                val = 0
            feature_values.append(val)
        
        X = np.array(feature_values).reshape(1, -1)
        
        # Predict
        prob = self.model.predict_proba(X)[0]
        up_prob = prob[1]
        down_prob = prob[0]
        
        # Determine direction and bias
        if up_prob > down_prob:
            direction = "LONG"
            confidence = up_prob
            bias = (up_prob - 0.5) * 2  # Scale to -1 to 1
        else:
            direction = "SHORT"
            confidence = down_prob
            bias = -(down_prob - 0.5) * 2
        
        return Layer1Output(
            symbol=symbol,
            direction=direction,
            bias=bias,
            confidence=confidence,
            win_prob=confidence,  # Use confidence as win probability estimate
            features={col: feature_values[i] for i, col in enumerate(self.feature_cols[:5])}
        )
    
    def _layer2_size(
        self,
        layer1: Layer1Output,
        df: pd.DataFrame,
        equity: float
    ) -> Layer2Output:
        """
        Layer 2: Kelly Sizing + Game Theory Validation.
        
        Calculates optimal position size and validates against market structure.
        """
        current_price = df['close'].iloc[-1]
        
        # Kelly sizing
        kelly_f, shares = self.kelly.calculate(
            win_prob=layer1.win_prob,
            avg_win=self.avg_win,
            avg_loss=self.avg_loss,
            equity=equity,
            current_price=current_price
        )
        
        # Game theory analysis
        gt_score, edge, reasoning = self.game_theory.analyze(
            symbol=layer1.symbol,
            df=df,
            direction=layer1.direction,
            bias=layer1.bias
        )
        
        return Layer2Output(
            symbol=layer1.symbol,
            direction=layer1.direction,
            kelly_fraction=kelly_f,
            game_theory_score=gt_score,
            edge=edge,
            max_position_size=equity * self.kelly.max_position_pct,
            recommended_shares=shares
        )
    
    def _layer3_validate(
        self,
        layer1: Layer1Output,
        layer2: Layer2Output,
        equity: float
    ) -> Layer3Output:
        """
        Layer 3: Entry Gates.
        
        Final risk management and validation.
        """
        return self.gates.check(layer1, layer2, equity)
    
    def predict(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Run full 3-layer pipeline on market data.
        
        Args:
            data: Dict of symbol -> OHLCV DataFrame
            
        Returns:
            List of Signal objects (only passing all 3 layers)
        """
        signals = []
        
        # Get current equity (assume $100k if can't fetch)
        try:
            if self.alpaca_client:
                from alpaca.trading.client import TradingClient
                import os
                trading = TradingClient(
                    os.getenv('ALPACA_API_KEY'),
                    os.getenv('ALPACA_SECRET_KEY'),
                    paper=True
                )
                equity = float(trading.get_account().equity)
            else:
                equity = 100000
        except:
            equity = 100000
        
        print(f"[ClawdBrain] Running 3-layer pipeline on {len(data)} symbols")
        print(f"[ClawdBrain] Current equity: ${equity:,.2f}")
        print()
        
        for symbol, df in data.items():
            print(f"\n--- {symbol} ---")
            
            # Layer 1: Predict
            layer1 = self._layer1_predict(symbol, df)
            if layer1 is None:
                print("  Layer 1: No prediction")
                continue
            print(f"  Layer 1: {layer1.direction} | Confidence: {layer1.confidence:.2f} | Bias: {layer1.bias:+.2f}")
            
            # Layer 2: Size
            layer2 = self._layer2_size(layer1, df, equity)
            print(f"  Layer 2: Kelly {layer2.kelly_fraction:.3f} | GT Score: {layer2.game_theory_score:.2f} | Shares: {layer2.recommended_shares}")
            
            # Layer 3: Validate
            layer3 = self._layer3_validate(layer1, layer2, equity)
            
            if not layer3.passed:
                print(f"  Layer 3: REJECTED - {layer3.reject_reason}")
                continue
            
            print(f"  Layer 3: PASSED - Final shares: {layer3.final_shares}")
            
            # Create Signal
            signal = Signal(
                symbol=layer1.symbol,
                direction=layer1.direction,
                confidence=layer3.final_confidence,
                size=layer3.final_shares,
                metadata={
                    'brain': self.name,
                    'bias': layer1.bias,
                    'kelly_fraction': layer2.kelly_fraction,
                    'game_theory_score': layer2.game_theory_score,
                    'edge': layer2.edge,
                    'features': layer1.features
                }
            )
            
            signals.append(signal)
            self.gates.current_positions += 1
        
        print(f"\n[ClawdBrain] Generated {len(signals)} signals after 3-layer filtering")
        return signals


def test_clawd_brain():
    """Test the full ClawdBrain"""
    from ai_trading_bridge import AITradingBridge
    
    print("="*70)
    print("TESTING CLAWDBRAIN - 3 LAYER ARCHITECTURE")
    print("="*70)
    print()
    
    # Create brain
    brain = ClawdBrain(
        kelly_fraction=0.25,  # Quarter Kelly
        max_position_pct=0.10,  # Max 10% per position
        min_confidence=0.55
    )
    
    # Create bridge
    bridge = AITradingBridge(
        brain=brain,
        symbols=["SPY", "QQQ", "IWM", "NVDA", "AAPL"],
        timeframe="1D",
        lookback_days=60,
        paper=True
    )
    
    # Run cycle
    result = bridge.run_cycle()
    
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"Brain: {result['brain']}")
    print(f"Signals Passing All 3 Layers: {result['signals']}")
    print(f"Trades Executed: {result['executed']}")
    
    return result


if __name__ == '__main__':
    test_clawd_brain()
