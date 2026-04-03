"""
ClawdBrain v3.1 - Full 3-Layer AI Trading System with Chi-Squared Gate

Architecture:
  Layer 1: XGBoost Bias Prediction (direction + confidence)
  Layer 2: Kelly Position Sizing (with real win/loss stats) + Game Theory Validation
  Layer 3: Entry Gates (including Chi-Squared Gate 6) + Risk Checks
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
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from ai_trading_bridge import AIBrain, Signal, AITradingBridge
from training.feature_generator import FeatureGenerator
from data.alpaca_client import AlpacaDataClient

# Chi-Squared Validation (embedded to avoid import issues)
from scipy import stats

def validate_prediction_against_outcomes(
    prediction: Dict, outcomes: List[bool], expected_wr: float, alpha: float = 0.05
):
    """Chi-squared test: (O-E)²/E"""
    n = len(outcomes)
    if n < 30:
        return type('Result', (), {'passed': False, 'p_value': 1.0, 'rejection_reason': 'Sample < 30'})()
    
    obs_wins = sum(outcomes)
    obs_loss = n - obs_wins
    exp_wins = expected_wr * n
    exp_loss = (1 - expected_wr) * n
    
    chi_sq = ((obs_wins - exp_wins) ** 2 / max(exp_wins, 0.001) + 
              (obs_loss - exp_loss) ** 2 / max(exp_loss, 0.001))
    p_val = 1 - stats.chi2.cdf(chi_sq, df=1)
    
    passed = p_val >= alpha
    return type('Result', (), {
        'passed': passed, 
        'p_value': p_val, 
        'chi_squared_stat': chi_sq,
        'rejection_reason': '' if passed else f'p={p_val:.3f} < alpha'
    })()

CHI2_AVAILABLE = True


@dataclass
class Layer1Output:
    """Output from XGBoost bias layer"""
    symbol: str
    direction: str
    bias: float
    confidence: float
    win_prob: float
    features: Dict[str, float]


@dataclass  
class Layer2Output:
    """Output from Kelly + Game Theory layer"""
    symbol: str
    direction: str
    kelly_fraction: float
    game_theory_score: float
    edge: float
    max_position_size: float
    recommended_shares: int
    avg_win: float  # Real from backtest
    avg_loss: float  # Real from backtest


@dataclass
class Layer3Output:
    """Output from Entry Gates"""
    passed: bool
    reject_reason: Optional[str]
    final_shares: int
    final_confidence: float
    chi2_result: Optional[Dict]
    metadata: Dict


@dataclass
class TradeOutcome:
    """Record of trade outcome for chi-squared tracking"""
    symbol: str
    prediction_confidence: float
    predicted_direction: str
    actual_return: float
    won: bool
    timestamp: str


class KellyCalculator:
    """Kelly Criterion with real statistics from backtests"""
    
    def __init__(self, fraction: float = 0.25, max_position_pct: float = 0.10):
        self.fraction = fraction
        self.max_position_pct = max_position_pct
        self.trade_history: List[TradeOutcome] = []
        self._cached_stats = None
        
    def compute_stats(self) -> Tuple[float, float]:
        """Compute real avg win/loss from trade history"""
        if len(self.trade_history) < 10:
            # Not enough data, use conservative defaults
            return 0.015, 0.010  # 1.5% win, 1.0% loss
        
        wins = [t.actual_return for t in self.trade_history if t.won]
        losses = [abs(t.actual_return) for t in self.trade_history if not t.won]
        
        avg_win = np.mean(wins) if wins else 0.015
        avg_loss = np.mean(losses) if losses else 0.010
        
        return avg_win, avg_loss
    
    def calculate(
        self,
        win_prob: float,
        equity: float,
        current_price: float,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> Tuple[float, int, float, float]:
        """
        Calculate Kelly-optimal position size using real statistics.
        
        Returns:
            (kelly_fraction, shares, avg_win_used, avg_loss_used)
        """
        # Use provided stats or compute from history
        if avg_win is None or avg_loss is None:
            avg_win, avg_loss = self.compute_stats()
        
        if avg_loss == 0:
            return 0, 0, avg_win, avg_loss
            
        b = avg_win / avg_loss  # Win/loss ratio
        q = 1 - win_prob
        
        # Full Kelly
        kelly_f = (b * win_prob - q) / b if b > 0 else 0
        
        # Fractional Kelly for safety
        kelly_f *= self.fraction
        
        # Clamp
        kelly_f = min(kelly_f, self.max_position_pct)
        kelly_f = max(kelly_f, 0)
        
        # Calculate shares
        position_dollars = equity * kelly_f
        shares = int(position_dollars / current_price)
        
        return kelly_f, shares, avg_win, avg_loss
    
    def record_outcome(self, outcome: TradeOutcome):
        """Record trade outcome for future Kelly calculations"""
        self.trade_history.append(outcome)
        # Keep last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
        
        # Save to disk
        self._save_history()
    
    def _save_history(self):
        """Persist trade history"""
        history_path = Path("logs/trade_history.json")
        history_path.parent.mkdir(exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump([{
                'symbol': t.symbol,
                'confidence': t.prediction_confidence,
                'direction': t.predicted_direction,
                'return': t.actual_return,
                'won': t.won,
                'timestamp': t.timestamp
            } for t in self.trade_history], f)
    
    def load_history(self):
        """Load trade history from disk"""
        history_path = Path("logs/trade_history.json")
        if history_path.exists():
            with open(history_path, 'r') as f:
                data = json.load(f)
                self.trade_history = [
                    TradeOutcome(
                        symbol=d['symbol'],
                        prediction_confidence=d['confidence'],
                        predicted_direction=d['direction'],
                        actual_return=d['return'],
                        won=d['won'],
                        timestamp=d['timestamp']
                    ) for d in data
                ]


class ChiSquaredGate:
    """
    Gate 6: Chi-Squared Validation
    Only trades from validated confidence buckets pass.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.prediction_history: List[Dict] = []
        self.validated_buckets = set()
        
    def update_validation(self, predictions: List[float], actuals: List[bool]):
        """Update which confidence buckets are validated"""
        if not CHI2_AVAILABLE or len(predictions) < 30:
            return
        
        # Group by confidence buckets
        buckets = {f"{i*20}-{(i+1)*20}%": [] for i in range(5)}
        
        for pred, actual in zip(predictions, actuals):
            bucket_idx = min(int(pred * 5), 4)
            bucket_name = f"{bucket_idx*20}-{(bucket_idx+1)*20}%"
            buckets[bucket_name].append((pred, actual))
        
        # Validate each bucket with chi-squared
        for bucket_name, data in buckets.items():
            if len(data) < 10:
                continue
            
            preds, acts = zip(*data)
            expected_wr = np.mean(preds)
            
            result = validate_prediction_against_outcomes(
                {'id': bucket_name}, list(acts), expected_wr, self.alpha
            )
            
            if result.passed:
                self.validated_buckets.add(bucket_name)
                print(f"[Chi2Gate] ✓ Bucket {bucket_name} validated (p={result.p_value:.3f})")
            else:
                print(f"[Chi2Gate] ✗ Bucket {bucket_name} rejected (p={result.p_value:.3f})")
    
    def check(self, confidence: float) -> Tuple[bool, str]:
        """Check if confidence bucket is validated"""
        if not CHI2_AVAILABLE:
            return True, "Chi2 not available, bypassed"
        
        bucket_idx = min(int(confidence * 5), 4)
        bucket_name = f"{bucket_idx*20}-{(bucket_idx+1)*20}%"
        
        if bucket_name in self.validated_buckets:
            return True, f"Bucket {bucket_name} validated"
        else:
            return False, f"Bucket {bucket_name} not yet validated (need 30+ samples)"


class EntryGates:
    """Layer 3 - Entry validation gates (now with Gate 6: Chi-Squared)"""
    
    def __init__(
        self,
        min_confidence: float = 0.55,
        min_kelly_fraction: float = 0.01,
        max_positions: int = 5,
        daily_loss_limit: float = 0.03,
        use_chi2: bool = True
    ):
        self.min_confidence = min_confidence
        self.min_kelly_fraction = min_kelly_fraction
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit
        self.use_chi2 = use_chi2
        
        self.current_positions = 0
        self.daily_pnl = 0
        self.chi2_gate = ChiSquaredGate() if use_chi2 else None
        
    def check(self, layer1: Layer1Output, layer2: Layer2Output, 
              current_equity: float) -> Layer3Output:
        """Run all 6 entry gates"""
        
        # Gate 1: Confidence check
        if layer1.confidence < self.min_confidence:
            return self._reject(f"Confidence {layer1.confidence:.2f} < {self.min_confidence}", layer1)
        
        # Gate 2: Kelly fraction check
        if layer2.kelly_fraction < self.min_kelly_fraction:
            return self._reject(f"Kelly {layer2.kelly_fraction:.3f} too small", layer1)
        
        # Gate 3: Game theory check
        if layer2.game_theory_score < 0.3:
            return self._reject(f"GT score {layer2.game_theory_score:.2f} too low", layer1)
        
        # Gate 4: Position limit
        if self.current_positions >= self.max_positions:
            return self._reject(f"Max positions ({self.max_positions}) reached", layer1)
        
        # Gate 5: Daily loss limit
        if self.daily_pnl < -current_equity * self.daily_loss_limit:
            return self._reject("Daily loss limit hit", layer1)
        
        # Gate 6: Chi-Squared validation (THE BIG ONE)
        chi2_passed = True
        chi2_reason = "Chi2 gate not enabled"
        chi2_result = None
        
        if self.chi2_gate and CHI2_AVAILABLE:
            chi2_passed, chi2_reason = self.chi2_gate.check(layer1.confidence)
            chi2_result = {'passed': chi2_passed, 'reason': chi2_reason}
            
            if not chi2_passed:
                return self._reject(f"Chi2 Gate: {chi2_reason}", layer1, chi2_result)
        
        # All gates passed!
        return Layer3Output(
            passed=True,
            reject_reason=None,
            final_shares=layer2.recommended_shares,
            final_confidence=layer1.confidence * layer2.game_theory_score,
            chi2_result=chi2_result,
            metadata={'layer1': layer1, 'layer2': layer2, 'gates_passed': 6}
        )
    
    def _reject(self, reason: str, layer1: Layer1Output, chi2=None):
        return Layer3Output(
            passed=False, reject_reason=reason, final_shares=0,
            final_confidence=layer1.confidence, chi2_result=chi2, metadata={}
        )


# Keep GameTheoryValidator as before (simplified version)
class GameTheoryValidator:
    def analyze(self, symbol: str, df: pd.DataFrame, direction: str, bias: float):
        if len(df) < 20:
            return 0, 0, "Insufficient data"
        
        scores = []
        returns_5d = (df['close'].iloc[-1] / df['close'].iloc[-5]) - 1
        momentum_aligned = (direction == "LONG" and returns_5d > 0) or (direction == "SHORT" and returns_5d < 0)
        scores.append(1.0 if momentum_aligned else 0.3)
        
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
        scores.append(1.0 if volatility < 0.02 else 0.7 if volatility < 0.04 else 0.4)
        
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        scores.append(1.0 if df['volume'].iloc[-1] > vol_sma * 1.2 else 0.6)
        
        return np.mean(scores), bias * np.mean(scores) * 0.02, "Basic GT"


class ClawdBrain(AIBrain):  # v3.1 - Canonical
    """Full 3-Layer AI Trading Brain with Chi-Squared Gate 6"""
    
    def __init__(self, model_path: str = None, kelly_fraction: float = 0.25,
                 max_position_pct: float = 0.10, min_confidence: float = 0.55,
                 paper: bool = True, use_chi2: bool = True):
        
        self.model_path = model_path or Path(__file__).parent / 'training' / 'xgb_model.pkl'
        self.paper = paper
        
        self.kelly = KellyCalculator(fraction=kelly_fraction, max_position_pct=max_position_pct)
        self.kelly.load_history()  # Load previous trade stats
        
        self.game_theory = GameTheoryValidator()
        self.gates = EntryGates(min_confidence=min_confidence, use_chi2=use_chi2)
        
        self.feature_gen = FeatureGenerator()
        self.model = None
        self.feature_cols = None
        self.alpaca_client = AlpacaDataClient() if paper else None
        
        self._name = "ClawdBrain-v3.1-chi2"
        
    @property
    def name(self): return self._name
    
    def load_model(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            saved = pickle.load(f)
            self.model = saved['model']
            self.feature_cols = saved['features']
        print(f"[ClawdBrain] Loaded model: {len(self.feature_cols)} features")
        return self
    
    def _get_equity(self) -> float:
        """Fetch equity with proper error handling"""
        try:
            from alpaca.trading.client import TradingClient
            import os
            trading = TradingClient(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                paper=True
            )
            equity = float(trading.get_account().equity)
            print(f"[ClawdBrain] Equity fetched: ${equity:,.2f}")
            return equity
        except Exception as e:
            print(f"[ClawdBrain] WARNING: Failed to fetch equity ({e}), using default $100k")
            return 100000.0
    
    def _layer1_predict(self, symbol: str, df: pd.DataFrame):
        if self.model is None:
            self.load_model()
        if len(df) < 30:
            return None
        
        # Fetch cross-asset data for market features
        all_market_data = {symbol: df}
        cross_assets = ['SPY', 'VIXY', 'SQQQ', 'SPXU', 'GLD', 'TLT']
        
        for asset in cross_assets:
            if asset != symbol:
                try:
                    asset_df = self.alpaca_client.get_historical_bars(asset, timeframe='1D', days=60)
                    if not asset_df.empty:
                        all_market_data[asset] = asset_df
                except Exception as e:
                    print(f"[ClawdBrain] Warning: Could not fetch {asset}: {e}")
        
        features_df = self.feature_gen.generate_features(df, all_market_data)
        if features_df.empty:
            return None
        
        latest = features_df.iloc[-1]
        feature_values = [latest.get(col, 0) if not pd.isna(latest.get(col, 0)) else 0 
                         for col in self.feature_cols]
        
        X = np.array(feature_values).reshape(1, -1)
        prob = self.model.predict_proba(X)[0]
        up_prob = prob[1]
        
        direction = "LONG" if up_prob > 0.5 else "SHORT"
        confidence = max(up_prob, 1 - up_prob)
        bias = (confidence - 0.5) * 2
        
        return Layer1Output(symbol, direction, bias, confidence, confidence, {})
    
    def _layer2_size(self, layer1: Layer1Output, df: pd.DataFrame, equity: float):
        current_price = df['close'].iloc[-1]
        
        # Use real Kelly stats
        kelly_f, shares, avg_win, avg_loss = self.kelly.calculate(
            win_prob=layer1.win_prob,
            equity=equity,
            current_price=current_price
        )
        
        gt_score, edge, _ = self.game_theory.analyze(
            layer1.symbol, df, layer1.direction, layer1.bias
        )
        
        return Layer2Output(
            symbol=layer1.symbol, direction=layer1.direction,
            kelly_fraction=kelly_f, game_theory_score=gt_score, edge=edge,
            max_position_size=equity * 0.10, recommended_shares=shares,
            avg_win=avg_win, avg_loss=avg_loss
        )
    
    def predict(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals = []
        equity = self._get_equity()
        
        print(f"[ClawdBrain] Running 3-layer pipeline on {len(data)} symbols")
        print(f"[ClawdBrain] Kelly stats: avg_win={self.kelly.compute_stats()[0]:.3f}, avg_loss={self.kelly.compute_stats()[1]:.3f}")
        
        for symbol, df in data.items():
            print(f"\n--- {symbol} ---")
            
            layer1 = self._layer1_predict(symbol, df)
            if layer1 is None:
                print("  Layer 1: No prediction")
                continue
            print(f"  Layer 1: {layer1.direction} | Conf: {layer1.confidence:.2f}")
            
            layer2 = self._layer2_size(layer1, df, equity)
            print(f"  Layer 2: Kelly {layer2.kelly_fraction:.3f} | GT {layer2.game_theory_score:.2f} | Shares: {layer2.recommended_shares}")
            print(f"           (using avg_win={layer2.avg_win:.3f}, avg_loss={layer2.avg_loss:.3f})")
            
            layer3 = self.gates.check(layer1, layer2, equity)
            
            if not layer3.passed:
                print(f"  Layer 3: REJECTED - {layer3.reject_reason}")
                continue
            
            print(f"  Layer 3: PASSED (Gate 6: {layer3.chi2_result})")
            
            signals.append(Signal(
                symbol=layer1.symbol, direction=layer1.direction,
                confidence=layer3.final_confidence, size=layer3.final_shares,
                metadata={'brain': self.name, 'chi2': layer3.chi2_result}
            ))
            self.gates.current_positions += 1
        
        print(f"\n[ClawdBrain] Generated {len(signals)} signals")
        return signals


if __name__ == '__main__':
    from ai_trading_bridge import AITradingBridge
    
    print("="*70)
    print("TESTING CLAWDBRAIN v3.1 WITH CHI-SQUARED GATE")
    print("="*70)
    
    brain = ClawdBrain(use_chi2=True)
    bridge = AITradingBridge(brain=brain, symbols=["SPY", "QQQ"], timeframe="1D", paper=True)
    result = bridge.run_cycle()
    
    print(f"\nSignals: {result['signals']}")
