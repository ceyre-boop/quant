"""
Universal AI Trading Bridge for Alpaca

Drop in ANY AI model - just implement predict() method.
Bridge handles: data fetch → signal generation → trade execution → tracking
"""
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from data.alpaca_client import AlpacaDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce


@dataclass
class Signal:
    """Standard signal format - any AI brain outputs this"""
    symbol: str
    direction: str  # "LONG" | "SHORT" | "FLAT"
    confidence: float  # 0.0 to 1.0
    size: int = 10  # Number of shares
    metadata: Dict[str, Any] = None  # Extra info (model version, features used, etc.)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AIBrain(ABC):
    """
    Abstract base class for AI trading models.
    
    Just implement predict() - return list of Signals.
    The bridge handles everything else.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging"""
        pass
    
    @abstractmethod
    def predict(self, data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals from market data.
        
        Args:
            data: Dict mapping symbol -> DataFrame (OHLCV)
        
        Returns:
            List of Signal objects
        """
        pass
    
    def warmup(self, data: Dict[str, Any]) -> None:
        """Optional: Warm up model with historical data"""
        pass


class MomentumBrain(AIBrain):
    """Example: Simple momentum model"""
    
    @property
    def name(self) -> str:
        return "MomentumBrain-v1"
    
    def predict(self, data: Dict[str, Any]) -> List[Signal]:
        signals = []
        
        for symbol, df in data.items():
            if len(df) < 20:
                continue
            
            # Simple 5-period momentum
            current = df['close'].iloc[-1]
            prev = df['close'].iloc[-5]
            change = (current - prev) / prev
            
            if abs(change) > 0.01:  # 1% threshold
                direction = "LONG" if change > 0 else "SHORT"
                confidence = min(abs(change) * 100, 1.0)
                
                signals.append(Signal(
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    size=10,
                    metadata={
                        'model': self.name,
                        'change_pct': change * 100,
                        'price': current
                    }
                ))
        
        return signals


class XGBoostBrain(AIBrain):
    """XGBoost model wrapper - plug in your trained model here"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.feature_cols = None
        
    @property
    def name(self) -> str:
        return "XGBoostBrain-v1"
    
    def load_model(self, path: str):
        """Load trained XGBoost model"""
        import pickle
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"[XGBoostBrain] Loaded model from {path}")
    
    def predict(self, data: Dict[str, Any]) -> List[Signal]:
        """Generate signals using XGBoost predictions"""
        if self.model is None:
            print("[XGBoostBrain] No model loaded, returning empty signals")
            return []
        
        signals = []
        
        # TODO: Build features from data
        # TODO: Run model.predict_proba()
        # TODO: Convert to Signal objects
        
        return signals


class AITradingBridge:
    """
    Universal bridge: Connect ANY AI brain to Alpaca for live trading.
    
    Usage:
        brain = YourModel()  # Implements AIBrain
        bridge = AITradingBridge(brain)
        bridge.run_cycle()   # Fetch data → Get signals → Execute trades
    """
    
    def __init__(
        self,
        brain: AIBrain,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1H",
        lookback_days: int = 30,
        paper: bool = True,
        min_confidence: float = 0.5,
        log_level: str = "INFO"
    ):
        self.brain = brain
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.paper = paper
        self.min_confidence = min_confidence
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Load env
        load_dotenv()
        
        # Initialize Alpaca
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY required")
        
        self.data_client = AlpacaDataClient(api_key, secret_key, paper=paper)
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        
        # Verify connection
        account = self.trading_client.get_account()
        self.logger.info(f"[{brain.name}] Bridge connected")
        self.logger.info(f"  Account: {account.id}")
        self.logger.info(f"  Equity: ${account.equity}")
        
        # Set symbols
        self.symbols = symbols or self._default_symbols()
        
        # Stats
        self.cycles_run = 0
        self.signals_generated = 0
        self.trades_executed = 0
    
    def _setup_logging(self, level: str) -> logging.Logger:
        logger = logging.getLogger(f"AITradingBridge.{self.brain.name}")
        logger.setLevel(getattr(logging, level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # File handler
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler(
                f"logs/bridge_{self.brain.name.lower()}.log"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _default_symbols(self) -> List[str]:
        """Default tradeable symbols"""
        return ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "MSFT", "AMD"]
    
    def fetch_data(self) -> Dict[str, Any]:
        """Fetch market data for all symbols"""
        self.logger.info(f"Fetching {len(self.symbols)} symbols ({self.timeframe}, {self.lookback_days}d)")
        
        data = {}
        for symbol in self.symbols:
            try:
                df = self.data_client.get_historical_bars(
                    symbol, 
                    timeframe=self.timeframe, 
                    days=self.lookback_days
                )
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                self.logger.warning(f"Failed to fetch {symbol}: {e}")
        
        self.logger.info(f"Retrieved {len(data)}/{len(self.symbols)} symbols")
        return data
    
    def execute_signal(self, signal: Signal) -> bool:
        """Execute a signal via Alpaca"""
        if signal.confidence < self.min_confidence:
            self.logger.info(f"Skipping {signal.symbol} (confidence {signal.confidence:.2f} < {self.min_confidence})")
            return False
        
        if signal.direction == "FLAT":
            return False
        
        try:
            side = OrderSide.BUY if signal.direction == "LONG" else OrderSide.SELL
            
            order = MarketOrderRequest(
                symbol=signal.symbol,
                qty=signal.size,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            result = self.trading_client.submit_order(order)
            
            self.logger.info(f"✓ TRADE EXECUTED: {signal.symbol} {signal.direction}")
            self.logger.info(f"  Order ID: {result.id}")
            self.logger.info(f"  Confidence: {signal.confidence:.2%}")
            self.logger.info(f"  Size: {signal.size} shares")
            
            if signal.metadata:
                self.logger.info(f"  Metadata: {signal.metadata}")
            
            self.trades_executed += 1
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Trade failed for {signal.symbol}: {e}")
            return False
    
    def run_cycle(self) -> Dict[str, Any]:
        """
        Run one complete trading cycle:
        1. Fetch data
        2. Generate signals (via AI brain)
        3. Execute trades
        4. Return stats
        """
        self.logger.info("="*60)
        self.logger.info(f"TRADING CYCLE #{self.cycles_run + 1}")
        self.logger.info(f"Brain: {self.brain.name}")
        self.logger.info(f"Time: {datetime.now().isoformat()}")
        self.logger.info("="*60)
        
        # 1. Fetch data
        data = self.fetch_data()
        
        if not data:
            self.logger.error("No data fetched - aborting cycle")
            return {"status": "error", "reason": "no_data"}
        
        # 2. Generate signals via AI brain
        self.logger.info(f"Running {self.brain.name} prediction...")
        signals = self.brain.predict(data)
        self.logger.info(f"Generated {len(signals)} signals")
        
        # 3. Execute trades
        executed = 0
        for signal in signals:
            self.logger.info(f"  Signal: {signal.symbol} {signal.direction} (conf: {signal.confidence:.2f})")
            if self.execute_signal(signal):
                executed += 1
        
        self.cycles_run += 1
        self.signals_generated += len(signals)
        
        result = {
            "status": "success",
            "cycle": self.cycles_run,
            "symbols_data": len(data),
            "signals": len(signals),
            "executed": executed,
            "brain": self.brain.name,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info("="*60)
        self.logger.info("CYCLE COMPLETE")
        self.logger.info(f"  Signals: {len(signals)}")
        self.logger.info(f"  Executed: {executed}")
        self.logger.info("="*60)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            "brain": self.brain.name,
            "cycles_run": self.cycles_run,
            "signals_generated": self.signals_generated,
            "trades_executed": self.trades_executed,
            "symbols": self.symbols
        }


# Example usage
def example_momentum():
    """Example: Run momentum strategy"""
    print("="*60)
    print("EXAMPLE: Momentum Brain")
    print("="*60)
    
    # 1. Create your AI brain
    brain = MomentumBrain()
    
    # 2. Create bridge with your brain
    bridge = AITradingBridge(
        brain=brain,
        symbols=["SPY", "QQQ", "IWM", "NVDA", "AAPL"],
        timeframe="1H",
        min_confidence=0.6,
        paper=True
    )
    
    # 3. Run trading cycle
    result = bridge.run_cycle()
    
    print("\nResult:", result)
    return result


def example_custom_model():
    """Template: Plug in your custom model"""
    
    # TODO: Implement your model
    class MyCustomBrain(AIBrain):
        @property
        def name(self):
            return "MyModel-v1"
        
        def predict(self, data):
            signals = []
            # Your prediction logic here
            # Return list of Signal objects
            return signals
    
    # Use it
    brain = MyCustomBrain()
    bridge = AITradingBridge(brain)
    bridge.run_cycle()


if __name__ == "__main__":
    example_momentum()
