"""
QUICK START: Plug Your AI Model Into Alpaca Trading

3 steps. 10 lines of code. Done.
"""

from ai_trading_bridge import AIBrain, Signal, AITradingBridge
import pandas as pd
from typing import Dict, List, Any


# ═══════════════════════════════════════════════════════════
# STEP 1: Create your AI brain (implement 2 methods)
# ═══════════════════════════════════════════════════════════

class MyAIBrain(AIBrain):
    """Your custom AI model - just implement predict()"""
    
    @property
    def name(self) -> str:
        return "MyAwesomeModel-v1"
    
    def predict(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Args:
            data: Dict of symbol -> DataFrame with OHLCV columns
                  
        Returns:
            List of Signal objects (what to trade)
        """
        signals = []
        
        for symbol, df in data.items():
            # Your AI logic here
            # df has: open, high, low, close, volume
            
            # Example: Buy if price > 20-day MA
            if len(df) >= 20:
                ma20 = df['close'].rolling(20).mean().iloc[-1]
                current = df['close'].iloc[-1]
                
                if current > ma20 * 1.02:  # 2% above MA
                    signals.append(Signal(
                        symbol=symbol,
                        direction="LONG",
                        confidence=0.75,  # Your model's confidence
                        size=10,          # Shares to buy
                        metadata={
                            'model': self.name,
                            'ma20': ma20,
                            'price': current
                        }
                    ))
        
        return signals


# ═══════════════════════════════════════════════════════════
# STEP 2: Create bridge with your brain
# ═══════════════════════════════════════════════════════════

brain = MyAIBrain()

bridge = AITradingBridge(
    brain=brain,
    symbols=["SPY", "QQQ", "NVDA", "AAPL"],  # What to trade
    timeframe="1H",                           # 1H, 1D, etc
    min_confidence=0.6,                       # Only trade if conf > 60%
    paper=True                                # Paper trading (set False for LIVE)
)


# ═══════════════════════════════════════════════════════════
# STEP 3: Run it - fetches data, generates signals, trades
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = bridge.run_cycle()
    
    print("\n" + "="*50)
    print(f"Brain: {result['brain']}")
    print(f"Signals: {result['signals']}")
    print(f"Trades Executed: {result['executed']}")
    print("="*50)
