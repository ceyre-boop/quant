"""
Backtest Lifecycle - Historical testing with real Alpaca data

Runs the full 3-layer system on historical data to:
1. Validate signal quality
2. Generate training data for XGBoost
3. Calculate Sharpe and win rate

Usage:
    python orchestrator/backtest_lifecycle.py --start 2025-06-01 --end 2025-08-01 --symbols SPY,QQQ
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.alpaca_client import AlpacaDataClient
from layer1.bias_engine import BiasEngine
from layer1.feature_builder import FeatureBuilder
from layer2.risk_engine import RiskEngine
from layer3.game_engine import GameEngine
from contracts.types import MarketData, Direction, BiasOutput, RiskOutput, GameOutput


@dataclass
class BacktestTrade:
    """Record of a backtest trade."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    direction: Direction
    position_size: float
    pnl: float
    pnl_pct: float
    features: Dict[str, Any]
    bias_confidence: float
    game_score: float


class BacktestLifecycle:
    """
    Backtest coordinator using real Alpaca historical data.
    
    Free tier limit: 365 days of historical data
    """
    
    def __init__(
        self,
        start_date: str = "2025-06-01",
        end_date: str = "2025-08-01",
        symbols: List[str] = None,
        timeframe: str = "1D"
    ):
        """
        Initialize backtest.
        
        Args:
            start_date: Start date (YYYY-MM-DD) - must be within 365 days
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to test
            timeframe: "1H", "1D", etc.
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.symbols = symbols or ["SPY", "QQQ"]
        self.timeframe = timeframe
        
        # Initialize real components
        self.alpaca = AlpacaDataClient()
        self.bias_engine = BiasEngine()
        self.feature_builder = FeatureBuilder()
        self.risk_engine = RiskEngine()
        self.game_engine = GameEngine()
        
        # Results storage
        self.trades: List[BacktestTrade] = []
        self.signals: List[Dict] = []
        
        print(f"[BACKTEST] Initialized: {start_date} to {end_date}")
        print(f"[BACKTEST] Symbols: {', '.join(self.symbols)}")
        
    def run(self) -> Dict[str, Any]:
        """
        Run full backtest.
        
        Returns:
            Statistics dict with Sharpe, win rate, etc.
        """
        print(f"\n{'='*60}")
        print("[BACKTEST] Starting historical run...")
        print(f"{'='*60}\n")
        
        for symbol in self.symbols:
            self._backtest_symbol(symbol)
        
        # Calculate statistics
        stats = self._calculate_stats()
        
        # Save results
        self._save_results()
        
        return stats
    
    def _backtest_symbol(self, symbol: str):
        """Backtest a single symbol."""
        print(f"[BACKTEST] Processing {symbol}...")
        
        # Fetch historical bars
        try:
            bars = self.alpaca.get_bars(
                symbol=symbol,
                start=self.start_date,
                end=self.end_date,
                timeframe=self.timeframe
            )
        except Exception as e:
            print(f"[ERROR] Failed to fetch {symbol}: {e}")
            return
        
        if bars is None or len(bars) == 0:
            print(f"[WARNING] No data for {symbol}")
            return
        
        print(f"[BACKTEST] {symbol}: {len(bars)} bars")
        
        # Run through each bar
        position = None
        
        for i in range(len(bars)):
            if i < 20:  # Need enough history for indicators
                continue
                
            current_bar = bars.iloc[i]
            
            # Create market data
            market_data = MarketData(
                symbol=symbol,
                current_price=current_bar['close'],
                bid=current_bar['close'] - 0.01,
                ask=current_bar['close'] + 0.01,
                spread=0.02,
                volume_24h=current_bar['volume'],
                atr_14=self._calculate_atr(bars, i),
                timestamp=current_bar.name if hasattr(current_bar, 'name') else datetime.now()
            )
            
            # Build features
            features = self.feature_builder.build(symbol, market_data, bars.iloc[:i+1])
            
            # Layer 1: Bias
            bias = self.bias_engine.predict(symbol, features, features.regime)
            
            # Skip if no confidence
            if bias.confidence < 0.5:
                continue
            
            # Layer 2: Risk (simplified account)
            account = self._create_mock_account()
            risk = self.risk_engine.calculate(symbol, bias, features, account, market_data)
            
            # Layer 3: Game Theory
            game = self.game_engine.analyze(symbol, bias, risk, market_data)
            
            # Record signal
            signal = {
                'date': market_data.timestamp,
                'symbol': symbol,
                'direction': bias.direction.value if hasattr(bias.direction, 'value') else str(bias.direction),
                'confidence': bias.confidence,
                'entry_price': market_data.current_price,
                'stop_price': risk.stop_price,
                'tp1_price': risk.tp1_price,
                'game_score': game.composite_score if hasattr(game, 'composite_score') else 0.5,
                'features': features.features if hasattr(features, 'features') else {}
            }
            self.signals.append(signal)
            
            # Simulate trade outcome (simplified - exit next day for now)
            if i + 1 < len(bars):
                next_bar = bars.iloc[i + 1]
                exit_price = next_bar['close']
                
                if bias.direction == Direction.LONG:
                    pnl = exit_price - market_data.current_price
                else:
                    pnl = market_data.current_price - exit_price
                
                pnl_pct = (pnl / market_data.current_price) * 100
                
                trade = BacktestTrade(
                    symbol=symbol,
                    entry_date=market_data.timestamp,
                    exit_date=next_bar.name if hasattr(next_bar, 'name') else datetime.now(),
                    entry_price=market_data.current_price,
                    exit_price=exit_price,
                    direction=bias.direction,
                    position_size=risk.position_size if hasattr(risk, 'position_size') else 1.0,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    features=features.features if hasattr(features, 'features') else {},
                    bias_confidence=bias.confidence,
                    game_score=game.composite_score if hasattr(game, 'composite_score') else 0.5
                )
                self.trades.append(trade)
    
    def _calculate_atr(self, bars: pd.DataFrame, idx: int, period: int = 14) -> float:
        """Calculate ATR for a given bar."""
        if idx < period:
            return 1.0
        
        highs = bars['high'].iloc[idx-period:idx]
        lows = bars['low'].iloc[idx-period:idx]
        closes = bars['close'].iloc[idx-period:idx]
        
        tr1 = highs - lows
        tr2 = abs(highs - closes.shift(1))
        tr3 = abs(lows - closes.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.mean()
    
    def _create_mock_account(self):
        """Create mock account state for backtest."""
        from contracts.types import AccountState
        return AccountState(
            equity=100000.0,
            buying_power=100000.0,
            open_positions={},
            daily_pnl=0.0,
            weekly_pnl=0.0,
            margin_used=0.0
        )
    
    def _calculate_stats(self) -> Dict[str, Any]:
        """Calculate backtest statistics."""
        if not self.trades:
            return {"error": "No trades generated"}
        
        pnls = [t.pnl_pct for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        stats = {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "total_pnl_pct": sum(pnls),
            "sharpe_ratio": self._calculate_sharpe(pnls)
        }
        
        print(f"\n{'='*60}")
        print("[BACKTEST] RESULTS")
        print(f"{'='*60}")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1%}")
        print(f"Avg P&L: {stats['avg_pnl']:.2f}%")
        print(f"Total P&L: {stats['total_pnl_pct']:.2f}%")
        print(f"Sharpe: {stats['sharpe_ratio']:.2f}")
        print(f"{'='*60}\n")
        
        return stats
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / std_dev
    
    def _save_results(self):
        """Save backtest results to disk."""
        output_dir = Path("data/backtest_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame([
                {
                    'symbol': t.symbol,
                    'entry_date': t.entry_date,
                    'exit_date': t.exit_date,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'direction': str(t.direction),
                    'pnl_pct': t.pnl_pct,
                    'bias_confidence': t.bias_confidence,
                    'game_score': t.game_score
                }
                for t in self.trades
            ])
            trades_file = output_dir / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"[BACKTEST] Trades saved: {trades_file}")
        
        # Save signals for training
        if self.signals:
            signals_df = pd.DataFrame(self.signals)
            signals_file = output_dir / f"signals_{timestamp}.csv"
            signals_df.to_csv(signals_file, index=False)
            print(f"[BACKTEST] Signals saved: {signals_file}")


def main():
    parser = argparse.ArgumentParser(description="Run backtest with Alpaca data")
    parser.add_argument("--start", default="2025-06-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-08-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", default="SPY,QQQ", help="Comma-separated symbols")
    parser.add_argument("--timeframe", default="1D", help="1H, 1D, etc.")
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(",")
    
    backtest = BacktestLifecycle(
        start_date=args.start,
        end_date=args.end,
        symbols=symbols,
        timeframe=args.timeframe
    )
    
    stats = backtest.run()
    
    # Exit code based on results
    if stats.get("win_rate", 0) > 0.55 and stats.get("sharpe_ratio", 0) > 1.0:
        print("[BACKTEST] ✅ Performance threshold met - ready for paper trading")
        sys.exit(0)
    else:
        print("[BACKTEST] ⚠️ Performance below threshold - needs tuning")
        sys.exit(1)


if __name__ == "__main__":
    main()
