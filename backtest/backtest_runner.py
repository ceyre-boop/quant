"""Backtest Runner - Historical replay engine for the Clawd Trading System.

Replays historical market data through the trading pipeline to evaluate strategy performance.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np

from contracts.types import (
    BiasOutput, RiskOutput, GameOutput, RegimeState,
    PositionState, EntrySignal, Direction, Magnitude, AdversarialRisk
)

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtest execution modes."""
    VECTORIZED = "vectorized"    # Fast, loop-based
    EVENT_DRIVEN = "event_driven"  # More realistic, tick-by-tick


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    symbol: str
    start_date: datetime
    end_date: datetime
    timeframe: str = "5m"
    mode: BacktestMode = BacktestMode.EVENT_DRIVEN
    initial_capital: float = 50000.0
    commission_per_trade: float = 5.0  # USD
    spread_pips: float = 0.5
    slippage_model: str = "random"  # "none", "random", "market_impact"
    use_margin: bool = True
    margin_requirement: float = 0.02  # 50:1 leverage
    max_positions: int = 5
    
    def __post_init__(self):
        if isinstance(self.start_date, str):
            self.start_date = datetime.fromisoformat(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = datetime.fromisoformat(self.end_date)


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    trade_id: str
    symbol: str
    direction: Direction
    entry_time: datetime
    entry_price: float
    position_size: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = "open"
    realized_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "position_size": self.position_size,
            "stop_price": self.stop_price,
            "tp1_price": self.tp1_price,
            "tp2_price": self.tp2_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "realized_pnl": self.realized_pnl,
            "commission": self.commission,
            "slippage": self.slippage
        }


@dataclass
class BacktestResult:
    """Complete backtest results."""
    config: BacktestConfig
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    total_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    def calculate_metrics(self):
        """Calculate performance metrics from trades."""
        if not self.trades:
            return
        
        self.total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t.exit_time is not None]
        
        if not closed_trades:
            return
        
        # Win/Loss stats
        winning = [t for t in closed_trades if t.realized_pnl > 0]
        losing = [t for t in closed_trades if t.realized_pnl <= 0]
        
        self.winning_trades = len(winning)
        self.losing_trades = len(losing)
        self.win_rate = self.winning_trades / len(closed_trades) if closed_trades else 0.0
        
        self.avg_win = np.mean([t.realized_pnl for t in winning]) if winning else 0.0
        self.avg_loss = np.mean([t.realized_pnl for t in losing]) if losing else 0.0
        
        # Profit factor
        gross_profit = sum(t.realized_pnl for t in winning)
        gross_loss = abs(sum(t.realized_pnl for t in losing))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Total return
        total_pnl = sum(t.realized_pnl for t in closed_trades)
        self.total_return = (total_pnl / self.config.initial_capital) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "symbol": self.config.symbol,
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "initial_capital": self.config.initial_capital,
                "commission_per_trade": self.config.commission_per_trade
            },
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": self.equity_curve,
            "metrics": {
                "total_return_pct": round(self.total_return, 2),
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": round(self.win_rate * 100, 2),
                "avg_win": round(self.avg_win, 2),
                "avg_loss": round(self.avg_loss, 2),
                "profit_factor": round(self.profit_factor, 2),
                "sharpe_ratio": round(self.sharpe_ratio, 2),
                "max_drawdown": round(self.max_drawdown, 2),
                "max_drawdown_pct": round(self.max_drawdown_pct, 2)
            }
        }


class BacktestRunner:
    """Historical replay engine for backtesting."""
    
    def __init__(
        self,
        config: BacktestConfig,
        data_loader: Optional[Callable] = None,
        bias_engine: Optional[Callable] = None,
        risk_engine: Optional[Callable] = None,
        game_engine: Optional[Callable] = None,
        entry_validator: Optional[Callable] = None
    ):
        """Initialize backtest runner.
        
        Args:
            config: Backtest configuration
            data_loader: Function to load historical data
            bias_engine: Layer 1 bias engine
            risk_engine: Layer 2 risk engine
            game_engine: Layer 3 game engine
            entry_validator: Entry validation function
        """
        self.config = config
        self.data_loader = data_loader
        self.bias_engine = bias_engine
        self.risk_engine = risk_engine
        self.game_engine = game_engine
        self.entry_validator = entry_validator
        
        # Backtest state
        self.current_capital: float = config.initial_capital
        self.equity: float = config.initial_capital
        self.open_position: Optional[TradeRecord] = None
        self.trade_counter: int = 0
        self.equity_history: List[Dict[str, Any]] = []
        
        # Results
        self.result = BacktestResult(config=config)
        
        logger.info(f"BacktestRunner initialized for {config.symbol}")
        logger.info(f"Period: {config.start_date.date()} to {config.end_date.date()}")
    
    def load_data(self) -> pd.DataFrame:
        """Load historical data for backtest period.
        
        Returns:
            DataFrame with OHLCV data
        """
        if self.data_loader:
            return self.data_loader(
                self.config.symbol,
                self.config.start_date,
                self.config.end_date,
                self.config.timeframe
            )
        
        # Default: generate mock data for testing
        return self._generate_mock_data()
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """Generate mock OHLCV data for testing."""
        np.random.seed(42)
        
        # Generate date range
        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='5min'
        )
        
        # Filter trading hours (9:30 - 16:00 EST)
        dates = dates[
            (dates.hour >= 9) & 
            (dates.hour < 16) & 
            (~dates.weekday.isin([5, 6]))  # Exclude weekends
        ]
        
        n = len(dates)
        
        # Generate price series with trend and noise
        returns = np.random.normal(0.0001, 0.001, n)
        prices = 21900 * np.exp(np.cumsum(returns))
        
        # Generate OHLC
        noise = np.random.normal(0, 5, n)
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices + noise,
            'high': prices + abs(noise) + np.random.uniform(2, 10, n),
            'low': prices - abs(noise) - np.random.uniform(2, 10, n),
            'close': prices,
            'volume': np.random.uniform(100000, 500000, n)
        })
        
        df.set_index('timestamp', inplace=True)
        return df
    
    def run(self) -> BacktestResult:
        """Execute the backtest.
        
        Returns:
            BacktestResult with all trades and metrics
        """
        logger.info("Starting backtest...")
        
        # Load data
        data = self.load_data()
        logger.info(f"Loaded {len(data)} bars of data")
        
        if self.config.mode == BacktestMode.EVENT_DRIVEN:
            self._run_event_driven(data)
        else:
            self._run_vectorized(data)
        
        # Calculate final metrics
        self.result.calculate_metrics()
        self._calculate_sharpe_and_drawdown()
        
        logger.info(f"Backtest complete: {self.result.total_trades} trades, "
                   f"{self.result.win_rate*100:.1f}% win rate")
        
        return self.result
    
    def _run_event_driven(self, data: pd.DataFrame):
        """Run event-driven backtest (tick-by-tick)."""
        for timestamp, bar in data.iterrows():
            self._process_bar(timestamp, bar)
            self._record_equity(timestamp)
    
    def _run_vectorized(self, data: pd.DataFrame):
        """Run vectorized backtest (faster, less realistic)."""
        # Simplified implementation - process bars in batch
        for timestamp, bar in data.iterrows():
            self._process_bar(timestamp, bar)
            self._record_equity(timestamp)
    
    def _process_bar(self, timestamp: datetime, bar: pd.Series):
        """Process a single price bar.
        
        Args:
            timestamp: Bar timestamp
            bar: OHLCV data
        """
        # Check for exits if in position
        if self.open_position:
            self._check_exits(timestamp, bar)
        
        # Look for entries if not in position
        else:
            self._check_entry(timestamp, bar)
    
    def _check_entry(self, timestamp: datetime, bar: pd.Series):
        """Check for entry signals."""
        if not all([self.bias_engine, self.risk_engine, self.game_engine]):
            return
        
        try:
            # Create market data snapshot
            market_data = {
                'timestamp': timestamp,
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar.get('volume', 0)
            }
            
            # Run through layers
            bias = self.bias_engine(self.config.symbol, market_data)
            game = self.game_engine(self.config.symbol, market_data, bias)
            risk = self.risk_engine(self.config.symbol, bias, None, None, market_data)
            
            # Check for valid entry
            if bias and risk and game:
                if (bias.confidence >= 0.55 and 
                    risk.ev_positive and
                    game.adversarial_risk.value != "EXTREME"):
                    
                    self._open_position(timestamp, bar, bias, risk, game)
                    
        except Exception as e:
            logger.error(f"Error checking entry: {e}")
    
    def _open_position(
        self,
        timestamp: datetime,
        bar: pd.Series,
        bias: BiasOutput,
        risk: RiskOutput,
        game: GameOutput
    ):
        """Open a new position."""
        self.trade_counter += 1
        trade_id = f"{self.config.symbol}_{self.trade_counter:04d}"
        
        # Apply slippage
        entry_price = bar['close']
        if self.config.slippage_model == "random":
            slippage = np.random.uniform(0, self.config.spread_pips)
            entry_price += slippage if bias.direction == Direction.LONG else -slippage
        
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=self.config.symbol,
            direction=bias.direction,
            entry_time=timestamp,
            entry_price=entry_price,
            position_size=risk.position_size,
            stop_price=risk.stop_price,
            tp1_price=risk.tp1_price,
            tp2_price=risk.tp2_price,
            commission=self.config.commission_per_trade,
            slippage=entry_price - bar['close']
        )
        
        self.open_position = trade
        self.current_capital -= trade.commission
        
        logger.debug(f"Opened {trade.direction.name} position at {entry_price}")
    
    def _check_exits(self, timestamp: datetime, bar: pd.Series):
        """Check if position should be closed."""
        if not self.open_position:
            return
        
        trade = self.open_position
        exit_triggered = False
        exit_price = None
        exit_reason = ""
        
        # Check stop loss
        if trade.direction == Direction.LONG:
            if bar['low'] <= trade.stop_price:
                exit_triggered = True
                exit_price = max(bar['open'], trade.stop_price)
                exit_reason = "stop_loss"
            elif bar['high'] >= trade.tp2_price:
                exit_triggered = True
                exit_price = trade.tp2_price
                exit_reason = "tp2"
            elif bar['high'] >= trade.tp1_price:
                # Partial exit at TP1 - for simplicity, close full position
                exit_triggered = True
                exit_price = trade.tp1_price
                exit_reason = "tp1"
        else:  # SHORT
            if bar['high'] >= trade.stop_price:
                exit_triggered = True
                exit_price = min(bar['open'], trade.stop_price)
                exit_reason = "stop_loss"
            elif bar['low'] <= trade.tp2_price:
                exit_triggered = True
                exit_price = trade.tp2_price
                exit_reason = "tp2"
            elif bar['low'] <= trade.tp1_price:
                exit_triggered = True
                exit_price = trade.tp1_price
                exit_reason = "tp1"
        
        if exit_triggered:
            self._close_position(timestamp, exit_price, exit_reason)
    
    def _close_position(self, timestamp: datetime, exit_price: float, exit_reason: str):
        """Close the open position."""
        if not self.open_position:
            return
        
        trade = self.open_position
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Calculate PnL
        if trade.direction == Direction.LONG:
            pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            pnl = (trade.entry_price - exit_price) * trade.position_size
        
        trade.realized_pnl = pnl - trade.commission
        
        # Update capital
        self.current_capital += pnl - trade.commission
        
        # Save trade
        self.result.trades.append(trade)
        
        logger.debug(f"Closed position: {exit_reason} at {exit_price}, PnL: {pnl:.2f}")
        
        self.open_position = None
    
    def _record_equity(self, timestamp: datetime):
        """Record equity at this point in time."""
        equity = self.current_capital
        
        # Add unrealized PnL
        if self.open_position:
            # Will be calculated from last known price
            pass
        
        self.equity_history.append({
            'timestamp': timestamp.isoformat(),
            'equity': equity,
            'cash': self.current_capital
        })
    
    def _calculate_sharpe_and_drawdown(self):
        """Calculate Sharpe ratio and max drawdown."""
        if not self.equity_history:
            return
        
        equity_values = [e['equity'] for e in self.equity_history]
        
        # Max drawdown
        peak = equity_values[0]
        max_dd = 0
        max_dd_pct = 0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak) * 100 if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        self.result.max_drawdown = max_dd
        self.result.max_drawdown_pct = max_dd_pct
        
        # Sharpe ratio (simplified - assuming daily returns)
        if len(equity_values) > 1:
            returns = pd.Series(equity_values).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                self.result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        
        self.result.equity_curve = self.equity_history
    
    def save_results(self, filepath: str):
        """Save backtest results to file.
        
        Args:
            filepath: Path to save results
        """
        results = self.result.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def print_summary(self):
        """Print backtest summary to console."""
        print("\n" + "=" * 60)
        print(f"BACKTEST RESULTS: {self.config.symbol}")
        print("=" * 60)
        print(f"Period: {self.config.start_date.date()} to {self.config.end_date.date()}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print("-" * 60)
        print("PERFORMANCE METRICS:")
        print(f"  Total Return:      {self.result.total_return:+.2f}%")
        print(f"  Total Trades:      {self.result.total_trades}")
        print(f"  Win Rate:          {self.result.win_rate*100:.1f}%")
        print(f"  Avg Win:           ${self.result.avg_win:,.2f}")
        print(f"  Avg Loss:          ${self.result.avg_loss:,.2f}")
        print(f"  Profit Factor:     {self.result.profit_factor:.2f}")
        print(f"  Sharpe Ratio:      {self.result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown:      ${self.result.max_drawdown:,.2f} ({self.result.max_drawdown_pct:.1f}%)")
        print("=" * 60 + "\n")


def run_simple_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    data: Optional[pd.DataFrame] = None
) -> BacktestResult:
    """Run a simple backtest with mock engines.
    
    Args:
        symbol: Trading symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data: Optional historical data
        
    Returns:
        BacktestResult
    """
    config = BacktestConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    def mock_bias_engine(symbol, market_data):
        import random
        close = market_data.get('close', 100)
        # Simple trend-following mock
        direction = Direction.LONG if random.random() > 0.5 else Direction.SHORT
        return BiasOutput(
            direction=direction,
            magnitude=Magnitude.NORMAL,
            confidence=0.6 + random.random() * 0.3,
            regime_override=False,
            rationale=["TREND_STRENGTH"],
            model_version="mock",
            feature_snapshot={}
        )
    
    def mock_risk_engine(symbol, bias, features, account, market_data):
        close = market_data.get('close', 100)
        atr = close * 0.002
        direction_mult = 1 if bias.direction == Direction.LONG else -1
        
        return RiskOutput(
            position_size=1.0,
            kelly_fraction=0.25,
            stop_price=close - (atr * 2 * direction_mult),
            stop_method="atr",
            tp1_price=close + (atr * 2 * direction_mult),
            tp2_price=close + (atr * 3.5 * direction_mult),
            trail_config={},
            expected_value=0.5,
            ev_positive=True,
            size_breakdown={}
        )
    
    def mock_game_engine(symbol, market_data, bias):
        from contracts.types import LiquidityPool, TrappedPositions
        return GameOutput(
            liquidity_map={'equal_highs': [], 'equal_lows': []},
            nearest_unswept_pool=None,
            trapped_positions=TrappedPositions(
                trapped_longs=[], trapped_shorts=[],
                total_long_pain=0, total_short_pain=0, squeeze_probability=0.0
            ),
            forced_move_probability=0.5,
            nash_zones=[],
            kyle_lambda=0.3,
            game_state_aligned=True,
            game_state_summary="NEUTRAL",
            adversarial_risk=AdversarialRisk.LOW
        )
    
    runner = BacktestRunner(
        config=config,
        data_loader=lambda s, start, end, tf: data if data is not None else pd.DataFrame(),
        bias_engine=mock_bias_engine,
        risk_engine=mock_risk_engine,
        game_engine=mock_game_engine
    )
    
    return runner.run()
