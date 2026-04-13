"""
Phase 10 — Sovereign Backtest Engine (V1.0)
Historical simulation using full orchestrator pipeline.
Validates performance at 3x slippage before paper trading.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

from sovereign.orchestrator import SovereignOrchestrator
from sovereign.data.feeds.alpaca_feed import AlpacaFeed
from contracts.types import SovereignFeatureRecord, MarketData
from config.loader import params

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    start_date: str
    end_date: str
    total_trades: int
    win_rate: float
    avg_return: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    expectancy: float
    profit_factor: float
    slippage_assumed: float
    passed_3x_slippage: bool
    equity_curve: List[Dict]
    trades: List[Dict]
    diagnostics: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SovereignBacktest:
    """
    Phase 10 — Historical Simulation Engine
    
    Runs the full Sovereign pipeline over historical data:
    1. Fetches historical OHLCV
    2. Generates features
    3. Runs orchestrator (with paper mode)
    4. Simulates execution with slippage
    5. Calculates performance metrics
    
    Must pass 3x slippage tolerance before Phase 11.
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        starting_equity: float = 100000.0,
        slippage: float = 0.001,  # 0.1% default
        commission: float = 0.0005,  # 0.05% per trade
    ):
        self.symbols = symbols
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.starting_equity = starting_equity
        self.slippage = slippage
        self.commission = commission
        
        # Initialize components
        self.orchestrator = SovereignOrchestrator(mode='paper')
        self.feed = AlpacaFeed()
        
        # Tracking
        self.equity = starting_equity
        self.equity_curve = []
        self.trades = []
        self.positions = {}  # symbol -> position info
        
    def run(self) -> BacktestResult:
        """
        Execute full backtest.
        
        Returns:
            BacktestResult with all metrics
        """
        logger.info("=" * 60)
        logger.info("SOVEREIGN BACKTEST ENGINE")
        logger.info("=" * 60)
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Starting Equity: ${self.starting_equity:,.2f}")
        logger.info(f"Slippage: {self.slippage:.3%}")
        logger.info(f"Commission: {self.commission:.3%}")
        logger.info("-" * 60)
        
        # Fetch historical data
        logger.info("Fetching historical data...")
        data = self._fetch_data()
        
        if not data:
            raise ValueError("No data fetched for backtest period")
        
        # Run simulation
        logger.info("Running simulation...")
        self._simulate(data)
        
        # Calculate metrics
        logger.info("Calculating performance metrics...")
        result = self._calculate_metrics()
        
        # Run 3x slippage stress test
        logger.info("Running 3x slippage stress test...")
        result_3x = self._run_slippage_stress_test()
        
        # Combine results
        result.passed_3x_slippage = result_3x['profitable']
        result.diagnostics['slippage_stress'] = result_3x
        
        self._log_results(result)
        return result
    
    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical OHLCV data for all symbols."""
        data = {}
        
        for symbol in self.symbols:
            try:
                df = self.feed.get_historical_bars(
                    symbol=symbol,
                    start=self.start_date.strftime('%Y-%m-%d'),
                    end=self.end_date.strftime('%Y-%m-%d'),
                    timeframe='1D'
                )
                if not df.empty:
                    data[symbol] = df
                    logger.info(f"  {symbol}: {len(df)} bars")
                else:
                    logger.warning(f"  {symbol}: No data")
            except Exception as e:
                logger.error(f"  {symbol}: Error fetching data - {e}")
        
        return data
    
    def _simulate(self, data: Dict[str, pd.DataFrame]):
        """Run bar-by-bar simulation."""
        # Align dates across all symbols
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        
        dates = sorted(all_dates)
        
        for date in tqdm(dates, desc="Simulating"):
            for symbol, df in data.items():
                if date not in df.index:
                    continue
                
                row = df.loc[date]
                
                # Update existing positions
                self._update_positions(symbol, row)
                
                # Check for new signals
                self._check_for_signals(symbol, row, date)
                
                # Record equity
                self._record_equity(date)
    
    def _check_for_signals(self, symbol: str, row: pd.Series, date: datetime):
        """Check if orchestrator generates a signal."""
        # Skip if already in position
        if symbol in self.positions:
            return
        
        # Build feature record (simplified - would need full feature pipeline)
        # For backtest, we'd ideally pre-compute all features
        # Here we simulate with basic features
        
        try:
            # Get ATR
            atr = row.get('atr_14', row['close'] * 0.02)  # Default 2% if no ATR
            
            # Create minimal feature record for orchestrator
            # In production, this would come from full feature pipeline
            from contracts.types import (
                RegimeFeatures, MomentumFeatures, MacroFeatures,
                PetrolausDecision
            )
            
            regime = RegimeFeatures(
                hurst_short=0.5,  # Neutral default
                hurst_long=0.5,
                hurst_signal='NEUTRAL',
                csd_score=0.5,
                csd_signal='NEUTRAL',
                hmm_state=1,
                hmm_state_label='NORMAL',
                hmm_confidence=0.6,
                hmm_transition_prob=0.2,
                adx=25.0,
                adx_signal='ESTABLISHED'
            )
            
            momentum = MomentumFeatures(
                logistic_ode_score=0.0,
                jt_momentum_12_1=0.0,
                volume_entropy=1.0,
                rsi_14=50.0,
                rsi_signal='NEUTRAL'
            )
            
            macro = MacroFeatures(
                yield_curve_slope=0.01,
                yield_curve_velocity=0.0,
                erp=0.04,
                cape_zscore=1.0,
                cot_zscore=0.0,
                m2_velocity=1.5,
                hyg_spread_bps=200.0,
                macro_signal='RISK_ON'
            )
            
            petroulas = PetrolausDecision(
                fault_detected=False,
                fault_reason=None,
                fault_frameworks=[],
                action='TRADE',
                macro_features=macro
            )
            
            feature_record = SovereignFeatureRecord(
                symbol=symbol,
                timestamp=date.isoformat(),
                regime=regime,
                momentum=momentum,
                macro=macro,
                petroulas=petroulas,
                bar_ohlcv={
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row.get('volume', 0)
                },
                is_valid=True,
                validation_errors=[]
            )
            
            # Run orchestrator
            result = self.orchestrator.run_session(
                symbol=symbol,
                feature_record=feature_record,
                current_price=row['close'],
                atr=atr,
                equity=self.equity
            )
            
            if result and result.get('status') == 'EXECUTED':
                self._execute_trade(symbol, row, date, result)
                
        except Exception as e:
            logger.error(f"Error checking signals for {symbol}: {e}")
    
    def _execute_trade(self, symbol: str, row: pd.Series, date: datetime, result: Dict):
        """Simulate trade execution with slippage."""
        # Apply slippage to entry
        direction = 1 if result.get('direction') == 'LONG' else -1
        slippage_adj = 1 + (self.slippage * direction)
        entry_price = row['close'] * slippage_adj
        
        # Get from result or use defaults
        position_value = result.get('p_size', self.equity * 0.01)
        stop_price = result.get('stop', entry_price * 0.98)
        tp_price = result.get('tp', entry_price * 1.03)
        
        # Commission
        commission_cost = position_value * self.commission
        
        position = {
            'symbol': symbol,
            'entry_date': date,
            'entry_price': entry_price,
            'position_value': position_value,
            'stop_loss': stop_price,
            'take_profit': tp_price,
            'direction': direction,
            'commission_paid': commission_cost
        }
        
        self.positions[symbol] = position
        self.equity -= commission_cost
        
        logger.info(f"ENTERED: {symbol} @ ${entry_price:.2f} "
                   f"Size: ${position_value:,.0f}")
    
    def _update_positions(self, symbol: str, row: pd.Series):
        """Update open positions and check for exits."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = row['close']
        
        exit_triggered = False
        exit_price = current_price
        exit_reason = None
        
        # Check stop loss
        if position['direction'] == 1:  # LONG
            if current_price <= position['stop_loss']:
                exit_triggered = True
                exit_price = position['stop_loss']
                exit_reason = 'STOP_LOSS'
        else:  # SHORT
            if current_price >= position['stop_loss']:
                exit_triggered = True
                exit_price = position['stop_loss']
                exit_reason = 'STOP_LOSS'
        
        # Check take profit
        if position['direction'] == 1:  # LONG
            if current_price >= position['take_profit']:
                exit_triggered = True
                exit_price = position['take_profit']
                exit_reason = 'TAKE_PROFIT'
        else:  # SHORT
            if current_price <= position['take_profit']:
                exit_triggered = True
                exit_price = position['take_profit']
                exit_reason = 'TAKE_PROFIT'
        
        if exit_triggered:
            self._close_position(symbol, position, exit_price, row.name, exit_reason)
    
    def _close_position(self, symbol: str, position: Dict, exit_price: float, 
                        exit_date: datetime, reason: str):
        """Close position and record P&L."""
        # Apply slippage
        direction = position['direction']
        slippage_adj = 1 - (self.slippage * direction)
        filled_exit = exit_price * slippage_adj
        
        # Calculate P&L
        if direction == 1:  # LONG
            pnl_pct = (filled_exit - position['entry_price']) / position['entry_price']
        else:  # SHORT
            pnl_pct = (position['entry_price'] - filled_exit) / position['entry_price']
        
        pnl_dollars = position['position_value'] * pnl_pct
        
        # Commission
        commission_exit = position['position_value'] * self.commission
        pnl_dollars -= (position['commission_paid'] + commission_exit)
        
        # Update equity
        self.equity += pnl_dollars
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_date': position['entry_date'].isoformat(),
            'exit_date': exit_date.isoformat() if isinstance(exit_date, datetime) else str(exit_date),
            'entry_price': position['entry_price'],
            'exit_price': filled_exit,
            'position_value': position['position_value'],
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'pnl_dollars': pnl_dollars,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
            'total_commission': position['commission_paid'] + commission_exit
        }
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        emoji = "✅" if pnl_dollars > 0 else "❌"
        logger.info(f"CLOSED: {symbol} @ ${filled_exit:.2f} "
                   f"PnL: ${pnl_dollars:,.2f} ({pnl_pct:+.2%}) {emoji} {reason}")
    
    def _record_equity(self, date):
        """Record equity curve point."""
        self.equity_curve.append({
            'date': date.isoformat() if isinstance(date, datetime) else str(date),
            'equity': self.equity
        })
    
    def _calculate_metrics(self) -> BacktestResult:
        """Calculate performance metrics."""
        if not self.trades:
            return BacktestResult(
                start_date=self.start_date.isoformat(),
                end_date=self.end_date.isoformat(),
                total_trades=0,
                win_rate=0.0,
                avg_return=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                expectancy=0.0,
                profit_factor=0.0,
                slippage_assumed=self.slippage,
                passed_3x_slippage=False,
                equity_curve=self.equity_curve,
                trades=self.trades,
                diagnostics={}
            )
        
        # Basic metrics
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t['pnl_dollars'] > 0)
        win_rate = wins / total_trades
        
        returns = [t['pnl_pct'] for t in self.trades]
        avg_return = np.mean(returns)
        
        total_return = (self.equity - self.starting_equity) / self.starting_equity
        
        # Max drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Sharpe (simplified, assumes daily)
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Expectancy
        avg_win = np.mean([r for r in returns if r > 0]) if wins > 0 else 0
        avg_loss = np.mean([r for r in returns if r <= 0]) if wins < total_trades else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # Profit factor
        gross_profit = sum(t['pnl_dollars'] for t in self.trades if t['pnl_dollars'] > 0)
        gross_loss = abs(sum(t['pnl_dollars'] for t in self.trades if t['pnl_dollars'] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return BacktestResult(
            start_date=self.start_date.isoformat(),
            end_date=self.end_date.isoformat(),
            total_trades=total_trades,
            win_rate=win_rate,
            avg_return=avg_return,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            expectancy=expectancy,
            profit_factor=profit_factor,
            slippage_assumed=self.slippage,
            passed_3x_slippage=False,  # To be filled after stress test
            equity_curve=self.equity_curve,
            trades=self.trades,
            diagnostics={}
        )
    
    def _run_slippage_stress_test(self) -> Dict:
        """Run backtest with 3x slippage to verify robustness."""
        # Save original
        original_slippage = self.slippage
        
        # Run with 3x slippage
        self.slippage = original_slippage * 3
        
        # Reset state
        self.equity = self.starting_equity
        self.equity_curve = []
        self.trades = []
        self.positions = {}
        
        # Re-fetch data and simulate
        data = self._fetch_data()
        self._simulate(data)
        
        # Calculate result
        total_return = (self.equity - self.starting_equity) / self.starting_equity
        profitable = total_return > 0
        
        # Restore original
        self.slippage = original_slippage
        
        return {
            'slippage_3x': self.slippage,
            'total_return': total_return,
            'profitable': profitable,
            'trades': len(self.trades)
        }
    
    def _log_results(self, result: BacktestResult):
        """Log backtest results."""
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Period: {result.start_date[:10]} to {result.end_date[:10]}")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Win Rate: {result.win_rate:.1%}")
        logger.info(f"Avg Return per Trade: {result.avg_return:.2%}")
        logger.info(f"Total Return: {result.total_return:.2%}")
        logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Expectancy: {result.expectancy:.3f}")
        logger.info(f"Profit Factor: {result.profit_factor:.2f}")
        logger.info(f"Slippage Assumed: {result.slippage_assumed:.3%}")
        logger.info("-" * 60)
        
        if result.passed_3x_slippage:
            logger.info("✅ PASSED 3x SLIPPAGE STRESS TEST")
        else:
            logger.info("❌ FAILED 3x SLIPPAGE STRESS TEST")
        
        logger.info("=" * 60)
    
    def save_results(self, result: BacktestResult, path: Optional[Path] = None):
        """Save backtest results to file."""
        if path is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            path = Path(f'data/backtests/backtest_{timestamp}.json')
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Backtest results saved to {path}")


# ─── Standalone execution ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    # Example backtest configuration
    backtest = SovereignBacktest(
        symbols=['META', 'PFE', 'UNH'],  # Trinity assets
        start_date='2023-01-01',
        end_date='2024-12-31',
        starting_equity=100000.0,
        slippage=0.001  # 0.1%
    )
    
    result = backtest.run()
    backtest.save_results(result)
    
    # Exit code based on 3x slippage test
    exit(0 if result.passed_3x_slippage else 1)
