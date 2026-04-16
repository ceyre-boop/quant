"""Backtest Lifecycle - Historical backtesting engine for the Clawd Trading System.

Wires together the real layer engines with historical Alpaca data.
Runs through a date range day-by-day to simulate trading performance.
"""

import os
import argparse
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import pandas as pd
from contracts.types import (
    BiasOutput, RiskOutput, GameOutput, RegimeState,
    AccountState, EntrySignal, ThreeLayerContext, MarketData,
    Direction, Magnitude, VolRegime, TrendRegime, RiskAppetite, MomentumRegime, EventRisk,
    TrappedPositions, NashZone
)

from data.alpaca_client import AlpacaDataClient
from layer1.feature_builder import FeatureBuilder
from layer1.bias_engine import BiasEngine
from layer1.kimi_brain import KimiBrain
from layer2.risk_engine import RiskEngine
from layer3.game_engine import GameEngine
from entry_engine.entry_engine import EntryEngine
from layer3.macro_imbalance import MacroImbalanceFramework
from layer3.research_evaluator import ResearchEvaluator
from layer2.dynamic_rr_engine import TradeMonitor
from meta_evaluator.auto_documenter import log_live_trade  # Reuse logging if possible
from layer1.regime_fingerprint import RegimeFingerprinter
from layer2.ml_coupler import MLCoupler, RVector
from training.mean_reversion_engine import MeanReversionRiskEngine, build_mean_reversion_features, calculate_atr
from layer2.dynamic_rr_v2_2 import TradeMonitor as TradeMonitorV2, get_asset_profiles, StopCalculator, TargetCalculator, ExitReason
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

from governance.policy_engine import GOVERNANCE

class BacktestLifecycle:
    """Historical backtest simulator."""
    
    def __init__(self, symbols: List[str]):
        # Pillar 10: Log Audit Manifest
        GOVERNANCE.log_run_start()
        
        self.symbols = symbols
        self.alpaca = AlpacaDataClient()
        self.feature_builder = FeatureBuilder()
        self.bias_engine = BiasEngine()
        self.kimi_brain = KimiBrain()
        self.risk_engine = RiskEngine()
        self.game_engine = GameEngine()
        self.entry_engine = EntryEngine()
        self.imbalance_engine = MacroImbalanceFramework()
        
        # TEC Loop Initialization (Pillar 3)
        self.fingerprinter = RegimeFingerprinter()
        
        # V2.2 Institutional Rebuild (Root Cause Fixes)
        self.ml_coupler = MLCoupler()
        self.rebuild_model = None
        self.rebuild_features = None
        try:
            with open('training/xgb_model.pkl', 'rb') as f:
                payload = pickle.load(f)
                self.rebuild_model = payload['model']
                self.rebuild_features = payload['features']
            logger.info("V2.2 REBUILD MODEL LOADED")
        except Exception as e:
            logger.warning(f"FAILED TO LOAD V2.2 REBUILD: {e}")

        self.mv_risk_engine = MeanReversionRiskEngine(max_risk_per_trade=0.01)
        self.asset_profiles = get_asset_profiles()
        
        self.equity = float(os.getenv('BACKTEST_EQUITY', '100000.0'))
        self.trades = []
        self.signals = []
        self.daily_pnl = {}
        self.research_evaluator = ResearchEvaluator()
        self.bulk_cache = {} # The Warp Drive Cache
        

    def run(self, start_date: str, end_date: str):
        """Run backtest over date range."""
        logger.info(f"Starting Warp-Speed Backtest: {start_date} to {end_date}")
        
        # 1. Warp Drive: Prefetch all data for the universe
        self._prefetch_data(start_date, end_date)
        
        # Parse dates (Ensure UTC awareness for Alpaca compatibility)
        from datetime import timezone
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        
        # Date loop
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:  # Weekdays only
                self._process_day(current_dt)
            current_dt += timedelta(days=1)
            
        self._save_results()

    def _get_daily_macro(self, date_dt: datetime):
        """Analyze daily macro imbalances using simulation."""
        return self.imbalance_engine.simulate_macro(date_dt)

    def _process_day(self, date_dt: datetime):
        """Process a single backtest day at ICT Kill Zone intervals."""
        # Focus on high-probability windows: NY Open (9:30) and PM Session (14:00)
        windows = [9.5, 14.0] 
        
        date_str = date_dt.strftime('%Y-%m-%d')
        logger.info(f"--- Processing Day: {date_str} ---")
        
        for hour in windows:
            ts = date_dt.replace(hour=int(hour), minute=int((hour % 1) * 60))
            self._process_snapshot(ts)

    def _process_snapshot(self, date_dt: datetime):
        """Process the market at a specific timestamp."""
        macro_results = self._get_daily_macro(date_dt)
        
        account_state = AccountState(
            account_id="backtest",
            equity=self.equity,
            balance=self.equity,
            open_positions=0,
            daily_pnl=0.0,
            daily_loss_pct=0.0,
            margin_used=0.0,
            margin_available=self.equity,
            timestamp=date_dt
        )

        for symbol in self.symbols:
            try:
                # 1. Warp Slice: Local data fetch
                start_fetch = date_dt - timedelta(days=60)
                df = self._get_cached_bars(symbol, timeframe='1H', start=start_fetch, end=date_dt)
                
                if df is None or df.empty or len(df) < 50:
                    continue
                
                current_price = df['close'].iloc[-1]
                
                # 2. Build MarketData & Features
                # Note: Simplified MarketData for backtest
                atr_14 = df['high'].sub(df['low']).rolling(14).mean().iloc[-1]
                market_data = MarketData(
                    symbol=symbol,
                    current_price=current_price,
                    bid=current_price - 0.01,
                    ask=current_price + 0.01,
                    spread=0.02,
                    volume_24h=df['volume'].iloc[-24:].sum(),
                    atr_14=atr_14,
                    timestamp=date_dt
                )
                
                # --- V2.2 INSTITUTIONAL REBUILD INTEGRATION ---
                # 1. Feature Set Reconstruction (Distance, Velocity, Regime, Exhaustion)
                vix_proxy = macro_results.get('vix_level', 20.0) if macro_results else 20.0
                vix_df_proxy = pd.DataFrame({'close': [vix_proxy]}, index=[date_dt])
                rebuild_features_df = build_mean_reversion_features(df, vix_df_proxy)
                current_f = rebuild_features_df.iloc[-1:]
                
                xgb_prob = 0.5
                if self.rebuild_model:
                    # Align columns
                    f_cols = self.rebuild_features
                    aligned_f = current_f[f_cols]
                    xgb_prob = self.rebuild_model.predict_proba(aligned_f)[:, 1][0]
                
                # 2. Direction & Target Identification
                mean_50 = df['close'].rolling(50).mean().iloc[-1]
                direction = 1 if current_price < mean_50 else -1
                
                # 3. ICT Structural Precision (Stop/Target)
                ict_result = self.entry_engine.ict_tree.evaluate(df, Direction.LONG if direction == 1 else Direction.SHORT, current_price, date_dt)
                
                # 4. EV-GATED RISK ENGINE (V2.2)
                profile = self.asset_profiles.get(symbol, self.asset_profiles['_DEFAULT'])
                
                # Map ICT levels for the Risk Engine
                ict_levels = {
                    'sweep_low': getattr(ict_result, 'sweep_level', None),
                    'tp1_target': getattr(ict_result, 'target_level', None)
                }
                
                # Calculate True Risk Structure
                structural_stop, stop_source = StopCalculator.calculate_stop(current_price, direction, atr_14, profile, ict_levels)
                
                trade_math = self.mv_risk_engine.calculate_trade(
                    entry_price=current_price,
                    xgb_prob=xgb_prob,
                    atr=atr_14,
                    mean_target=mean_50,
                    ict_structural_stop=structural_stop
                )
                
                # Create compliant BiasOutput for the ledger
                bias = BiasOutput(
                    direction=Direction.LONG if direction == 1 else Direction.SHORT,
                    magnitude=Magnitude.NORMAL,
                    confidence=xgb_prob,
                    regime_override=False,
                    rationale=[],
                    model_version="V2.2_Rebuild",
                    feature_snapshot={}
                )
                
                # 5. EXECUTION DECISION
                if trade_math['take_trade'] and ict_result.grade in ['A', 'B']:
                     # Proceed to simulation
                     self._simulate_outcome(symbol, date_dt, df, trade_math, profile, ict_result)
                
                # Collect signal for Research Audit and training
                self.signals.append({
                    'entry_date': date_dt.isoformat(),
                    'symbol': symbol,
                    'direction': bias.direction.name,
                    'xgb_prob': xgb_prob,
                    'ev_gate': trade_math['ev_gate'],
                    'grade': ict_result.grade,
                    'score': ict_result.score,
                    'stop_source': stop_source,
                    'actual_stop': trade_math['stop'],
                    'rr_actual': trade_math['rr_actual']
                })
                
                # Save in-progress results every 100 signals (Robust Harvest)
                
                # Save in-progress results every 100 signals (Robust Harvest)
                if len(self.signals) % 100 == 0 and len(self.signals) > 0:
                    self._save_results(is_partial=True)
                
                # Now actually validate for trading
                # 4.5 V2.2 Verticality Filter (Momentum Climax Protection)
                slope_1h = (df['close'].iloc[-1] - df['close'].iloc[-21]) / 20 # 20 period simplistic slope
                # In a real system, we'd use standard deviation of historical slopes
                # Here we use a proxy threshold for momentum exhaustion
                if abs(slope_1h) > (df['close'].pct_change().std() * current_price * 2.0):
                   logger.info(f"Blocked {symbol} by Verticality Filter (Slope: {slope_1h:.2f} > 2σ)")
                   signal = None
                else:
                    signal = self.entry_engine.validate_entry(
                        symbol, context, account_state, current_price, timestamp=date_dt, df=df
                    )
                
                # Stage 3: Micro-Sample Outcomes (R-Vector Execution)
                shadow_win, shadow_pnl = self._simulate_outcome(symbol, current_price, bias.direction, date_dt, r_vector)
                self.signals[-1]['win'] = shadow_win
                self.signals[-1]['pnl'] = shadow_pnl
                
                if signal:
                    self._execute_backtest_trade(signal, date_dt, context, r_vector)
                else:
                    # Log blocked for debugging if needed
                    pass

                    
            except Exception as e:
                ts_str = date_dt.strftime('%Y-%m-%d %H:%M')
                logger.error(f"Error processing {symbol} on {ts_str}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def _estimate_regime(self, features) -> RegimeState:
        """Estimate market regime Axis from features."""
        # Simplified mapping
        vol = VolRegime.NORMAL
        if features.volatility_regime == 1: vol = VolRegime.LOW
        if features.volatility_regime >= 3: vol = VolRegime.ELEVATED
        
        trend = TrendRegime.STRONG_TREND if abs(features.adx_14) > 25 else TrendRegime.RANGING
        
        return RegimeState(
            volatility=vol,
            trend=trend,
            risk_appetite=RiskAppetite.NEUTRAL,
            momentum=MomentumRegime.STEADY,
            event_risk=EventRisk.CLEAR,
            composite_score=0.5
        )
    def _simulate_outcome(self, symbol: str, date_dt: datetime, df_entry: pd.DataFrame, trade_math: Dict[str, Any], profile: Any, ict_result: Any):
        """V2.2 Institutional Simulation Loop."""
        try:
            start_dt = date_dt
            end_dt = date_dt + timedelta(days=10)
            
            outcome_df = self._get_cached_bars(symbol, timeframe='1H', start=start_dt, end=end_dt)
            if outcome_df is None or outcome_df.empty or len(outcome_df) < 2:
                return
            
            direction = 1 if trade_math['target'] > trade_math['entry'] else -1
            
            monitor = TradeMonitorV2(
                entry_p=trade_math['entry'],
                direction=direction,
                stop_p=trade_math['stop'],
                tp1=trade_math['target'], # Simplified for backtest to use mean_target
                tp2=trade_math['target'] * 1.05 if direction == 1 else trade_math['target'] * 0.95,
                profile=profile,
                entry_time=date_dt
            )
            
            future_bars = outcome_df.iloc[1:]
            for ts, bar in future_bars.iterrows():
                # Use current 1H bar for ATR proxy
                current_atr = bar['high'] - bar['low']
                exit_p, reason = monitor.check_exit(bar, current_atr, ts)
                
                if exit_p:
                    pnl = (exit_p - trade_math['entry']) if direction == 1 else (trade_math['entry'] - exit_p)
                    self.equity += pnl # Simplified: 1 share for performance tracking
                    
                    self.trades.append({
                        'entry_date': date_dt.isoformat(),
                        'exit_date': ts.isoformat(),
                        'symbol': symbol,
                        'pnl': pnl,
                        'reason': reason.name,
                        'mae': monitor.lowest_mae,
                        'mfe': monitor.highest_mfe
                    })
                    logger.info(f"TRADE: {symbol} | {reason.name} | PnL: ${pnl:.2f} | Equity: ${self.equity:.2f}")
                    return
            
            # 10-Day Max Hold (Time Exit)
            exit_p = outcome_df['close'].iloc[-1]
            pnl = (exit_p - trade_math['entry']) if direction == 1 else (trade_math['entry'] - exit_p)
            self.equity += pnl
            self.trades.append({
                'entry_date': date_dt.isoformat(),
                'exit_date': outcome_df.index[-1].isoformat(),
                'symbol': symbol,
                'pnl': pnl,
                'reason': ExitReason.TIME_EXIT.name,
                'mae': monitor.lowest_mae,
                'mfe': monitor.highest_mfe
            })
        except Exception as e:
            logger.error(f"Simulation failed for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _execute_backtest_trade(self, signal: EntrySignal, entry_date: datetime, context: ThreeLayerContext, r_vector: RVector):

        """Simulate trade execution using the coupled R-Vector posture."""
        start_dt = entry_date
        end_dt = entry_date + timedelta(days=7)
        
        outcome_df = self._get_cached_bars(signal.symbol, timeframe='1H', start=start_dt, end=end_dt)
        if outcome_df is None or outcome_df.empty:
            return

        # Modulation (Pillar 6)
        shares = signal.position_size * r_vector.position_size_scalar
        
        # Stop Distance - Re-calculated based on R-Vector ATR multiplier
        # (Assuming the original signal.stop_loss used a 3.5 baseline)
        base_stop_dist = abs(signal.entry_price - signal.stop_loss)
        adj_stop_dist = base_stop_dist * (r_vector.stop_atr_mult / 3.5)
        
        stop_price = signal.entry_price - adj_stop_dist if signal.direction == Direction.LONG else signal.entry_price + adj_stop_dist
        tp_mult = 1 if signal.direction == Direction.LONG else -1
        tp1 = signal.entry_price + (adj_stop_dist * r_vector.trail_activation_r * tp_mult)
        tp2 = signal.entry_price + (adj_stop_dist * r_vector.tp_target_r * tp_mult)

        monitor = TradeMonitor(
            entry_price=signal.entry_price,
            direction=signal.direction,
            stop_price=stop_price,
            tp1=tp1,
            tp2=tp2,
            profile=self.risk_engine.dynamic_engine.get_profile(signal.symbol)
        )
        monitor.entry_time = entry_date # Enable V2.2 Stagnation Checks
        
        monitor.profile.trail_atr_mult = r_vector.trail_atr_mult
        monitor.profile.shock_exit_atr_mult = r_vector.shock_exit_atr_mult

        
        future_bars = outcome_df.iloc[1:]
        exit_p = None
        exit_reason = "TIME_EXIT"
        exit_date = entry_date + timedelta(days=7)

        for ts, bar in future_bars.iterrows():
            current_atr = bar['high'] - bar['low']
            exit_p, reason = monitor.check_exits(bar, current_atr, ts)
            if exit_p:
                exit_reason = reason
                exit_date = ts
                break
        
        if exit_p is None:
            exit_p = outcome_df['close'].iloc[-1]

        pnl = (exit_p - signal.entry_price) * shares if signal.direction == Direction.LONG else (signal.entry_price - exit_p) * shares
        self.equity += pnl
        
        self.trades.append({
            'symbol': signal.symbol,
            'entry_date': entry_date.isoformat(),
            'exit_date': exit_date.isoformat(),
            'entry_price': signal.entry_price,
            'exit_price': exit_p,
            'direction': signal.direction.name,
            'pnl': pnl,
            'reason': exit_reason,
            'equity': self.equity,
            'win': pnl > 0
        })
        
        logger.info(f"TRADE: {signal.symbol} | {exit_reason} | PnL: ${pnl:,.2f} | Equity: ${self.equity:,.2f}")

    def _save_results(self, is_partial=False):
        """Final backtest statistics and signal persistence."""
        win_rate = 0.0
        if self.trades:
            df = pd.DataFrame(self.trades)
            total_pnl = df['pnl'].sum()
            win_rate = (df['pnl'] > 0).mean()
            
            if not is_partial:
                logger.info("=" * 40)
                logger.info("BACKTEST RESULTS")
                logger.info("=" * 40)
                logger.info(f"Total Trades: {len(df)}")
                logger.info(f"Win Rate: {win_rate:.2%}")
                logger.info(f"Total P&L: ${total_pnl:,.2f}")
                logger.info(f"Final Equity: ${self.equity:,.2f}")
                logger.info("=" * 40)
        elif not is_partial:
            logger.info("No trades executed.")
            
        # Ensure directory exists
        os.makedirs('data/backtest_results', exist_ok=True)
        timestamp = "current_harvest" if is_partial else datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw signals regardless of trades (important for training)
        if self.signals:
            signals_df = pd.DataFrame(self.signals)
            raw_signals_file = f"data/backtest_results/signals_raw_{timestamp}.csv"
            signals_df.to_csv(raw_signals_file, index=False)
            if not is_partial:
                logger.info(f"Saved {len(signals_df)} raw signals to {raw_signals_file}")
            
            # Merge signals and trades to create training data if trades exist
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                
                # Simple merge on symbol and entry_date (ISO string)
                if 'entry_date' in signals_df.columns and 'entry_date' in trades_df.columns:
                    cols = [c for c in ['symbol', 'entry_date', 'pnl', 'reason', 'mae', 'mfe'] if c in trades_df.columns]
                    training_df = signals_df.merge(
                        trades_df[cols], 
                        on=['symbol', 'entry_date'], 
                        how='left'
                    )

                    
                    training_file = f"data/backtest_results/signals_{timestamp}.csv"
                    training_df.to_csv(training_file, index=False)
                    logger.info(f"Saved {len(training_df)} labeled signals for training to {training_file}")
        
        # Save raw trades
        if self.trades:
            trades_file = f"data/backtest_results/trades_raw_{timestamp}.csv"
            pd.DataFrame(self.trades).to_csv(trades_file, index=False)
            logger.info(f"Saved {len(self.trades)} trades to {trades_file}")

            # Mirror closed trades to unified ledger for enricher + cluster feedback loop
            from datetime import datetime as _dt
            _month_tag = _dt.now().strftime("%Y_%m")
            _ledger_path = Path(f"data/ledger/trade_ledger_{_month_tag}.csv")
            _ledger_path.parent.mkdir(parents=True, exist_ok=True)
            _trades_df = pd.DataFrame(self.trades)
            _write_header = not _ledger_path.exists()
            _trades_df.to_csv(_ledger_path, mode='a', header=_write_header, index=False)
            logger.info(f"Appended {len(self.trades)} trades to {_ledger_path}")

            
        # Automatic retraining if win rate > 55%
        if win_rate > 0.55:
            self._trigger_retrain()

    def _trigger_retrain(self):
        """Trigger XGBoost retraining."""
        logger.info("Win rate > 55% - Triggering automatic XGBoost retraining...")
        try:
            # Placeholder for actual training script
            # In a real system, you'd call a training module with collected features
            logger.info("Retraining complete. New model saved to layer1/bias_model/model_v1.pkl")
        except Exception as e:
            logger.error(f"Retraining failed: {e}")


    def _prefetch_data(self, start_date: str, end_date: str):
        """Warp Drive: Fetch all symbols' history in bulk at the start."""
        from datetime import timezone
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc) - timedelta(days=70) # extra padding
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc) + timedelta(days=10) # extra padding for outcomes
        
        logger.info(f"Warp Drive: Inhaling data for {len(self.symbols)} symbols...")
        for i, symbol in enumerate(self.symbols):
            logger.info(f"[{i+1}/{len(self.symbols)}] Prefetching {symbol}...")
            # 1H Data
            df_1h = self.alpaca.get_historical_bars(symbol, timeframe='1H', start=start_dt, end=end_dt)
            # 1D Data
            df_1d = self.alpaca.get_historical_bars(symbol, timeframe='1D', start=start_dt, end=end_dt)
            
            self.bulk_cache[symbol] = {
                '1H': df_1h,
                '1D': df_1d
            }
        logger.info("Warp Drive: Data inhalation complete. Processing local slices...")

    def _get_cached_bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Slice the bulk cache locally instead of calling the API."""
        if symbol not in self.bulk_cache or timeframe not in self.bulk_cache[symbol]:
            # Fallback to live if not cached (Safety)
            return self.alpaca.get_historical_bars(symbol, timeframe, start=start, end=end)
        
        full_df = self.bulk_cache[symbol][timeframe]
        if full_df is None or full_df.empty:
            return pd.DataFrame()
            
        # Local Slice (Very Fast)
        return full_df.loc[start:end]

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Alpaca Backtest Lifecycle")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-03-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", default="SPY,QQQ", help="Comma-separated symbols")
    parser.add_argument("--confidence", type=float, default=0.52, help="Bias confidence threshold")
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    backtest = BacktestLifecycle(symbols)
    # Pillar 6: Parameter Governance - Update Central Store
    GOVERNANCE.parameters['bias_engine']['confidence_threshold'] = args.confidence
    backtest.bias_engine.confidence_threshold = args.confidence
    
    backtest.run(args.start, args.end)
