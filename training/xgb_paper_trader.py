"""
XGBoost Paper Trading Bridge
Wires the trained XGBoost model into the paper trading system
"""
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.alpaca_client import AlpacaDataClient
from training.feature_generator import FeatureGenerator
from execution.paper_trading import PaperTradingEngine, PaperPosition, TradeStatus


class XGBoostEntrySignal:
    """Entry signal from XGBoost model prediction"""
    
    def __init__(self):
        self.client = AlpacaDataClient()
        self.feature_gen = FeatureGenerator()
        
        # Load trained model
        model_path = Path(__file__).parent.parent / 'training' / 'xgb_model.pkl'
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_cols = data['features']
        
        print(f"[XGBBridge] Loaded model with {len(self.feature_cols)} features")
    
    def generate_signals(self, symbols: list = None, threshold: float = 0.55) -> list:
        """
        Generate entry signals for all symbols
        
        Args:
            symbols: List of symbols (defaults to ALL_SYMBOLS)
            threshold: Min probability to trigger signal (0.55 = 55%)
            
        Returns:
            List of signal dicts
        """
        if symbols is None:
            symbols = self.client.ALL_SYMBOLS
        
        print(f"[XGBBridge] Scanning {len(symbols)} symbols...")
        
        # Fetch all data
        all_data = self.client.get_multiple_assets(symbols, timeframe='1D', days=30)
        
        signals = []
        for symbol, df in all_data.items():
            if df.empty or len(df) < 20:
                continue
            
            # Generate features
            features_df = self.feature_gen.generate_features(df, all_data)
            
            # Get latest features
            latest = features_df[self.feature_cols].iloc[-1:].values
            
            # Predict
            proba = self.model.predict_proba(latest)[0][1]  # Probability of UP
            prediction = 1 if proba >= threshold else 0
            
            if prediction == 1:  # Model predicts UP
                # Calculate entry parameters
                current_price = df['close'].iloc[-1]
                atr = features_df['atr_pct'].iloc[-1] if 'atr_pct' in features_df else 0.02
                
                # Risk-based stop (1R = 1x ATR)
                stop_distance = current_price * atr
                stop_loss = current_price - stop_distance
                
                # Take profits (1.5R, 2.5R, 3.5R)
                tp1 = current_price + (stop_distance * 1.5)
                tp2 = current_price + (stop_distance * 2.5)
                tp3 = current_price + (stop_distance * 3.5)
                
                signal = {
                    'symbol': symbol,
                    'direction': 'LONG',
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit_1': tp1,
                    'take_profit_2': tp2,
                    'take_profit_3': tp3,
                    'probability': proba,
                    'confidence': proba,  # Same as probability
                    'atr_pct': atr,
                    'model': 'XGBoost_v1'
                }
                signals.append(signal)
                print(f"[SIGNAL] {symbol} LONG @ ${current_price:.2f} (prob: {proba:.2%})")
        
        print(f"[XGBBridge] Generated {len(signals)} signals")
        return signals


class XGBoostPaperTrader:
    """Paper trader using XGBoost signals with entry/exit rules"""
    
    def __init__(self, starting_equity: float = 100000.0):
        self.signal_gen = XGBoostEntrySignal()
        self.paper_engine = PaperTradingEngine(starting_equity)
        
        # Parameters
        self.max_positions = 5
        self.risk_per_trade = 0.02  # 2%
        self.min_probability = 0.55
        
        print(f"[XGBTrader] Initialized with ${starting_equity:,.2f}")
    
    def scan_and_trade(self):
        """Scan for signals and execute paper trades"""
        print("\n" + "="*70)
        print("XGBOOST PAPER TRADING SCAN")
        print("="*70)
        
        # Check if at max positions
        open_count = len(self.paper_engine.positions)
        if open_count >= self.max_positions:
            print(f"[XGBTrader] At max positions ({open_count}/{self.max_positions})")
            return
        
        # Generate signals
        signals = self.signal_gen.generate_signals(threshold=self.min_probability)
        
        # Filter to top N signals by probability
        signals.sort(key=lambda x: x['probability'], reverse=True)
        available_slots = self.max_positions - open_count
        top_signals = signals[:available_slots]
        
        # Execute trades
        for signal in top_signals:
            self._execute_signal(signal)
        
        # Print summary
        summary = self.paper_engine.get_summary()
        print(f"\n[SUMMARY]")
        print(f"  Equity: ${summary['current_equity']:,.2f}")
        print(f"  Open positions: {summary['open_positions']}")
        if 'total_trades' in summary and summary['total_trades'] > 0:
            print(f"  Total trades: {summary['total_trades']}")
            print(f"  Win rate: {summary['win_rate']:.1%}")
            print(f"  Total P&L: ${summary['total_pnl']:,.2f}")
    
    def _execute_signal(self, signal: dict):
        """Execute a single signal through paper trading"""
        symbol = signal['symbol']
        
        # Check if already in position
        if symbol in self.paper_engine.positions:
            print(f"[SKIP] Already in position: {symbol}")
            return
        
        # Calculate position size (2% risk)
        account_value = self.paper_engine.current_equity
        risk_amount = account_value * self.risk_per_trade
        
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            print(f"[SKIP] Invalid risk for {symbol}")
            return
        
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry_price
        
        # Create paper position
        from datetime import datetime
        position = PaperPosition(
            trade_id=f"XGB_{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            direction='LONG',
            entry_price=entry_price,
            position_size=position_value,
            stop_loss=stop_loss,
            take_profit_1=signal['take_profit_1'],
            take_profit_2=signal['take_profit_2'],
            entry_time=datetime.now(),
            status=TradeStatus.OPEN,
            entry_model='XGBoost',
            expected_r=2.0
        )
        
        # Add to engine
        self.paper_engine.positions[symbol] = position
        self.paper_engine.daily_trades += 1
        
        print(f"[EXECUTED] {symbol} LONG {shares} shares @ ${entry_price:.2f}")
        print(f"           Stop: ${stop_loss:.2f} | TP1: ${signal['take_profit_1']:.2f}")
    
    def update_positions(self):
        """Update all positions with current prices and check exits"""
        print("\n" + "="*70)
        print("UPDATING POSITIONS")
        print("="*70)
        
        if not self.paper_engine.positions:
            print("[XGBTrader] No open positions")
            return
        
        # Fetch current prices
        symbols = list(self.paper_engine.positions.keys())
        prices = {}
        for symbol in symbols:
            price = self.signal_gen.client.get_latest_price(symbol)
            if price:
                prices[symbol] = price
        
        # Update positions
        closed = self.paper_engine.update_positions(prices)
        
        print(f"[XGBTrader] Updated {len(symbols)} positions, {len(closed)} closed")
        
        # Show open positions
        for symbol, pos in self.paper_engine.positions.items():
            if symbol in prices:
                current = prices[symbol]
                pnl_pct = (current - pos.entry_price) / pos.entry_price
                print(f"  {symbol}: ${current:.2f} ({pnl_pct:+.2%})")
    
    def run_once(self):
        """Run one scan-and-trade cycle"""
        self.update_positions()
        self.scan_and_trade()


if __name__ == '__main__':
    trader = XGBoostPaperTrader(starting_equity=100000.0)
    trader.run_once()
