"""
Institutional Label Repair - Pillar 2 (Scientific Integrity)
Re-evaluates legacy signal outcomes using the Doctrine-compliant Execution Engine.
Prevents lookahead-leakage from contaminating the model's training data.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv() # Load institutional credentials

from layer3.macro_imbalance import MacroImbalanceFramework
from governance.policy_engine import GOVERNANCE
from layer2.dynamic_rr_engine import DynamicRREngine, TradeMonitor
from data.alpaca_client import AlpacaDataClient
from contracts.types import Direction, MarketData, RegimeState

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def relabel_ledger(input_path: str, output_path: str):
    logger.info(f"Starting Optimized Relabel Process: {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df.iloc[:, 0], utc=True)
    
    # Initialize Engines
    alpaca = AlpacaDataClient()
    dynamic_engine = DynamicRREngine()
    
    rebuilt_results = []
    symbols = df['symbol'].unique()
    total_symbols = len(symbols)
    
    for s_idx, symbol in enumerate(symbols):
        logger.info(f"[{s_idx+1}/{total_symbols}] Prefetching bulk data for {symbol}...")
        
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # Determine range for this symbol
        min_date = symbol_df['timestamp'].min()
        max_date = symbol_df['timestamp'].max() + timedelta(days=8)
        
        # 1. Bulk Fetch (Pillar 1: Optimization)
        # BUG FIX: Use keyword arguments to avoid positional mismatch (days vs start)
        full_outcome_df = alpaca.get_historical_bars(symbol, '1H', start=min_date, end=max_date)
        if full_outcome_df is None or full_outcome_df.empty:
            logger.warning(f"No data for {symbol}, skipping...")
            continue
            
        profile = dynamic_engine.get_profile(symbol)
        
        # 2. Process signals for this symbol locally
        for i, row in symbol_df.iterrows():
            entry_date = row['timestamp']
            direction = Direction.LONG if row['direction'] == 'LONG' else Direction.SHORT
            atr_14 = row.get('atr_14', 1.0)
            
            # Slice 7-day window from bulk data
            end_date = entry_date + timedelta(days=7)
            outcome_df = full_outcome_df.loc[entry_date:end_date]
            
            if len(outcome_df) < 2:
                continue
                
            entry_price = outcome_df['open'].iloc[0]
            stop_dist = atr_14 * profile.stop_atr_multiplier
            stop_price = entry_price - stop_dist if direction == Direction.LONG else entry_price + stop_dist
            
            tp1_dist = stop_dist * profile.tp_min_r
            tp1 = entry_price + tp1_dist if direction == Direction.LONG else entry_price - tp1_dist
            
            tp2_dist = stop_dist * profile.tp_target_r
            tp2 = entry_price + tp2_dist if direction == Direction.LONG else entry_price - tp2_dist
            
            monitor = TradeMonitor(entry_price, direction, stop_price, tp1, tp2, profile)
            
            # Skip first bar (Pillar 2)
            future_bars = outcome_df.iloc[1:]
            final_pnl = 0.0
            final_win = False
            
            for ts, bar in future_bars.iterrows():
                current_atr = bar['high'] - bar['low']
                exit_p, reason = monitor.check_exits(bar, current_atr, ts)
                if exit_p:
                    final_pnl = (exit_p - entry_price) if direction == Direction.LONG else (entry_price - exit_p)
                    final_win = final_pnl > 0
                    break
            else:
                exit_p = future_bars['close'].iloc[-1]
                final_pnl = (exit_p - entry_price) if direction == Direction.LONG else (entry_price - exit_p)
                final_win = final_pnl > 0

            new_row = row.copy()
            new_row['win'] = final_win
            new_row['pnl'] = final_pnl
            rebuilt_results.append(new_row)
        
    rebuilt_df = pd.DataFrame(rebuilt_results)
    rebuilt_df.to_csv(output_path, index=False)
    logger.info(f"Relabeling Complete. Saved to {output_path}")

if __name__ == "__main__":
    relabel_ledger(
        "data/backtest_results/signals_raw_20260409_222335.csv",
        "data/backtest_results/signals_truth_verified.csv"
    )
