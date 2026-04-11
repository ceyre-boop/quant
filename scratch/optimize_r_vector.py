"""
Institutional R-Vector Optimizer (Pillar 7: Research Workflow)
Performs sensitivity analysis on the 6-D Risk Posture (R-Vector) 
against a small sample of truth-verified trade setups.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
from orchestrator.backtest_lifecycle import BacktestLifecycle
from contracts.types import Direction
from layer2.ml_coupler import RVector
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

def optimize_r_mapping(trades_csv: str, symbols: List[str]):
    """
    Stage 3: Sensitivity Sweep.
    Iterates through R-Vector permutations to find the 'Regime-Optimal' coupling.
    """
    logger.info(f"Initiating Sensitivity Sweep for {trades_csv}...")
    
    # 1. Load the 'Ground Truth' trade setups
    df = pd.read_csv(trades_csv)
    if df.empty:
        logger.error("No trades found to optimize.")
        return

    # To run a fast sweep, we bypass the full backtest and just simulate outcomes
    # using a headless version of the TradeMonitor.
    
    # Grid Search Parameters (Small Sample Efficient)
    STOP_ATR_STEPS = [2.5, 3.5, 4.5, 5.5, 6.5]
    TP_R_STEPS = [2.0, 4.0, 6.0, 8.0]
    TRAIL_ACT_STEPS = [0.5, 1.0, 1.5]
    
    results = []
    
    # Initialize Backtest context for data access
    bt = BacktestLifecycle(symbols)
    
    # 2. Warp Inhalation: Pre-fetch outcomes (Pillar 1)
    trade_symbols = df['symbol'].unique().tolist()
    min_entry = pd.to_datetime(df['entry_date']).min()
    max_exit = pd.to_datetime(df['entry_date']).max() + timedelta(days=8)
    
    logger.info(f"Warp Inhalation: Pre-fetching data for {len(trade_symbols)} symbols...")
    bars_df = bt.alpaca.get_historical_bars(trade_symbols, '1H', start=min_entry, end=max_exit)
    
    # Populate Bulk Cache manually for the headless BT instance
    for sym in trade_symbols:
        proxy_sym = bt.alpaca.FUTURES_PROXIES.get(sym, sym)
        try:
            if hasattr(bars_df, 'index') and proxy_sym in bars_df.index.get_level_values(0):
                bt.bulk_cache[sym] = {'1H': bars_df.loc[proxy_sym]}
                bt.bulk_cache[proxy_sym] = {'1H': bars_df.loc[proxy_sym]}
        except:
            continue

    
    for stop_mult in STOP_ATR_STEPS:
        for tp_r in TP_R_STEPS:
            for trail_act in TRAIL_ACT_STEPS:
                r_candidate = RVector(
                    stop_atr_mult=stop_mult,
                    tp_target_r=tp_r,
                    trail_activation_r=trail_act,
                    trail_atr_mult=3.0,
                    position_size_scalar=1.0, # Target pure R:R first
                    shock_exit_atr_mult=4.0
                )
                
                total_pnl = 0.0
                wins = 0
                
                for _, trade in df.iterrows():
                    # Re-simulate outcome with candidate R-Vector
                    win, pnl = bt._simulate_outcome(
                        trade['symbol'], 
                        trade['entry_price'], 
                        Direction.LONG if trade['direction'] == 'LONG' else Direction.SHORT,
                        pd.to_datetime(trade['entry_date']),
                        r_candidate
                    )
                    if win: 
                        wins += 1
                        # print(f"  WIN: {trade['symbol']} | PnL: {pnl:.2f}")
                    total_pnl += pnl
                
                print(f"RESULT: Stop {stop_mult} | TP {tp_r} | Trail {trail_act} -> Wins: {wins} | PnL: {total_pnl:.2f}")

                
                results.append({
                    'stop_atr': stop_mult,
                    'tp_r': tp_r,
                    'trail_act': trail_act,
                    'net_pnl_units': total_pnl,
                    'win_rate': wins / len(df)
                })
                
    results_df = pd.DataFrame(results)
    best_pnl = results_df.sort_values(by='net_pnl_units', ascending=False).iloc[0]
    
    print("\n" + "="*40)
    print("SENSITIVITY SWEEP COMPLETE")
    print("="*40)
    print(f"Optimal Stop ATR: {best_pnl['stop_atr']}")
    print(f"Optimal TP Ratio: {best_pnl['tp_r']}")
    print(f"Optimal Trail Activation: {best_pnl['trail_act']}")
    print(f"Projected Units: {best_pnl['net_pnl_units']:.2f}")
    print(f"Projected Win Rate: {best_pnl['win_rate']:.2%}")
    print("="*40)

if __name__ == "__main__":
    import os
    # Find the latest trade file
    trade_files = [f for f in os.listdir('data/backtest_results') if f.startswith('trades_raw_')]
    if trade_files:
        latest_file = os.path.join('data/backtest_results', sorted(trade_files)[-1])
        optimize_r_mapping(latest_file, ["SPY"]) # Placeholder symbols, BT loads what it needs
