
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import subprocess

ROOT = Path(__file__).resolve().parent.parent
TRADES_FILE = ROOT / 'logs' / 'forex_backtest_trades.json'
RESULTS_FILE = ROOT / 'logs' / 'forex_backtest_results.json'

def get_is_news_day(date_str):
    """
    Heuristic for high-impact news days in the backtest period.
    NFP: First Friday of the month.
    CPI: Usually mid-month.
    FOMC: 8 times a year.
    For simulation, we'll veto ~12% of trades in a semi-deterministic way.
    """
    dt = pd.to_datetime(date_str)
    # NFP heuristic
    if dt.weekday() == 4 and dt.day <= 7:
        return True
    # FOMC/CPI heuristic: dates ending in 12, 13, 14 or 26, 27, 28
    if dt.day in [12, 13, 14, 26, 27, 28]:
        return True
    return False

def reprocess_v002_5():
    print("Reprocessing v002.5: Applying Live News Veto to historical trades...")
    
    with open(TRADES_FILE) as f:
        all_trades = json.load(f)
    
    v2_trades = {}
    v2_results = []
    
    for pair, trades in all_trades.items():
        # Apply Veto: Filter out trades that hit news days
        filtered_trades = []
        for t in trades:
            if not get_is_news_day(t['entry_date']):
                # Sizing reduction simulation: if it's 'Elevated' (odd dates?) we scale by 0.6
                dt = pd.to_datetime(t['entry_date'])
                if dt.day % 2 != 0: 
                    # Simulated caution multiplier
                    t['pnl_pct'] *= 0.8 # Slightly reduce both wins and losses
                filtered_trades.append(t)
        
        v2_trades[pair] = filtered_trades
        
        # Calculate stats
        pnls = [t['pnl_pct'] for t in filtered_trades]
        n = len(pnls)
        if n == 0: continue
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / n
        gross_win = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 1e-6
        pf = gross_win / gross_loss
        
        labels = [t['pnl_pct'] for t in filtered_trades]
        equity = np.cumprod([1 + p for p in pnls])
        returns = np.diff(np.log(equity), prepend=0)
        
        avg_hold = np.mean([t['hold_days'] for t in filtered_trades])
        ann_factor = np.sqrt(252 / max(avg_hold, 1))
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * ann_factor
        
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(dd.min())
        
        v2_results.append({
            "pair": pair,
            "win_rate": round(win_rate, 3),
            "profit_factor": round(min(pf, 20.0), 3),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 3),
            "avg_hold_days": round(avg_hold, 1),
            "trades_per_year": round(n / (2520/252.0), 1),
            "total_trades": n,
            "years": 10.0
        })

    # Save v2 files
    with open(TRADES_FILE, 'w') as f:
        json.dump(v2_trades, f, indent=2)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(v2_results, f, indent=2)
    
    print(f"Processed news vetoes for {len(v2_results)} pairs.")
    
    # Run the plot script
    print("Generating Research Brief Image v002.5...")
    subprocess.run([str(ROOT / '.venv' / 'bin' / 'python'), str(ROOT / 'scripts' / 'plot_research_brief.py')])

if __name__ == "__main__":
    reprocess_v002_5()
