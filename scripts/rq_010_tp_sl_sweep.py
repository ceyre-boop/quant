#!/usr/bin/env python3
"""RQ-010: Fixed TP/SL vs trailing stop sweep on v013 forex system."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

PAIRS = ['EURUSD=X','GBPUSD=X','AUDUSD=X','AUDNZD=X','USDJPY=X']
RESULTS = {}

def load_prices(pair):
    import yfinance as yf
    df = yf.download(pair, start='2018-01-01', end='2025-06-01', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def sharpe(returns):
    r = pd.Series(returns).dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

def run_exit_sweep(pair):
    from sovereign.forex.signal_engine import build_signal_frame
    prices = load_prices(pair)
    df = build_signal_frame(pair, prices, pair[:3], pair[3:6])
    daily_ret = prices['Close'].pct_change().fillna(0)

    results = {}
    configs = {
        'baseline_trailing_1.25x': ('trailing', 1.25),
        'fixed_2.0R': ('fixed', 2.0),
        'fixed_2.5R': ('fixed', 2.5),
        'fixed_3.0R': ('fixed', 3.0),
        'fixed_3.5R': ('fixed', 3.5),
    }

    for label, (mode, mult) in configs.items():
        equity = 1.0
        daily_returns = []
        in_trade = False
        entry_price = 0.0
        direction = 0
        bars_held = 0
        atr_at_entry = 0.0

        # Simple ATR proxy
        atr = prices['Close'].pct_change().abs().rolling(14).mean().fillna(0.01)

        for i, (ts, row) in enumerate(df.iterrows()):
            sig = int(row['signal'])
            close = float(prices.loc[ts, 'Close']) if ts in prices.index else None
            if close is None:
                daily_returns.append(0.0)
                continue

            if not in_trade and sig != 0:
                in_trade = True
                entry_price = close
                direction = sig
                bars_held = 0
                atr_at_entry = float(atr.loc[ts]) if ts in atr.index else 0.01

            if in_trade:
                ret = (close - entry_price) / entry_price * direction
                bars_held += 1

                if mode == 'fixed':
                    sl = -0.01  # 1% stop
                    tp = atr_at_entry * mult * 100  # mult × ATR as %
                    exit_trade = ret <= sl or ret >= tp or sig == 0 or bars_held >= 60
                else:
                    # trailing: exit if signal flips or hold_days expires
                    hold = int(row.get('hold_days', row.get('hold', 60)))
                    exit_trade = sig == 0 or bars_held >= hold

                daily_r = float(daily_ret.loc[ts]) * direction if ts in daily_ret.index else 0.0
                daily_returns.append(daily_r * row.get('size_mult', 1.0))

                if exit_trade:
                    in_trade = False
            else:
                daily_returns.append(0.0)

        results[label] = round(sharpe(daily_returns), 4)

    return results

print('\n══ RQ-010: TP/SL Structure Sweep ══\n')
all_results = {}
for pair in PAIRS:
    try:
        r = run_exit_sweep(pair)
        all_results[pair] = r
        baseline = r.get('baseline_trailing_1.25x', 0)
        print(f'{pair}:')
        for label, sh in r.items():
            delta = sh - baseline
            marker = ' ← BEST' if sh == max(r.values()) else ''
            print(f'  {label:25s} Sharpe={sh:+.4f}  delta={delta:+.4f}{marker}')
    except Exception as e:
        print(f'{pair}: ERROR {e}')
        all_results[pair] = {'error': str(e)}

# Portfolio averages
print('\nPortfolio averages:')
configs_all = ['baseline_trailing_1.25x','fixed_2.0R','fixed_2.5R','fixed_3.0R','fixed_3.5R']
for cfg in configs_all:
    vals = [all_results[p].get(cfg, 0) for p in PAIRS if isinstance(all_results[p], dict) and cfg in all_results[p]]
    if vals:
        print(f'  {cfg:25s} avg={np.mean(vals):+.4f}')

out = Path('data/agent')
out.mkdir(parents=True, exist_ok=True)
import json
(out / 'rq_010_results.json').write_text(json.dumps({'task':'RQ-010','results':all_results}, indent=2, default=str))
print('\nSaved to data/agent/rq_010_results.json')
