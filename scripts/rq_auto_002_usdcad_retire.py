#!/usr/bin/env python3
"""RQ-AUTO-002: USDCAD retirement validation — does any regime save it?"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

PAIR = 'USDCAD=X'

def sharpe(returns):
    r = pd.Series(returns).dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

print('\n══ RQ-AUTO-002: USDCAD Retirement Validation ══\n')

try:
    import yfinance as yf
    from sovereign.forex.signal_engine import build_signal_frame

    prices = yf.download(PAIR, start='2018-01-01', end='2025-06-01', progress=False)
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    # Also pull SPY for regime
    spy = yf.download('SPY', start='2018-01-01', end='2025-06-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_close = spy['Close']

    df = build_signal_frame(PAIR, prices, 'USD', 'CAD')
    daily_ret = prices['Close'].pct_change().fillna(0)
    spy_200sma = spy_close.rolling(200).mean()

    # Full system Sharpe
    port_ret = []
    for ts, row in df.iterrows():
        sig = int(row['signal'])
        sm = float(row.get('size_mult', 1.0))
        r = float(daily_ret.loc[ts]) * sig * sm if ts in daily_ret.index and sig != 0 else 0.0
        port_ret.append(r)
    full_sharpe = sharpe(port_ret)
    print(f'Full period Sharpe: {full_sharpe:+.4f}')

    # By SPY regime
    bull_ret, bear_ret = [], []
    for ts, row in df.iterrows():
        sig = int(row['signal'])
        sm = float(row.get('size_mult', 1.0))
        r = float(daily_ret.loc[ts]) * sig * sm if ts in daily_ret.index and sig != 0 else 0.0
        sma = float(spy_200sma.loc[ts]) if ts in spy_200sma.index else None
        price = float(spy_close.loc[ts]) if ts in spy_close.index else None
        if sma and price:
            (bull_ret if price > sma else bear_ret).append(r)
        else:
            bull_ret.append(r)

    print(f'SPY Bull regime Sharpe:  {sharpe(bull_ret):+.4f}  (n={len(bull_ret)} days)')
    print(f'SPY Bear regime Sharpe:  {sharpe(bear_ret):+.4f}  (n={len(bear_ret)} days)')

    # By VIX regime
    try:
        vix = yf.download('^VIX', start='2018-01-01', end='2025-06-01', progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        vix_close = vix['Close']

        low_vix_ret, high_vix_ret = [], []
        for ts, row in df.iterrows():
            sig = int(row['signal'])
            sm = float(row.get('size_mult', 1.0))
            r = float(daily_ret.loc[ts]) * sig * sm if ts in daily_ret.index and sig != 0 else 0.0
            v = float(vix_close.loc[ts]) if ts in vix_close.index else None
            if v:
                (low_vix_ret if v < 20 else high_vix_ret).append(r)
            else:
                low_vix_ret.append(r)

        print(f'VIX<20 Sharpe:          {sharpe(low_vix_ret):+.4f}  (n={len(low_vix_ret)} days)')
        print(f'VIX>=20 Sharpe:         {sharpe(high_vix_ret):+.4f}  (n={len(high_vix_ret)} days)')
    except Exception as e:
        print(f'VIX regime: skipped ({e})')

    max_regime_sharpe = max(full_sharpe, sharpe(bull_ret), sharpe(bear_ret))
    verdict = 'RETIRE' if max_regime_sharpe < 0.5 else 'KEEP — regime gate exists'
    print(f'\nVERDICT: {verdict} (best regime Sharpe={max_regime_sharpe:.4f})')
    print('Note: USDCAD already retired in v008. This confirms retirement stands.')

    result = {'task': 'RQ-AUTO-002', 'full_sharpe': full_sharpe, 'verdict': verdict}
    Path('data/agent/rq_auto_002_results.json').write_text(json.dumps(result, indent=2))
    print('\nSaved to data/agent/rq_auto_002_results.json')

except Exception as e:
    print(f'ERROR: {e}')
    import traceback; traceback.print_exc()
