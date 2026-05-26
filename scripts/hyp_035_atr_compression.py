#!/usr/bin/env python3
"""
HYP-035: Volatility Compression Entry Gate
Does entering when ATR < 20th percentile produce better R?

H0: ATR compression at entry has no relationship to subsequent R
H1: COMPRESSED entries improve avg R by > 0.30R vs NORMAL/EXTENDED
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd

print('\n══ HYP-035: ATR Compression Entry Gate ══\n')

try:
    import yfinance as yf

    PAIRS = {
        'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'AUDUSD': 'AUDUSD=X',
        'NZDUSD': 'NZDUSD=X', 'USDJPY': 'USDJPY=X',
    }
    HOLD_DAYS = 5
    STOP_MULT = 2.0  # 2 ATR stop

    all_trades = []

    for pair, ticker in PAIRS.items():
        df = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        close = df['Close'].dropna()

        if len(close) < 300:
            continue

        # ATR_14 as % of price
        atr14 = close.pct_change().abs().rolling(14).mean()
        # 252-day rolling percentile of ATR
        atr_pct = atr14.rolling(252).apply(
            lambda x: (x[-1] >= x).mean() * 100 if len(x) == 252 else np.nan,
            raw=True
        )

        # Simulate v013 macro signal: simplified momentum direction
        # 20d momentum as direction proxy (consistent with v013 signal stack)
        mom20 = close.pct_change(20)
        mom5  = close.pct_change(5)

        # Entry rule: signal fires on macro momentum alignment
        # For this test we use all days with a clear direction signal
        for i in range(260, len(close) - HOLD_DAYS, HOLD_DAYS):
            if np.isnan(atr_pct.iloc[i]) or np.isnan(mom20.iloc[i]):
                continue

            direction = 1 if mom20.iloc[i] > 0 else -1
            entry_price = close.iloc[i]
            exit_price  = close.iloc[i + HOLD_DAYS]

            raw_ret = (exit_price - entry_price) / entry_price * direction
            atr_val = atr14.iloc[i]
            stop = atr_val * STOP_MULT
            if stop == 0:
                continue

            r_multiple = raw_ret / stop
            r_multiple = max(min(r_multiple, 5.0), -2.0)  # cap extremes

            bucket = (
                'COMPRESSED' if atr_pct.iloc[i] < 20 else
                'EXTENDED'   if atr_pct.iloc[i] > 80 else
                'NORMAL'
            )

            all_trades.append({
                'pair': pair,
                'date': close.index[i],
                'atr_pct': float(atr_pct.iloc[i]),
                'bucket': bucket,
                'r_multiple': float(r_multiple),
            })

    print(f'Total simulated trades: {len(all_trades)}')

    df_t = pd.DataFrame(all_trades)

    print('\nResults by ATR compression bucket:')
    print(f'{"Bucket":12s}  {"N":>5s}  {"WR%":>6s}  {"AvgR":>7s}  {"Sharpe":>8s}')
    print('─' * 50)

    bucket_stats = {}
    for bucket in ['COMPRESSED', 'NORMAL', 'EXTENDED']:
        subset = df_t[df_t['bucket'] == bucket]['r_multiple']
        if len(subset) < 10:
            continue
        wr  = (subset > 0).mean() * 100
        avg = subset.mean()
        std = subset.std()
        sharpe = avg / std * np.sqrt(252 / HOLD_DAYS) if std > 0 else 0
        bucket_stats[bucket] = {'n': len(subset), 'wr': wr, 'avg_r': avg, 'sharpe': sharpe}
        print(f'  {bucket:12s}  {len(subset):5d}  {wr:6.1f}%  {avg:+7.3f}R  {sharpe:8.3f}')

    normal_avg = bucket_stats.get('NORMAL', {}).get('avg_r', 0)
    compressed_avg = bucket_stats.get('COMPRESSED', {}).get('avg_r', 0)
    delta = compressed_avg - normal_avg

    print(f'\n  Delta (COMPRESSED vs NORMAL): {delta:+.4f}R')
    print(f'  H1 threshold: > +0.30R')

    if delta > 0.30:
        verdict = 'CONFIRMED — compression entries outperform by > 0.30R'
    elif delta > 0.10:
        verdict = 'PARTIAL — improvement real but below H1 threshold'
    elif delta > -0.05:
        verdict = 'INCONCLUSIVE — within noise band'
    else:
        verdict = 'REJECTED — compression does not improve R-multiple'

    print(f'\nVERDICT: {verdict}')

    result = {
        'task': 'HYP-035',
        'n_trades': len(all_trades),
        'bucket_stats': bucket_stats,
        'compressed_vs_normal_delta': round(delta, 4),
        'verdict': verdict,
    }
    Path('data/agent/hyp_035_results.json').write_text(json.dumps(result, indent=2, default=float))
    print('\nSaved to data/agent/hyp_035_results.json')

except Exception as e:
    print(f'ERROR: {e}')
    import traceback; traceback.print_exc()
