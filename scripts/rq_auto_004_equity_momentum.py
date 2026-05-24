#!/usr/bin/env python3
"""RQ-AUTO-004: Equity counter-momentum feature importance study."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

TICKERS = ['META', 'UNH', 'AMD', 'BAC', 'JPM']

print('\n══ RQ-AUTO-004: Equity Counter-Momentum Feature Importance ══\n')

try:
    import yfinance as yf

    frames = {}
    for t in TICKERS:
        df = yf.download(t, start='2023-01-01', end='2025-06-01', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        frames[t] = df['Close']

    all_ic = {}
    for ticker, close in frames.items():
        features = pd.DataFrame(index=close.index)
        features['mom_5d']    = close.pct_change(5)
        features['mom_10d']   = close.pct_change(10)
        features['mom_20d']   = close.pct_change(20)
        features['dist_50sma']  = (close / close.rolling(50).mean()) - 1
        features['dist_200sma'] = (close / close.rolling(200).mean()) - 1

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # Target: next 10d return
        target = close.pct_change(10).shift(-10)
        features['target'] = target
        features = features.dropna()

        # Information coefficient (Spearman rank correlation)
        ic = {}
        for feat in ['mom_5d','mom_10d','mom_20d','dist_50sma','dist_200sma','rsi_14']:
            if feat in features.columns:
                corr = features[feat].corr(features['target'], method='spearman')
                ic[feat] = round(float(corr), 4)

        all_ic[ticker] = ic
        print(f'{ticker}:')
        for feat, val in sorted(ic.items(), key=lambda x: abs(x[1]), reverse=True):
            bar = '█' * int(abs(val) * 20)
            sign = '+' if val > 0 else '-'
            print(f'  {feat:15s}  IC={val:+.4f}  {"→ counter-momentum" if val < -0.05 else "→ momentum" if val > 0.05 else "→ weak"}')

    # Aggregate: average IC across all tickers
    print('\nAggregate IC (avg across tickers):')
    feat_names = ['mom_5d','mom_10d','mom_20d','dist_50sma','dist_200sma','rsi_14']
    agg = {}
    for feat in feat_names:
        vals = [all_ic[t].get(feat, 0) for t in TICKERS if feat in all_ic.get(t, {})]
        if vals:
            agg[feat] = round(sum(vals)/len(vals), 4)

    for feat, val in sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f'  {feat:15s}  avg_IC={val:+.4f}')

    # Best predictor for counter-momentum sizing
    best = min(agg.items(), key=lambda x: x[1])  # most negative = strongest counter-momentum
    print(f'\nBest counter-momentum feature: {best[0]} (IC={best[1]:+.4f})')
    verdict = f'USE {best[0]} for counter-momentum sizing' if best[1] < -0.05 else 'No strong counter-momentum signal found'
    print(f'VERDICT: {verdict}')

    result = {'task': 'RQ-AUTO-004', 'ic_by_ticker': all_ic, 'aggregate_ic': agg, 'verdict': verdict}
    Path('data/agent/rq_auto_004_results.json').write_text(json.dumps(result, indent=2))
    print('\nSaved to data/agent/rq_auto_004_results.json')

except Exception as e:
    print(f'ERROR: {e}')
    import traceback; traceback.print_exc()
