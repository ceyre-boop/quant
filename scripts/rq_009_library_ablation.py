#!/usr/bin/env python3
"""
RQ-009 / SUG-006: Alexandrian Library feature criticality ranking.
Ablation test — remove each of the 23 features one at a time,
measure threat score delta on known CRITICAL events.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

print('\n══ SUG-006 / RQ-009: Library Feature Criticality Ablation ══\n')

try:
    from sovereign.risk.market_memory import extract_features, _cosine_similarity, FEATURE_NAMES, N_FEATURES
    from sovereign.risk.alexandrian_library import AlexandrianLibrary

    print(f'Features: {N_FEATURES}')
    print('Feature names:', FEATURE_NAMES)

    lib = AlexandrianLibrary()

    import yfinance as yf
    import pandas as pd

    # Pull recent data — use 2022 bear market as test window (should score CRITICAL)
    spy = yf.download('SPY', start='2022-01-01', end='2022-12-31', progress=False)
    vix = yf.download('^VIX', start='2022-01-01', end='2022-12-31', progress=False)
    gld = yf.download('GLD', start='2022-01-01', end='2022-12-31', progress=False)

    for df in [spy, vix, gld]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)

    # Take a 90-day window from June 2022 (peak stress)
    spy_arr = spy['Close'].loc['2022-03-01':'2022-06-16'].values[-90:]
    vix_arr = vix['Close'].loc['2022-03-01':'2022-06-16'].values[-90:]
    gld_arr = gld['Close'].loc['2022-03-01':'2022-06-16'].values[-90:]

    if len(spy_arr) < 60:
        print('Insufficient data window — need 60+ bars')
        sys.exit(0)

    # Baseline feature vector
    baseline_feats = extract_features(spy_arr, vix_prices=vix_arr, gold_prices=gld_arr)
    print(f'\nBaseline feature vector computed (len={len(baseline_feats)})')
    print('Feature values:')
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, baseline_feats)):
        print(f'  {i:2d} {name:30s} = {val:+.4f}')

    # Ablation: zero each feature, measure similarity shift to all library entries
    from sovereign.risk.market_memory import _cosine_similarity

    # Build stored entries from library
    if not lib._patterns:
        lib.build_from_history()

    if not lib._patterns:
        print('\nLibrary has no stored entries — run build_from_history() first')
        # Still report feature values
        sys.exit(0)

    print(f'\nLibrary entries: {len(lib._patterns)}')

    # Baseline max similarity
    baseline_sims = [_cosine_similarity(baseline_feats, np.array(e.features)) for e in lib._patterns]
    baseline_max = max(baseline_sims) if baseline_sims else 0.0
    print(f'Baseline max similarity: {baseline_max:.4f}')

    impacts = []
    for i, feat_name in enumerate(FEATURE_NAMES):
        ablated = baseline_feats.copy()
        ablated[i] = 0.0
        sims = [_cosine_similarity(ablated, np.array(e.features)) for e in lib._patterns]
        max_sim = max(sims) if sims else 0.0
        delta = max_sim - baseline_max
        impacts.append((feat_name, delta, max_sim))

    # Rank by impact (most negative = most critical)
    impacts.sort(key=lambda x: x[1])
    print('\nFeature criticality ranking (most critical first):')
    print(f'{"Feature":30s}  {"Delta":>8s}  {"Ablated Sim":>12s}')
    for name, delta, sim in impacts:
        marker = ' ← CRITICAL' if delta < -0.05 else ' ← important' if delta < -0.02 else ''
        print(f'{name:30s}  {delta:+.4f}   {sim:.4f}{marker}')

    top3 = [x[0] for x in impacts[:3]]
    print(f'\nTop 3 most critical features: {top3}')
    print('VERDICT: Features above with delta < -0.05 are load-bearing. Others can be simplified.')

    result = {
        'task': 'SUG-006/RQ-009',
        'n_features': N_FEATURES,
        'baseline_max_similarity': baseline_max,
        'feature_impacts': [{'name': n, 'delta': d, 'ablated_sim': s} for n, d, s in impacts],
        'top_critical': top3,
    }
    Path('data/agent/rq_009_library_ablation.json').write_text(json.dumps(result, indent=2, default=float))
    print('\nSaved to data/agent/rq_009_library_ablation.json')

except Exception as e:
    print(f'ERROR: {e}')
    import traceback; traceback.print_exc()
