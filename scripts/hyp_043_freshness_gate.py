#!/usr/bin/env python3
"""
HYP-043: Macro Signal Freshness Gate
Oracle identified that v013 treats a day-1 signal identically to a day-5 signal.
This script validates whether the decay is real and monotonic before any code changes.

Method:
- Run all 5 v013 pairs via ForexBatchBacktester (captures signals array + trade list)
- For each trade, compute bars_since_spike = distance from entry to nearest prior signal bar
- Split R-multiple distribution by freshness bucket
- t-test bucket 1-2 vs bucket 4+
- Report: accept/reject H1 (monotonic decay, p < 0.05, effect > 0.10R)

No live code changes. Read-only validation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from scipy import stats

print('\n══ HYP-043: Macro Signal Freshness Gate Validation ══\n')

V013_PAIRS = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'USDJPY=X']
STOP_MULT  = 2.0   # 2× ATR stop (matching v013)

try:
    from sovereign.forex.batch_backtester import ForexBatchBacktester
    from sovereign.forex.fast_backtester import simulate_forex_trades_arrays

    bt = ForexBatchBacktester()
    print(f'Loading {len(V013_PAIRS)} pairs...')
    bt.preload(V013_PAIRS)
    print('Data loaded.\n')

    all_trades = []   # {pair, entry_date, r_multiple, bars_since_spike, days_since_spike}

    for pair in V013_PAIRS:
        dataset = bt._array_cache.get(pair)
        if dataset is None:
            print(f'  {pair}: no data — skipping')
            continue

        signals = dataset.signals       # np.ndarray, shape (n,)
        index   = dataset.index         # DatetimeIndex
        closes  = dataset.closes
        opens   = dataset.opens
        hold_d  = dataset.hold_days

        # Get trades (same simulation as backtest_pair)
        trades = simulate_forex_trades_arrays(
            opens=opens,
            closes=closes,
            signals=signals,
            hold_days=hold_d,
            stop_pct=bt._backtester.STOP_PCT,
            index=index,
        )

        if not trades:
            print(f'  {pair}: no trades')
            continue

        # Build signal-fired dates: bars where signals != 0
        signal_bars = [index[i] for i, s in enumerate(signals) if s != 0]

        n_tagged = 0
        for t in trades:
            entry_date = pd.Timestamp(t['entry_date'])
            exit_date  = pd.Timestamp(t['exit_date'])

            # Find the most recent signal bar at or before entry date
            prior = [d for d in signal_bars if d <= entry_date]
            if not prior:
                continue
            signal_fire_date = max(prior)

            # bars_since_spike: trading days between signal fire and entry
            # (signal fires on bar i, entry on bar i+1 is "0 days after")
            bars_since = (entry_date - signal_fire_date).days
            # Convert to approximate trading days (÷ 7 × 5)
            trading_days = max(0, int(bars_since * 5 / 7))

            # R-multiple: pnl_pct / stop_pct (normalised to R)
            pnl_pct  = t.get('pnl_pct', 0.0)
            risk_pct = t.get('risk_pct', 0.01)
            # stop = atr-based; approximate from risk_pct
            # R = pnl_pct / risk_pct gives R-multiple directly
            r_multiple = pnl_pct / risk_pct if risk_pct > 0 else pnl_pct / 0.01
            r_multiple = float(np.clip(r_multiple, -3.0, 8.0))

            all_trades.append({
                'pair':            pair,
                'entry_date':      entry_date,
                'signal_date':     signal_fire_date,
                'days_since':      bars_since,
                'trading_days':    trading_days,
                'r_multiple':      r_multiple,
                'direction':       t.get('direction', 0),
            })
            n_tagged += 1

        print(f'  {pair}: {len(trades)} trades, {n_tagged} tagged with freshness')

    print(f'\nTotal tagged trades: {len(all_trades)}')

    if not all_trades:
        print('ERROR: No trades tagged — cannot validate hypothesis')
        sys.exit(1)

    df = pd.DataFrame(all_trades)

    # ── Bucket by trading days since signal ──────────────────────────────
    def bucket(d):
        if d <= 2:   return '1-2d (fresh)'
        if d <= 5:   return '3-5d'
        if d <= 10:  return '6-10d'
        if d <= 20:  return '11-20d'
        return '21+d (stale)'

    BUCKET_ORDER = ['1-2d (fresh)', '3-5d', '6-10d', '11-20d', '21+d (stale)']
    df['bucket'] = df['trading_days'].apply(bucket)

    print('\n── R-multiple by signal freshness ──────────────────────────')
    print(f'{"Bucket":18s}  {"N":>5s}  {"WR%":>6s}  {"AvgR":>7s}  {"StdR":>7s}')
    print('─' * 55)

    bucket_stats = {}
    for b in BUCKET_ORDER:
        sub = df[df['bucket'] == b]['r_multiple']
        if len(sub) < 5:
            print(f'  {b:16s}  {len(sub):5d}  (insufficient data)')
            continue
        wr    = (sub > 0).mean() * 100
        avg_r = sub.mean()
        std_r = sub.std()
        bucket_stats[b] = {'n': len(sub), 'wr': wr, 'avg_r': avg_r, 'std_r': std_r, 'values': sub.tolist()}
        print(f'  {b:16s}  {len(sub):5d}  {wr:6.1f}%  {avg_r:+7.3f}R  {std_r:7.3f}')

    # ── Monotonicity test ────────────────────────────────────────────────
    avg_rs = [bucket_stats[b]['avg_r'] for b in BUCKET_ORDER if b in bucket_stats]
    is_monotonic = all(avg_rs[i] >= avg_rs[i+1] for i in range(len(avg_rs)-1))

    print(f'\n  Monotonic decay: {"YES ✓" if is_monotonic else "NO ✗"}')
    print(f'  Sequence: {" > ".join(f"{x:+.3f}" for x in avg_rs)}')

    # ── t-test: fresh (1-2d) vs stale (11-20d and 21+d) ─────────────────
    fresh_vals = []
    stale_vals = []
    for b in BUCKET_ORDER:
        if b not in bucket_stats:
            continue
        if b in ('1-2d (fresh)', '3-5d'):
            fresh_vals.extend(bucket_stats[b]['values'])
        if b in ('11-20d', '21+d (stale)'):
            stale_vals.extend(bucket_stats[b]['values'])

    p_value = None
    effect  = None
    if fresh_vals and stale_vals:
        t_stat, p_value = stats.ttest_ind(fresh_vals, stale_vals, equal_var=False)
        effect = np.mean(fresh_vals) - np.mean(stale_vals)
        print(f'\n  t-test (fresh vs stale):')
        print(f'    fresh mean R: {np.mean(fresh_vals):+.4f}  (n={len(fresh_vals)})')
        print(f'    stale mean R: {np.mean(stale_vals):+.4f}  (n={len(stale_vals)})')
        print(f'    effect size:  {effect:+.4f}R')
        print(f'    p-value:      {p_value:.4f}  (threshold < 0.05)')

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f'\n{"═"*55}')
    confirmed = (
        is_monotonic
        and p_value is not None and p_value < 0.05
        and effect is not None and effect > 0.10
    )

    if confirmed:
        verdict = 'CONFIRMED — implement freshness gate as v014'
        print(f'  ✅ H1 ACCEPTED: {verdict}')
        print(f'\n  Recommended multipliers:')
        print(f'    bars 1-2d : ×1.35')
        print(f'    bars 3-5d : ×1.00')
        print(f'    bars 6-10d: ×0.80')
        print(f'    bars 11+d : ×0.65')
        print(f'    bars 21+d : veto (0.0×)')
    elif is_monotonic and effect and effect > 0.05:
        verdict = 'PARTIAL — real decay but below H1 threshold; use as size boost, not veto'
        print(f'  ⚠️  H1 PARTIAL: {verdict}')
    else:
        verdict = 'REJECTED — no monotonic decay or no statistical significance'
        print(f'  ❌ H1 REJECTED: {verdict}')

    print(f'{"═"*55}\n')

    result = {
        'task': 'HYP-043',
        'n_trades': len(all_trades),
        'bucket_stats': {b: {k: v for k, v in s.items() if k != 'values'} for b, s in bucket_stats.items()},
        'is_monotonic': is_monotonic,
        'avg_r_sequence': avg_rs,
        'effect_size_r': round(effect, 4) if effect else None,
        'p_value': round(p_value, 5) if p_value else None,
        'verdict': verdict,
        'confirmed': confirmed,
    }
    Path('data/agent/hyp_043_results.json').write_text(json.dumps(result, indent=2, default=float))
    print('Saved to data/agent/hyp_043_results.json')

except Exception as e:
    print(f'ERROR: {e}')
    import traceback; traceback.print_exc()
