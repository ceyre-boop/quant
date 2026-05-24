#!/usr/bin/env python3
"""RQ-AUTO-006: ICT bear-market allocation validation."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

LEDGER_FILES = list(Path('data/ledger').glob('ict_paper_trades*.json')) + \
               list(Path('data/ledger').glob('trade_ledger*.jsonl'))

def load_ict_trades():
    trades = []
    for f in LEDGER_FILES:
        try:
            if f.suffix == '.json':
                d = json.loads(f.read_text())
                t = d if isinstance(d, list) else d.get('trades', [])
                trades.extend(t)
            else:
                for line in f.read_text().strip().split('\n'):
                    if line.strip():
                        trades.append(json.loads(line))
        except Exception:
            pass
    return trades

print('\n══ RQ-AUTO-006: ICT Bear-Market Allocation ══\n')

trades = load_ict_trades()
print(f'Total trades: {len(trades)}')

try:
    import yfinance as yf
    import pandas as pd

    spy = yf.download('SPY', start='2022-01-01', end='2025-06-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_200sma = spy['Close'].rolling(200).mean()

    def get_regime(ts_str):
        try:
            from datetime import datetime
            for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
                try:
                    dt = pd.Timestamp(datetime.strptime(ts_str[:19], fmt))
                    close_val = spy['Close'].asof(dt) if dt in spy.index or dt > spy.index[0] else None
                    sma_val = spy_200sma.asof(dt) if dt in spy_200sma.index or dt > spy_200sma.index[0] else None
                    if close_val and sma_val:
                        return 'BULL' if float(close_val) > float(sma_val) else 'BEAR'
                except Exception:
                    pass
        except Exception:
            pass
        return 'UNKNOWN'

    bull_trades, bear_trades = [], []
    for t in trades:
        ts = t.get('entry_time', t.get('ts', t.get('timestamp', '')))
        regime = get_regime(str(ts))
        if regime == 'BULL':
            bull_trades.append(t)
        elif regime == 'BEAR':
            bear_trades.append(t)

    def stats(tlist, label):
        rs = [float(t.get('r_multiple', t.get('r', 0))) for t in tlist
              if t.get('r_multiple') is not None or t.get('r') is not None]
        if not rs:
            print(f'{label}: n={len(tlist)} — no r data')
            return None
        wr = len([r for r in rs if r > 0]) / len(rs) * 100
        avg_r = sum(rs) / len(rs)
        print(f'{label}: n={len(rs):3d}  WR={wr:.1f}%  avgR={avg_r:+.3f}')
        return avg_r

    bull_avg = stats(bull_trades, 'SPY BULL regime')
    bear_avg = stats(bear_trades, 'SPY BEAR regime')

    if bull_avg is not None and bear_avg is not None:
        if bear_avg < -0.1:
            verdict = 'CONFIRMED: reduce ICT allocation in bear market (SPY < 200SMA)'
        elif abs(bear_avg - bull_avg) < 0.1:
            verdict = 'REJECTED: no significant regime difference'
        else:
            verdict = f'PARTIAL: bear avg={bear_avg:+.3f}, monitor with more data'
        print(f'\nVERDICT: {verdict}')

    result = {'task': 'RQ-AUTO-006', 'n_bull': len(bull_trades), 'n_bear': len(bear_trades)}
    Path('data/agent/rq_auto_006_results.json').write_text(json.dumps(result, indent=2))
    print('\nSaved to data/agent/rq_auto_006_results.json')

except Exception as e:
    print(f'ERROR: {e}')
    import traceback; traceback.print_exc()
