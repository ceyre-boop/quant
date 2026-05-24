#!/usr/bin/env python3
"""RQ-REST-002: UTC 15xx London close edge study — is it a valid second window?"""
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

def utc_hour(ts_str):
    try:
        from datetime import datetime
        for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S'):
            try:
                return datetime.strptime(ts_str[:19], fmt).hour
            except Exception:
                pass
    except Exception:
        pass
    return None

trades = load_ict_trades()
print(f'\n══ RQ-REST-002: UTC 15xx London Close Edge ══\n')
print(f'Total trades loaded: {len(trades)}')

# Group by UTC hour
by_hour = {}
for t in trades:
    ts = t.get('entry_time', t.get('ts', t.get('timestamp', '')))
    h = utc_hour(str(ts))
    if h is not None:
        by_hour.setdefault(h, []).append(t)

# Show stats by hour
print('\nHourly breakdown:')
for h in sorted(by_hour.keys()):
    tlist = by_hour[h]
    rs = [float(t.get('r_multiple', t.get('r', 0))) for t in tlist
          if t.get('r_multiple') is not None or t.get('r') is not None]
    if not rs:
        continue
    wr = len([r for r in rs if r > 0]) / len(rs) * 100
    avg_r = sum(rs) / len(rs)
    window = ''
    if 2 <= h <= 5: window = ' ← London'
    elif 13 <= h <= 16: window = ' ← London Close/NY'
    print(f'  UTC {h:02d}xx  n={len(rs):3d}  WR={wr:.1f}%  avgR={avg_r:+.3f}{window}')

# Key comparison: UTC 03 vs UTC 15
print('\nKey comparison:')
windows = {
    'UTC 03xx (confirmed)': [h for h in [3] if h in by_hour],
    'UTC 15xx (candidate)': [h for h in [15] if h in by_hour],
    'UTC 02-05 (full London)': [h for h in range(2, 6) if h in by_hour],
    'UTC 13-16 (London close)': [h for h in range(13, 17) if h in by_hour],
}

for label, hours in windows.items():
    tlist = []
    for h in hours:
        tlist.extend(by_hour.get(h, []))
    if not tlist:
        print(f'  {label}: no data')
        continue
    rs = [float(t.get('r_multiple', t.get('r', 0))) for t in tlist
          if t.get('r_multiple') is not None or t.get('r') is not None]
    if not rs:
        print(f'  {label}: n={len(tlist)} — no r data')
        continue
    wr = len([r for r in rs if r > 0]) / len(rs) * 100
    avg_r = sum(rs) / len(rs)
    print(f'  {label}: n={len(rs):3d}  WR={wr:.1f}%  avgR={avg_r:+.3f}')

# Grade A within UTC 15xx
utc15_a = [t for t in by_hour.get(15, []) if t.get('grade', '').upper() in ('A', 'A+')]
if utc15_a:
    rs = [float(t.get('r_multiple', t.get('r', 0))) for t in utc15_a
          if t.get('r_multiple') is not None or t.get('r') is not None]
    if rs:
        print(f'\n  UTC 15xx Grade A: n={len(rs)}  WR={len([r for r in rs if r>0])/len(rs)*100:.1f}%  avgR={sum(rs)/len(rs):+.3f}')
        verdict = 'CONFIRMED — add UTC 15xx as second window' if sum(rs)/len(rs) > 0.3 and len(rs) >= 10 else 'INSUFFICIENT DATA' if len(rs) < 10 else 'REJECTED'
        print(f'  VERDICT: {verdict}')

result = {'task': 'RQ-REST-002', 'hours_with_data': sorted(by_hour.keys())}
Path('data/agent/rq_rest_002_results.json').write_text(json.dumps(result, indent=2))
print('\nSaved to data/agent/rq_rest_002_results.json')
