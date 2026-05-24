#!/usr/bin/env python3
"""RQ-REST-001: Test displacement=0 as standalone veto gate on ICT trades."""
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
                    if line:
                        trades.append(json.loads(line))
        except Exception:
            pass
    return trades

trades = load_ict_trades()
print(f'\n══ RQ-REST-001: Displacement=0 Veto Gate ══\n')
print(f'Total trades loaded: {len(trades)}')

# Classify by displacement
with_disp = [t for t in trades if float(t.get('displacement', t.get('displacement_score', 1))) > 0]
no_disp   = [t for t in trades if float(t.get('displacement', t.get('displacement_score', 1))) == 0]

def stats(tlist, label):
    if not tlist:
        print(f'{label}: n=0')
        return
    rs = [float(t.get('r_multiple', t.get('r', 0))) for t in tlist if t.get('r_multiple') is not None or t.get('r') is not None]
    if not rs:
        print(f'{label}: n={len(tlist)} — no r_multiple field')
        return
    wins = [r for r in rs if r > 0]
    wr = len(wins)/len(rs)*100
    avg_r = sum(rs)/len(rs)
    print(f'{label}: n={len(rs):3d}  WR={wr:.1f}%  avgR={avg_r:+.3f}')

stats(with_disp, 'displacement > 0')
stats(no_disp,   'displacement = 0')

# Test: what if we veto all displacement=0 trades?
all_rs = [float(t.get('r_multiple', t.get('r', 0))) for t in trades if t.get('r_multiple') is not None or t.get('r') is not None]
gated_rs = [float(t.get('r_multiple', t.get('r', 0))) for t in with_disp if t.get('r_multiple') is not None or t.get('r') is not None]

if all_rs and gated_rs:
    print(f'\nBaseline (all trades):    n={len(all_rs)}  avgR={sum(all_rs)/len(all_rs):+.3f}  WR={len([r for r in all_rs if r>0])/len(all_rs)*100:.1f}%')
    print(f'Gated  (disp>0 only):   n={len(gated_rs)}  avgR={sum(gated_rs)/len(gated_rs):+.3f}  WR={len([r for r in gated_rs if r>0])/len(gated_rs)*100:.1f}%')
    removed_pct = (len(all_rs) - len(gated_rs)) / len(all_rs) * 100
    avg_delta = sum(gated_rs)/len(gated_rs) - sum(all_rs)/len(all_rs)
    print(f'\nRemoves {removed_pct:.1f}% of trades, avgR delta={avg_delta:+.3f}')
    verdict = 'CONFIRMED — veto displacement=0' if avg_delta > 0.05 else 'REJECTED — delta too small'
    print(f'\nVERDICT: {verdict}')
else:
    print('\nInsufficient r_multiple data in ledger — check field names')
    # Show available fields
    if trades:
        print('Available fields:', list(trades[0].keys())[:10])

result = {
    'task': 'RQ-REST-001',
    'n_total': len(trades),
    'n_with_displacement': len(with_disp),
    'n_no_displacement': len(no_disp),
}
Path('data/agent/rq_rest_001_results.json').write_text(json.dumps(result, indent=2))
print('\nSaved to data/agent/rq_rest_001_results.json')
