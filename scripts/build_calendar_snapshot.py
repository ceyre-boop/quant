#!/usr/bin/env python3
"""
Build a committed calendar snapshot so the dashboard Calendar grid shows real monthly P&L
when the live backend is cold/unreachable (Render free-tier sleep). This mirrors commit
6253bac, which gave the TRADE/TRADES panels a proof_of_life snapshot fallback.

It captures the EXACT output of live_signals_server._calendar_data() for the current month
and a few prior months, so the committed snapshot and the live /calendar endpoint render
identically through the same calRender() path in index.html.

Re-run:  python3 scripts/build_calendar_snapshot.py
"""
import json
import os
import sys
from datetime import datetime, timezone

# Run from repo root so the server module's relative data paths resolve.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts')
import live_signals_server as srv  # noqa: E402  (import-safe: server start is __main__-guarded)


def _months_back(n):
    now = datetime.now(timezone.utc)
    y, m = now.year, now.month
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return out


def main():
    months = _months_back(4)  # current month + 3 prior (covers calendar back-navigation)
    data = {}
    for mo in months:
        try:
            data[mo] = srv._calendar_data(mo)
        except Exception as e:  # never let one bad month abort the snapshot
            data[mo] = {'month': mo, 'days': {},
                        'month_total': {'pnl': 0, 'n': 0, 'wins': 0, 'closed': 0},
                        'error': str(e)}

    out = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'note': ('Committed calendar P&L snapshot — the dashboard falls back to this when the '
                 'live /calendar endpoint is cold (Render free-tier sleep). '
                 'Re-run scripts/build_calendar_snapshot.py to refresh.'),
        'months': data,
    }
    path = 'data/calendar_snapshot.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

    total_days = sum(len(v.get('days', {})) for v in data.values())
    print(f"wrote {path}: {len(data)} months, {total_days} days with data")
    for mo in months:
        v = data[mo]
        mt = v.get('month_total', {})
        print(f"  {mo}: {len(v.get('days', {}))} days · pnl={mt.get('pnl')} · n={mt.get('n')}"
              + (f" · ERROR {v['error']}" if v.get('error') else ""))


if __name__ == '__main__':
    main()
