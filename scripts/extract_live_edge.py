"""
scripts/extract_live_edge.py
=============================
Extract the real TP distribution from 30 days of live paper trading and
write it to logs/live_edge.json for use by prop_challenge_optimizer.py.

Reads:  logs/ict_paper_trade_log.csv  (produced by ict/paper_trader.py)
   OR:  logs/ict_backtest_window_A.json / ict_backtest_window_B.json
         (use --source-json to read from the ICT backtest JSON format)
Writes: logs/live_edge.json

Outcome mapping (paper_trader.py semantics → optimizer model):
    TP2              → tp2 category   (big winner)
    TP1, BE, TP1_TIMEOUT → tp1 category  (partial winner)
    STOP, TIMEOUT    → stop category  (full loss or 0R)

Usage:
    # From live paper trade CSV (after 30 days of running paper trader):
    python3 scripts/extract_live_edge.py
    python3 scripts/extract_live_edge.py --days 30 --log logs/ict_paper_trade_log.csv

    # From existing backtest JSON windows (run right now, no waiting):
    python3 scripts/extract_live_edge.py --source-json logs/ict_backtest_window_A.json
    python3 scripts/extract_live_edge.py --source-json logs/ict_backtest_window_A.json --days 30
    python3 scripts/extract_live_edge.py --source-json logs/ict_backtest_window_A.json --days 365

    # Combine both windows (maximum statistical power):
    python3 scripts/extract_live_edge.py \\
        --source-json logs/ict_backtest_window_A.json \\
        --source-json logs/ict_backtest_window_B.json

    python3 scripts/extract_live_edge.py --min-trades 15
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Backtest reference edge (for tolerance check) ─────────────────────────── #
BACKTEST_TP2_RATE  = 0.195
BACKTEST_TP1_RATE  = 0.120
BACKTEST_STOP_RATE = 0.685
BACKTEST_TP2_R     = 4.5
BACKTEST_TP1_R     = 1.5
BACKTEST_STOP_R    = -1.0
BACKTEST_EV        = 0.40   # R per trade

TOLERANCE_RATE = 0.08   # ±8 percentage points before flagging drift
TOLERANCE_EV   = 0.20   # ±0.20R EV before flagging drift

DEFAULT_LOG  = 'logs/ict_paper_trade_log.csv'
OUTPUT_PATH  = Path('logs/live_edge.json')

# Column names written by ict/paper_trader.py
COL_DATE    = 'date'
COL_OUTCOME = 'outcome'
COL_PNL_R   = 'pnl_r'
COL_PAIR    = 'pair'

# Outcome groups — covers both paper_trader.py and ICT backtest JSON formats
TP2_OUTCOMES  = {'TP2'}
TP1_OUTCOMES  = {'TP1', 'BE', 'TP1_TIMEOUT'}
STOP_OUTCOMES = {'STOP', 'TIMEOUT'}


def parse_date(s: str) -> Optional[datetime]:
    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S%z'):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def load_trades(log_file: str, days: int) -> List[dict]:
    """Load and filter closed trades from the paper trade CSV."""
    path = Path(log_file)
    if not path.exists():
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    trades = []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # date column may be close_time or trade date
            dt = parse_date(row.get(COL_DATE, '') or row.get('close_time', ''))
            if dt and dt < cutoff:
                continue  # older than window
            outcome = row.get(COL_OUTCOME, '').strip()
            if not outcome:
                continue
            try:
                pnl_r = float(row.get(COL_PNL_R, 0.0))
            except (ValueError, TypeError):
                pnl_r = 0.0
            trades.append({
                'date':    dt,
                'pair':    row.get(COL_PAIR, ''),
                'outcome': outcome,
                'pnl_r':   pnl_r,
            })
    return trades


def load_trades_from_json(json_file: str, days: int) -> List[dict]:
    """
    Load and filter trades from an ICT backtest JSON file.

    The JSON format (produced by ict/backtester.py) has a top-level 'trades'
    array where each entry contains:
        entry_dt  : 'YYYY-MM-DD HH:MM'
        outcome   : 'TP1' | 'TP2' | 'STOP' | 'TIMEOUT'
        pnl_r     : float
        pair      : str

    When days=365 (or any value larger than the window span), all trades are
    returned — i.e., the entire window is treated as the live sample.
    When days < window span, only the most recent N days are returned.
    """
    path = Path(json_file)
    if not path.exists():
        print(f'  ⚠️  File not found: {json_file}')
        return []

    data = json.loads(path.read_text())
    raw_trades = data.get('trades', [])

    # Parse all dates first so we can find the window's own most-recent date
    parsed: List[tuple] = []
    for t in raw_trades:
        dt_str = t.get('entry_dt', '')
        dt = None
        for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d'):
            try:
                dt = datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        if dt:
            parsed.append((dt, t))

    if not parsed:
        return []

    # Cutoff = most recent trade date minus `days` (not wall-clock now, since
    # backtest data may be months old relative to today)
    latest_date = max(dt for dt, _ in parsed)
    cutoff = latest_date - timedelta(days=days)

    trades = []
    for dt, t in parsed:
        if dt < cutoff:
            continue
        outcome = t.get('outcome', '').strip()
        if not outcome:
            continue
        try:
            pnl_r = float(t.get('pnl_r', 0.0))
        except (ValueError, TypeError):
            pnl_r = 0.0
        trades.append({
            'date':    dt,
            'pair':    t.get('pair', ''),
            'outcome': outcome,
            'pnl_r':   pnl_r,
        })
    return trades


def compute_distribution(trades: List[dict]) -> dict:
    """
    Compute TP distribution from closed trade list.
    Returns dict with rates and average R values per category.
    """
    n = len(trades)
    if n == 0:
        return {}

    tp2_trades  = [t for t in trades if t['outcome'] in TP2_OUTCOMES]
    tp1_trades  = [t for t in trades if t['outcome'] in TP1_OUTCOMES]
    stop_trades = [t for t in trades if t['outcome'] in STOP_OUTCOMES]

    def _mean_r(group: List[dict]) -> float:
        if not group:
            return 0.0
        return sum(t['pnl_r'] for t in group) / len(group)

    def _safe_rate(group: List[dict]) -> float:
        return len(group) / n

    tp2_rate  = _safe_rate(tp2_trades)
    tp1_rate  = _safe_rate(tp1_trades)
    stop_rate = _safe_rate(stop_trades)

    # Normalize so rates sum to exactly 1.0
    total = tp2_rate + tp1_rate + stop_rate
    if total > 0:
        tp2_rate  /= total
        tp1_rate  /= total
        stop_rate /= total

    tp2_r  = _mean_r(tp2_trades)  if tp2_trades  else BACKTEST_TP2_R
    tp1_r  = _mean_r(tp1_trades)  if tp1_trades  else BACKTEST_TP1_R
    stop_r = _mean_r(stop_trades) if stop_trades else BACKTEST_STOP_R

    ev_per_trade = tp2_rate * tp2_r + tp1_rate * tp1_r + stop_rate * stop_r

    # Trades per month estimate (based on actual date span)
    dates = [t['date'] for t in trades if t['date']]
    if len(dates) >= 2:
        span_days = (max(dates) - min(dates)).days or 1
        trades_per_month = round(n / span_days * 30, 1)
    else:
        trades_per_month = None

    return {
        'n_trades':           n,
        'tp2_rate':           round(tp2_rate,  4),
        'tp1_rate':           round(tp1_rate,  4),
        'stop_rate':          round(stop_rate, 4),
        'tp2_r':              round(tp2_r,     3),
        'tp1_r':              round(tp1_r,     3),
        'stop_r':             round(stop_r,    3),
        'ev_per_trade_r':     round(ev_per_trade, 4),
        'trades_per_month':   trades_per_month,
        'outcome_counts': {
            'TP2':  len(tp2_trades),
            'TP1':  len(tp1_trades),   # BE + TP1_TIMEOUT
            'STOP': len(stop_trades),  # STOP + TIMEOUT
        },
    }


def tolerance_check(live: dict) -> Tuple[bool, List[str]]:
    """
    Compare live edge to backtest reference.
    Returns (within_tolerance, list_of_warnings).
    """
    warnings = []

    diff_tp2 = abs(live['tp2_rate'] - BACKTEST_TP2_RATE)
    if diff_tp2 > TOLERANCE_RATE:
        warnings.append(
            f'TP2 rate drift: live={live["tp2_rate"]:.1%}  '
            f'backtest={BACKTEST_TP2_RATE:.1%}  '
            f'diff={diff_tp2*100:.1f}pp  (threshold={TOLERANCE_RATE*100:.0f}pp)'
        )

    diff_stop = abs(live['stop_rate'] - BACKTEST_STOP_RATE)
    if diff_stop > TOLERANCE_RATE:
        warnings.append(
            f'Stop rate drift: live={live["stop_rate"]:.1%}  '
            f'backtest={BACKTEST_STOP_RATE:.1%}  '
            f'diff={diff_stop*100:.1f}pp  (threshold={TOLERANCE_RATE*100:.0f}pp)'
        )

    diff_ev = abs(live['ev_per_trade_r'] - BACKTEST_EV)
    if diff_ev > TOLERANCE_EV:
        warnings.append(
            f'EV drift: live={live["ev_per_trade_r"]:.3f}R  '
            f'backtest={BACKTEST_EV:.2f}R  '
            f'diff={diff_ev:.3f}R  (threshold={TOLERANCE_EV:.2f}R)'
        )

    return len(warnings) == 0, warnings


def print_report(dist: dict, warnings: List[str], within_tol: bool,
                 days: int, source_label: str = '') -> None:
    width = 60
    day_str = f'last {days} days' if days < 365 else 'full window'
    src_str = f' [{source_label}]' if source_label else ''
    print(f'\n{"="*width}')
    print(f'  LIVE EDGE EXTRACTION — {day_str}{src_str}')
    print(f'{"="*width}')
    print(f'  Trades analysed:       {dist["n_trades"]}')
    print(f'  Outcomes:              TP2={dist["outcome_counts"]["TP2"]}  '
          f'TP1={dist["outcome_counts"]["TP1"]}  '
          f'STOP/TIMEOUT={dist["outcome_counts"]["STOP"]}')
    print()
    print(f'  {"":20}  {"LIVE":>8}  {"BACKTEST":>10}  {"DELTA":>8}')
    print(f'  {"-"*50}')

    def row(label, live_val, bt_val, fmt='.1%'):
        delta = live_val - bt_val
        sign  = '+' if delta >= 0 else ''
        print(f'  {label:20}  {live_val:{fmt}}  {bt_val:{fmt}}    {sign}{delta:{fmt}}')

    row('TP2 rate',   dist['tp2_rate'],  BACKTEST_TP2_RATE)
    row('TP1 rate',   dist['tp1_rate'],  BACKTEST_TP1_RATE)
    row('Stop rate',  dist['stop_rate'], BACKTEST_STOP_RATE)
    row('TP2 R',      dist['tp2_r'],     BACKTEST_TP2_R,  '.2f')
    row('TP1 R',      dist['tp1_r'],     BACKTEST_TP1_R,  '.2f')
    row('Stop R',     dist['stop_r'],    BACKTEST_STOP_R, '.2f')
    row('EV / trade', dist['ev_per_trade_r'], BACKTEST_EV, '.3f')
    if dist['trades_per_month']:
        print(f'  {"Trades / month":20}  {dist["trades_per_month"]:>8.1f}')

    print()
    if within_tol:
        print(f'  EDGE CHECK: ✅ WITHIN TOLERANCE — live edge matches backtest')
    else:
        print(f'  EDGE CHECK: ⚠️  DRIFT DETECTED')
        for w in warnings:
            print(f'    → {w}')
    print(f'{"="*width}\n')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract real TP distribution from paper trade log or backtest JSON.'
    )
    parser.add_argument('--log',         default=DEFAULT_LOG,
                        help=f'Path to paper trade CSV (default: {DEFAULT_LOG})')
    parser.add_argument('--source-json', action='append', dest='source_jsons',
                        metavar='FILE', default=None,
                        help='Path to ICT backtest JSON (e.g. logs/ict_backtest_window_A.json). '
                             'Can be specified multiple times to combine windows. '
                             'When provided, --log is ignored.')
    parser.add_argument('--days',        type=int, default=30,
                        help='Rolling window in days relative to the most recent trade in the '
                             'source data. Use 365 to include the full window (default: 30).')
    parser.add_argument('--min-trades',  type=int, default=20,
                        help='Minimum trades required for valid edge (default: 20)')
    parser.add_argument('--out',         default=str(OUTPUT_PATH),
                        help=f'Output JSON path (default: {OUTPUT_PATH})')
    args = parser.parse_args()

    # ── Load trades ─────────────────────────────────────────────────────────── #
    if args.source_jsons:
        # JSON mode: read from one or more backtest window files
        trades: List[dict] = []
        source_labels = []
        for jfile in args.source_jsons:
            t = load_trades_from_json(jfile, args.days)
            trades.extend(t)
            label = Path(jfile).stem
            source_labels.append(f'{label}({len(t)})')
            print(f'  Loaded {len(t)} trades from {jfile} '
                  f'(last {args.days} days of that window)')
        source_label = ', '.join(source_labels)
        source_path  = ', '.join(args.source_jsons)
    else:
        # CSV mode: live paper trade log
        trades = load_trades(args.log, args.days)
        source_label = Path(args.log).name
        source_path  = args.log

    # ── Validate count ───────────────────────────────────────────────────────── #
    if not trades:
        if args.source_jsons:
            print(f'\n  ⚠️  No trades found in the specified JSON file(s).')
        else:
            print(f'\n  ⚠️  No trades found in {args.log}')
            print(f'  The paper trading run has not produced data yet.')
            print(f'  Use --source-json logs/ict_backtest_window_A.json to use existing data.\n')
        return

    if len(trades) < args.min_trades:
        print(f'\n  ⚠️  Only {len(trades)} trades found (minimum: {args.min_trades}).')
        if not args.source_jsons:
            print(f'  Tip: use --source-json to pull from backtest windows, or '
                  f'--days 365 for the full window.')
        print(f'  Proceeding — treat results with caution (wide confidence intervals).\n')

    # ── Compute, check, report ───────────────────────────────────────────────── #
    dist = compute_distribution(trades)
    within_tol, warnings = tolerance_check(dist)

    print_report(dist, warnings, within_tol, args.days, source_label)

    # ── Write output ─────────────────────────────────────────────────────────── #
    out = {
        **dist,
        'days':             args.days,
        'source':           source_path,
        'source_type':      'backtest_json' if args.source_jsons else 'paper_trade_csv',
        'within_tolerance': within_tol,
        'tolerance_warnings': warnings,
        'computed_at':      datetime.now(timezone.utc).isoformat(),
        'backtest_reference': {
            'tp2_rate':  BACKTEST_TP2_RATE,
            'tp1_rate':  BACKTEST_TP1_RATE,
            'stop_rate': BACKTEST_STOP_RATE,
            'tp2_r':     BACKTEST_TP2_R,
            'tp1_r':     BACKTEST_TP1_R,
            'stop_r':    BACKTEST_STOP_R,
            'ev':        BACKTEST_EV,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f'  Saved to {args.out}')
    print(f'  Feed to optimizer:  python3 scripts/prop_challenge_optimizer.py '
          f'--live-edge-file {args.out}\n')


if __name__ == '__main__':
    main()
