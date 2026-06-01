#!/usr/bin/env python3
"""
Alta Investments — System Version Snapshot Tool

Saves a complete, reproducible snapshot of the current system state.
Every version must prove improvement over the previous in a declared metric.

Usage:
    python3 scripts/save_version.py --label "improved USDCHF with SNB filter"
    python3 scripts/save_version.py --compare   (show current vs last saved)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT        = Path(__file__).parents[1]
RESEARCH    = ROOT / 'logs' / 'research'
RESULTS     = ROOT / 'logs' / 'forex_backtest_results.json'
TRADES      = ROOT / 'logs' / 'forex_backtest_trades.json'
BRIEF       = ROOT / 'logs' / 'charts' / 'alta_research_brief.png'


def _next_version() -> str:
    existing = sorted([d.name for d in RESEARCH.iterdir() if d.is_dir() and d.name.startswith('v')]) if RESEARCH.exists() else []
    if not existing:
        return 'v001'
    last_num = int(existing[-1][1:])
    return f'v{last_num + 1:03d}'


def _last_version() -> str | None:
    if not RESEARCH.exists():
        return None
    existing = sorted([d.name for d in RESEARCH.iterdir() if d.is_dir() and d.name.startswith('v')])
    return existing[-1] if existing else None


def _load_results() -> list[dict]:
    with open(RESULTS) as f:
        return json.load(f)


def _weighted_avg_sharpe(results: list[dict]) -> float:
    """√n-weighted portfolio Sharpe (SE(Sharpe) ∝ 1/√n) — thin pairs count less."""
    import math
    pairs = [(r['sharpe'], r.get('total_trades', 0)) for r in results
             if r.get('total_trades', 0) > 0]
    if not pairs:
        return 0.0
    weights = [math.sqrt(n) for _, n in pairs]
    return sum(s * w for (s, _), w in zip(pairs, weights)) / sum(weights)


def _metrics(results: list[dict]) -> dict:
    return {
        'sharpe_by_pair':        {r['pair']: r['sharpe'] for r in results},
        'win_rate_by_pair':      {r['pair']: r['win_rate'] for r in results},
        'pf_by_pair':            {r['pair']: r['profit_factor'] for r in results},
        'max_dd_by_pair':        {r['pair']: r['max_drawdown'] for r in results},
        'trades_by_pair':        {r['pair']: r['total_trades'] for r in results},
        'positive_sharpe_count': sum(1 for r in results if r['sharpe'] > 0),
        'total_pairs':           len(results),
        'avg_sharpe':            round(_weighted_avg_sharpe(results), 4),
        'best_pair':             max(results, key=lambda r: r['sharpe'])['pair'],
        'best_sharpe':           round(max(r['sharpe'] for r in results), 4),
        'worst_pair':            min(results, key=lambda r: r['sharpe'])['pair'],
        'worst_sharpe':          round(min(r['sharpe'] for r in results), 4),
        'total_trades_all':      sum(r['total_trades'] for r in results),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT
        ).decode().strip()
    except Exception:
        return 'unknown'


def compare() -> None:
    last = _last_version()
    if not last:
        print("No saved versions yet.")
        return

    meta_path = RESEARCH / last / 'metadata.json'
    with open(meta_path) as f:
        prev = json.load(f)

    current = _metrics(_load_results())
    prev_m  = prev['metrics']

    print(f"\n{'─'*60}")
    print(f"  COMPARISON: {last}  →  CURRENT")
    print(f"{'─'*60}")
    print(f"  {'PAIR':12s}  {'PREV':>8s}  {'NOW':>8s}  {'DELTA':>8s}")
    print(f"  {'─'*44}")

    deltas = []
    for pair in current['sharpe_by_pair']:
        prev_s = prev_m['sharpe_by_pair'].get(pair, 0)
        now_s  = current['sharpe_by_pair'][pair]
        delta  = now_s - prev_s
        deltas.append(delta)
        flag = '▲' if delta > 0.05 else ('▼' if delta < -0.05 else '·')
        lbl  = pair.replace('=X', '')
        print(f"  {lbl:12s}  {prev_s:>+8.3f}  {now_s:>+8.3f}  {delta:>+8.3f}  {flag}")

    avg_delta = sum(deltas) / len(deltas)
    print(f"\n  Avg Sharpe  {prev_m['avg_sharpe']:>+8.4f}  {current['avg_sharpe']:>+8.4f}  {avg_delta:>+8.4f}")
    print(f"  Positive pairs  {prev_m['positive_sharpe_count']:>3d}  →  {current['positive_sharpe_count']:>3d}")
    print(f"\n  To qualify as v{int(last[1:])+1:03d}: avg_sharpe must exceed "
          f"{prev_m['avg_sharpe'] + prev['to_beat_in_v002']['minimum_improvement']:.4f}")
    verdict = '✓ IMPROVEMENT' if avg_delta > 0 else '✗ NO IMPROVEMENT'
    print(f"  Verdict: {verdict}  (Δ avg_sharpe = {avg_delta:+.4f})\n")


def save(label: str) -> None:
    version = _next_version()
    out_dir = RESEARCH / version
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results()
    current = _metrics(results)

    # Check improvement against last version
    last = _last_version()
    prev_avg = None
    if last and (RESEARCH / last / 'metadata.json').exists():
        with open(RESEARCH / last / 'metadata.json') as f:
            prev_meta = json.load(f)
        prev_avg = prev_meta['metrics']['avg_sharpe']
        min_improvement = prev_meta.get('to_beat_in_v002', {}).get('minimum_improvement', 0.05)
        delta = current['avg_sharpe'] - prev_avg
        if delta < min_improvement:
            print(f"\n⚠  Improvement gate: avg_sharpe delta = {delta:+.4f} "
                  f"(need >{min_improvement:+.4f})")
            print(f"   Current: {current['avg_sharpe']:.4f}  |  Previous ({last}): {prev_avg:.4f}")
            ans = input("   Save anyway? This overrides the improvement gate. [y/N] ").strip().lower()
            if ans != 'y':
                print("   Aborted. Improve the system first.")
                return

    # Copy artifacts
    if BRIEF.exists():
        shutil.copy(BRIEF, out_dir / f'chart_{version}.png')
    if RESULTS.exists():
        shutil.copy(RESULTS, out_dir / 'backtest_results.json')
    if TRADES.exists():
        shutil.copy(TRADES, out_dir / 'backtest_trades.json')

    worst_4 = sorted(results, key=lambda r: r['sharpe'])[:4]

    meta = {
        'version':           version,
        'label':             label,
        'date':              datetime.now().strftime('%Y-%m-%d %H:%M'),
        'git_commit':        _git_commit(),
        'backtest_period':   '2015-01-01 to 2024-12-31',
        'metrics':           current,
        'improvement_vs_prev': {
            'prev_version':  last,
            'prev_avg_sharpe': prev_avg,
            'delta_avg_sharpe': round(current['avg_sharpe'] - prev_avg, 4) if prev_avg else None,
        },
        'to_beat_in_next': {
            'target_metric':       'avg_sharpe across all 11 pairs',
            'current_value':       current['avg_sharpe'],
            'minimum_improvement': 0.05,
            'target_positive_pairs': current['positive_sharpe_count'] + 1,
            'weakest_pairs':       [r['pair'] for r in worst_4],
            'notes': f"Focus on {', '.join(r['pair'].replace('=X','') for r in worst_4[:2])} first",
        },
        'chart': f'chart_{version}.png',
    }

    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓  Saved {version}: '{label}'")
    print(f"   Avg Sharpe: {current['avg_sharpe']:.4f}  |  "
          f"Positive pairs: {current['positive_sharpe_count']}/{current['total_pairs']}")
    print(f"   Next target: avg_sharpe > {current['avg_sharpe'] + 0.05:.4f}")
    print(f"   Weakest to fix: {', '.join(r['pair'].replace('=X','') for r in worst_4[:2])}")
    print(f"   Saved to: {out_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description='Alta Investments — version snapshot')
    parser.add_argument('--label', type=str, default='',
                        help='Description of what changed in this version')
    parser.add_argument('--compare', action='store_true',
                        help='Compare current state to last saved version')
    args = parser.parse_args()

    if args.compare:
        compare()
    elif args.label:
        save(args.label)
    else:
        print("Usage:")
        print("  python3 scripts/save_version.py --label 'what changed'")
        print("  python3 scripts/save_version.py --compare")


if __name__ == '__main__':
    main()
