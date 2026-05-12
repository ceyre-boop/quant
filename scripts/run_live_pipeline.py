"""
scripts/run_live_pipeline.py
=============================
End-to-end pipeline: extract live edge → optimize → validate → GO/NO-GO.

This is the full scientific process in one command:

  1. Extract real TP distribution from 30-day paper trade log
  2. Feed live edge into prop_challenge_optimizer.py
  3. Compare Monte Carlo vs walk-forward
  4. Print GO/NO-GO for a real prop challenge

Usage:
    python3 scripts/run_live_pipeline.py
    python3 scripts/run_live_pipeline.py --fast          # 2,000 MC trials (5 min scan)
    python3 scripts/run_live_pipeline.py --days 30       # default window
    python3 scripts/run_live_pipeline.py --skip-extract  # use existing live_edge.json
    python3 scripts/run_live_pipeline.py --workers 4

Output:
    logs/live_edge.json           — extracted real edge
    logs/prop_optimizer_results.json — full sweep results
    logs/prop_optimal_config.json    — best realistic config
    Final console verdict: GO / NO-GO
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


LIVE_EDGE_FILE  = 'logs/live_edge.json'
RESULTS_FILE    = 'logs/prop_optimizer_results.json'
OPTIMAL_FILE    = 'logs/prop_optimal_config.json'
PAPER_LOG       = 'logs/ict_paper_trade_log.csv'

# Validation thresholds (spec)
WF_WITHIN_5PP  = 5.0    # walk-forward within 5pp → VALIDATED
WF_WITHIN_10PP = 10.0   # 5–10pp → reduce risk
WF_OVER_10PP   = 10.0   # >10pp → DO NOT attempt challenge

RISK_ADJUST_PP = 0.25   # pp to subtract when clustering risk is borderline


def _run(cmd: list, desc: str) -> int:
    """Run a subprocess command, streaming output to stdout."""
    print(f'\n  ▶  {desc}')
    print(f'  $ {" ".join(cmd)}\n')
    result = subprocess.run(cmd, check=False)
    return result.returncode


def step1_extract(days: int, min_trades: int,
                  source_jsons: list[str] | None = None) -> bool:
    """Step 1: Extract live edge from paper trade log or backtest JSON."""
    print(f'\n{"="*65}')
    if source_jsons:
        src_desc = ', '.join(source_jsons)
        print(f'  STEP 1 — EXTRACT LIVE EDGE  (backtest JSON, last {days} days)')
        print(f'  Source: {src_desc}')
    else:
        print(f'  STEP 1 — EXTRACT LIVE EDGE  ({days}-day window)')
    print(f'{"="*65}')

    cmd = [sys.executable, 'scripts/extract_live_edge.py',
           '--days', str(days),
           '--min-trades', str(min_trades),
           '--out', LIVE_EDGE_FILE]
    if source_jsons:
        for jf in source_jsons:
            cmd += ['--source-json', jf]
    else:
        cmd += ['--log', PAPER_LOG]

    _run(cmd, 'Extracting TP distribution')

    if not Path(LIVE_EDGE_FILE).exists():
        print(f'\n  ❌ {LIVE_EDGE_FILE} was not produced.')
        if not source_jsons:
            print(f'     Ensure the paper trader has been running and producing '
                  f'data at {PAPER_LOG}')
            print(f'     Or use --source-json logs/ict_backtest_window_A.json '
                  f'to use existing backtest data.')
        return False

    data = json.loads(Path(LIVE_EDGE_FILE).read_text())
    n = data.get('n_trades', 0)
    if n < min_trades:
        print(f'\n  ❌ Only {n} trades found (need {min_trades}+).')
        if source_jsons:
            print(f'     Try --days 365 to use the full window.')
        else:
            print(f'     Continue paper trading and re-run when you have more data.')
        return False

    return True


def step2_optimize(fast: bool, workers: int | None) -> bool:
    """Step 2: Run full MC sweep with live edge."""
    print(f'\n{"="*65}')
    print(f'  STEP 2 — PROP CHALLENGE OPTIMIZER (live edge)')
    print(f'{"="*65}')

    cmd = [sys.executable, 'scripts/prop_challenge_optimizer.py',
           '--live-edge-file', LIVE_EDGE_FILE]
    if fast:
        cmd.append('--fast')
    if workers:
        cmd += ['--workers', str(workers)]

    rc = _run(cmd, 'Running 31,752 combinations × MC trials')
    return rc == 0


def step3_verdict() -> None:
    """Step 3: Read results and print GO/NO-GO verdict."""
    print(f'\n{"="*65}')
    print(f'  STEP 3 — FINAL VERDICT')
    print(f'{"="*65}')

    if not Path(RESULTS_FILE).exists():
        print(f'  ❌ Results file not found: {RESULTS_FILE}')
        return

    results = json.loads(Path(RESULTS_FILE).read_text())
    best    = results.get('best_realistic') or results.get('best', {})
    wf      = results.get('walk_forward', {})
    live    = results.get('live_edge_input') or {}

    mc_pass = best.get('pass_rate', 0) * 100
    wfa     = wf.get('A')
    wfb     = wf.get('B')
    validated = wf.get('validated', False)

    # Build a dynamic label from what was actually loaded
    n_trades   = live.get('n_trades', '?')
    days_used  = live.get('days', '?')
    src_type   = live.get('source_type', 'unknown')
    if src_type == 'backtest_json':
        edge_label = f'{n_trades} backtest trades, {days_used}-day window'
    else:
        edge_label = f'{n_trades} paper trades, last {days_used} days'

    print(f'\n  LIVE EDGE ({edge_label}):')
    print(f'  TP2 rate:           {live.get("tp2_rate", "?"):.1%}'
          if isinstance(live.get('tp2_rate'), float) else f'  TP2 rate: ?')
    print(f'  TP1 rate:           {live.get("tp1_rate", "?"):.1%}'
          if isinstance(live.get('tp1_rate'), float) else f'  TP1 rate: ?')
    print(f'  Stop rate:          {live.get("stop_rate", "?"):.1%}'
          if isinstance(live.get('stop_rate'), float) else f'  Stop rate: ?')
    print(f'  EV / trade:         {live.get("ev_per_trade_r", "?"):.3f}R'
          if isinstance(live.get('ev_per_trade_r'), float) else f'  EV / trade: ?')

    print(f'\n  OPTIMAL CONFIG (realistic signal frequency):')
    print(f'  Risk per trade:     {best.get("risk_pct", "?"):.2f}%'
          if isinstance(best.get('risk_pct'), float) else '  Risk: ?')
    print(f'  Trades / month:     {best.get("trades_per_month", "?")}')
    print(f'  Challenge window:   {best.get("challenge_days", "?")} days')
    print(f'  Profit target:      {best.get("profit_target", 0)*100:.0f}%'
          if isinstance(best.get('profit_target'), float) else '  Target: ?')
    print(f'  n_simultaneous:     {best.get("n_simultaneous", "?")}')
    print(f'  Single pass rate:   {mc_pass:.1f}%')
    print(f'  Portfolio pass:     {best.get("portfolio_pass", 0)*100:.1f}%'
          if isinstance(best.get('portfolio_pass'), float) else '  Portfolio: ?')
    print(f'  Expected cost:      ${best.get("exp_cost", 0):.0f}'
          if isinstance(best.get('exp_cost'), float) else '  Expected cost: ?')

    print(f'\n  WALK-FORWARD VALIDATION:')
    print(f'  Monte Carlo:        {mc_pass:.1f}%')
    print(f'  Walk-forward A:     {wfa if wfa else "n/a"}%')
    print(f'  Walk-forward B:     {wfb if wfb else "n/a"}%')

    # Determine verdict
    print(f'\n{"="*65}')
    if not wfa or not wfb:
        print(f'  ⚠️  Walk-forward data unavailable — cannot fully validate.')
        print(f'     Use the optimizer output as guidance only.')
        print(f'     Verdict: CONDITIONAL — review walk-forward data manually')
    else:
        avg_wf = (wfa + wfb) / 2
        diff = mc_pass - avg_wf  # positive means MC over-estimates walk-forward

        if abs(diff) <= WF_WITHIN_5PP:
            print(f'  ✅  VALIDATED — walk-forward within {abs(diff):.1f}pp of Monte Carlo')
            print(f'\n  ════════════════════════════════════════════════════')
            print(f'  🟢  GO — ATTEMPT ONE REAL PROP CHALLENGE')
            print(f'  ════════════════════════════════════════════════════')
            print(f'\n  Configuration:')
            print(f'  Risk per trade:  {best.get("risk_pct", "?"):.2f}%'
                  if isinstance(best.get('risk_pct'), float) else '  Risk: ?')
            print(f'  Trades / month:  {best.get("trades_per_month", "?")}')
            print(f'  Challenge days:  {best.get("challenge_days", "?")}')
            print(f'  Accounts:        1  (capital-constrained protocol)')
            print(f'  Firm:            FunderPro $10k challenge (~$99)')
            print(f'\n  Trust the optimizer. Not your gut.')

        elif abs(diff) <= WF_OVER_10PP:
            risk_pct = best.get('risk_pct')
            if risk_pct is None:
                adj_risk_str = '(current risk - 0.25%)'
            else:
                adj_risk_str = f'{round(risk_pct - RISK_ADJUST_PP, 2):.2f}%'
            print(f'  ⚠️  MARGINAL — walk-forward {abs(diff):.1f}pp below Monte Carlo')
            print(f'     Clustering risk is borderline.')
            print(f'\n  ════════════════════════════════════════════════════')
            print(f'  🟡  CONDITIONAL GO — reduce risk to {adj_risk_str} and re-run')
            print(f'  ════════════════════════════════════════════════════')
            print(f'\n  Action:')
            print(f'  1. Set risk_pct to {adj_risk_str} in logs/live_edge.json')
            print(f'  2. Re-run:  python3 scripts/run_live_pipeline.py --skip-extract')
            print(f'  3. If walk-forward is then within 5pp → attempt challenge')

        else:
            print(f'  ❌  EDGE UNSTABLE — walk-forward {abs(diff):.1f}pp below Monte Carlo')
            print(f'     (threshold: >{WF_OVER_10PP}pp = do not attempt)')
            print(f'\n  ════════════════════════════════════════════════════')
            print(f'  🔴  NO-GO — DO NOT ATTEMPT CHALLENGE')
            print(f'  ════════════════════════════════════════════════════')
            print(f'\n  Your real-world trade clustering is degrading the edge.')
            print(f'  Continue paper trading.  Investigate stop clustering.')
            print(f'  Re-run this pipeline in 2 weeks with more data.')

    print(f'{"="*65}\n')

    # Append pipeline run to log
    log_entry = {
        'timestamp':  datetime.now(timezone.utc).isoformat(),
        'mc_pass':    mc_pass,
        'wfa':        wfa,
        'wfb':        wfb,
        'validated':  validated,
        'best_risk':  best.get('risk_pct'),
        'live_ev':    live.get('ev_per_trade_r'),
        'live_n':     live.get('n_trades'),
    }
    log_path = Path('logs/pipeline_runs.jsonl')
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    print(f'  Pipeline run logged to {log_path}\n')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='End-to-end live pipeline: extract → optimize → validate → verdict'
    )
    parser.add_argument('--days',         type=int, default=30,
                        help='Rolling window in days (relative to most recent trade). '
                             'Use 365 to include full backtest windows. (default: 30)')
    parser.add_argument('--min-trades',   type=int, default=20)
    parser.add_argument('--fast',         action='store_true',
                        help='2,000 MC trials instead of 10,000 (quick scan)')
    parser.add_argument('--workers',      type=int, default=None)
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip Step 1 — use existing logs/live_edge.json')
    parser.add_argument('--source-json',  action='append', dest='source_jsons',
                        metavar='FILE', default=None,
                        help='Use ICT backtest JSON instead of paper trade CSV. '
                             'Can be specified multiple times. '
                             'e.g. --source-json logs/ict_backtest_window_A.json')
    args = parser.parse_args()

    print(f'\n{"="*65}')
    print(f'  LIVE EDGE → PROP CHALLENGE PIPELINE')
    print(f'  {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
    print(f'{"="*65}')
    if args.source_jsons:
        print(f'  Data source: backtest JSON windows (last {args.days} days each)')
        for jf in args.source_jsons:
            print(f'    {jf}')
    else:
        print(f'  Data source: paper trade CSV (last {args.days} days)')
    print(f'  Step 1: Extract edge from {args.days}-day sample')
    print(f'  Step 2: Re-run optimizer with live edge (31,752 param combinations)')
    print(f'  Step 3: Compare Monte Carlo vs walk-forward → GO/NO-GO')

    # Step 1
    if not args.skip_extract:
        ok = step1_extract(args.days, args.min_trades, args.source_jsons)
        if not ok:
            sys.exit(1)
    else:
        if not Path(LIVE_EDGE_FILE).exists():
            print(f'\n  ❌ --skip-extract used but {LIVE_EDGE_FILE} not found.')
            sys.exit(1)
        print(f'\n  ⏭  Skipping extraction — using existing {LIVE_EDGE_FILE}')

    # Step 2
    ok = step2_optimize(args.fast, args.workers)
    if not ok:
        print(f'\n  ❌ Optimizer failed. Check output above.')
        sys.exit(1)

    # Step 3
    step3_verdict()


if __name__ == '__main__':
    main()
