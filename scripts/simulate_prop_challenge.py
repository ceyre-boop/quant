"""
scripts/simulate_prop_challenge.py
===================================
Monte Carlo simulation of FunderPro challenge pass probability.

Runs 10,000 simulated challenge attempts using the ICT backtest trade
distribution and answers:
  - What % of attempts pass before hitting DD limit?
  - What's the median days to pass?
  - What's the biggest bottleneck (DD limit vs profit target)?
  - What improvement to avgR/WR would push pass rate above 80%?

FunderPro standard challenge rules (configurable):
  Account size:      $10,000
  Profit target:     8%  ($800)
  Max daily DD:      4%  ($400 from daily high)
  Max total DD:      8%  ($800 from initial)
  Min trading days:  4
  Time limit:        30 calendar days

Usage:
  python3 scripts/simulate_prop_challenge.py
  python3 scripts/simulate_prop_challenge.py --target 10 --max-dd 5 --trials 20000
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

# ── Challenge parameters ──────────────────────────────────────────────────── #
ACCOUNT         = 10_000.0
PROFIT_TARGET   = 0.08     # 8%
MAX_TOTAL_DD    = 0.08     # 8% from initial
MAX_DAILY_DD    = 0.04     # 4% from daily open
MIN_TRADE_DAYS  = 4
MAX_DAYS        = 30
RISK_PER_TRADE  = 0.01     # 1% per trade (ICT protocol)
TRADES_PER_DAY  = 0.8      # avg trades per day (4 pairs × ~0.2 trades/pair/day)

# ── ICT edge parameters (from backtest) ────────────────────────────────────── #
# Source: forex v004 macro-filtered results + ICT backtest OOS results
PAIR_STATS = {
    # GBPUSD: London + NY PM (BOE/FED structure), regime TP 3R/6R when trending
    'GBPUSD': {'wr': 0.594, 'avg_win_r': 2.4,  'avg_loss_r': -1.0, 'trades_per_month': 8},
    # EURUSD: NY PM only, regime TP, memory+heatmap filters improve WR
    'EURUSD': {'wr': 0.555, 'avg_win_r': 2.1,  'avg_loss_r': -1.0, 'trades_per_month': 4},
    # AUDUSD: NY PM only, risk-on regime filter
    'AUDUSD': {'wr': 0.535, 'avg_win_r': 2.0,  'avg_loss_r': -1.0, 'trades_per_month': 3},
    # AUDNZD: NY PM only, RBA/RBNZ differential filter
    'AUDNZD': {'wr': 0.573, 'avg_win_r': 2.0,  'avg_loss_r': -1.0, 'trades_per_month': 3},
}

# Portfolio average (used for baseline simulation)
ALL_WR      = statistics.mean(s['wr']      for s in PAIR_STATS.values())
ALL_WIN_R   = statistics.mean(s['avg_win_r'] for s in PAIR_STATS.values())
ALL_LOSS_R  = statistics.mean(s['avg_loss_r'] for s in PAIR_STATS.values())
TOTAL_TRADES_MONTH = sum(s['trades_per_month'] for s in PAIR_STATS.values())


def simulate_challenge(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float,
    trades_per_month: float,
    account: float = ACCOUNT,
    profit_target_pct: float = PROFIT_TARGET,
    max_total_dd_pct: float = MAX_TOTAL_DD,
    max_daily_dd_pct: float = MAX_DAILY_DD,
    max_days: int = MAX_DAYS,
    min_trade_days: int = MIN_TRADE_DAYS,
    risk_pct: float = RISK_PER_TRADE,
    seed: int = None,
) -> dict:
    """Simulate one challenge attempt. Returns result dict."""
    if seed is not None:
        random.seed(seed)

    balance         = account
    peak_balance    = account
    profit_target   = account * (1 + profit_target_pct)
    max_dd_floor    = account * (1 - max_total_dd_pct)
    trades_per_day  = trades_per_month / 30.0
    trade_days      = set()
    day             = 0
    total_trades    = 0

    while day < max_days:
        day += 1
        daily_open  = balance
        daily_floor = daily_open * (1 - max_daily_dd_pct)

        # How many trades today? Poisson-ish
        n_trades = 0
        for _ in range(4):  # max 4 pairs per day
            if random.random() < trades_per_day / 4:
                n_trades += 1

        for _ in range(n_trades):
            # Check balance before each trade
            if balance <= max_dd_floor:
                return _result('BUST_TOTAL_DD', day, balance, account, total_trades, trade_days)
            if balance <= daily_floor:
                break  # daily DD limit — stop trading today

            risk_dollars = balance * risk_pct
            win = random.random() < win_rate

            # Add variance: winners/losers have std dev of ~0.5R
            if win:
                r = avg_win_r + random.gauss(0, 0.4)
                r = max(r, 0.1)
            else:
                r = avg_loss_r + random.gauss(0, 0.2)
                r = min(r, -0.1)

            pnl = risk_dollars * r
            balance += pnl
            total_trades += 1
            trade_days.add(day)

            # Check after trade
            if balance <= max_dd_floor:
                return _result('BUST_TOTAL_DD', day, balance, account, total_trades, trade_days)

        if balance >= profit_target and len(trade_days) >= min_trade_days:
            return _result('PASS', day, balance, account, total_trades, trade_days)

        peak_balance = max(peak_balance, balance)

    # Time limit reached
    if balance >= profit_target and len(trade_days) >= min_trade_days:
        return _result('PASS', day, balance, account, total_trades, trade_days)

    return _result('TIMEOUT', day, balance, account, total_trades, trade_days)


def _result(outcome, days, balance, account, n_trades, trade_days):
    return {
        'outcome':     outcome,
        'days':        days,
        'final_pct':   round((balance / account - 1) * 100, 2),
        'n_trades':    n_trades,
        'trade_days':  len(trade_days),
        'passed':      outcome == 'PASS',
    }


def run_simulation(
    n_trials: int = 10_000,
    win_rate: float = None,
    avg_win_r: float = None,
    **kwargs
) -> dict:
    wr  = win_rate  or ALL_WR
    awr = avg_win_r or ALL_WIN_R

    results = [
        simulate_challenge(wr, awr, ALL_LOSS_R, TOTAL_TRADES_MONTH, **kwargs)
        for _ in range(n_trials)
    ]

    passed   = [r for r in results if r['passed']]
    busted   = [r for r in results if r['outcome'] == 'BUST_TOTAL_DD']
    timeout  = [r for r in results if r['outcome'] == 'TIMEOUT']

    pass_rate = len(passed) / n_trials

    days_to_pass = sorted(r['days'] for r in passed) if passed else []
    median_days  = days_to_pass[len(days_to_pass)//2] if days_to_pass else None

    return {
        'trials':       n_trials,
        'win_rate':     round(wr, 3),
        'avg_win_r':    round(awr, 3),
        'pass_rate':    round(pass_rate, 3),
        'pass_pct':     round(pass_rate * 100, 1),
        'bust_pct':     round(len(busted) / n_trials * 100, 1),
        'timeout_pct':  round(len(timeout) / n_trials * 100, 1),
        'median_days_to_pass': median_days,
        'avg_trades_when_passing': round(statistics.mean(r['n_trades'] for r in passed), 1) if passed else 0,
    }


def sensitivity_table(n_trials: int = 5_000) -> list:
    """Show how pass rate changes as WR and avgR improve."""
    rows = []
    for wr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.57]:
        for avg_win in [1.5, 1.8, 2.0, 2.2, 2.5]:
            r = run_simulation(n_trials=n_trials, win_rate=wr, avg_win_r=avg_win)
            rows.append({
                'wr': wr, 'avg_win_r': avg_win,
                'pass_pct': r['pass_pct'],
                'bust_pct': r['bust_pct'],
                'median_days': r['median_days_to_pass'],
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials',   type=int,   default=10_000)
    parser.add_argument('--target',   type=float, default=8.0,  help='Profit target %%')
    parser.add_argument('--max-dd',   type=float, default=8.0,  help='Max total DD %%')
    parser.add_argument('--daily-dd', type=float, default=4.0,  help='Max daily DD %%')
    parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args()

    print(f"\n{'='*62}")
    print(f"  FUNDERPRO CHALLENGE SIMULATOR — ICT + Macro Stack")
    print(f"{'='*62}")
    print(f"  Universe: GBPUSD / EURUSD / AUDUSD / AUDNZD")
    print(f"  Edge:     WR={ALL_WR:.1%}  avgWin={ALL_WIN_R:.2f}R  avgLoss={ALL_LOSS_R:.2f}R")
    print(f"  Trades:   ~{TOTAL_TRADES_MONTH}/month across 4 pairs")
    print(f"  Rules:    Target +{args.target}%  MaxDD -{args.max_dd}%  DailyDD -{args.daily_dd}%")
    print(f"  Trials:   {args.trials:,}")
    print(f"{'='*62}")

    base = run_simulation(
        n_trials=args.trials,
        profit_target_pct=args.target/100,
        max_total_dd_pct=args.max_dd/100,
        max_daily_dd_pct=args.daily_dd/100,
    )

    print(f"\n  CURRENT EDGE (macro-filtered v004):")
    print(f"  Pass rate:         {base['pass_pct']:>6.1f}%")
    print(f"  Bust rate:         {base['bust_pct']:>6.1f}%")
    print(f"  Timeout rate:      {base['timeout_pct']:>6.1f}%")
    if base['median_days_to_pass']:
        print(f"  Median days to pass: {base['median_days_to_pass']} days")
    print(f"  Avg trades to pass:  {base['avg_trades_when_passing']}")

    verdict = (
        "READY TO ATTEMPT" if base['pass_pct'] >= 70
        else "BORDERLINE — improve avgR first" if base['pass_pct'] >= 45
        else "NOT READY — significant edge gap"
    )
    print(f"\n  VERDICT: {verdict}")

    # Per-pair breakdown
    print(f"\n  PER-PAIR PASS RATES (isolated, 1 pair only):")
    for pair, stats in PAIR_STATS.items():
        r = run_simulation(
            n_trials=args.trials // 4,
            win_rate=stats['wr'],
            avg_win_r=stats['avg_win_r'],
            profit_target_pct=args.target/100,
            max_total_dd_pct=args.max_dd/100,
        )
        bar = '█' * int(r['pass_pct'] / 5)
        print(f"  {pair}: {r['pass_pct']:>5.1f}% pass  {bar}")

    if args.sensitivity:
        print(f"\n  SENSITIVITY TABLE (what WR + avgR gets you to 80% pass):")
        print(f"  {'WR':>6}  {'avgWinR':>7}  {'Pass%':>6}  {'Bust%':>6}  {'Days':>5}")
        print(f"  {'-'*40}")
        rows = sensitivity_table(n_trials=args.trials // 4)
        for row in rows:
            marker = ' ◄ TARGET' if row['pass_pct'] >= 80 else ''
            print(f"  {row['wr']:.0%}    {row['avg_win_r']:.1f}R      "
                  f"{row['pass_pct']:>5.1f}%  {row['bust_pct']:>5.1f}%  "
                  f"  {row['median_days'] or '—':>4}{marker}")

    # Save results
    out = Path('logs/prop_challenge_simulation.json')
    out.write_text(json.dumps({'base': base, 'pair_stats': PAIR_STATS,
                               'rules': {'target': args.target, 'max_dd': args.max_dd}},
                              indent=2))
    print(f"\n  Results saved to {out}")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    main()
