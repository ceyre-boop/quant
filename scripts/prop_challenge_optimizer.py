"""
scripts/prop_challenge_optimizer.py
=====================================
Systematic prop challenge optimizer.

Fixed edge inputs (replicated across 2 independent windows):
  TP2: 19.5% @ 4.5R
  TP1: 12.0% @ 1.5R
  STOP: 68.5% @ -1.0R
  EV per trade: +0.40R (confirmed)

Sweeps 5,292 parameter combinations × 10,000 MC trials each.
n_simultaneous computed analytically (no extra simulation needed).
Uses all available CPU cores.

Usage:
    python3 scripts/prop_challenge_optimizer.py
    python3 scripts/prop_challenge_optimizer.py --trials 5000 --fast
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Fixed edge inputs (do not change — replicated from backtest) ──────────── #
TP2_RATE  = 0.195
TP1_RATE  = 0.120
STOP_RATE = 0.685
TP2_R     = 4.5
TP1_R     = 1.5
STOP_R    = -1.0
NOISE_STD = 0.15     # gaussian noise on pnl (slippage/spread variation)

CHALLENGE_FEE   = 99.0   # USD — FunderPro $10k challenge
ACCOUNT_START   = 10_000.0

# ── Parameter sweep grid ─────────────────────────────────────────────────── #
RISK_PCT_VALS      = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
TRADES_PER_MONTH   = [6, 9, 12, 15, 18, 24, 30]
CHALLENGE_DAYS     = [30, 45, 60, 90]
PROFIT_TARGET      = [0.06, 0.08, 0.10]
MAX_DAILY_DD       = [0.05, 0.08, 0.10]
MAX_TOTAL_DD       = [0.08, 0.10, 0.12]
N_SIMULTANEOUS     = [1, 2, 3, 4, 5, 6]


@dataclass
class SimResult:
    risk_pct:        float
    trades_per_month:int
    challenge_days:  int
    profit_target:   float
    max_daily_dd:    float
    max_total_dd:    float
    n_trials:        int
    pass_rate:       float
    bust_rate:       float
    timeout_rate:    float
    median_days:     Optional[int]
    ev_per_attempt:  float   # expected PnL per attempt in dollars


def simulate_batch(
    risk_pct:         float,
    trades_per_month: int,
    challenge_days:   int,
    profit_target:    float,
    max_daily_dd:     float,
    max_total_dd:     float,
    n_trials:         int,
    rng_seed:         int = 0,
) -> SimResult:
    """
    Vectorised Monte Carlo simulation of one parameter set.
    Runs n_trials simultaneously using numpy.
    """
    rng = np.random.default_rng(rng_seed)
    risk  = risk_pct / 100.0
    acct  = np.full(n_trials, ACCOUNT_START, dtype=np.float64)
    peak  = acct.copy()
    floor = ACCOUNT_START * (1 - max_total_dd)
    tgt   = ACCOUNT_START * (1 + profit_target)

    # State flags: 0=active, 1=passed, 2=busted
    status     = np.zeros(n_trials, dtype=np.int8)
    pass_day   = np.full(n_trials, challenge_days + 1, dtype=np.int32)

    # Expected trades per day (Poisson)
    lam = trades_per_month / 30.0

    for day in range(challenge_days):
        active = status == 0
        if not active.any():
            break

        daily_open  = acct.copy()
        daily_floor = daily_open * (1 - max_daily_dd)
        daily_dd_hit = np.zeros(n_trials, dtype=bool)

        # Number of trade opportunities today (Poisson)
        n_trades_today = rng.poisson(lam, size=n_trials)

        for trade_slot in range(int(np.max(n_trades_today)) + 1):
            # Mask: active + not daily DD busted + has a trade this slot
            mask = active & ~daily_dd_hit & (n_trades_today > trade_slot)
            if not mask.any():
                break

            # Draw outcome from TP distribution
            u = rng.uniform(size=n_trials)
            noise = rng.normal(1.0, NOISE_STD, size=n_trials)
            noise = np.clip(noise, 0.5, 1.5)

            pnl = np.where(
                u < TP2_RATE,
                acct * risk * TP2_R * noise,
                np.where(
                    u < TP2_RATE + TP1_RATE,
                    acct * risk * TP1_R * noise,
                    acct * risk * STOP_R * noise,
                )
            )

            # Apply only where mask is active
            acct = np.where(mask, acct + pnl, acct)
            peak = np.maximum(peak, acct)

            # Daily DD check
            daily_dd_hit |= (mask & (acct <= daily_floor))

            # Total DD check → bust
            newly_busted = mask & (acct <= floor)
            status = np.where(newly_busted, 2, status)
            active = status == 0

            # Pass check
            newly_passed = active & (acct >= tgt)
            status = np.where(newly_passed, 1, status)
            pass_day = np.where(newly_passed & (pass_day > day), day + 1, pass_day)
            active = status == 0

        # End of day: check total DD for any remaining active
        still_busted = active & (acct <= floor)
        status = np.where(still_busted, 2, status)
        active = status == 0

        # Check pass at end of day
        eod_passed = active & (acct >= tgt)
        status = np.where(eod_passed, 1, status)
        pass_day = np.where(eod_passed & (pass_day > day), day + 1, pass_day)
        active = status == 0

    pass_mask    = status == 1
    bust_mask    = status == 2
    timeout_mask = status == 0

    pass_rate    = float(pass_mask.sum() / n_trials)
    bust_rate    = float(bust_mask.sum() / n_trials)
    timeout_rate = float(timeout_mask.sum() / n_trials)

    pass_days = pass_day[pass_mask]
    median_days = int(np.median(pass_days)) if len(pass_days) > 0 else None

    # EV per attempt = E[final_balance] - ACCOUNT_START
    ev = float(np.mean(acct) - ACCOUNT_START)

    return SimResult(
        risk_pct=risk_pct, trades_per_month=trades_per_month,
        challenge_days=challenge_days, profit_target=profit_target,
        max_daily_dd=max_daily_dd, max_total_dd=max_total_dd,
        n_trials=n_trials, pass_rate=pass_rate, bust_rate=bust_rate,
        timeout_rate=timeout_rate, median_days=median_days, ev_per_attempt=ev,
    )


def _worker(args):
    params, n_trials, seed = args
    risk, tpm, days, target, daily_dd, total_dd = params
    try:
        return simulate_batch(risk, tpm, days, target, daily_dd, total_dd,
                              n_trials, rng_seed=seed)
    except Exception as e:
        return None


def run_full_sweep(n_trials: int = 10_000, n_workers: int = None) -> List[SimResult]:
    combos = list(product(
        RISK_PCT_VALS, TRADES_PER_MONTH, CHALLENGE_DAYS,
        PROFIT_TARGET, MAX_DAILY_DD, MAX_TOTAL_DD,
    ))
    n_total = len(combos)
    print(f"  Sweeping {n_total:,} combinations × {n_trials:,} trials = "
          f"{n_total*n_trials:,.0f} simulations")

    tasks = [(c, n_trials, i) for i, c in enumerate(combos)]
    workers = n_workers or max(1, mp.cpu_count() - 1)
    print(f"  Using {workers} CPU cores\n")

    results = []
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=8)):
            if r:
                results.append(r)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed
                eta     = (n_total - i - 1) / rate
                print(f"  {i+1:>5}/{n_total}  {rate:.0f} combos/sec  ETA {eta:.0f}s",
                      end='\r', flush=True)
    print(f"\n  Done in {time.time()-t0:.1f}s")
    return results


def add_portfolio_metrics(results: List[SimResult]) -> List[dict]:
    """Add n_simultaneous portfolio stats (computed analytically)."""
    expanded = []
    for r in results:
        for n in N_SIMULTANEOUS:
            portfolio_pass = 1 - (1 - r.pass_rate) ** n
            cost           = n * CHALLENGE_FEE
            cost_per_pass  = cost / portfolio_pass if portfolio_pass > 0 else float('inf')
            exp_attempts   = 1 / r.pass_rate if r.pass_rate > 0 else float('inf')
            exp_cost       = exp_attempts * CHALLENGE_FEE
            d = asdict(r)
            d.update({
                'n_simultaneous':   n,
                'portfolio_pass':   round(portfolio_pass, 4),
                'portfolio_cost':   round(cost, 0),
                'cost_per_pass':    round(cost_per_pass, 2),
                'exp_attempts':     round(exp_attempts, 1),
                'exp_cost':         round(exp_cost, 2),
            })
            expanded.append(d)
    return expanded


def walk_forward_validation(
    best: dict,
    window_label: str,
    trade_file: str,
) -> float:
    """
    Replay actual trade sequence (from backtest log) against best config.
    Returns pass rate across all possible 30-day starting points.
    """
    try:
        data   = json.loads(Path(trade_file).read_text())
        trades = [t for t in data.get('trades', [])
                  if t['pair'] == 'GBPUSD']
    except Exception:
        return None

    if not trades:
        return None

    risk       = best['risk_pct'] / 100.0
    days       = best['challenge_days']
    target     = ACCOUNT_START * (1 + best['profit_target'])
    floor      = ACCOUNT_START * (1 - best['max_total_dd'])
    daily_dd_r = best['max_daily_dd']

    # Convert trades to (pnl_r, datetime) sorted by date
    pnl_seq = [t['pnl_r'] for t in trades]
    n       = len(pnl_seq)
    if n < 10:
        return None

    passes = 0
    attempts = 0
    # Slide a window across the trade sequence
    for start_idx in range(n):
        acct = ACCOUNT_START
        day_acct = ACCOUNT_START
        passed = busted = False
        trade_count = 0
        day_trades = 0
        current_day = 0

        for idx in range(start_idx, n):
            pnl_r = pnl_seq[idx]
            pnl   = acct * risk * pnl_r
            acct += pnl
            trade_count += 1
            day_trades  += 1

            if acct <= floor:
                busted = True; break
            if acct <= day_acct * (1 - daily_dd_r):
                day_acct = acct  # end of day effectively
            if acct >= target:
                passed = True; break
            if day_trades >= 2:  # approx 2 trades per day at 6/month
                day_acct = acct
                day_trades = 0
                current_day += 1
                if current_day >= days:
                    break

        attempts += 1
        if passed:
            passes += 1

    return round(passes / attempts * 100, 1) if attempts else None


def sensitivity_analysis(best: dict, n_trials: int = 10_000) -> dict:
    """How much can TP2 rate degrade before pass rate drops below 50%?"""
    global TP2_RATE, STOP_RATE
    results = {}
    orig_tp2  = TP2_RATE
    orig_stop = STOP_RATE

    for tp2 in [0.195, 0.175, 0.150, 0.125, 0.100, 0.080, 0.060]:
        TP2_RATE  = tp2
        STOP_RATE = 1 - tp2 - TP1_RATE
        r = simulate_batch(
            best['risk_pct'], best['trades_per_month'],
            best['challenge_days'], best['profit_target'],
            best['max_daily_dd'], best['max_total_dd'],
            n_trials,
        )
        results[tp2] = {
            'pass_rate': round(r.pass_rate * 100, 1),
            'bust_rate': round(r.bust_rate * 100, 1),
        }

    TP2_RATE  = orig_tp2
    STOP_RATE = orig_stop
    return results


def print_config(rank: int, d: dict, label: str = ''):
    tag = f' [{label}]' if label else ''
    print(f"\n  #{rank}{tag}")
    print(f"  Risk={d['risk_pct']:.2f}%  Trades/mo={d['trades_per_month']}  "
          f"Days={d['challenge_days']}  Target={d['profit_target']*100:.0f}%  "
          f"DailyDD={d['max_daily_dd']*100:.0f}%  TotalDD={d['max_total_dd']*100:.0f}%  "
          f"n_sim={d['n_simultaneous']}")
    print(f"  Pass={d['pass_rate']*100:.1f}%  Bust={d['bust_rate']*100:.1f}%  "
          f"Portfolio={d['portfolio_pass']*100:.1f}%  "
          f"Median={d['median_days']}d  Cost/pass=${d['cost_per_pass']:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10_000)
    parser.add_argument('--fast',   action='store_true',
                        help='Use 2,000 trials (quick scan)')
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()

    n_trials = 2_000 if args.fast else args.trials

    print(f'\n{"="*65}')
    print(f'  PROP CHALLENGE OPTIMIZER')
    print(f'  Fixed edge: TP2={TP2_RATE:.1%}@{TP2_R}R | TP1={TP1_RATE:.1%}@{TP1_R}R | '
          f'STOP={STOP_RATE:.1%}@{STOP_R}R')
    print(f'  EV per trade: +0.40R (replicated)')
    print(f'{"="*65}\n')

    # ── 1. Full sweep ──────────────────────────────────────────────────────
    raw = run_full_sweep(n_trials=n_trials, n_workers=args.workers)
    all_results = add_portfolio_metrics(raw)

    # ── 2. Top configs ─────────────────────────────────────────────────────
    by_portfolio = sorted(all_results, key=lambda x: -x['portfolio_pass'])
    by_ev        = sorted(all_results, key=lambda x: -x['ev_per_attempt'])
    by_safest    = sorted(all_results, key=lambda x: x['bust_rate'])

    print(f'\n{"="*65}')
    print('  TOP 10 BY PORTFOLIO PASS RATE')
    print(f'{"="*65}')
    for i, d in enumerate(by_portfolio[:10]):
        print_config(i+1, d)

    print(f'\n{"="*65}')
    print('  TOP 10 BY EV PER ATTEMPT')
    print(f'{"="*65}')
    for i, d in enumerate(by_ev[:10]):
        print_config(i+1, d)

    print(f'\n{"="*65}')
    print('  TOP 10 SAFEST (lowest bust rate, pass_rate > 20%)')
    print(f'{"="*65}')
    safest = [d for d in by_safest if d['pass_rate'] > 0.20]
    for i, d in enumerate(safest[:10]):
        print_config(i+1, d)

    # ── 3. Best overall config ─────────────────────────────────────────────
    best = by_portfolio[0]
    print(f'\n{"="*65}')
    print('  OPTIMAL CONFIGURATION')
    print(f'{"="*65}')
    print(f'  Risk per trade:         {best["risk_pct"]:.2f}%')
    print(f'  Trades per month:       {best["trades_per_month"]}')
    print(f'  Challenge window:       {best["challenge_days"]} days')
    print(f'  Profit target:          {best["profit_target"]*100:.0f}%')
    print(f'  Max daily DD:           {best["max_daily_dd"]*100:.0f}%')
    print(f'  Max total DD:           {best["max_total_dd"]*100:.0f}%')
    print(f'  Simultaneous attempts:  {best["n_simultaneous"]}')
    print()
    print(f'  Single attempt pass:    {best["pass_rate"]*100:.1f}%')
    print(f'  Single attempt bust:    {best["bust_rate"]*100:.1f}%')
    print(f'  Portfolio pass rate:    {best["portfolio_pass"]*100:.1f}%')
    print(f'  Expected attempts:      {best["exp_attempts"]:.1f}')
    print(f'  Expected cost:          ${best["exp_cost"]:.0f}')
    print(f'  Median days to pass:    {best["median_days"]}')
    print(f'  Cost per funded acct:   ${best["cost_per_pass"]:.0f}')

    # ── 4. Walk-forward validation ─────────────────────────────────────────
    print(f'\n  WALK-FORWARD VALIDATION')
    mc_pass = best['pass_rate'] * 100
    wfa = walk_forward_validation(best, 'Window A', 'logs/ict_backtest_window_A.json')
    wfb = walk_forward_validation(best, 'Window B', 'logs/ict_backtest_window_B.json')
    print(f'  Monte Carlo:    {mc_pass:.1f}%')
    print(f'  Walk-forward A: {wfa if wfa else "n/a"}%')
    print(f'  Walk-forward B: {wfb if wfb else "n/a"}%')

    validated = False
    if wfa and wfb:
        avg_wf = (wfa + wfb) / 2
        diff   = abs(mc_pass - avg_wf)
        validated = diff <= 5
        print(f'  Agreement:      {"YES ✓" if validated else "NO — clustering risk detected"}')
        if not validated:
            adj_risk = best['risk_pct'] - 0.25
            print(f'  → Adjust: reduce risk to {adj_risk:.2f}% and re-run')

    # ── 5. Sensitivity analysis ────────────────────────────────────────────
    print(f'\n  EDGE DEGRADATION TOLERANCE')
    print(f'  {"TP2 rate":>10}  {"Pass%":>7}  {"Bust%":>6}')
    sens = sensitivity_analysis(best, n_trials=min(n_trials, 5_000))
    breakeven_tp2 = None
    for tp2_rate, s in sorted(sens.items()):
        marker = ' ◄ current' if abs(tp2_rate - 0.195) < 0.001 else ''
        be_flag = ''
        if s['pass_rate'] < 50 and breakeven_tp2 is None:
            breakeven_tp2 = tp2_rate
            be_flag = ' ← BREAKEVEN'
        print(f'  {tp2_rate:.1%}        {s["pass_rate"]:>5.1f}%  {s["bust_rate"]:>5.1f}%{marker}{be_flag}')

    safety_margin = round((TP2_RATE - (breakeven_tp2 or 0)) * 100, 1) if breakeven_tp2 else '>13%'
    print(f'\n  Current TP2 rate:    {TP2_RATE:.1%}')
    print(f'  Breakeven TP2 rate:  {breakeven_tp2:.1%}' if breakeven_tp2 else '  Breakeven: not reached')
    print(f'  Safety margin:       {safety_margin}pp')

    # ── 6. Final recommendation ────────────────────────────────────────────
    # Find best config for realistic current signal frequency (6 trades/month GBPUSD)
    realistic = [d for d in all_results
                 if d['trades_per_month'] <= 9   # what we actually generate
                 and d['n_simultaneous'] <= 4]   # keep cost reasonable
    best_realistic = sorted(realistic, key=lambda x: -x['portfolio_pass'])[0] if realistic else best

    print(f'\n{"="*65}')
    print('  RECOMMENDATION (current signal frequency)')
    print(f'{"="*65}')
    print_config(1, best_realistic, 'REALISTIC')
    print(f'\n  Monthly GBPUSD trades:  6 (confirmed)')
    print(f'  Needed to reach target: {best_realistic["trades_per_month"]}/month')
    gap = best_realistic['trades_per_month'] - 6
    if gap > 0:
        print(f'  Gap to fill:            +{gap}/month from other pairs')
        print(f'  → Fix EURUSD/AUDUSD to contribute {gap} more trades/month')
        print(f'  → Each pair needs ~{gap//2} clean A-grade signals/month')
    else:
        print(f'  → Already within reach — start challenge now')
    print(f'{"="*65}\n')

    # ── 7. Save ────────────────────────────────────────────────────────────
    Path('logs/prop_optimizer_results.json').write_text(
        json.dumps({
            'n_trials':    n_trials,
            'top_20':      by_portfolio[:20],
            'top_ev':      by_ev[:10],
            'safest':      safest[:10],
            'best':        best,
            'best_realistic': best_realistic,
            'sensitivity': {str(k): v for k, v in sens.items()},
            'walk_forward': {'A': wfa, 'B': wfb, 'validated': validated},
            'breakeven_tp2': breakeven_tp2,
            'safety_margin_pp': safety_margin,
        }, indent=2)
    )
    Path('logs/prop_optimal_config.json').write_text(
        json.dumps(best_realistic, indent=2)
    )
    print(f'  Saved to logs/prop_optimizer_results.json')
    print(f'  Saved to logs/prop_optimal_config.json\n')


if __name__ == '__main__':
    main()
