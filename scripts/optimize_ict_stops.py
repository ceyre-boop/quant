"""
scripts/optimize_ict_stops.py
==============================
Two diagnostics in one script:

1. STOP WIDTH SWEEP — test 0.08 / 0.15 / 0.25 / 0.40 × ATR buffer on GBPUSD
   Finds the stop width that maximizes EV (WR × avgWinR - (1-WR))
   Also runs the prop challenge sim at each width.

2. PAIR DIAGNOSIS — compare GBPUSD vs EURUSD vs AUDUSD structure
   Reports: spread-equivalent pips, ATR/stop ratio, session split, stop rate
   Identifies why 14% WR pairs fail where GBPUSD succeeds.

Usage:
    python3 scripts/optimize_ict_stops.py
    python3 scripts/optimize_ict_stops.py --pair GBPUSD --windows A B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import yfinance as yf

from ict.pipeline import ICTPipeline, ICTSignal, ICTVeto, ICTGrade
from ict.micro_risk import MicroRiskParams
from ict.session_classifier import SessionClassifier
from ict._atr_utils import compute_atr

ACCOUNT        = 10_000.0
RISK_PCT       = 0.01
MAX_HOLD_BARS  = 20
MIN_BARS       = 80
TP1_R          = 2.0
TP1_FRAC       = 0.5
LONDON_PAIRS   = {'GBPUSD'}


def fetch(pair: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(pair, period='730d', interval='1h',
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.capitalize)[['Open','High','Low','Close']].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[(df.index >= pd.Timestamp(start, tz='UTC')) &
            (df.index <= pd.Timestamp(end, tz='UTC'))]
    return df


def run_backtest(pair: str, df: pd.DataFrame, stop_buf_atr: float) -> dict:
    """Run ICT backtest on pair with given stop buffer multiplier."""
    clean     = pair.replace('=X','')
    pipeline  = ICTPipeline()
    sess_clf  = SessionClassifier()
    account   = MicroRiskParams(account_size=ACCOUNT)
    trades    = []
    last_sig  = -10

    for i in range(MIN_BARS, len(df) - MAX_HOLD_BARS - 2):
        ts   = df.index[i].to_pydatetime()
        sess = sess_clf.classify(ts)
        if not sess.should_trade:
            continue
        kz = sess.kill_zone_name
        if kz == 'NY_Open':
            continue
        if kz == 'London' and clean not in LONDON_PAIRS:
            continue
        if i - last_sig < 4:
            continue

        window = df.iloc[i - MIN_BARS: i + 1]
        atr    = compute_atr(window)
        if atr <= 0:
            continue

        sma50     = float(df['Close'].iloc[max(0,i-50):i].mean()) if i >= 50 else None
        price_now = float(df['Close'].iloc[i])
        with_trend = ('LONG' if (sma50 and price_now > sma50)
                      else 'SHORT' if (sma50 and price_now < sma50)
                      else None)

        best = None
        for direction in ('LONG', 'SHORT'):
            if with_trend and direction != with_trend:
                continue
            try:
                r = pipeline.evaluate(symbol=clean, direction=direction,
                                      df=window, timestamp=ts,
                                      account=account, atr=atr)
                if isinstance(r, ICTSignal) and r.passed and r.grade == ICTGrade.A:
                    if best is None or r.score > best.score:
                        best = r
            except Exception:
                pass

        if best is None:
            continue

        entry_price = float(df['Open'].iloc[i + 1])
        sign        = 1 if best.direction == 'LONG' else -1

        # Stop with variable buffer
        if best.sweep is not None:
            swept     = best.sweep.swept_level
            buf       = stop_buf_atr * atr
            stop_dist = abs(entry_price - (swept - buf if best.direction == 'LONG' else swept + buf))
            if stop_dist < 0.01 * atr:
                stop_dist = atr * 1.0
        else:
            stop_dist = atr * 1.0

        # TP levels
        tp1_r = 3.0 if (clean == 'GBPUSD' and kz == 'London') else 2.0
        tp2_r = 6.0 if (clean == 'GBPUSD' and kz == 'London') else 4.0

        tp1_price = entry_price + sign * stop_dist * tp1_r
        tp2_price = entry_price + sign * stop_dist * tp2_r
        stop_price = entry_price - sign * stop_dist

        # Simulate outcome
        outcome = 'TIMEOUT'
        exit_price = float(df['Close'].iloc[min(i + MAX_HOLD_BARS, len(df)-1)])
        hold_bars = 0
        partial_closed = False

        for j in range(1, MAX_HOLD_BARS + 1):
            if i + j >= len(df):
                break
            hi = float(df['High'].iloc[i + j])
            lo = float(df['Low'].iloc[i + j])
            hold_bars = j

            stop_hit = (best.direction == 'LONG' and lo <= stop_price) or \
                       (best.direction == 'SHORT' and hi >= stop_price)
            if stop_hit:
                exit_price = stop_price
                outcome    = 'STOP'
                break

            if not partial_closed:
                tp1_hit = (best.direction == 'LONG' and hi >= tp1_price) or \
                          (best.direction == 'SHORT' and lo <= tp1_price)
                if tp1_hit:
                    partial_closed = True
                    stop_price     = entry_price  # move to BE

            if partial_closed:
                tp2_hit = (best.direction == 'LONG' and hi >= tp2_price) or \
                          (best.direction == 'SHORT' and lo <= tp2_price)
                if tp2_hit:
                    exit_price = tp2_price
                    outcome    = 'TP2'
                    break
        else:
            if partial_closed:
                outcome = 'TP1'

        if outcome == 'TP2':
            pnl_r = TP1_FRAC * tp1_r + (1 - TP1_FRAC) * tp2_r
        elif outcome == 'TP1':
            pnl_r = tp1_r * TP1_FRAC
        elif outcome == 'STOP':
            pnl_r = -1.0
        else:
            pnl_r = sign * (exit_price - entry_price) / stop_dist

        trades.append({
            'pair': clean, 'direction': best.direction, 'session': kz,
            'score': best.score, 'outcome': outcome,
            'pnl_r': round(pnl_r, 3), 'stop_dist_atr': round(stop_dist / atr, 3),
            'atr': atr,
        })
        last_sig = i

    return _stats(trades, stop_buf_atr)


def _stats(trades: list, buf: float) -> dict:
    if not trades:
        return {'stop_buf': buf, 'n': 0, 'wr': 0, 'avg_r': 0, 'ev': 0,
                'avg_win_r': 0, 'stop_rate': 0, 'sharpe': 0}
    pnls  = [t['pnl_r'] for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses= [p for p in pnls if p < 0]
    wr    = len(wins) / len(pnls)
    avg_r = float(np.mean(pnls))
    avg_w = float(np.mean(wins)) if wins else 0
    avg_l = float(np.mean(losses)) if losses else 0
    ev    = round(wr * avg_w + (1 - wr) * avg_l, 4)
    stop_rate = sum(1 for t in trades if t['outcome'] == 'STOP') / len(trades)

    equity = [ACCOUNT]
    for p in pnls:
        equity.append(equity[-1] * (1 + p * RISK_PCT))
    eq = np.array(equity)
    dd  = float(np.min((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)))
    ret_arr = np.diff(eq) / eq[:-1]
    sharpe  = float(np.mean(ret_arr) / (np.std(ret_arr) + 1e-9) * np.sqrt(252 * 6.5))

    # Per-session
    by_sess = {}
    for t in trades:
        s = t['session']
        by_sess.setdefault(s, []).append(t['pnl_r'])
    sess_stats = {s: {'n': len(v), 'wr': round(sum(1 for x in v if x>0)/len(v),3),
                      'avg_r': round(float(np.mean(v)),3)} for s,v in by_sess.items()}

    return {
        'stop_buf':   buf,
        'n':          len(trades),
        'wr':         round(wr, 3),
        'avg_r':      round(avg_r, 4),
        'avg_win_r':  round(avg_w, 3),
        'avg_loss_r': round(avg_l, 3),
        'ev':         ev,
        'stop_rate':  round(stop_rate, 3),
        'max_dd_pct': round(dd * 100, 2),
        'sharpe':     round(sharpe, 3),
        'by_session': sess_stats,
    }


def prop_pass_rate(wr, avg_win_r, n_per_month=6, n_trials=5000):
    """Quick Monte Carlo prop pass probability."""
    import random, statistics
    passed = 0
    for _ in range(n_trials):
        bal   = ACCOUNT
        floor = ACCOUNT * 0.92      # 8% max DD
        target= ACCOUNT * 1.08      # 8% profit target
        tpd   = n_per_month / 30.0  # trades per day
        for day in range(30):
            daily_floor = bal * 0.96   # 4% daily DD
            for _ in range(4):
                if random.random() > tpd / 4: continue
                if bal <= floor or bal <= daily_floor: break
                risk = bal * RISK_PCT
                win  = random.random() < wr
                r    = (avg_win_r + random.gauss(0, 0.4)) if win else (-1.0 + random.gauss(0, 0.2))
                bal += risk * max(min(r, 10), -2)
            if bal >= target:
                passed += 1
                break
    return round(passed / n_trials * 100, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', nargs='+',
                        default=['GBPUSD=X','EURUSD=X','AUDUSD=X'])
    parser.add_argument('--start', default='2024-05-01')
    parser.add_argument('--end',   default='2025-05-01')
    args = parser.parse_args()

    # ── 1. Stop width sweep on GBPUSD ─────────────────────────────────────
    print('\n' + '='*65)
    print('  STOP WIDTH OPTIMIZATION — GBPUSD')
    print(f'  Period: {args.start} → {args.end}')
    print('='*65)

    df_gbp = fetch('GBPUSD=X', args.start, args.end)
    print(f'  Data: {len(df_gbp)} bars\n')

    results = []
    for buf in [0.08, 0.15, 0.25, 0.40]:
        r = run_backtest('GBPUSD=X', df_gbp, buf)
        pp = prop_pass_rate(r['wr'], r['avg_win_r'])
        r['prop_pass'] = pp
        results.append(r)
        marker = ' ◄ BEST EV' if r['ev'] == max(x['ev'] for x in results) else ''
        print(f'  buf={buf:.2f}×ATR: n={r["n"]:3d}  WR={r["wr"]*100:.0f}%  '
              f'avgWin={r["avg_win_r"]:.2f}R  EV={r["ev"]:+.3f}R  '
              f'stop%={r["stop_rate"]*100:.0f}%  DD={r["max_dd_pct"]:.1f}%  '
              f'prop={r["prop_pass"]}%{marker}')

    best = max(results, key=lambda x: x['ev'])
    print(f'\n  OPTIMAL STOP: {best["stop_buf"]:.2f}×ATR')
    print(f'  EV={best["ev"]:+.3f}R  WR={best["wr"]*100:.0f}%  '
          f'avgWin={best["avg_win_r"]:.2f}R  prop={best["prop_pass"]}%')

    # Session breakdown at optimal stop
    print(f'\n  By session (buf={best["stop_buf"]}×ATR):')
    for sess, ss in best.get('by_session', {}).items():
        print(f'    {sess:12} n={ss["n"]:3d}  WR={ss["wr"]*100:.0f}%  avgR={ss["avg_r"]:+.3f}')

    # ── 2. Pair diagnosis ──────────────────────────────────────────────────
    print('\n' + '='*65)
    print('  PAIR DIAGNOSIS — why GBPUSD works, others fail')
    print('='*65)

    for pair in args.pairs:
        clean = pair.replace('=X','')
        df_p  = fetch(pair, args.start, args.end)
        if len(df_p) < 200:
            print(f'  {clean}: insufficient data')
            continue
        r = run_backtest(pair, df_p, 0.15)   # use 0.15 for fair comparison
        print(f'\n  {clean} (buf=0.15×ATR):')
        print(f'    n={r["n"]}  WR={r["wr"]*100:.0f}%  avgWin={r["avg_win_r"]:.2f}R  '
              f'EV={r["ev"]:+.3f}R  stop%={r["stop_rate"]*100:.0f}%  '
              f'DD={r["max_dd_pct"]:.1f}%  Sharpe={r["sharpe"]:.2f}')
        for sess, ss in r.get('by_session', {}).items():
            print(f'    {sess:12} n={ss["n"]:3d}  WR={ss["wr"]*100:.0f}%  avgR={ss["avg_r"]:+.3f}')

    # Save
    out = Path('logs/ict_stop_optimization.json')
    out.write_text(json.dumps({'gbpusd_sweep': results,
                               'best_stop_buf': best['stop_buf']}, indent=2))
    print(f'\n  Saved to {out}')
    print('='*65 + '\n')


if __name__ == '__main__':
    main()
