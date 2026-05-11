"""
scripts/run_ict_backtest.py
===========================
Walk-forward backtest of the ICT Micro-Edge Pipeline on 1 year of hourly forex data.

For each pair:
  1. Download 1yr hourly OHLCV via yfinance
  2. Step through every bar inside a killzone window
  3. Run ICTPipeline on trailing 100-bar window
  4. If signal fires (grade A/A+): simulate outcome
     - Entry: next bar open
     - Stop:  ATR × 1.0 below/above entry
     - TP1:   2R  (50% of position)
     - TP2:   4R  (50% of position)
     - Max hold: 20 bars (20 hours)
  5. Record result per trade
  6. Compute portfolio-level stats
  7. Push to Firebase → dashboard shows real numbers

Usage:
    python3 scripts/run_ict_backtest.py
    python3 scripts/run_ict_backtest.py --pairs GBPUSD EURUSD  # subset
    python3 scripts/run_ict_backtest.py --no-push              # local only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from ict.pipeline import ICTPipeline, ICTSignal, ICTVeto
from ict.micro_risk import MicroRiskParams
from ict.session_classifier import SessionClassifier
from ict._atr_utils import compute_atr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

PAIRS = [
    'GBPUSD=X', 'EURUSD=X', 'AUDUSD=X',
    'USDJPY=X', 'GBPJPY=X', 'NZDUSD=X',
]

ACCOUNT_SIZE   = 10_000.0
RISK_PER_TRADE = 0.01        # 1% per trade for backtest sizing
TP1_R          = 2.0
TP2_R          = 4.0
TP1_FRAC       = 0.5         # close half at TP1
MAX_HOLD_BARS  = 20          # max 20 hours before flat
MIN_BARS       = 80          # need at least this many bars before scanning


# ── Trade record ─────────────────────────────────────────────────────────── #

@dataclass
class Trade:
    pair:       str
    direction:  str
    grade:      str
    score:      float
    session:    str
    entry_dt:   str
    entry:      float
    stop:       float
    tp1:        float
    tp2:        float
    atr:        float
    outcome:    str      # 'TP1' | 'TP2' | 'STOP' | 'TIMEOUT'
    exit_price: float
    pnl_r:      float    # R-multiple realised
    hold_bars:  int
    component_scores: dict


# ── Data ─────────────────────────────────────────────────────────────────── #

def fetch(pair: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    logger.info("Fetching %s …", pair)
    if start or end:
        df = yf.download(pair, start=start, end=end, interval='1h',
                         progress=False, auto_adjust=True)
    else:
        df = yf.download(pair, period='1y', interval='1h',
                         progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.capitalize)[['Open','High','Low','Close']].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    if start:
        df = df[df.index >= pd.Timestamp(start, tz='UTC')]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz='UTC')]
    return df


# ── Outcome simulation ────────────────────────────────────────────────────── #

def simulate_outcome(
    df: pd.DataFrame,
    signal_idx: int,
    direction: str,
    atr: float,
    fvg_limit: Optional[float] = None,
) -> tuple[str, float, float, int]:
    """
    Simulate trade outcome.

    If fvg_limit is provided, wait up to 6 bars for price to reach it
    (limit order at FVG midpoint).  If price never touches it, skip.
    Otherwise enter at next bar open (market).

    Returns (outcome, entry_price, exit_price, hold_bars).
    """
    if signal_idx + 1 >= len(df):
        return 'TIMEOUT', 0.0, 0.0, 0

    # Determine entry
    if fvg_limit is not None:
        # Wait for price to touch the FVG midpoint (limit entry)
        entry = None
        entry_bar = signal_idx + 1
        for j in range(1, 7):  # up to 6 bars for retracement
            bar_idx = signal_idx + j
            if bar_idx >= len(df):
                break
            bar = df.iloc[bar_idx]
            lo, hi = float(bar['Low']), float(bar['High'])
            if direction == 'LONG' and lo <= fvg_limit:
                entry     = fvg_limit
                entry_bar = bar_idx
                break
            elif direction == 'SHORT' and hi >= fvg_limit:
                entry     = fvg_limit
                entry_bar = bar_idx
                break
        if entry is None:
            return 'TIMEOUT', 0.0, 0.0, 0   # price never reached FVG — no trade
        signal_idx = entry_bar - 1
    else:
        entry = float(df['Open'].iloc[signal_idx + 1])
    stop_dist = atr * 1.0
    sign = 1 if direction == 'LONG' else -1

    stop = entry - sign * stop_dist
    tp1  = entry + sign * stop_dist * TP1_R
    tp2  = entry + sign * stop_dist * TP2_R

    # Walk forward bar by bar
    partial_tp1 = False
    for j in range(1, MAX_HOLD_BARS + 1):
        bar_idx = signal_idx + 1 + j
        if bar_idx >= len(df):
            break
        bar = df.iloc[bar_idx]
        hi, lo = float(bar['High']), float(bar['Low'])

        if direction == 'LONG':
            if lo <= stop:
                ep = stop if not partial_tp1 else (tp1 * TP1_FRAC + stop * (1 - TP1_FRAC))
                pnl = (ep - entry) / stop_dist
                return 'STOP', entry, ep, j
            if not partial_tp1 and hi >= tp1:
                partial_tp1 = True
            if partial_tp1 and hi >= tp2:
                ep = tp2
                pnl = TP1_FRAC * TP1_R + (1 - TP1_FRAC) * TP2_R
                return 'TP2', entry, tp2, j
        else:
            if hi >= stop:
                ep = stop if not partial_tp1 else (tp1 * TP1_FRAC + stop * (1 - TP1_FRAC))
                pnl = (entry - ep) / stop_dist
                return 'STOP', entry, ep, j
            if not partial_tp1 and lo <= tp1:
                partial_tp1 = True
            if partial_tp1 and lo <= tp2:
                return 'TP2', entry, tp2, j

    # TP1 hit but TP2 never reached in time
    if partial_tp1:
        return 'TP1', entry, tp1, MAX_HOLD_BARS

    # Neither hit — exit at last close
    last_close = float(df['Close'].iloc[min(signal_idx + MAX_HOLD_BARS, len(df) - 1)])
    return 'TIMEOUT', entry, last_close, MAX_HOLD_BARS


# ── Per-pair backtest ─────────────────────────────────────────────────────── #

def backtest_pair(pair: str, start: Optional[str] = None, end: Optional[str] = None) -> List[Trade]:
    df = fetch(pair, start=start, end=end)
    if df.empty or len(df) < MIN_BARS:
        logger.warning("%s: not enough data (%d bars)", pair, len(df))
        return []

    pipeline = ICTPipeline()
    sess_clf = SessionClassifier()
    account  = MicroRiskParams(account_size=ACCOUNT_SIZE)
    trades: List[Trade] = []
    last_signal_bar = -10   # prevent re-entering immediately after a signal

    clean = pair.replace('=X', '')
    logger.info("%s: scanning %d bars …", clean, len(df))

    for i in range(MIN_BARS, len(df) - MAX_HOLD_BARS - 2):
        ts   = df.index[i].to_pydatetime()
        sess = sess_clf.classify(ts)

        if not sess.should_trade:
            continue

        # Data says NY Open (07:00-10:00 ET) has 22% WR — not tradeable
        # London and NY PM remain active
        if sess.kill_zone_name == 'NY_Open':
            continue

        if i - last_signal_bar < 4:   # 4-bar cooldown after any signal
            continue

        # HTF trend filter: 50-bar SMA — only trade with the trend
        if i >= 50:
            sma50 = float(df['Close'].iloc[i-50:i].mean())
            price_now = float(df['Close'].iloc[i])
            # Try LONG only if above SMA, SHORT only if below
            # (enforced per-direction below, not as a full skip)

        window = df.iloc[i - MIN_BARS: i + 1]
        atr    = compute_atr(window)
        if atr <= 0:
            continue

        # HTF trend: above 50-SMA = prefer LONG, below = prefer SHORT
        sma50 = float(df['Close'].iloc[max(0,i-50):i].mean()) if i >= 50 else None
        price_now = float(df['Close'].iloc[i])
        with_trend = ('LONG' if (sma50 and price_now > sma50)
                      else 'SHORT' if (sma50 and price_now < sma50)
                      else None)

        # Try both directions but only keep with-trend signal unless both score high
        best = None
        for direction in ('LONG', 'SHORT'):
            # Skip counter-trend signals — data shows they underperform consistently
            if with_trend and direction != with_trend:
                continue
            try:
                result = pipeline.evaluate(
                    symbol=clean, direction=direction,
                    df=window, timestamp=ts, account=account, atr=atr,
                )
                score = result.score if hasattr(result, 'score') else 0.0
                if best is None or score > (best.score if hasattr(best, 'score') else 0):
                    best = result
            except Exception as e:
                logger.debug("%s %s pipeline error: %s", clean, direction, e)

        if not isinstance(best, ICTSignal) or not best.passed:
            continue

        # Signal fired — simulate outcome
        # Use FVG limit entry level if pipeline provided one
        fvg_entry = getattr(best, 'entry_level', None)
        outcome, entry_price, exit_price, hold_bars = simulate_outcome(
            df, i, best.direction, atr, fvg_limit=fvg_entry
        )
        if entry_price == 0:
            continue

        sign = 1 if best.direction == 'LONG' else -1

        # Structural stop: below/above swept level (ICT invalidation point)
        # Falls back to 1×ATR if no sweep available
        if best.sweep is not None:
            swept = best.sweep.swept_level
            buf   = 0.08 * atr
            structural_stop = swept - buf if best.direction == 'LONG' else swept + buf
            stop_dist = abs(entry_price - structural_stop)
            if stop_dist < 0.01 * atr:  # degenerate — fallback
                stop_dist = atr * 1.0
        else:
            stop_dist = atr * 1.0

        if outcome == 'TP2':
            pnl_r = TP1_FRAC * TP1_R + (1 - TP1_FRAC) * TP2_R   # 1.0 + 2.0 = 3.0R
        elif outcome == 'TP1':
            pnl_r = TP1_R * TP1_FRAC - (1 - TP1_FRAC) * 0.5       # partial TP1, rest BE ≈ 0.5R net
        elif outcome == 'STOP':
            pnl_r = -1.0
        else:  # TIMEOUT
            pnl_r = round(sign * (exit_price - entry_price) / stop_dist, 3)

        sz = best.sizing
        trades.append(Trade(
            pair=clean, direction=best.direction,
            grade=best.grade.value, score=round(best.score, 2),
            session=sess.kill_zone_name or 'UNKNOWN',
            entry_dt=ts.strftime('%Y-%m-%d %H:%M'),
            entry=round(entry_price, 5),
            stop=round(entry_price - sign * stop_dist, 5),
            tp1=round(entry_price + sign * stop_dist * TP1_R, 5),
            tp2=round(entry_price + sign * stop_dist * TP2_R, 5),
            atr=round(atr, 5),
            outcome=outcome,
            exit_price=round(exit_price, 5),
            pnl_r=round(pnl_r, 3),
            hold_bars=hold_bars,
            component_scores=best.component_scores,
        ))

        last_signal_bar = i

    logger.info("%s: %d signals → %d trades", clean, i, len(trades))
    return trades


# ── Portfolio stats ───────────────────────────────────────────────────────── #

def compute_stats(trades: List[Trade]) -> dict:
    if not trades:
        return {'n_trades': 0}

    pnls = [t.pnl_r for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls)

    # Running equity curve (1% risk per trade on starting $10k)
    equity = [ACCOUNT_SIZE]
    for p in pnls:
        equity.append(equity[-1] * (1 + p * RISK_PER_TRADE))
    equity_arr = np.array(equity)

    # Sharpe (annualised, assuming ~252 trading days, avg 2 trades/day during killzones)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252 * 4) if len(returns) > 1 else 0.0

    # Max drawdown
    peak = equity_arr[0]
    max_dd = 0.0
    for e in equity_arr:
        peak = max(peak, e)
        dd = (e - peak) / peak
        max_dd = min(max_dd, dd)

    # By pair
    by_pair = {}
    for t in trades:
        by_pair.setdefault(t.pair, []).append(t.pnl_r)
    pair_stats = {
        p: {
            'n': len(rs),
            'win_rate': round(sum(1 for r in rs if r > 0) / len(rs), 3),
            'avg_r': round(np.mean(rs), 3),
            'total_r': round(sum(rs), 3),
        }
        for p, rs in by_pair.items()
    }

    # By session
    by_session = {}
    for t in trades:
        by_session.setdefault(t.session, []).append(t.pnl_r)
    session_stats = {
        s: {
            'n': len(rs),
            'win_rate': round(sum(1 for r in rs if r > 0) / len(rs), 3),
            'avg_r': round(np.mean(rs), 3),
        }
        for s, rs in by_session.items()
    }

    # By grade
    by_grade = {}
    for t in trades:
        by_grade.setdefault(t.grade, []).append(t.pnl_r)
    grade_stats = {
        g: {
            'n': len(rs),
            'win_rate': round(sum(1 for r in rs if r > 0) / len(rs), 3),
            'avg_r': round(np.mean(rs), 3),
        }
        for g, rs in by_grade.items()
    }

    return {
        'n_trades':      len(trades),
        'win_rate':      round(win_rate, 4),
        'avg_r':         round(np.mean(pnls), 4),
        'total_r':       round(sum(pnls), 3),
        'sharpe':        round(float(sharpe), 3),
        'max_dd_pct':    round(float(max_dd) * 100, 2),
        'final_equity':  round(float(equity_arr[-1]), 2),
        'return_pct':    round((float(equity_arr[-1]) / ACCOUNT_SIZE - 1) * 100, 2),
        'avg_win_r':     round(np.mean(wins), 3) if wins else 0,
        'avg_loss_r':    round(np.mean(losses), 3) if losses else 0,
        'profit_factor': round(sum(wins) / abs(sum(losses)) + 1e-9, 3) if losses else 999.0,
        'tp2_rate':      round(sum(1 for t in trades if t.outcome == 'TP2') / len(trades), 3),
        'tp1_rate':      round(sum(1 for t in trades if t.outcome == 'TP1') / len(trades), 3),
        'stop_rate':     round(sum(1 for t in trades if t.outcome == 'STOP') / len(trades), 3),
        'timeout_rate':  round(sum(1 for t in trades if t.outcome == 'TIMEOUT') / len(trades), 3),
        'avg_hold_bars': round(np.mean([t.hold_bars for t in trades]), 1),
        'by_pair':       pair_stats,
        'by_session':    session_stats,
        'by_grade':      grade_stats,
        'equity_curve':  [round(e, 2) for e in equity_arr.tolist()[::max(1, len(equity_arr)//100)]],
        'period':        '1Y hourly · 6 forex pairs',
        'run_at':        datetime.now(timezone.utc).isoformat(),
    }


# ── Firebase push ─────────────────────────────────────────────────────────── #

def push_to_firebase(stats: dict, trades: List[Trade]):
    try:
        import firebase_admin
        from firebase_admin import credentials, db as rtdb

        sa_path = 'config/firebase_service_account.json'
        if not firebase_admin._apps:
            cred = credentials.Certificate(sa_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://clawd-trading-7b8de-default-rtdb.firebaseio.com'
            })
        ref = rtdb.reference('/')

        # Backtest stats
        ref.child('signals/ICT_ENGINE/backtest').set(stats)

        # Recent trades (last 50)
        recent = [asdict(t) for t in trades[-50:]]
        ref.child('signals/ICT_ENGINE/backtest_trades').set(recent)

        logger.info("Pushed backtest results to Firebase ✓")
    except Exception as e:
        logger.warning("Firebase push failed: %s", e)


# ── Main ──────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs',   nargs='+', default=None)
    parser.add_argument('--no-push', action='store_true')
    parser.add_argument('--start',   default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end',     default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--label',   default='', help='Label for this run (e.g. H1, H2, OOS)')
    args = parser.parse_args()

    pairs = [f'{p}=X' if '=X' not in p else p for p in args.pairs] if args.pairs else PAIRS

    all_trades: List[Trade] = []
    for pair in pairs:
        try:
            trades = backtest_pair(pair, start=args.start, end=args.end)
            all_trades.extend(trades)
        except Exception as e:
            logger.error("Backtest failed for %s: %s", pair, e)

    stats = compute_stats(all_trades)

    # Print summary
    print('\n' + '='*60)
    print(f'  ICT ENGINE — 1-YEAR BACKTEST RESULTS')
    print('='*60)
    print(f'  Pairs:       {", ".join(p.replace("=X","") for p in pairs)}')
    print(f'  Trades:      {stats["n_trades"]}')
    print(f'  Win rate:    {stats.get("win_rate",0)*100:.1f}%')
    print(f'  Avg R:       {stats.get("avg_r",0):+.3f}R')
    print(f'  Total R:     {stats.get("total_r",0):+.1f}R')
    print(f'  Sharpe:      {stats.get("sharpe",0):.3f}')
    print(f'  Max DD:      {stats.get("max_dd_pct",0):.1f}%')
    print(f'  Return:      {stats.get("return_pct",0):+.1f}%  (${ACCOUNT_SIZE:,.0f} → ${stats.get("final_equity",ACCOUNT_SIZE):,.0f})')
    print(f'  Profit factor: {stats.get("profit_factor",0):.2f}')
    print()
    print(f'  Outcomes:  TP2={stats.get("tp2_rate",0)*100:.0f}%  TP1={stats.get("tp1_rate",0)*100:.0f}%  STOP={stats.get("stop_rate",0)*100:.0f}%  TIMEOUT={stats.get("timeout_rate",0)*100:.0f}%')
    print()

    if stats.get('by_pair'):
        print('  By pair:')
        for pair_name, ps in sorted(stats['by_pair'].items(), key=lambda x: -x[1]['total_r']):
            print(f'    {pair_name:8}  n={ps["n"]:3d}  WR={ps["win_rate"]*100:.0f}%  avgR={ps["avg_r"]:+.3f}  totalR={ps["total_r"]:+.1f}')

    if stats.get('by_session'):
        print('\n  By session:')
        for sess, ss in sorted(stats['by_session'].items(), key=lambda x: -x[1]['n']):
            print(f'    {sess:12}  n={ss["n"]:3d}  WR={ss["win_rate"]*100:.0f}%  avgR={ss["avg_r"]:+.3f}')

    if stats.get('by_grade'):
        print('\n  By grade:')
        for grade, gs in sorted(stats['by_grade'].items()):
            print(f'    {grade:6}  n={gs["n"]:3d}  WR={gs["win_rate"]*100:.0f}%  avgR={gs["avg_r"]:+.3f}')

    print('='*60)

    # Save locally
    out = Path('logs/ict_backtest_results.json')
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'stats': stats, 'trades': [asdict(t) for t in all_trades]}, f, indent=2, default=str)
    logger.info("Saved to %s", out)

    if not args.no_push:
        push_to_firebase(stats, all_trades)


if __name__ == '__main__':
    main()
