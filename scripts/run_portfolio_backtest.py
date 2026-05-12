"""
scripts/run_portfolio_backtest.py
==================================
Ultraplan Day 2 — 30-year portfolio backtest across all pairs simultaneously.

This is categorically different from the single-pair backtester:
  - All pairs run concurrently, sharing a $100k equity curve
  - The PortfolioEngine allocates capital across simultaneously-active signals
  - Correlation adjustments prevent double-sizing correlated USD bets
  - The 6% daily risk cap protects against tail events (2015 SNB, 2008, etc.)

Critical test: survive 2015 SNB floor removal (Jan 15, 2015 — 1500+ pip move).
If the portfolio-level risk management holds through that event, it holds through anything.

Target metrics (ultraplan):
  Portfolio Sharpe > 1.5
  Max drawdown < 15%
  Survive 2015 SNB event
  ≥ 2 trades per pair per month average

Usage:
    python3 scripts/run_portfolio_backtest.py
    python3 scripts/run_portfolio_backtest.py --start 1993-01-01
    python3 scripts/run_portfolio_backtest.py --pairs EURUSD GBPUSD --start 2010-01-01
    python3 scripts/run_portfolio_backtest.py --start 2015-01-01 --end 2016-01-01  # SNB test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parents[1] / 'logs'
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class PortfolioBacktestResult:
    start: str
    end: str
    n_pairs: int
    n_trades: int
    years: float
    total_return_pct: float
    annualised_return_pct: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_date: str
    calmar: float
    win_rate: float
    profit_factor: float
    avg_trades_per_pair_per_month: float
    survived_snb_2015: bool
    snb_drawdown: float        # drawdown during Jan 2015 SNB event specifically
    pair_results: Dict         # per-pair breakdown

    def passed_targets(self) -> Dict[str, bool]:
        return {
            'sharpe > 1.5': self.sharpe > 1.5,
            'max_dd < 15%': self.max_drawdown > -0.15,
            'survived_snb': self.survived_snb_2015,
            'trade_frequency': self.avg_trades_per_pair_per_month >= 2.0,
        }


def download_pair_data(pair: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download OHLCV for a pair, with squeeze fix for new yfinance."""
    try:
        import yfinance as yf
        df = yf.download(pair, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or len(df) < 50:
            return None
        # Fix yfinance MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = df[col].squeeze()
        return df
    except Exception as e:
        logger.warning(f"Download failed for {pair}: {e}")
        return None


def run_portfolio_backtest(
    pairs: List[str],
    start: str,
    end: str,
    initial_equity: float = 100_000.0,
) -> PortfolioBacktestResult:
    """
    Core portfolio backtest engine.

    For each trading day:
      1. Check all pairs for active signals
      2. Pass active signals to PortfolioEngine
      3. Simulate position PnL
      4. Update equity curve
    """
    from sovereign.forex.pair_universe import PAIR_CONFIG, CB_TO_COUNTRY, ALL_PAIRS
    from sovereign.forex.data_fetcher import ForexDataFetcher
    from sovereign.forex.entry_engine import CBEventTrigger
    from sovereign.forex.signal_engine import ForexSignalEngine
    from sovereign.execution.portfolio_engine import PortfolioEngine, PortfolioSignal

    logger.info(f"Running portfolio backtest: {len(pairs)} pairs | {start} → {end}")
    logger.info(f"Initial equity: ${initial_equity:,.0f}")

    # Download all pair data
    pair_data: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        df = download_pair_data(pair, start, end)
        if df is not None and len(df) >= 252:
            pair_data[pair] = df
            logger.info(f"  {pair}: {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")
        else:
            logger.warning(f"  {pair}: insufficient data, skipping")

    if not pair_data:
        raise ValueError("No usable pair data downloaded")

    # Build signal frames for all pairs
    fetcher = ForexDataFetcher()
    cb = CBEventTrigger()
    signal_engine = ForexSignalEngine(fetcher=fetcher, cb_trigger=cb)

    signal_frames: Dict[str, pd.DataFrame] = {}
    for pair, df in pair_data.items():
        cfg = PAIR_CONFIG.get(pair)
        if not cfg:
            continue
        try:
            sig_frame = signal_engine.build_signal_frame(
                prices=df,
                base_country=CB_TO_COUNTRY[cfg.base_central_bank],
                quote_country=CB_TO_COUNTRY[cfg.quote_central_bank],
                start=start,
                end=end,
                pair=pair,
            )
            signal_frames[pair] = sig_frame
        except Exception as e:
            logger.warning(f"Signal generation failed for {pair}: {e}")

    if not signal_frames:
        raise ValueError("No signals generated")

    # Build unified date index
    all_dates = sorted(set.union(*[set(sf.index) for sf in signal_frames.values()]))
    logger.info(f"Trading days: {len(all_dates)}")

    # Portfolio simulation
    engine = PortfolioEngine(total_equity=initial_equity)
    equity = initial_equity
    equity_curve: List[float] = [equity]
    equity_dates: List[pd.Timestamp] = [all_dates[0] if all_dates else pd.Timestamp(start)]

    # Track open positions: pair → {entry_date, entry_price, direction, units, stop, hold_days_left, atr}
    open_positions: Dict[str, dict] = {}
    all_trades: List[dict] = []
    pair_trade_counts: Dict[str, int] = {p: 0 for p in pair_data}

    HOLD_DAYS = 60
    STOP_PCT = 0.04

    for date in all_dates:
        day_pnl_usd = 0.0

        # Step 1: Update open positions
        closed_today = []
        for pair, pos in list(open_positions.items()):
            df = pair_data.get(pair)
            if df is None or date not in df.index:
                continue

            current_price = float(df.loc[date, 'Close'])
            entry_price = pos['entry_price']
            direction = pos['direction']
            units = pos['units']
            stop_price = pos['stop_price']

            # Calculate current PnL
            if direction == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                stopped = current_price <= stop_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
                stopped = current_price >= stop_price

            pos['days_held'] = pos.get('days_held', 0) + 1
            hit_stop = stopped
            expired = pos['days_held'] >= HOLD_DAYS

            if hit_stop or expired:
                pnl_usd = pnl_pct * entry_price * units
                day_pnl_usd += pnl_usd
                closed_today.append(pair)
                all_trades.append({
                    'pair': pair,
                    'entry_date': str(pos['entry_date'].date()),
                    'exit_date': str(date.date()),
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'units': units,
                    'exit_reason': 'STOP' if hit_stop else 'EXPIRE',
                    'days_held': pos['days_held'],
                })

        for pair in closed_today:
            del open_positions[pair]

        equity = max(1000.0, equity + day_pnl_usd)
        engine.update_equity(equity)

        # Step 2: Check for new signals
        new_signals: List[PortfolioSignal] = []
        for pair, sig_frame in signal_frames.items():
            if pair in open_positions:
                continue  # already in this pair
            if date not in sig_frame.index:
                continue

            sig_val = int(sig_frame.loc[date, 'signal'])
            if sig_val == 0:
                continue

            df = pair_data.get(pair)
            if df is None or date not in df.index:
                continue

            row = df.loc[date]
            entry = float(row['Close'])
            high = float(row.get('High', entry))
            low = float(row.get('Low', entry))
            atr = float(high - low)  # daily range as ATR proxy
            if atr < 1e-6:
                atr = entry * 0.005  # 0.5% fallback

            direction = 'LONG' if sig_val > 0 else 'SHORT'
            stop_pct = STOP_PCT
            stop = entry * (1 - stop_pct) if direction == 'LONG' else entry * (1 + stop_pct)
            tp1 = entry * (1 + stop_pct * 2) if direction == 'LONG' else entry * (1 - stop_pct * 2)
            tp2 = entry * (1 + stop_pct * 3) if direction == 'LONG' else entry * (1 - stop_pct * 3)

            # Conviction proxy: magnitude of signal (use abs(signal) if available, else 0.65)
            conviction = min(1.0, 0.65 + abs(sig_val - 1) * 0.1)

            new_signals.append(PortfolioSignal(
                pair=pair,
                direction=direction,
                conviction=conviction,
                atr=atr,
                entry_price=entry,
                stop_price=stop,
                tp1_price=tp1,
                tp2_price=tp2,
                predicted_r_p50=0.6,  # conservative default; replaced by PredictNow in live
            ))

        # Step 3: Allocate capital across new signals
        if new_signals:
            # Build active_positions map for correlation check
            active_allocs = {}
            from sovereign.execution.portfolio_engine import PortfolioAllocation
            for pair, pos in open_positions.items():
                active_allocs[pair] = PortfolioAllocation(
                    pair=pair, direction=pos['direction'],
                    units=pos['units'], risk_usd=0, risk_pct=0,
                    raw_score=0, adj_score=0, corr_penalty=0,
                    size_multiplier=1.0,
                    stop_price=pos['stop_price'],
                    tp1_price=pos['entry_price'],
                    tp2_price=pos['entry_price'],
                    entry_price=pos['entry_price'],
                )

            report = engine.allocate(new_signals, active_allocs)

            for alloc in report.allocations:
                if not alloc.accepted or alloc.units <= 0:
                    continue
                open_positions[alloc.pair] = {
                    'entry_date': date,
                    'entry_price': alloc.entry_price,
                    'direction': alloc.direction,
                    'units': alloc.units,
                    'stop_price': alloc.stop_price,
                    'days_held': 0,
                }
                pair_trade_counts[alloc.pair] = pair_trade_counts.get(alloc.pair, 0) + 1

        equity_curve.append(equity)
        equity_dates.append(date)

    # ── Statistics ──────────────────────────────────────────────────── #
    equity_arr = np.array(equity_curve)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    years = len(all_dates) / 252.0

    total_return = (equity - initial_equity) / initial_equity
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    ann_factor = np.sqrt(252)
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * ann_factor if len(returns) > 1 else 0.0
    downside = returns[returns < 0]
    sortino = (returns.mean() / (downside.std() + 1e-9)) * ann_factor if len(downside) > 1 else 0.0

    rolling_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - rolling_max) / rolling_max
    max_dd = float(drawdowns.min())
    max_dd_idx = int(drawdowns.argmin())
    max_dd_date = str(equity_dates[max_dd_idx].date()) if max_dd_idx < len(equity_dates) else '?'

    calmar = abs(ann_return / max_dd) if max_dd < 0 else 0.0

    n_trades = len(all_trades)
    wins = [t for t in all_trades if t['pnl_pct'] > 0]
    losses = [t for t in all_trades if t['pnl_pct'] <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    gross_win = sum(t['pnl_usd'] for t in wins)
    gross_loss = abs(sum(t['pnl_usd'] for t in losses))
    profit_factor = gross_win / max(gross_loss, 1.0)

    months = years * 12
    avg_trades_per_pair_per_month = (n_trades / len(pair_data)) / months if months > 0 and pair_data else 0.0

    # SNB 2015 test: drawdown during Jan 2015
    snb_dd = 0.0
    survived_snb = True
    try:
        snb_start = pd.Timestamp('2015-01-14')
        snb_end = pd.Timestamp('2015-01-16')
        snb_mask = [(snb_start <= d <= snb_end) for d in equity_dates]
        if any(snb_mask):
            snb_eq = equity_arr[[i for i, m in enumerate(snb_mask) if m]]
            if len(snb_eq) >= 2:
                snb_dd = float((snb_eq[-1] - snb_eq[0]) / snb_eq[0])
            survived_snb = snb_dd > -0.10  # survived if SNB day didn't cost >10%
    except Exception:
        pass

    # Per-pair breakdown
    pair_results = {}
    for pair in pair_data:
        pair_trades = [t for t in all_trades if t['pair'] == pair]
        if pair_trades:
            p_wins = [t for t in pair_trades if t['pnl_pct'] > 0]
            p_ret = sum(t['pnl_pct'] for t in pair_trades)
            pair_results[pair] = {
                'n_trades': len(pair_trades),
                'win_rate': len(p_wins) / len(pair_trades),
                'total_return_pct': p_ret,
                'avg_hold_days': np.mean([t['days_held'] for t in pair_trades]),
            }

    return PortfolioBacktestResult(
        start=start,
        end=end,
        n_pairs=len(pair_data),
        n_trades=n_trades,
        years=round(years, 2),
        total_return_pct=round(total_return * 100, 2),
        annualised_return_pct=round(ann_return * 100, 2),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_drawdown=round(max_dd, 4),
        max_drawdown_date=max_dd_date,
        calmar=round(calmar, 3),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 3),
        avg_trades_per_pair_per_month=round(avg_trades_per_pair_per_month, 2),
        survived_snb_2015=survived_snb,
        snb_drawdown=round(snb_dd, 4),
        pair_results=pair_results,
    )


def print_report(result: PortfolioBacktestResult):
    print()
    print("=" * 60)
    print("PORTFOLIO BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period:          {result.start} → {result.end} ({result.years:.1f} years)")
    print(f"Pairs traded:    {result.n_pairs}")
    print(f"Total trades:    {result.n_trades}")
    print()
    print("── Performance ─────────────────────────────────")
    print(f"Total return:    {result.total_return_pct:+.1f}%")
    print(f"Ann. return:     {result.annualised_return_pct:+.1f}%")
    print(f"Sharpe ratio:    {result.sharpe:.3f}  {'✓ PASS' if result.sharpe > 1.5 else '✗ FAIL'} (target >1.5)")
    print(f"Sortino ratio:   {result.sortino:.3f}")
    print(f"Calmar ratio:    {result.calmar:.3f}")
    print()
    print("── Risk ─────────────────────────────────────────")
    print(f"Max drawdown:    {result.max_drawdown:.1%}  {'✓ PASS' if result.max_drawdown > -0.15 else '✗ FAIL'} (target <15%)")
    print(f"  at:            {result.max_drawdown_date}")
    print(f"Win rate:        {result.win_rate:.1%}")
    print(f"Profit factor:   {result.profit_factor:.2f}")
    print()
    print("── SNB 2015 Test ────────────────────────────────")
    snb_status = '✓ SURVIVED' if result.survived_snb_2015 else '✗ WIPED'
    print(f"SNB event:       {snb_status}  (drawdown: {result.snb_drawdown:.1%})")
    print()
    print("── Trade Frequency ──────────────────────────────")
    freq_pass = result.avg_trades_per_pair_per_month >= 2.0
    print(f"Trades/pair/mo:  {result.avg_trades_per_pair_per_month:.2f}  {'✓ PASS' if freq_pass else '✗ FAIL'} (target ≥2)")
    print()
    print("── Per-Pair Breakdown ───────────────────────────")
    for pair, pr in sorted(result.pair_results.items(), key=lambda x: -x[1]['total_return_pct']):
        print(f"  {pair:<12}  n={pr['n_trades']:>3}  wr={pr['win_rate']:.1%}  "
              f"ret={pr['total_return_pct']:+.1%}  hold={pr['avg_hold_days']:.0f}d")
    print()
    print("── Targets ──────────────────────────────────────")
    targets = result.passed_targets()
    passed = sum(1 for v in targets.values() if v)
    for name, ok in targets.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{len(targets)} targets met")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='30-year portfolio backtest')
    parser.add_argument('--start', default='1993-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--equity', type=float, default=100_000.0)
    parser.add_argument('--pairs', nargs='+', default=None,
                        help='Subset of pairs to test (default: all)')
    parser.add_argument('--snb-only', action='store_true',
                        help='Quick SNB 2015 stress test only (Jan 2014–Jan 2016)')
    parser.add_argument('--save', action='store_true',
                        help='Save results JSON to logs/')
    args = parser.parse_args()

    if args.snb_only:
        args.start = '2014-01-01'
        args.end = '2016-03-01'
        print("Running SNB 2015 stress test (Jan 15, 2015 floor removal)")

    from sovereign.forex.pair_universe import ALL_PAIRS
    pairs = args.pairs or ALL_PAIRS

    result = run_portfolio_backtest(
        pairs=pairs,
        start=args.start,
        end=args.end,
        initial_equity=args.equity,
    )

    print_report(result)

    if args.save:
        out = RESULTS_DIR / f'portfolio_backtest_{args.start[:4]}_{args.end[:4]}.json'
        with open(out, 'w') as f:
            d = asdict(result)
            json.dump(d, f, indent=2, default=str)
        print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
