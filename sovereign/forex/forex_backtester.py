"""
Forex backtester — macro swing strategy.

Monthly signal → enter at open next day → hold HOLD_DAYS or until reversal.
Uses direct pandas trade simulation (not fast_engine) because macro FX signals
have 40-90 day holds where ATR stops are counterproductive.

fast_engine is still available for equity-style sub-strategies.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from sovereign.forex.pair_universe import ALL_PAIRS, PAIR_CONFIG, CB_TO_COUNTRY
from sovereign.forex.data_fetcher import ForexDataFetcher, FALLBACK_RATES, FALLBACK_CPI
from sovereign.forex.fair_value import FairValueModel
from sovereign.forex.cycle_detector import CycleDetector
logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).parents[2] / 'logs' / 'forex_backtest_results.json'
RESULTS_PATH.parent.mkdir(exist_ok=True)


@dataclass
class ForexBacktestResult:
    pair: str
    win_rate: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    avg_hold_days: float
    trades_per_year: float
    total_trades: int
    years: float


class ForexBacktester:

    HOLD_DAYS = 60          # hold per signal — macro signals play out over 2-3 months
    STOP_PCT = 0.04         # 4% hard stop — wide enough for FX swing, tight enough to limit ruin
    SIGNAL_THRESHOLD = 0.20

    def __init__(self, start: str = '2015-01-01', end: str = '2024-12-31'):
        self.start = start
        self.end = end
        self._fetcher = ForexDataFetcher()

    def backtest_pair(self, pair: str) -> Optional[ForexBacktestResult]:
        cfg = PAIR_CONFIG.get(pair)
        if not cfg:
            return None

        df = self._download_price(pair)
        if df is None or len(df) < 252:
            logger.warning(f"Insufficient data for {pair}")
            return None

        base_country = CB_TO_COUNTRY[cfg.base_central_bank]
        quote_country = CB_TO_COUNTRY[cfg.quote_central_bank]

        signals = self._generate_monthly_signals(df, base_country, quote_country)
        trades = self._simulate_trades(df, signals)

        if not trades:
            return None

        return self._compute_stats(pair, trades, len(df))

    def backtest_all(self) -> List[ForexBacktestResult]:
        results = []
        for pair in ALL_PAIRS:
            try:
                r = self.backtest_pair(pair)
                if r:
                    results.append(r)
                    print(
                        f"  {pair:12s}  win={r.win_rate:.1%}  pf={r.profit_factor:.2f}"
                        f"  sharpe={r.sharpe:.2f}  dd={r.max_drawdown:.1%}"
                        f"  tpy={r.trades_per_year:.0f}"
                    )
            except Exception as e:
                logger.warning(f"Backtest failed for {pair}: {e}")

        output = [asdict(r) for r in results]
        with open(RESULTS_PATH, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {RESULTS_PATH}")
        return results

    # ── Signal generation ─────────────────────────────────────────────── #

    def _generate_monthly_signals(
        self, prices: pd.DataFrame, base: str, quote: str
    ) -> pd.Series:
        """
        Generate a monthly signal series: +1 = long, -1 = short, 0 = flat.
        Uses FRED rate history when available, falls back to current snapshot.
        """
        close = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
        monthly_dates = close.resample('BMS').first().index
        signals = pd.Series(0.0, index=monthly_dates)

        # Get rate histories (real or flat fallback)
        base_rates = self._fetcher.get_rate_history(base, start='2014-01-01')
        quote_rates = self._fetcher.get_rate_history(quote, start='2014-01-01')
        base_cpi_h = self._fetcher.get_cpi_history(base, start='2014-01-01')
        quote_cpi_h = self._fetcher.get_cpi_history(quote, start='2014-01-01')

        for date in monthly_dates:
            spot = float(close.asof(date)) if hasattr(close, 'asof') else float(close.loc[:date].iloc[-1])

            b_rate = float(base_rates.asof(date)) if date in base_rates.index or date > base_rates.index[0] else FALLBACK_RATES[base]
            q_rate = float(quote_rates.asof(date)) if date in quote_rates.index or date > quote_rates.index[0] else FALLBACK_RATES[quote]
            b_cpi  = float(base_cpi_h.asof(date)) if len(base_cpi_h) and date >= base_cpi_h.index[0] else FALLBACK_CPI[base]
            q_cpi  = float(quote_cpi_h.asof(date)) if len(quote_cpi_h) and date >= quote_cpi_h.index[0] else FALLBACK_CPI[quote]

            rate_diff = b_rate - q_rate
            real_rate_diff = (b_rate - b_cpi) - (q_rate - q_cpi)

            # IRP deviation vs trailing 252-day history
            hist = close.loc[:date]
            irp_fv = spot * (1 + q_rate / 100) / (1 + b_rate / 100)
            irp_dev = (spot - irp_fv) / irp_fv if irp_fv != 0 else 0.0
            if len(hist) > 252:
                sigma = hist.pct_change().std() * np.sqrt(252)
                irp_z = irp_dev / (sigma + 1e-8)
            else:
                irp_z = 0.0

            # Real rate differential normalized
            rdm = np.clip(real_rate_diff / 4.0, -1, 1)

            # IRP mean reversion component
            irp_component = np.clip(-irp_z / 1.5, -1, 1)

            macro_score = 0.50 * irp_component + 0.50 * rdm

            # Momentum confirmation: 3-month price trend (63 days)
            if len(hist) > 63:
                mom = float(hist.iloc[-1] / hist.iloc[-63] - 1)
                mom_sign = np.sign(mom) if abs(mom) > 0.005 else 0  # ignore tiny moves
            else:
                mom_sign = 0

            # Only fire if macro signal and momentum agree
            macro_sign = np.sign(macro_score) if abs(macro_score) > self.SIGNAL_THRESHOLD else 0
            if macro_sign != 0 and (mom_sign == 0 or mom_sign == macro_sign):
                signals[date] = macro_sign

        return signals

    def _simulate_trades(
        self, df: pd.DataFrame, signals: pd.Series
    ) -> list:
        """
        Simulate trades from monthly signals.

        Rules:
          - Enter at next-day open after signal fires
          - Exit after HOLD_DAYS, or if signal reverses, or -STOP_PCT hit
          - One trade at a time; new signal while in trade → ignore until exit
        """
        close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
        opens = df['Open'] if 'Open' in df.columns else close

        sig_map = signals[signals != 0].to_dict()
        trades = []
        in_trade = False
        entry_price = direction = entry_date = exit_date = None
        hold_count = 0

        for i, date in enumerate(close.index):
            if in_trade:
                hold_count += 1
                price = float(close.iloc[i])
                ret = direction * (price / entry_price - 1)

                # Exit: stop hit, hold expired, or signal reversed
                signal_today = sig_map.get(date, 0)
                stop_hit = ret <= -self.STOP_PCT
                time_exit = hold_count >= self.HOLD_DAYS
                reversal = signal_today != 0 and signal_today != direction

                if stop_hit or time_exit or reversal:
                    pnl_pct = direction * (price / entry_price - 1)
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'direction': direction,
                        'entry': entry_price,
                        'exit': price,
                        'pnl_pct': pnl_pct,
                        'hold_days': hold_count,
                        'exit_reason': 'stop' if stop_hit else ('reversal' if reversal else 'time'),
                    })
                    in_trade = False

                    # If reversal, open new trade in opposite direction immediately
                    if reversal and i + 1 < len(close.index):
                        in_trade = True
                        direction = int(signal_today)
                        entry_price = float(opens.iloc[min(i + 1, len(opens) - 1)])
                        entry_date = close.index[min(i + 1, len(close.index) - 1)]
                        hold_count = 0

            if not in_trade:
                signal_today = sig_map.get(date, 0)
                if signal_today != 0 and i + 1 < len(close.index):
                    in_trade = True
                    direction = int(signal_today)
                    entry_price = float(opens.iloc[i + 1])
                    entry_date = close.index[i + 1]
                    hold_count = 0

        return trades

    def _compute_stats(
        self, pair: str, trades: list, n_bars: int
    ) -> ForexBacktestResult:
        pnls = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        n = len(pnls)
        years = n_bars / 252.0

        win_rate = len(wins) / n if n else 0.0
        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-6
        profit_factor = gross_win / gross_loss

        avg_hold = np.mean([t['hold_days'] for t in trades]) if trades else 0.0

        # Equity curve → Sharpe and max drawdown
        equity = np.cumprod([1 + p for p in pnls])
        returns = np.diff(np.log(equity), prepend=0)
        ann_factor = np.sqrt(252 / max(avg_hold, 1))
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * ann_factor if n > 1 else 0.0

        rolling_max = np.maximum.accumulate(equity)
        drawdowns = (equity - rolling_max) / rolling_max
        max_dd = float(drawdowns.min()) if len(drawdowns) else 0.0

        return ForexBacktestResult(
            pair=pair,
            win_rate=round(win_rate, 3),
            profit_factor=round(min(profit_factor, 20.0), 3),
            sharpe=round(sharpe, 3),
            max_drawdown=round(max_dd, 3),
            avg_hold_days=round(avg_hold, 1),
            trades_per_year=round(n / max(years, 1), 1),
            total_trades=n,
            years=round(years, 1),
        )

    # ── Price download ────────────────────────────────────────────────── #

    def _download_price(self, pair: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(
                pair, start=self.start, end=self.end,
                progress=False, auto_adjust=True
            )
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.dropna()
        except Exception as e:
            logger.warning(f"Download failed for {pair}: {e}")
            return None


def main():
    print("\nFOREX BACKTEST — 2015–2024")
    print('=' * 60)
    bt = ForexBacktester()
    results = bt.backtest_all()

    if results:
        best = max(results, key=lambda r: r.sharpe)
        print(f"\nBACKTEST: {len(results)} pairs")
        print(f"BEST SHARPE: {best.pair}  sharpe={best.sharpe:.2f}  "
              f"win={best.win_rate:.1%}  pf={best.profit_factor:.2f}")
    print()


if __name__ == '__main__':
    main()
