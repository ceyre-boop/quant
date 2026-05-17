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
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from sovereign.forex.pair_universe import ALL_PAIRS, PAIR_CONFIG, CB_TO_COUNTRY
from sovereign.forex.data_fetcher import ForexDataFetcher
from sovereign.forex.entry_engine import CBEventTrigger, CB_MIN_SURPRISE_BPS
from sovereign.forex.fast_backtester import simulate_forex_trades
from sovereign.forex.signal_engine import ForexSignalEngine, SignalConfig
from sovereign.forex.compliance import (
    ForexComplianceConfig,
    score_compliance,
    block_live_mode_if_needed,
)
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
    STOP_PCT = 0.04         # compatibility fallback only; strict mode uses ATR stop
    STOP_ATR_MULT = 2.0
    TRAILING_ATR_MULT = 1.25  # forensics v1: 1.25x beats 1.0x (Sharpe 1.024 vs 0.884)
    DONCHIAN_EXIT_DAYS = 10
    SIGNAL_THRESHOLD = 0.15   # lowered from 0.20 — more macro signals for statistical validity
    CB_SURPRISE_THRESHOLD = 20  # 20bp in backtest (vs 25bp live) for adequate sample size
    MAX_RISK_PER_TRADE_PCT = 0.01
    MAX_SHARED_JPY_POSITIONS = 2

    def __init__(
        self,
        start: str = '2015-01-01',
        end: str = '2024-12-31',
        *,
        strict_mode: bool = False,
        use_macro_overlay: bool = False,
        allow_pyramiding: bool = True,
        max_pyramid_units: int = 4,
    ):
        self.start = start
        self.end = end
        self.strict_mode = strict_mode
        self.allow_pyramiding = allow_pyramiding and strict_mode
        self.max_pyramid_units = max_pyramid_units if strict_mode else 1
        self._compliance = ForexComplianceConfig(
            strict_mode=strict_mode,
            max_risk_per_trade_pct=self.MAX_RISK_PER_TRADE_PCT,
            max_shared_jpy_positions=self.MAX_SHARED_JPY_POSITIONS,
            max_pyramid_units=self.max_pyramid_units,
            use_macro_overlay=use_macro_overlay,
        )
        self._compliance.validate_startup()
        self._fetcher = ForexDataFetcher()
        self._cb = CBEventTrigger()
        self._signals = ForexSignalEngine(
            fetcher=self._fetcher,
            cb_trigger=self._cb,
            config=SignalConfig(
                hold_days=self.HOLD_DAYS,
                signal_threshold=self.SIGNAL_THRESHOLD,
                cb_surprise_threshold=self.CB_SURPRISE_THRESHOLD,
                strict_mode=strict_mode,
                use_macro_overlay=use_macro_overlay,
            ),
        )

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

        signals = self._signals.build_signal_frame(
            prices=df,
            base_country=base_country,
            quote_country=quote_country,
            start=self.start,
            end=self.end,
            pair=pair,
        )
        trades = self._simulate_trades(df, signals)

        if not trades:
            return None

        return self._compute_stats(pair, trades, len(df))

    def backtest_all(self) -> List[ForexBacktestResult]:
        results = []
        all_trades: dict[str, list] = {}
        pair_bars: dict[str, int] = {}

        for pair in ALL_PAIRS:
            try:
                cfg = PAIR_CONFIG.get(pair)
                if not cfg:
                    continue
                df = self._download_price(pair)
                if df is None or len(df) < 252:
                    continue
                base_country  = CB_TO_COUNTRY[cfg.base_central_bank]
                quote_country = CB_TO_COUNTRY[cfg.quote_central_bank]
                signals = self._signals.build_signal_frame(
                    prices=df, base_country=base_country,
                    quote_country=quote_country,
                    start=self.start, end=self.end, pair=pair,
                )
                trades = self._simulate_trades(df, signals)
                if not trades:
                    continue
                all_trades[pair] = trades
                pair_bars[pair] = len(df)
            except Exception as e:
                logger.warning(f"Backtest failed for {pair}: {e}")

        if self.strict_mode:
            all_trades = self._apply_correlation_caps(all_trades)

        for pair, trades in all_trades.items():
            if not trades:
                continue
            n_bars = pair_bars.get(pair, 0)
            if n_bars <= 0:
                continue
            r = self._compute_stats(pair, trades, n_bars)
            results.append(r)
            print(
                f"  {pair:12s}  win={r.win_rate:.1%}  pf={r.profit_factor:.2f}"
                f"  sharpe={r.sharpe:.2f}  dd={r.max_drawdown:.1%}"
                f"  tpy={r.trades_per_year:.0f}"
            )

        output = [asdict(r) for r in results]
        with open(RESULTS_PATH, 'w') as f:
            json.dump(output, f, indent=2)

        trades_path = RESULTS_PATH.parent / 'forex_backtest_trades.json'
        with open(trades_path, 'w') as f:
            # Serialise timestamps
            serialisable = {}
            for pair, trades in all_trades.items():
                serialisable[pair] = [
                    {k: (str(v) if hasattr(v, 'date') else v)
                     for k, v in t.items()} for t in trades
                ]
            json.dump(serialisable, f, indent=2, default=str)

        return results

    def _simulate_trades(
        self, df: pd.DataFrame, signals: pd.DataFrame
    ) -> list:
        close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
        atr_series = self._signals._compute_atr_pct(close, df)
        return simulate_forex_trades(
            df,
            signals,
            stop_pct=self.STOP_PCT,
            atr_series=atr_series,
            stop_atr_mult=self.STOP_ATR_MULT,
            trailing_atr_mult=self.TRAILING_ATR_MULT,
            strict_mode=self.strict_mode,
            donchian_exit_days=self.DONCHIAN_EXIT_DAYS,
            allow_pyramiding=self.allow_pyramiding,
            max_pyramid_units=self.max_pyramid_units,
            risk_pct=self.MAX_RISK_PER_TRADE_PCT,
            max_risk_pct=self.MAX_RISK_PER_TRADE_PCT,
            enable_cb_refresh=not self.strict_mode,
        )

    def _apply_correlation_caps(self, all_trades: dict[str, list]) -> dict[str, list]:
        flattened = []
        for pair, trades in all_trades.items():
            for t in trades:
                flattened.append((pair, t))
        flattened.sort(key=lambda x: x[1]['entry_date'])

        accepted: dict[str, list] = {pair: [] for pair in all_trades.keys()}
        active = []
        for pair, trade in flattened:
            entry_dt = trade['entry_date']
            active = [a for a in active if a['exit_date'] >= entry_dt]
            if 'JPY' in pair:
                jpy_active = sum(1 for a in active if 'JPY' in a['pair'])
                if jpy_active >= self.MAX_SHARED_JPY_POSITIONS:
                    continue
            trade_with_pair = dict(trade)
            trade_with_pair['pair'] = pair
            active.append(trade_with_pair)
            accepted[pair].append(trade)
        return accepted

    def generate_compliance_report(self, mode: str = 'paper') -> dict:
        report = score_compliance(self._compliance)
        block_live_mode_if_needed(mode=mode, report=report)
        return report

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
    parser = argparse.ArgumentParser(description="Forex backtester")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--strict-mode", action="store_true")
    parser.add_argument("--macro-overlay", action="store_true")
    args = parser.parse_args()

    print("\nFOREX BACKTEST — 2015–2024")
    print('=' * 60)
    bt = ForexBacktester(strict_mode=args.strict_mode, use_macro_overlay=args.macro_overlay)
    report = bt.generate_compliance_report(mode=args.mode)
    print(
        f"Compliance: score={report['score']} status={report['status']} "
        f"rules={report['rule_set_version']}"
    )
    try:
        from governance.policy_engine import GOVERNANCE
        GOVERNANCE.update_forex_compliance(
            rule_set_version=report['rule_set_version'],
            status=report['status'],
            score=report['score'],
        )
    except Exception as exc:
        logger.warning(f"Governance compliance update skipped: {exc}")

    results = bt.backtest_all()

    if results:
        best = max(results, key=lambda r: r.sharpe)
        print(f"\nBACKTEST: {len(results)} pairs")
        print(f"BEST SHARPE: {best.pair}  sharpe={best.sharpe:.2f}  "
              f"win={best.win_rate:.1%}  pf={best.profit_factor:.2f}")
    print()


if __name__ == '__main__':
    main()
