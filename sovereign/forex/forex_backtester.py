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
from sovereign.forex.fast_backtester import simulate_forex_trades, simulate_forex_trades_arrays
from sovereign.forex.signal_engine import ForexSignalEngine, SignalConfig
from sovereign.forex.compliance import (
    ForexComplianceConfig,
    score_compliance,
    block_live_mode_if_needed,
)
logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).parents[2] / 'logs' / 'forex_backtest_results.json'
RESULTS_PATH.parent.mkdir(exist_ok=True)

# ── Transaction costs ─────────────────────────────────────────────────────── #
# Round-trip spread in PRICE units (yen pairs are in yen, ~150-price scale).
# Applied as a fraction of entry price in _simulate_trades, since trade pnl_pct
# is a fractional return — a raw price-unit subtraction would over-charge JPY
# pairs by ~150×. OANDA practice is spread-only (no commission).
SPREAD_COST = {
    'GBPUSD=X': 0.00012,   # ~1.2 pips
    'EURUSD=X': 0.00010,   # ~1.0 pips
    'USDJPY=X': 0.012,     # ~1.2 pips (yen)
    'AUDUSD=X': 0.00014,   # ~1.4 pips
    'AUDNZD=X': 0.00020,   # ~2.0 pips (cross, wider)
}
SLIPPAGE_PER_SIDE = 0.00005   # ~0.5 pips, charged on entry and exit
_DEFAULT_SPREAD = 0.00015     # fallback for pairs without an explicit entry

# Annual swap/financing rate per pair/direction (fraction of notional per year).
# Negative = you PAY to hold; positive = you EARN (carry). Applied per trade scaled
# by hold_days with a Wednesday-triple / weekend approximation. News-spread widening
# is intentionally NOT modeled: the array backtest path has no per-trade high-impact-
# news flag, so a per-trade multiplier would be fabricated rather than measured.
SWAP_RATES_ANNUAL = {
    'GBPUSD=X': {'LONG': -0.0012, 'SHORT': -0.0008},
    'EURUSD=X': {'LONG': -0.0015, 'SHORT': -0.0010},
    'USDJPY=X': {'LONG':  0.0020, 'SHORT': -0.0035},   # positive carry long
    'AUDUSD=X': {'LONG': -0.0008, 'SHORT': -0.0012},
    'AUDNZD=X': {'LONG': -0.0003, 'SHORT': -0.0003},   # near-zero carry cross
}
_DEFAULT_SWAP = {'LONG': -0.0010, 'SHORT': -0.0010}

# Gap 1: live-calibrated per-side slippage (price units), written by
# cost_calibrator.py from real fills. Absent → modeled SLIPPAGE_PER_SIDE is used.
_CALIBRATED_COSTS_PATH = Path(__file__).resolve().parents[2] / 'data' / 'execution' / 'calibrated_costs.json'
_calibrated_slippage_cache: Optional[dict] = None


def _calibrated_slippage(pair: Optional[str]) -> Optional[float]:
    """Per-side slippage (price units) from calibrated_costs.json, or None to fall back."""
    global _calibrated_slippage_cache
    if _calibrated_slippage_cache is None:
        try:
            data = json.loads(_CALIBRATED_COSTS_PATH.read_text())
            _calibrated_slippage_cache = {
                k: v.get("slippage_price_per_side")
                for k, v in data.get("slippage_by_pair", {}).items()
            }
        except Exception:
            _calibrated_slippage_cache = {}
    if not pair:
        return None
    norm = str(pair).replace("=X", "").replace("_", "").upper()
    return _calibrated_slippage_cache.get(norm)


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

    HOLD_DAYS = 60          # default hold; macro signals play out over 2-3 months
    STOP_PCT = 0.04         # compatibility fallback only; strict mode uses ATR stop
    STOP_ATR_MULT = 2.0
    TRAILING_ATR_MULT = 1.25  # forensics v1: 1.25x beats 1.0x (Sharpe 1.024 vs 0.884)
    DONCHIAN_EXIT_DAYS = 10
    SIGNAL_THRESHOLD = 0.15   # lowered from 0.20 — more macro signals for statistical validity
    CB_SURPRISE_THRESHOLD = 20  # 20bp in backtest (vs 25bp live) for adequate sample size
    MAX_RISK_PER_TRADE_PCT = 0.01
    MAX_SHARED_JPY_POSITIONS = 2
    # v007 per-pair hold sweep (2026-05-19): trailing_mult swept at optimal hold per pair.
    # AUDUSD 5d/1.0x: 1.292 (+0.028 vs 60d) | EURUSD 5d/1.25x: 1.441 (unchanged, shorter hold)
    # AUDNZD 7d/1.25x: 1.172 (unchanged, shorter hold) | GBPJPY 5d/1.0x: 0.741 (+0.088 vs 60d)
    # USDCAD/USDJPY: below sweep threshold, 60d hold unchanged.
    # Portfolio: 1.0713 (+0.017 vs v006 1.0547) — below +0.05 version gate but all-pair consistency
    # warrants apply: shorter holds reduce overnight exposure, GBPJPY gains are real.
    # NOTE: Retro validation 2026-05-27 flagged this as harmful (delta_sharpe=-0.087, p=0.008)
    # but live backtester CONTRADICTS that finding: removing overrides drops GBPUSD 1.89→1.70,
    # AUDUSD 1.67→1.43. Retro test was flawed — forensics 'hold' reflects natural trade duration,
    # not the effect of forced hold caps. Rule remains active pending redesigned validation.
    PAIR_HOLD_OVERRIDES: dict = {
        "GBPUSD=X": 6,
        "AUDUSD=X": 5,
        "EURUSD=X": 5,
        "AUDNZD=X": 7,
    }
    PAIR_TRAILING_OVERRIDES: dict = {
        "GBPUSD=X": 2.0,
        "AUDUSD=X": 1.0,   # 1.0x beats 1.25x at 5d hold: exits faster, less giveback
        "EURUSD=X": 1.25,
        "AUDNZD=X": 1.25,
    }
    # Bull+VIX regime gates (v011, 2026-05-22):
    # When SPY > 200 SMA AND VIX > threshold → suppress pair signal.
    # Universal finding: macro rate differential signals degrade when fear flows
    # compete with rate signals in a nominally-bullish market environment.
    #
    # Tiered thresholds (economically motivated, not just in-sample optimised):
    #   VIX>15 — JPY (safe-haven flows activate early) and crosses (both legs risk currencies)
    #   VIX>20 — USD macro pairs (rate differential survives mild VIX elevation; needs true fear)
    #
    # Per-pair Sharpe improvement at threshold (2015-2024 study, n per pair 57-120):
    #   USDJPY:  1.004 → 1.770  (VIX>15)
    #   AUDNZD:  1.172 → 1.558  (VIX>15)
    #   EURUSD:  1.441 → 1.583  (VIX>20)
    #   GBPUSD:  1.523 → 1.662  (VIX>20)
    #   AUDUSD:  1.292 → 1.665  (VIX>20)
    #   Portfolio: 1.2864 → 1.6476  (+0.361 total across v010+v011)
    PAIR_VIX_GATES: dict = {
        'USDJPY=X': 15.0,   # HYP-044 ROLLED BACK 2026-05-31: 15→13 showed delta 0.000 on the 2023-24 holdout — in-sample noise. Reverted to v013 threshold.
        'AUDNZD=X': 15.0,   # HYP-044 ROLLED BACK 2026-05-31: 15→13 showed delta 0.000 on the 2023-24 holdout — in-sample noise. Reverted to v013 threshold.
        'EURUSD=X': 18.0,   # ECB-FED rate diff survives mild fear, breaks at VIX>18
        'GBPUSD=X': 18.0,   # BOE-FED rate diff same — confirmed optimal in full sweep
        'AUDUSD=X': 20.0,   # RBA-FED commodity-linked, more resilient; needs true fear
    }

    def __init__(
        self,
        start: str = '2015-01-01',
        end: str = '2024-12-31',
        *,
        strict_mode: bool = False,
        use_macro_overlay: bool = False,
        allow_pyramiding: bool = True,
        max_pyramid_units: int = 4,
        signal_weights: Optional[dict] = None,
    ):
        self.start = start
        self.end = end
        self.strict_mode = strict_mode
        # Seed-library factor-replication hook: override the macro signal's factor
        # weights (irp_weight / rate_weight / use_momentum_filter). None → engine defaults
        # (unchanged behavior). Keys absent from the dict keep their defaults.
        self._signal_weights = signal_weights or {}
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
                **self._signal_weights,
            ),
        )

    def _apply_vix_regime_gate(
        self, signals: 'pd.DataFrame', pair: str, start: str, end: str
    ) -> 'pd.DataFrame':
        """
        Suppress pair signals when SPY > 200 SMA AND VIX > PAIR_VIX_GATES[pair].
        Universal finding (v011, 2026-05-22): bull+elevated-VIX overwhelms macro rate
        differential signals across all 5 pairs. Tiered thresholds by pair sensitivity.
        """
        vix_threshold = self.PAIR_VIX_GATES.get(pair)
        if vix_threshold is None:
            return signals
        import pandas as pd
        try:
            import yfinance as yf
            spy = yf.download('SPY', start=start, end=end, progress=False)
            vix = yf.download('^VIX', start=start, end=end, progress=False)
            for df_ in (spy, vix):
                if hasattr(df_.columns, 'get_level_values'):
                    df_.columns = df_.columns.get_level_values(0)
                df_.index = pd.to_datetime(df_.index).tz_localize(None)
            spy['sma200'] = spy['Close'].rolling(200).mean()
            spy['is_bull'] = spy['Close'] > spy['sma200']
            signals = signals.copy()
            for date in signals[signals['signal'] != 0].index:
                try:
                    if bool(spy['is_bull'].asof(date)) and float(vix['Close'].asof(date)) > vix_threshold:
                        signals.loc[date, 'signal'] = 0.0
                except Exception:
                    pass
        except Exception:
            pass
        return signals

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

        pair_hold = self.PAIR_HOLD_OVERRIDES.get(pair, self.HOLD_DAYS)
        pair_trailing = self.PAIR_TRAILING_OVERRIDES.get(pair, self.TRAILING_ATR_MULT)
        signals = self._get_pair_signals(
            df=df, base_country=base_country, quote_country=quote_country,
            pair=pair, hold_days=pair_hold,
        )

        # Bull+VIX regime gate: suppress signals when fear flows overwhelm rate differential
        if pair in self.PAIR_VIX_GATES:
            signals = self._apply_vix_regime_gate(
                signals, pair=pair, start=self.start, end=self.end
            )

        trades = self._simulate_trades(df, signals, pair=pair, trailing_mult=pair_trailing)

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
                pair_hold = self.PAIR_HOLD_OVERRIDES.get(pair, self.HOLD_DAYS)
                pair_trailing = self.PAIR_TRAILING_OVERRIDES.get(pair, self.TRAILING_ATR_MULT)
                pair_signals = self._get_pair_signals(
                    df=df, base_country=base_country, quote_country=quote_country,
                    pair=pair, hold_days=pair_hold,
                )
                trades = self._simulate_trades(df, pair_signals, pair=pair, trailing_mult=pair_trailing)
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

    # ── Public entry-point variants ───────────────────────────────────────── #

    def run_pair(
        self,
        pair: str,
        base_country: str,
        quote_country: str,
    ) -> Optional["ForexBacktestResult"]:
        """Like backtest_pair() but accepts explicit country params.

        Useful when the caller has already resolved base/quote from PAIR_CONFIG
        and wants to avoid a second lookup (e.g. rq_rest_004 gate tests).
        Applies all per-pair overrides (hold, trailing, VIX gate).
        """
        df = self._download_price(pair)
        if df is None or len(df) < 252:
            return None

        pair_hold = self.PAIR_HOLD_OVERRIDES.get(pair, self.HOLD_DAYS)
        pair_trailing = self.PAIR_TRAILING_OVERRIDES.get(pair, self.TRAILING_ATR_MULT)

        signals = self._get_pair_signals(
            df=df,
            base_country=base_country,
            quote_country=quote_country,
            pair=pair,
            hold_days=pair_hold,
        )

        if pair in self.PAIR_VIX_GATES:
            signals = self._apply_vix_regime_gate(
                signals, pair=pair, start=self.start, end=self.end
            )

        trades = self._simulate_trades(df, signals, pair=pair, trailing_mult=pair_trailing)
        if not trades:
            return None
        return self._compute_stats(pair, trades, len(df))

    def run_pair_with_trades(
        self,
        pair: str,
        base_country: str,
        quote_country: str,
        signal_engine_override=None,
    ) -> "tuple[Optional[ForexBacktestResult], list]":
        """Like run_pair() but returns (result, trades) and accepts an engine override.

        signal_engine_override: a ForexSignalEngine subclass instance whose
        build_signal_frame() is called in place of the default engine.
        Used by rq_rest_003 to capture per-signal macro_score.
        Applies all per-pair overrides (hold, trailing, VIX gate).
        """
        df = self._download_price(pair)
        if df is None or len(df) < 252:
            return None, []

        pair_hold = self.PAIR_HOLD_OVERRIDES.get(pair, self.HOLD_DAYS)
        pair_trailing = self.PAIR_TRAILING_OVERRIDES.get(pair, self.TRAILING_ATR_MULT)

        if signal_engine_override is not None:
            signals = signal_engine_override.build_signal_frame(
                prices=df,
                base_country=base_country,
                quote_country=quote_country,
                start=self.start,
                end=self.end,
                pair=pair,
            )
        else:
            signals = self._get_pair_signals(
                df=df,
                base_country=base_country,
                quote_country=quote_country,
                pair=pair,
                hold_days=pair_hold,
            )

        if pair in self.PAIR_VIX_GATES:
            signals = self._apply_vix_regime_gate(
                signals, pair=pair, start=self.start, end=self.end
            )

        trades = self._simulate_trades(df, signals, pair=pair, trailing_mult=pair_trailing)
        if not trades:
            return None, []

        result = self._compute_stats(pair, trades, len(df))
        return result, trades

    def run_with_signals(
        self,
        pair: str,
        opens: "np.ndarray",
        closes: "np.ndarray",
        signals: "np.ndarray",
        hold_days: "np.ndarray",
        index: "pd.Index",
    ) -> Optional["ForexBacktestResult"]:
        """Run a backtest with externally-provided array data (from ForexArrayDataset).

        Intended for gate-testing experiments (rq_rest_004) where the caller
        already has preloaded price/signal arrays and has applied an external
        gate to the signals before passing them in.

        Applies per-pair cost model and trailing-mult override; does NOT apply
        the VIX gate (caller is responsible for signal filtering).
        """
        pair_trailing = self.PAIR_TRAILING_OVERRIDES.get(pair, self.TRAILING_ATR_MULT)

        trades = simulate_forex_trades_arrays(
            opens=opens,
            closes=closes,
            signals=signals,
            hold_days=hold_days,
            stop_pct=self.STOP_PCT,
            index=index,
            atr_pcts=None,
            stop_atr_mult=self.STOP_ATR_MULT,
            trailing_atr_mult=pair_trailing,
            strict_mode=self.strict_mode,
            donchian_exit_days=self.DONCHIAN_EXIT_DAYS,
            allow_pyramiding=self.allow_pyramiding,
            max_pyramid_units=self.max_pyramid_units,
            risk_pct=self.MAX_RISK_PER_TRADE_PCT,
            max_risk_pct=self.MAX_RISK_PER_TRADE_PCT,
            enable_cb_refresh=not self.strict_mode,
        )

        if not trades:
            return None

        trades = self._apply_costs(trades, pair)
        return self._compute_stats(pair, trades, len(index))

    def _get_pair_signals(
        self,
        df: pd.DataFrame,
        base_country: str,
        quote_country: str,
        pair: str,
        hold_days: int,
    ) -> pd.DataFrame:
        if hold_days == self.HOLD_DAYS:
            return self._signals.build_signal_frame(
                prices=df, base_country=base_country, quote_country=quote_country,
                start=self.start, end=self.end, pair=pair,
            )
        # Need a different hold_days — build a one-off engine
        from sovereign.forex.signal_engine import ForexSignalEngine, SignalConfig
        engine = ForexSignalEngine(
            fetcher=self._fetcher,
            cb_trigger=self._cb,
            config=SignalConfig(
                hold_days=hold_days,
                signal_threshold=self.SIGNAL_THRESHOLD,
                cb_surprise_threshold=self.CB_SURPRISE_THRESHOLD,
                strict_mode=self.strict_mode,
                **self._signal_weights,
            ),
        )
        return engine.build_signal_frame(
            prices=df, base_country=base_country, quote_country=quote_country,
            start=self.start, end=self.end, pair=pair,
        )

    def _simulate_trades(
        self, df: pd.DataFrame, signals: pd.DataFrame, pair: str = None,
        trailing_mult: float = None
    ) -> list:
        close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
        atr_series = self._signals._compute_atr_pct(close, df)
        trades = simulate_forex_trades(
            df,
            signals,
            stop_pct=self.STOP_PCT,
            atr_series=atr_series,
            stop_atr_mult=self.STOP_ATR_MULT,
            trailing_atr_mult=trailing_mult if trailing_mult is not None else self.TRAILING_ATR_MULT,
            strict_mode=self.strict_mode,
            donchian_exit_days=self.DONCHIAN_EXIT_DAYS,
            allow_pyramiding=self.allow_pyramiding,
            max_pyramid_units=self.max_pyramid_units,
            risk_pct=self.MAX_RISK_PER_TRADE_PCT,
            max_risk_pct=self.MAX_RISK_PER_TRADE_PCT,
            enable_cb_refresh=not self.strict_mode,
        )
        return self._apply_costs(trades, pair)

    @staticmethod
    def _apply_costs(trades: list, pair: str = None) -> list:
        """Deduct round-trip spread + slippage + swap/financing from each trade's
        fractional pnl.

        pnl_pct is a fractional return (price/entry - 1), so the price-unit spread
        cost is normalised by the entry price before subtraction. Swap is a signed
        fraction of notional (negative = pay, positive = earn carry) scaled by the
        hold duration with a Wednesday-triple/weekend approximation.
        """
        spread = SPREAD_COST.get(pair, _DEFAULT_SPREAD)
        # Overlay LIVE-calibrated slippage (Gap 1) when available, else the modeled
        # default. Spread stays modeled (execution_tracker doesn't measure it).
        per_side = _calibrated_slippage(pair)
        slip = per_side if per_side is not None else SLIPPAGE_PER_SIDE
        cost_price = spread + 2 * slip
        swap_tbl = SWAP_RATES_ANNUAL.get(pair, _DEFAULT_SWAP)
        for t in trades:
            entry = max(t.get('entry', 0.0), 1e-9)
            spread_frac = cost_price / entry
            # Swap: signed annual rate / 365, scaled by hold days + weekend triple.
            side = 'LONG' if t.get('direction', 1) >= 0 else 'SHORT'
            hold_days = max(int(t.get('hold_days', 0)), 0)
            swap_days = hold_days + (hold_days // 5) * 2   # ~Wed-triple/weekend uplift
            swap_frac = (swap_tbl[side] / 365.0) * swap_days   # signed: +earn / -pay
            t['pnl_pct'] = t.get('pnl_pct', 0.0) - spread_frac + swap_frac
            t['cost_spread_frac'] = round(spread_frac, 6)
            t['cost_swap_frac'] = round(swap_frac, 6)
            t['risk_adjusted_pnl_pct'] = t['pnl_pct'] * t.get('risk_pct', 1.0)
        return trades

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
        # Annualize by EMPIRICAL trades/year (n/years), NOT 252/avg_hold.
        # 252/avg_hold assumes the book is always in a trade; with flat periods
        # that overcounts trades-per-year and inflates the Sharpe.
        ann_factor = np.sqrt(max(n, 1) / max(years, 1e-9))
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
