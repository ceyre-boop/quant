"""
Carry Engine — Track 2 (permanent background floor)

Implements the Lo binomial architecture's "carry base":
  • Runs every day regardless of regime, Library convergence, or macro gates.
  • Collects interest-rate differential on AUDCHF and NZDJPY.
  • Sized at 0.3% of account equity per pair — never larger, never counted
    in macro Sharpe statistics.
  • Uses 3× ATR stops to survive the occasional risk-off flush.
  • Logs to a dedicated carry ledger (NOT the signal ledger).

Design rationale (from TRADING_PHILOSOPHY.md Tenet 5):
  "The carry trade is the foundation."

This engine is NOT a signal filter.  It is infrastructure — the coupon on a
bond.  The coupon does not need to pass a conviction gate; it just pays.
The carry income funds patience for Track 1 (macro gates) and Track 3 (spike
surgeon) to deliver their larger, less frequent returns.

Threading model:
  Call ``CarryEngine.run_forever()`` in a daemon thread from your launcher.
  The engine sleeps between scans and is safe to run concurrently with the
  equity and forex macro engines.

Usage::

    engine = CarryEngine(account_equity=100_000)
    t = threading.Thread(target=engine.run_forever, daemon=True)
    t.start()

Or for a one-shot daily scan (e.g. from execute_daily.py)::

    engine = CarryEngine(account_equity=100_000)
    signals = engine.scan()           # returns list[CarrySignal]
    engine.log_signals(signals)
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Carry pair definitions
# ---------------------------------------------------------------------------
# Each entry: (yfinance ticker, base currency, quote currency, long direction)
# "Long" direction = buy the pair = hold the HIGH-yield currency.
# We long the pair when the high-yield currency is the BASE, short when it
# is the QUOTE.

@dataclass(frozen=True)
class CarryPairConfig:
    ticker: str
    base_currency: str       # e.g. 'AUD'
    quote_currency: str      # e.g. 'CHF'
    high_yield_side: str     # 'BASE' | 'QUOTE' — which side earns the carry
    description: str


CARRY_PAIRS: List[CarryPairConfig] = [
    CarryPairConfig(
        ticker='AUDCHF=X',
        base_currency='AUD',
        quote_currency='CHF',
        high_yield_side='BASE',   # AUD rate > CHF rate → long AUDCHF
        description='AUD/CHF — borrow CHF (≈0%), hold AUD (≈4%). '
                    'Classic high-yield vs safe-haven carry.',
    ),
    CarryPairConfig(
        ticker='NZDJPY=X',
        base_currency='NZD',
        quote_currency='JPY',
        high_yield_side='BASE',   # NZD rate > JPY rate → long NZDJPY
        description='NZD/JPY — borrow JPY (≈0.1%), hold NZD (≈5%). '
                    'Highest carry differential in G10.',
    ),
]

# ---------------------------------------------------------------------------
# Risk parameters (fixed — do not change without human approval)
# ---------------------------------------------------------------------------

CARRY_RISK_PER_PAIR  = 0.003   # 0.3% of account equity risked per carry pair
ATR_STOP_MULTIPLE    = 3.0     # stop = entry ± 3 × ATR(14)
ATR_PERIOD           = 14      # days for ATR calculation
MIN_CARRY_SPREAD_BPS = 100     # skip pair if carry differential < 1% (100 bps)
SCAN_INTERVAL_HOURS  = 4       # how often to re-scan in run_forever mode
LEDGER_DIR           = Path('data/ledger/carry')

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CarrySignal:
    ticker: str
    direction: str            # 'LONG' | 'SHORT' | 'FLAT'
    carry_spread_bps: float   # annualised interest-rate differential in bps
    atr: float                # 14-day ATR
    stop_distance: float      # 3 × ATR in price units
    position_size_pct: float  # fraction of account equity to risk (always 0.003)
    units: float              # estimated units given equity and stop
    reason: str               # human-readable rationale
    timestamp: str


@dataclass
class CarryPosition:
    ticker: str
    direction: str
    entry_price: float
    stop_price: float
    carry_spread_bps: float
    opened_at: str
    account_equity: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CarryEngine:
    """
    Permanent carry-income floor engine.

    Parameters
    ----------
    account_equity : float
        Current account equity in base currency (used for position sizing).
    rate_overrides : dict, optional
        Map of currency → annualised rate (%) for testing without live data.
        E.g. {'AUD': 4.35, 'CHF': 1.50, 'NZD': 5.50, 'JPY': 0.10}
    """

    def __init__(
        self,
        account_equity: float = 100_000.0,
        rate_overrides: Optional[Dict[str, float]] = None,
    ):
        self._equity = account_equity
        self._rate_overrides = rate_overrides or {}
        self._lock = threading.Lock()
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────── #

    def scan(self) -> List[CarrySignal]:
        """
        One-shot daily scan.  Returns a CarrySignal for each pair.

        Does NOT open or close positions — caller decides execution.
        Designed to be called from execute_daily.py each morning.
        """
        signals: List[CarrySignal] = []
        for cfg in CARRY_PAIRS:
            try:
                sig = self._evaluate_pair(cfg)
                signals.append(sig)
            except Exception as exc:
                logger.warning(f'[CarryEngine] Failed to evaluate {cfg.ticker}: {exc}')
        return signals

    def log_signals(self, signals: List[CarrySignal]) -> None:
        """Append scan results to the monthly carry ledger (separate from signal ledger)."""
        month = datetime.now(timezone.utc).strftime('%Y_%m')
        ledger_file = LEDGER_DIR / f'carry_ledger_{month}.jsonl'
        with self._lock:
            with open(ledger_file, 'a') as f:
                for sig in signals:
                    f.write(json.dumps(asdict(sig)) + '\n')
        logger.info(f'[CarryEngine] Logged {len(signals)} signals → {ledger_file}')

    def run_forever(self) -> None:
        """
        Daemon loop: scan every ``SCAN_INTERVAL_HOURS`` hours.

        Run in a daemon thread so it exits automatically when the main process
        ends.  Does not block — call from a Thread(daemon=True).
        """
        logger.info('[CarryEngine] Starting permanent carry-income floor loop '
                    f'(scan every {SCAN_INTERVAL_HOURS}h)')
        while True:
            try:
                signals = self.scan()
                self.log_signals(signals)
                self._print_summary(signals)
            except Exception as exc:
                logger.error(f'[CarryEngine] Scan error: {exc}')
            time.sleep(SCAN_INTERVAL_HOURS * 3600)

    def update_equity(self, new_equity: float) -> None:
        """Update account equity for position sizing (thread-safe)."""
        with self._lock:
            self._equity = new_equity

    # ── Internal ────────────────────────────────────────────────────────── #

    def _evaluate_pair(self, cfg: CarryPairConfig) -> CarrySignal:
        """Evaluate a single carry pair and return a CarrySignal."""
        base_rate  = self._get_rate(cfg.base_currency)
        quote_rate = self._get_rate(cfg.quote_currency)

        # Annualised spread in bps
        if cfg.high_yield_side == 'BASE':
            spread_bps = (base_rate - quote_rate) * 100.0
        else:
            spread_bps = (quote_rate - base_rate) * 100.0

        prices = self._fetch_prices(cfg.ticker)
        atr = self._compute_atr(prices)
        spot = float(prices['Close'].iloc[-1]) if prices is not None and len(prices) else 1.0

        if spread_bps < MIN_CARRY_SPREAD_BPS:
            return CarrySignal(
                ticker=cfg.ticker,
                direction='FLAT',
                carry_spread_bps=round(spread_bps, 1),
                atr=round(atr, 5),
                stop_distance=round(atr * ATR_STOP_MULTIPLE, 5),
                position_size_pct=0.0,
                units=0.0,
                reason=f'Spread {spread_bps:.0f}bps < minimum {MIN_CARRY_SPREAD_BPS}bps — no edge',
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Direction: long when high-yield side is base, short otherwise
        direction = 'LONG' if cfg.high_yield_side == 'BASE' else 'SHORT'

        # Position sizing: risk 0.3% of equity on the stop
        stop_distance = atr * ATR_STOP_MULTIPLE
        risk_amount   = self._equity * CARRY_RISK_PER_PAIR
        units = (risk_amount / stop_distance) if stop_distance > 0 else 0.0

        return CarrySignal(
            ticker=cfg.ticker,
            direction=direction,
            carry_spread_bps=round(spread_bps, 1),
            atr=round(atr, 5),
            stop_distance=round(stop_distance, 5),
            position_size_pct=CARRY_RISK_PER_PAIR,
            units=round(units, 2),
            reason=(
                f'{cfg.description} | '
                f'spread={spread_bps:.0f}bps | '
                f'stop={ATR_STOP_MULTIPLE}×ATR={stop_distance:.5f}'
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _get_rate(self, currency: str) -> float:
        """Return the current central bank policy rate (%) for a currency."""
        if currency in self._rate_overrides:
            return self._rate_overrides[currency]

        # Fallback to the same data source used by the macro signal engine
        try:
            from sovereign.forex.data_fetcher import ForexDataFetcher, FALLBACK_RATES
            from sovereign.forex.pair_universe import CB_TO_COUNTRY
            # Reverse map: currency → country code
            _CCY_TO_COUNTRY: Dict[str, str] = {
                'AUD': 'AU', 'NZD': 'NZ', 'JPY': 'JP', 'CHF': 'CH',
                'USD': 'US', 'EUR': 'EU', 'GBP': 'UK', 'CAD': 'CA',
            }
            country = _CCY_TO_COUNTRY.get(currency, currency[:2])
            fetcher = ForexDataFetcher()
            series = fetcher.get_rate_history(country, start='2020-01-01')
            if series is not None and len(series) > 0:
                return float(series.iloc[-1])
            return FALLBACK_RATES.get(country, 2.0)
        except Exception:
            # Hard fallback rates (as of 2024, update annually)
            _FALLBACK: Dict[str, float] = {
                'AUD': 4.35, 'NZD': 5.50, 'JPY': 0.10,
                'CHF': 1.50, 'USD': 5.25, 'EUR': 4.00,
                'GBP': 5.25, 'CAD': 5.00,
            }
            return _FALLBACK.get(currency, 2.0)

    @staticmethod
    def _fetch_prices(ticker: str):
        """Fetch recent OHLCV for ATR computation.  Returns None on failure."""
        try:
            import yfinance as yf
            df = yf.download(ticker, period='30d', progress=False, auto_adjust=True)
            if df is None or df.empty:
                return None
            if hasattr(df.columns, 'get_level_values'):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception:
            return None

    @staticmethod
    def _compute_atr(prices, period: int = ATR_PERIOD) -> float:
        """14-day ATR.  Returns a safe fallback if prices unavailable."""
        try:
            if prices is None or len(prices) < period + 1:
                return 0.001   # safe fallback for most G10 pairs
            h = prices['High']
            l = prices['Low']
            c = prices['Close']
            tr = (
                (h - l).abs()
                .combine((h - c.shift()).abs(), max)
                .combine((l - c.shift()).abs(), max)
            )
            atr = float(tr.rolling(period).mean().iloc[-1])
            return atr if not math.isnan(atr) else 0.001
        except Exception:
            return 0.001

    @staticmethod
    def _print_summary(signals: List[CarrySignal]) -> None:
        """Log a brief human-readable carry scan summary."""
        logger.info('─── CARRY ENGINE SCAN ───────────────────────────────────')
        for sig in signals:
            logger.info(
                f'  {sig.ticker:<12} {sig.direction:<5} '
                f'spread={sig.carry_spread_bps:>6.0f}bps  '
                f'stop={sig.stop_distance:.5f}  '
                f'units={sig.units:.2f}'
            )
        logger.info('─────────────────────────────────────────────────────────')
