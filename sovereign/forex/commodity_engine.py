"""
Commodity-currency correlation engine (Edge 5).

Cause-effect map: commodity prices move first, currency adjusts with a lag.
When correlation breaks (commodity up, currency flat/down) → currency catches up
within 5–15 trading days.

Pairs with commodity linkage:
  AUD → iron ore (BHP as proxy, also VALE, ^GSCI metals sub-index)
  CAD → crude oil (CL=F — WTI)
  NZD → dairy (DBA agriculture ETF as proxy — dairy futures not on yfinance)
  NOK → Brent crude (BZ=F) — not in our universe but included for future use
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Commodity proxy tickers for each currency
COMMODITY_PROXIES: Dict[str, str] = {
    'AUD': 'BHP',        # BHP Group — largest iron ore/coal miner; strong AUD proxy
    'CAD': 'CL=F',       # WTI crude oil futures
    'NZD': 'DBA',        # Agriculture ETF — dairy proxy (not perfect but available)
}

# Forex pairs where commodity signal applies (base or quote currency)
COMMODITY_PAIRS: Dict[str, tuple] = {
    'AUDUSD=X':  ('AUD', +1),   # +1 = higher commodity → higher pair price
    'USDCAD=X':  ('CAD', -1),   # -1 = higher oil → lower USD/CAD (CAD strengthens)
    'NZDUSD=X':  ('NZD', +1),
    'AUDNZD=X':  ('AUD', +1),   # iron ore vs dairy — weaker signal, use with caution
}

LOOKBACK_DAYS = 60   # rolling correlation window
LAG_THRESHOLD = 5.0  # commodity moved X% but currency didn't follow → signal
CORR_THRESHOLD = 0.4  # minimum correlation to flag as commodity-linked


@dataclass
class CommoditySignal:
    pair: str
    commodity: str
    commodity_ticker: str
    commodity_5d_return: float      # 5-day return of commodity proxy
    currency_5d_return: float       # 5-day return of the pair
    rolling_correlation: float      # 60-day rolling correlation
    lag_detected: bool              # commodity moved, currency lagged
    direction: str                  # LONG / SHORT / NEUTRAL
    conviction: float               # 0..1


class CommodityEngine:

    def __init__(self):
        self._price_cache: dict = {}

    def score_pair(self, pair: str) -> Optional[CommoditySignal]:
        if pair not in COMMODITY_PAIRS:
            return None

        currency, direction_sign = COMMODITY_PAIRS[pair]
        commodity_ticker = COMMODITY_PROXIES.get(currency)
        if not commodity_ticker:
            return None

        pair_prices = self._get_prices(pair, is_forex=True)
        comm_prices = self._get_prices(commodity_ticker, is_forex=False)

        if pair_prices is None or comm_prices is None:
            return None

        # Align on common dates
        aligned = pd.concat(
            [pair_prices.rename('pair'), comm_prices.rename('comm')], axis=1
        ).dropna()

        if len(aligned) < LOOKBACK_DAYS:
            return None

        aligned = aligned.tail(LOOKBACK_DAYS + 20)
        pair_ret = aligned['pair'].pct_change().dropna()
        comm_ret = aligned['comm'].pct_change().dropna()

        # Rolling correlation
        roll_corr = pair_ret.rolling(LOOKBACK_DAYS).corr(comm_ret).iloc[-1]
        if pd.isna(roll_corr):
            roll_corr = 0.0

        # 5-day returns
        comm_5d = float((aligned['comm'].iloc[-1] / aligned['comm'].iloc[-6] - 1)) if len(aligned) >= 6 else 0.0
        pair_5d = float((aligned['pair'].iloc[-1] / aligned['pair'].iloc[-6] - 1)) if len(aligned) >= 6 else 0.0

        # Lag signal: commodity moved significantly, currency hasn't caught up
        lag_detected = (
            abs(comm_5d) >= 0.04 and          # commodity moved 4%+
            abs(pair_5d) < abs(comm_5d) * 0.5  # currency moved less than half
        )

        # Direction: follow commodity move, adjusted by sign convention
        if lag_detected and abs(roll_corr) >= CORR_THRESHOLD:
            raw_signal = direction_sign * np.sign(comm_5d)
            conviction = min(abs(comm_5d) * 5, 0.8)  # scale: 4% move → 0.2, 10% move → 0.5
            if abs(roll_corr) > 0.6:
                conviction = min(conviction * 1.3, 0.85)
            direction = 'LONG' if raw_signal > 0 else 'SHORT'
        else:
            direction = 'NEUTRAL'
            conviction = 0.0

        return CommoditySignal(
            pair=pair,
            commodity=currency,
            commodity_ticker=commodity_ticker,
            commodity_5d_return=round(comm_5d, 4),
            currency_5d_return=round(pair_5d, 4),
            rolling_correlation=round(float(roll_corr), 3),
            lag_detected=lag_detected,
            direction=direction,
            conviction=round(conviction, 3),
        )

    def scan_commodity_pairs(self) -> list[CommoditySignal]:
        results = []
        for pair in COMMODITY_PAIRS:
            try:
                sig = self.score_pair(pair)
                if sig and sig.direction != 'NEUTRAL':
                    results.append(sig)
            except Exception as e:
                logger.warning(f"Commodity signal failed for {pair}: {e}")
        return sorted(results, key=lambda s: s.conviction, reverse=True)

    # ── Data ──────────────────────────────────────────────────────────── #

    def _get_prices(self, ticker: str, is_forex: bool) -> Optional[pd.Series]:
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        try:
            df = yf.download(
                ticker, period='6mo', progress=False, auto_adjust=True
            )
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            s = df['Close'].dropna()
            s.name = ticker
            self._price_cache[ticker] = s
            return s
        except Exception as e:
            logger.warning(f"Price download failed for {ticker}: {e}")
            return None
