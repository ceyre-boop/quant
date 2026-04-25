"""
ForexMacroEngine — composite macro scorer for all 11 forex pairs.

Weights:
    rate_differential_momentum:  0.30
    irp_z_score:                 0.25
    cycle_divergence:            0.25
    ppp_z_score:                 0.10
    hurst_regime:                0.10

Buffett lens: conviction < 0.35 → NEUTRAL (no trade on noise).
Risk sentiment layer: VIX > 25 + backwardation overrides macro for carry pairs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from sovereign.forex.pair_universe import ALL_PAIRS, PAIR_CONFIG, CB_TO_COUNTRY
from sovereign.forex.data_fetcher import ForexDataFetcher
from sovereign.forex.fair_value import FairValueModel
from sovereign.forex.cycle_detector import CycleDetector
from sovereign.forex.risk_sentiment import RiskSentimentEngine

logger = logging.getLogger(__name__)

WEIGHTS = {
    'rate_diff_momentum': 0.30,
    'irp_z':              0.25,
    'cycle_div':          0.25,
    'ppp_z':              0.10,
    'hurst':              0.10,
}

# Buffett lens thresholds (from cause-effect map Part 10)
CONVICTION_NEUTRAL_THRESHOLD = 0.35   # below this → NEUTRAL, no trade
CONVICTION_FULL_SIZE          = 0.70   # at or above → full size
CONVICTION_MAX_SIZE           = 0.85   # at or above → max size (1.5× normal)


@dataclass
class ForexSignal:
    pair: str
    direction: str        # LONG / SHORT / NEUTRAL
    conviction: float     # 0..1
    hold_period_estimate: int  # trading days
    primary_driver: str
    rate_differential: float
    irp_z: float
    ppp_z: float
    cycle_divergence: float
    hurst: float
    spot: float
    base_cycle: str
    quote_cycle: str


class ForexMacroEngine:

    def __init__(self):
        self._fetcher = ForexDataFetcher()
        self._fv = FairValueModel(self._fetcher)
        self._cycle = CycleDetector()
        self._risk = RiskSentimentEngine()
        self._price_cache: dict = {}

    def score_pair(self, pair: str) -> Optional[ForexSignal]:
        cfg = PAIR_CONFIG.get(pair)
        if not cfg:
            logger.warning(f"Unknown pair: {pair}")
            return None

        base_country = CB_TO_COUNTRY[cfg.base_central_bank]
        quote_country = CB_TO_COUNTRY[cfg.quote_central_bank]

        base_macro = self._fetcher.get_country_macro(base_country)
        quote_macro = self._fetcher.get_country_macro(quote_country)

        price_hist = self._get_price_history(pair)
        if price_hist is None or len(price_hist) < 60:
            logger.warning(f"Insufficient price history for {pair}")
            return None

        spot = float(price_hist.iloc[-1])

        # ── Fair value signals ──────────────────────────────────────── #
        fv_signal = self._fv.score_pair(
            pair=pair,
            spot=spot,
            base_rate=base_macro['rate'],
            quote_rate=quote_macro['rate'],
            base_cpi=base_macro['cpi_yoy'],
            quote_cpi=quote_macro['cpi_yoy'],
            price_history=price_hist,
        )

        # ── Cycle divergence ────────────────────────────────────────── #
        cycle_signal = self._cycle.score_pair(pair, base_macro, quote_macro)

        # ── Rate differential momentum ──────────────────────────────── #
        # The signal is the CHANGE in real rate differential, not the level.
        # Use historical rate data to compute 3M and 6M change in real rate diff.
        # Fall back to trajectory decisions if no history available.
        rate_diff = base_macro['rate'] - quote_macro['rate']
        real_rate_diff = (
            (base_macro['rate'] - base_macro.get('cpi_yoy', 2.0)) -
            (quote_macro['rate'] - quote_macro.get('cpi_yoy', 2.0))
        )
        rdm_score = self._real_rate_diff_momentum(
            base_country, quote_country,
            base_macro, quote_macro,
            real_rate_diff,
        )

        # ── Risk sentiment override ──────────────────────────────────── #
        # Edge 4: VIX > 25 + backwardation → carry unwind overrides macro
        risk_override = self._risk.override_for_pair(pair)

        # ── Hurst exponent ──────────────────────────────────────────── #
        hurst = self._compute_hurst(price_hist)
        if hurst > 0.55:
            hurst_score = 0.3
        elif hurst < 0.45:
            hurst_score = -0.1
        else:
            hurst_score = 0.0

        # ── Composite score ─────────────────────────────────────────── #
        irp_score = np.clip(-fv_signal.irp_z_score / 3.0, -1, 1)
        ppp_score = np.clip(-fv_signal.ppp_z_score / 3.0, -1, 1)
        cycle_score = np.clip(cycle_signal.divergence_score, -1, 1)

        raw = (
            WEIGHTS['rate_diff_momentum'] * rdm_score
            + WEIGHTS['irp_z']            * irp_score
            + WEIGHTS['cycle_div']        * cycle_score
            + WEIGHTS['ppp_z']            * ppp_score
            + WEIGHTS['hurst']            * hurst_score
        )

        conviction = min(abs(raw), 1.0)

        # Buffett lens: below conviction threshold = NEUTRAL (no noise trades)
        if conviction < CONVICTION_NEUTRAL_THRESHOLD and risk_override is None:
            direction = 'NEUTRAL'
        elif risk_override is not None:
            # VIX carry unwind overrides macro entirely
            direction = risk_override
            conviction = min(conviction + 0.3, 1.0)  # risk override adds conviction
        elif raw > 0:
            direction = 'LONG'
        else:
            direction = 'SHORT'

        # Skip if fair value models disagree
        if fv_signal.composite_direction == 'SKIP':
            conviction *= 0.5

        # ── Primary driver ──────────────────────────────────────────── #
        drivers = {
            'rate_diff_momentum': abs(WEIGHTS['rate_diff_momentum'] * rdm_score),
            'irp_mean_reversion': abs(WEIGHTS['irp_z'] * irp_score),
            'cycle_divergence':   abs(WEIGHTS['cycle_div'] * cycle_score),
            'ppp_deviation':      abs(WEIGHTS['ppp_z'] * ppp_score),
        }
        primary_driver = max(drivers, key=drivers.get)

        hold = self._estimate_hold(primary_driver, conviction)

        return ForexSignal(
            pair=pair,
            direction=direction,
            conviction=round(conviction, 3),
            hold_period_estimate=hold,
            primary_driver=primary_driver,
            rate_differential=round(rate_diff, 3),
            irp_z=round(fv_signal.irp_z_score, 3),
            ppp_z=round(fv_signal.ppp_z_score, 3),
            cycle_divergence=round(cycle_signal.divergence_score, 3),
            hurst=round(hurst, 3),
            spot=round(spot, 5),
            base_cycle=cycle_signal.base_cycle.phase,
            quote_cycle=cycle_signal.quote_cycle.phase,
        )

    def scan_all_pairs(self) -> List[ForexSignal]:
        signals = []
        for pair in ALL_PAIRS:
            try:
                sig = self.score_pair(pair)
                if sig and sig.direction != 'NEUTRAL':
                    signals.append(sig)
            except Exception as e:
                logger.warning(f"score_pair failed for {pair}: {e}")

        signals.sort(key=lambda s: s.conviction, reverse=True)

        print(f"\n{'PAIR':12s} {'DIR':6s} {'CONV':6s} {'DRIVER'}")
        print('-' * 60)
        for s in signals:
            print(
                f"{s.pair:12s} {s.direction:6s} "
                f"{s.conviction:.2f}   {s.primary_driver}"
            )
        print()

        top3 = [s for s in signals if s.conviction >= CONVICTION_NEUTRAL_THRESHOLD][:3]
        return top3 if top3 else signals[:3]

    # ── Helpers ──────────────────────────────────────────────────────── #

    def _real_rate_diff_momentum(
        self,
        base_country: str,
        quote_country: str,
        base_macro: dict,
        quote_macro: dict,
        current_real_diff: float,
    ) -> float:
        """
        Rate differential momentum = change in REAL rate differential.
        Cause-effect map: 'Not the level — the change.'
        Z-score the 3M change against 5-year history.
        Falls back to trajectory sum if no FRED history.
        """
        try:
            base_rates = self._fetcher.get_rate_history(base_country, start='2019-01-01')
            quote_rates = self._fetcher.get_rate_history(quote_country, start='2019-01-01')
            base_cpi_h = self._fetcher.get_cpi_history(base_country, start='2019-01-01')
            quote_cpi_h = self._fetcher.get_cpi_history(quote_country, start='2019-01-01')

            if (base_rates is None or quote_rates is None or
                    len(base_rates) < 63 or len(quote_rates) < 63):
                raise ValueError("insufficient rate history")

            # Align all series
            idx = base_rates.index.intersection(quote_rates.index)
            br = base_rates.reindex(idx).ffill()
            qr = quote_rates.reindex(idx).ffill()
            bc = base_cpi_h.reindex(idx).ffill() if len(base_cpi_h) > 10 else pd.Series(
                base_macro.get('cpi_yoy', 2.0), index=idx)
            qc = quote_cpi_h.reindex(idx).ffill() if len(quote_cpi_h) > 10 else pd.Series(
                quote_macro.get('cpi_yoy', 2.0), index=idx)

            real_diff = (br - bc) - (qr - qc)
            # 3-month change (63 business days)
            rdm_3m = real_diff.diff(63).dropna()
            if len(rdm_3m) < 63:
                raise ValueError("insufficient history for 3M diff")

            # Z-score the current 3M change against 5-year window
            window = min(1260, len(rdm_3m))  # 5 years
            mu = rdm_3m.tail(window).mean()
            sigma = rdm_3m.tail(window).std()
            current_change = float(rdm_3m.iloc[-1])
            z = (current_change - mu) / sigma if sigma > 0 else 0.0
            return float(np.clip(z / 2.0, -1, 1))  # normalize z to [-1, 1]

        except Exception:
            # Fallback: trajectory decisions
            base_net = sum(base_macro.get('rate_trajectory', [0, 0, 0]))
            quote_net = sum(quote_macro.get('rate_trajectory', [0, 0, 0]))
            return float(np.clip((base_net - quote_net) / 6.0, -1, 1))

    def _get_price_history(self, pair: str, years: int = 10) -> Optional[pd.Series]:
        if pair in self._price_cache:
            return self._price_cache[pair]
        try:
            start = pd.Timestamp.now() - pd.DateOffset(years=years)
            df = yf.download(
                pair, start=start.strftime('%Y-%m-%d'),
                progress=False, auto_adjust=True
            )
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            s = df['Close'].dropna()
            s.name = pair
            self._price_cache[pair] = s
            return s
        except Exception as e:
            logger.warning(f"Price download failed for {pair}: {e}")
            return None

    @staticmethod
    def _compute_hurst(prices: pd.Series, lags: int = 20) -> float:
        """R/S analysis hurst exponent estimate."""
        try:
            p = prices.dropna().values
            if len(p) < 100:
                return 0.5
            log_returns = np.log(p[1:] / p[:-1])
            tau = range(2, min(lags, len(log_returns) // 4))
            rs_list = []
            for t in tau:
                chunks = [log_returns[i:i+t] for i in range(0, len(log_returns)-t, t)]
                rs_vals = []
                for chunk in chunks:
                    if len(chunk) < 2:
                        continue
                    mean = chunk.mean()
                    std = chunk.std()
                    if std == 0:
                        continue
                    cumdev = np.cumsum(chunk - mean)
                    rs_vals.append((cumdev.max() - cumdev.min()) / std)
                if rs_vals:
                    rs_list.append(np.mean(rs_vals))
            if len(rs_list) < 4:
                return 0.5
            log_tau = np.log(list(tau)[:len(rs_list)])
            log_rs = np.log(rs_list)
            h = np.polyfit(log_tau, log_rs, 1)[0]
            return float(np.clip(h, 0.1, 0.9))
        except Exception:
            return 0.5

    @staticmethod
    def _estimate_hold(driver: str, conviction: float) -> int:
        base = {
            'irp_mean_reversion': 30,
            'rate_diff_momentum': 45,
            'cycle_divergence':   60,
            'ppp_deviation':      90,
        }.get(driver, 40)
        # Higher conviction → hold longer
        return int(base * (0.7 + 0.6 * conviction))
