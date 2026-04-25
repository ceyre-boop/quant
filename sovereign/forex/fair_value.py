"""
Fair value models for forex pairs.

Method A: Interest Rate Parity (IRP) — short-term, mean-reverting
Method B: Purchasing Power Parity (PPP) — long-term, structural

IRP: forward = spot × (1 + quote_rate/100) / (1 + base_rate/100)
PPP: relative CPI accumulation since a base year
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PPP base-year rates (OECD, 2020 reference)
# USD per 1 unit of foreign currency at PPP
PPP_BASE: dict = {
    'EURUSD=X': 1.08,
    'GBPUSD=X': 1.28,
    'USDJPY=X': 108.0,
    'USDCHF=X': 0.98,
    'AUDUSD=X': 0.72,
    'USDCAD=X': 1.22,
    'NZDUSD=X': 0.66,
    'EURGBP=X': 0.85,
    'EURJPY=X': 120.0,
    'GBPJPY=X': 142.0,
    'AUDNZD=X': 1.07,
}


@dataclass
class FairValueSignal:
    pair: str
    spot: float
    irp_fair_value: float
    irp_z_score: float
    irp_direction: str       # LONG / SHORT / NEUTRAL
    ppp_fair_value: float
    ppp_z_score: float
    ppp_direction: str
    rate_differential: float
    real_rate_differential: float
    composite_direction: str  # LONG / SHORT / SKIP (disagreement)
    composite_strength: float  # 0..1


class FairValueModel:

    IRP_Z_THRESHOLD = 1.5
    PPP_Z_THRESHOLD = 1.5
    HISTORY_WINDOW = 756   # 3 years business days for IRP z-score
    PPP_WINDOW = 2520      # 10 years for PPP z-score

    def __init__(self, fetcher):
        self._fetcher = fetcher

    def score_pair(
        self,
        pair: str,
        spot: float,
        base_rate: float,
        quote_rate: float,
        base_cpi: float,
        quote_cpi: float,
        price_history: Optional[pd.Series] = None,
    ) -> FairValueSignal:
        # ── IRP ────────────────────────────────────────────────────────
        irp_fv = spot * (1 + quote_rate / 100) / (1 + base_rate / 100)
        irp_dev = (spot - irp_fv) / irp_fv

        if price_history is not None and len(price_history) >= 60:
            ph = price_history.dropna()
            # Compute IRP FV for each historical point using current rates
            # (simplified: rates change slowly, current differential ok for zscore)
            irp_fv_hist = ph * (1 + quote_rate / 100) / (1 + base_rate / 100)
            irp_dev_hist = (ph - irp_fv_hist) / irp_fv_hist
            window = min(self.HISTORY_WINDOW, len(irp_dev_hist))
            mu = irp_dev_hist.tail(window).mean()
            sigma = irp_dev_hist.tail(window).std()
            irp_z = (irp_dev - mu) / sigma if sigma > 0 else 0.0
        else:
            irp_z = irp_dev * 10  # rough proxy

        irp_dir = self._direction(irp_z, self.IRP_Z_THRESHOLD)

        # ── PPP ────────────────────────────────────────────────────────
        ppp_fv = PPP_BASE.get(pair, spot)
        # Adjust PPP by cumulative CPI differential since 2020
        # Approximate: 4 years × avg CPI diff per year
        years_since_base = 4.0
        cpi_adjustment = (1 + base_cpi / 100) ** years_since_base / \
                         (1 + quote_cpi / 100) ** years_since_base
        ppp_fv_adj = ppp_fv * cpi_adjustment
        ppp_dev = (spot - ppp_fv_adj) / ppp_fv_adj

        if price_history is not None and len(price_history) >= 252:
            ph = price_history.dropna()
            window = min(self.PPP_WINDOW, len(ph))
            mu_p = ph.tail(window).mean()
            sigma_p = ph.tail(window).std()
            ppp_z = (spot - mu_p) / sigma_p if sigma_p > 0 else 0.0
        else:
            ppp_z = ppp_dev * 5

        ppp_dir = self._direction(ppp_z, self.PPP_Z_THRESHOLD)

        # ── Composite ──────────────────────────────────────────────────
        rate_diff = base_rate - quote_rate
        real_rate_diff = (base_rate - base_cpi) - (quote_rate - quote_cpi)

        irp_score = np.clip(-irp_z / 3.0, -1, 1)
        ppp_score = np.clip(-ppp_z / 3.0, -1, 1)
        composite_score = 0.6 * irp_score + 0.4 * ppp_score

        if irp_dir != 'NEUTRAL' and ppp_dir != 'NEUTRAL' and irp_dir != ppp_dir:
            composite_dir = 'SKIP'
        elif composite_score > 0.15:
            composite_dir = 'LONG'
        elif composite_score < -0.15:
            composite_dir = 'SHORT'
        else:
            composite_dir = 'NEUTRAL'

        return FairValueSignal(
            pair=pair,
            spot=spot,
            irp_fair_value=round(irp_fv, 5),
            irp_z_score=round(irp_z, 3),
            irp_direction=irp_dir,
            ppp_fair_value=round(ppp_fv_adj, 5),
            ppp_z_score=round(ppp_z, 3),
            ppp_direction=ppp_dir,
            rate_differential=round(rate_diff, 3),
            real_rate_differential=round(real_rate_diff, 3),
            composite_direction=composite_dir,
            composite_strength=round(abs(composite_score), 3),
        )

    @staticmethod
    def _direction(z: float, threshold: float) -> str:
        if z > threshold:
            return 'SHORT'   # overvalued → expect depreciation
        if z < -threshold:
            return 'LONG'    # undervalued → expect appreciation
        return 'NEUTRAL'
