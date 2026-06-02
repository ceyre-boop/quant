"""
Forex pair universe — static config, update annually.
"""
from dataclasses import dataclass
from typing import Dict, List

MAJOR_PAIRS: List[str] = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X',
    'AUDUSD=X',
    # USDCAD removed 2026-05-22 (regime study, 134 trades 2015-2026):
    # avg +0.071%/trade vs portfolio avg +0.204%. No regime where it clearly earns.
    # Bear+calm: only n=3 (too small). Bear+fear: +0.256% (n=24, lower than alternatives).
    # BOC policy historically lags Fed within 1 quarter — rate differential signal
    # redundant with USDJPY and GBPUSD. Oil correlation adds noise without edge.
    # NZDUSD removed Oracle audit 2026-05-17: Sharpe 0.22 across 2015-2024, max DD -11%.
    # Lowest in universe, drag on portfolio avg. RBNZ policy too closely tracks RBA
    # (AUDUSD already captures the Oceania macro signal). Slot freed for future pair.
    # USDCHF removed v004: SNB held -0.75% for 8 years (2014–2022), rate-diff signal
    # structurally broken for that entire window. Sharpe -0.45 across v003+v004.
]

CROSSES: List[str] = [
    # AUDNZD removed HYP-045 2026-06-02: OOS Sharpe -0.879 (2023-2024 holdout).
    # Both legs are RBA-driven (RBNZ tracks RBA within 1 quarter) — rate-differential
    # signal is redundant, no independent edge. Excluding lifts portfolio OOS Sharpe
    # 0.76 → 1.08 (p=0.002, decay 1.61 — ROBUST). Validated via canonical runner.
    # EURJPY removed: dual ECB+BOJ influence creates systematic signal conflicts.
    # CPI fades and calendar signals fight the carry trend → consistent loss.
    # JPY exposure maintained via USDJPY only.
    # EURGBP removed v004: ECB+BOE historically in lockstep — no consistent rate
    # divergence to capture. Sharpe -0.04 across v003+v004, profit factor 1.00.
    # GBPJPY removed 2026-05-22 (regime study, 133 trades 2015-2026):
    # avg +0.168%/trade vs portfolio avg +0.243%. Sharpe 0.741 vs portfolio 1.286.
    # 2022: -0.207% (BOE crisis); 2024: -0.038% — two most recent years both negative.
    # BOJ intervention risk compounds BOE volatility → dual-bank noise, no clean signal.
    # Portfolio improved from 1.1955 → 1.2864 (+0.091) on retirement.
]

ALL_PAIRS: List[str] = MAJOR_PAIRS + CROSSES

# Track 2 — carry base pairs (NOT in ALL_PAIRS; never run through macro signal gates).
# These are managed exclusively by sovereign.forex.carry_engine.CarryEngine.
CARRY_PAIRS: List[str] = [
    'AUDCHF=X',   # borrow CHF (~0-1.5%), hold AUD (~4-5%) — highest G10 carry
    'NZDJPY=X',   # borrow JPY (~0.1%), hold NZD (~5%) — second highest G10 carry
]


@dataclass(frozen=True)
class PairConfig:
    ticker: str
    base_currency: str
    quote_currency: str
    base_central_bank: str
    quote_central_bank: str


PAIR_CONFIG: Dict[str, PairConfig] = {
    'EURUSD=X': PairConfig('EURUSD=X', 'EUR', 'USD', 'ECB', 'FED'),
    'GBPUSD=X': PairConfig('GBPUSD=X', 'GBP', 'USD', 'BOE', 'FED'),
    'USDJPY=X': PairConfig('USDJPY=X', 'USD', 'JPY', 'FED', 'BOJ'),
    'USDCHF=X': PairConfig('USDCHF=X', 'USD', 'CHF', 'FED', 'SNB'),
    'AUDUSD=X': PairConfig('AUDUSD=X', 'AUD', 'USD', 'RBA', 'FED'),
    'USDCAD=X': PairConfig('USDCAD=X', 'USD', 'CAD', 'FED', 'BOC'),
    'NZDUSD=X': PairConfig('NZDUSD=X', 'NZD', 'USD', 'RBNZ', 'FED'),
    'EURGBP=X': PairConfig('EURGBP=X', 'EUR', 'GBP', 'ECB', 'BOE'),
    'EURJPY=X': PairConfig('EURJPY=X', 'EUR', 'JPY', 'ECB', 'BOJ'),
    'GBPJPY=X': PairConfig('GBPJPY=X', 'GBP', 'JPY', 'BOE', 'BOJ'),
    'AUDNZD=X': PairConfig('AUDNZD=X', 'AUD', 'NZD', 'RBA', 'RBNZ'),
}

# Central bank → country mapping for macro data lookups
CB_TO_COUNTRY: Dict[str, str] = {
    'FED':  'US',
    'ECB':  'EU',
    'BOE':  'UK',
    'BOJ':  'JP',
    'SNB':  'CH',
    'RBA':  'AU',
    'BOC':  'CA',
    'RBNZ': 'NZ',
}
