"""
Forex pair universe — static config, update annually.
"""
from dataclasses import dataclass
from typing import Dict, List

MAJOR_PAIRS: List[str] = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X',
    'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X',
    # USDCHF removed v004: SNB held -0.75% for 8 years (2014–2022), rate-diff signal
    # structurally broken for that entire window. Sharpe -0.45 across v003+v004.
]

CROSSES: List[str] = [
    'GBPJPY=X', 'AUDNZD=X',
    # EURJPY removed: dual ECB+BOJ influence creates systematic signal conflicts.
    # CPI fades and calendar signals fight the carry trend → consistent loss.
    # JPY exposure maintained via USDJPY and GBPJPY.
    # EURGBP removed v004: ECB+BOE historically in lockstep — no consistent rate
    # divergence to capture. Sharpe -0.04 across v003+v004, profit factor 1.00.
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
