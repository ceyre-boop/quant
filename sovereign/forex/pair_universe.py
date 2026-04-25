"""
Forex pair universe — static config, update annually.
"""
from dataclasses import dataclass
from typing import Dict, List

MAJOR_PAIRS: List[str] = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
    'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X',
]

CROSSES: List[str] = [
    'EURGBP=X', 'EURJPY=X', 'GBPJPY=X', 'AUDNZD=X',
]

ALL_PAIRS: List[str] = MAJOR_PAIRS + CROSSES


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
