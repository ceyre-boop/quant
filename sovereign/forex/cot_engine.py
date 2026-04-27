"""
COT Engine — CFTC Commitments of Traders positioning gate.

Dalio Q3: is the market already positioned for this trade?
If speculators are crowded in our direction → halve the size.

Source: CFTC Legacy Futures-Only report (no API key).
URL: https://www.cftc.gov/dea/newcot/c_disagg.txt
Cache: data/cache/cot/{currency}_cot.parquet, refreshed weekly.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parents[2] / 'data' / 'cache' / 'cot'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DAYS = 7
CROWDED_Z = 1.5
LOOKBACK_WEEKS = 156  # 3 years

FUTURES_CODES: dict[str, str] = {
    'EUR': '099741', 'GBP': '096742', 'JPY': '097741',
    'CHF': '092741', 'AUD': '232741', 'CAD': '090741', 'NZD': '112741',
}


class COTEngine:

    def get_positioning(self, currency: str) -> Optional[dict]:
        series = self._load_or_fetch(currency)
        if series is None or len(series) < 10:
            return None

        net = float(series.iloc[-1])
        window = series.tail(LOOKBACK_WEEKS)
        mu, sigma = window.mean(), window.std()
        z = float((net - mu) / sigma) if sigma > 0 else 0.0

        signal = ('CROWDED_LONG' if z > CROWDED_Z else
                  'CROWDED_SHORT' if z < -CROWDED_Z else 'NEUTRAL')

        return {
            'currency':     currency,
            'net_position': round(net, 0),
            'z_score':      round(z, 3),
            'signal':       signal,
            'as_of':        str(series.index[-1].date()),
        }

    def gate_signal(self, direction: str, currency: str) -> float:
        pos = self.get_positioning(currency)
        if pos is None:
            return 1.0
        if direction == 'LONG' and pos['signal'] == 'CROWDED_LONG':
            return 0.5
        if direction == 'SHORT' and pos['signal'] == 'CROWDED_SHORT':
            return 0.5
        return 1.0

    def _load_or_fetch(self, currency: str) -> Optional[pd.Series]:
        cache_path = CACHE_DIR / f'{currency}_cot.parquet'
        if cache_path.exists():
            age = (datetime.now() - datetime.fromtimestamp(
                cache_path.stat().st_mtime)).days
            if age < CACHE_DAYS:
                return pd.read_parquet(cache_path).squeeze()

        series = self._fetch_cftc(currency)
        if series is not None and not series.empty:
            series.to_frame('net_position').to_parquet(cache_path)
        return series

    def _fetch_cftc(self, currency: str) -> Optional[pd.Series]:
        code = FUTURES_CODES.get(currency)
        if not code:
            return None
        try:
            url = 'https://www.cftc.gov/dea/newcot/c_disagg.txt'
            df = pd.read_csv(url, low_memory=False)
            df.columns = [c.strip() for c in df.columns]

            market_col = next(c for c in df.columns if 'CFTC' in c.upper() and 'CODE' in c.upper())
            date_col   = next(c for c in df.columns if 'DATE' in c.upper())
            long_col   = next(c for c in df.columns if 'NONCOMM' in c.upper() and 'LONG' in c.upper() and 'ALL' in c.upper())
            short_col  = next(c for c in df.columns if 'NONCOMM' in c.upper() and 'SHORT' in c.upper() and 'ALL' in c.upper())

            sub = df[df[market_col].astype(str).str.strip() == code].copy()
            if sub.empty:
                logger.warning(f"No CFTC rows for {currency} (code {code})")
                return None

            sub[date_col] = pd.to_datetime(sub[date_col])
            sub = sub.sort_values(date_col)
            sub['net'] = (pd.to_numeric(sub[long_col], errors='coerce') -
                          pd.to_numeric(sub[short_col], errors='coerce'))
            s = sub.set_index(date_col)['net'].dropna()
            s.name = f'{currency}_cot_net'
            return s
        except Exception as e:
            logger.warning(f"CFTC fetch failed for {currency}: {e}")
            return None
