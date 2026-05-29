#!/usr/bin/env python3
"""
scripts/fetch_macro_cache.py
Fetch key macro series via yfinance (FRED fallback when accessible).
Saves data/macro/macro_snapshot.json for real-time dashboard use.

Run: python3 scripts/fetch_macro_cache.py
Schedule: every 4h via launchd (macro data is daily, 4h refresh is ample)
"""
import json
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, '.')

SNAPSHOT_PATH = 'data/macro/macro_snapshot.json'
HISTORY_DAYS = 252  # 1 trading year of history for z-scores

# yfinance symbols → internal key
YFINANCE_MAP = {
    '^VIX':    'vix',       # CBOE VIX
    '^TNX':    'dgs10',     # 10-Year Treasury yield (%)
    '^IRX':    'dgs3m',     # 13-Week T-Bill rate (%)
    '2YY=F':   'dgs2',      # 2-Year Treasury yield futures (%)
    'HYG':     'hyg',       # iShares HY Bond ETF (credit proxy)
    'TIP':     'tip',       # iShares TIPS ETF (real yield context)
}

# FRED series IDs to try (best-effort, skipped if timeout)
FRED_MAP = {
    'T10YIE':      'breakeven10y',   # 10Y Breakeven Inflation
    'WM2NS':       'm2',             # M2 Money Stock
    'BAMLH0A0HYM2':'hy_oas',         # HY OAS spread (bps)
    'FEDFUNDS':    'fedfunds',       # Fed Funds Rate
}

FRED_TIMEOUT = 12  # seconds — skip if FRED is slow


def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k, v)
    except FileNotFoundError:
        pass


def _fetch_yfinance() -> dict:
    import yfinance as yf
    import numpy as np
    import pandas as pd

    tickers = list(YFINANCE_MAP.keys())
    data = yf.download(
        tickers,
        period=f'{HISTORY_DAYS + 10}d',
        progress=False,
        auto_adjust=True,
        group_by='ticker',
    )

    series = {}
    latest = {}
    prev = {}
    history = {}

    # Multi-ticker download uses (ticker, field) MultiIndex
    import pandas as pd
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs('Close', axis=1, level=1)
    else:
        close = data['Close'] if 'Close' in data.columns else data

    for ticker, key in YFINANCE_MAP.items():
        try:
            col = close[ticker] if ticker in close.columns else None
            if col is None or col.dropna().empty:
                continue
            col = col.dropna()
            val = float(col.iloc[-1])
            prev_val = float(col.iloc[-2]) if len(col) > 1 else val

            # Normalize scale
            if key == 'dgs10':
                # ^TNX: old versions × 10 format
                if val > 20:
                    col = col / 10
                    val = float(col.iloc[-1])
                    prev_val = float(col.iloc[-2])
            elif key == 'dgs3m':
                # ^IRX is already a rate (e.g. 3.59 = 3.59%)
                pass

            # 252-day z-score
            mu = col.mean()
            sigma = col.std()
            z = (val - mu) / sigma if sigma > 0 else 0.0

            # Recent history (last 90 days) as [{t, v}] for chart
            h90 = col.tail(90)
            hist = [{'t': int(ts.timestamp() * 1000), 'v': round(float(v), 4)}
                    for ts, v in h90.items() if not (v != v)]  # skip NaN

            series[key] = {
                'value': round(val, 4),
                'prev':  round(prev_val, 4),
                'chg':   round(val - prev_val, 4),
                'z':     round(z, 2),
                'source': 'yfinance',
            }
            history[key] = hist
            latest[key] = val
            prev[key] = prev_val

        except Exception as e:
            pass

    # Derived series
    if 'dgs10' in latest and 'dgs2' in latest:
        spread = latest['dgs10'] - latest['dgs2']
        series['t10y2y'] = {
            'value': round(spread, 4),
            'prev':  round(prev['dgs10'] - prev['dgs2'], 4),
            'chg':   round(spread - (prev['dgs10'] - prev['dgs2']), 4),
            'z':     None,
            'source': 'computed',
        }

    if 'dgs10' in latest and 'dgs3m' in latest:
        spread = latest['dgs10'] - latest['dgs3m']
        series['t10y3m'] = {
            'value': round(spread, 4),
            'prev':  round(prev['dgs10'] - prev['dgs3m'], 4),
            'chg':   round(spread - (prev['dgs10'] - prev['dgs3m']), 4),
            'z':     None,
            'source': 'computed',
        }

    return series, history


def _fetch_fred(series: dict) -> dict:
    """Best-effort FRED fetch — adds to series dict, skips on timeout."""
    key = os.environ.get('FRED_API_KEY', '')
    if not key:
        return series
    try:
        from fredapi import Fred
        import signal as _sig

        fred = Fred(api_key=key)

        for fred_id, internal_key in FRED_MAP.items():
            try:
                data = fred.get_series(fred_id)
                if data is not None and not data.empty:
                    data = data.dropna()
                    val = float(data.iloc[-1])
                    prev_val = float(data.iloc[-2]) if len(data) > 1 else val
                    mu = data.mean()
                    sigma = data.std()
                    z = (val - mu) / sigma if sigma > 0 else 0.0
                    series[internal_key] = {
                        'value': round(val, 4),
                        'prev':  round(prev_val, 4),
                        'chg':   round(val - prev_val, 4),
                        'z':     round(z, 2),
                        'source': 'FRED',
                    }
            except Exception:
                pass
    except Exception:
        pass
    return series


def build_snapshot(series: dict, history: dict) -> dict:
    """Build the final snapshot with derived display metrics."""
    now = datetime.now(timezone.utc).isoformat()

    # Yield curve regime
    t10y2y = (series.get('t10y2y') or {}).get('value')
    t10y3m = (series.get('t10y3m') or {}).get('value')
    vix = (series.get('vix') or {}).get('value')
    z_vix = (series.get('vix') or {}).get('z')

    yield_curve_regime = 'NORMAL'
    if t10y2y is not None and t10y3m is not None:
        if t10y2y < 0 or t10y3m < 0:
            yield_curve_regime = 'INVERTED'
        elif t10y2y < 0.25 and t10y3m < 0.25:
            yield_curve_regime = 'FLAT'

    risk_regime = 'NORMAL'
    if vix is not None:
        if vix > 30:
            risk_regime = 'RISK_OFF'
        elif vix > 20:
            risk_regime = 'CAUTION'

    # Credit stress via HYG — HYG falling = spreads widening = stress
    hyg_z = (series.get('hyg') or {}).get('z')
    credit_regime = 'NORMAL'
    if hyg_z is not None:
        if hyg_z < -1.5:
            credit_regime = 'STRESS'
        elif hyg_z < -0.5:
            credit_regime = 'CAUTION'

    return {
        'fetched_at': now,
        'series': series,
        'history': history,
        'summary': {
            'yield_curve': yield_curve_regime,
            'risk_regime': risk_regime,
            'credit_regime': credit_regime,
            'vix': vix,
            'vix_z': z_vix,
            'dgs10': (series.get('dgs10') or {}).get('value'),
            'dgs2': (series.get('dgs2') or {}).get('value'),
            'dgs3m': (series.get('dgs3m') or {}).get('value'),
            't10y2y': t10y2y,
            't10y3m': t10y3m,
            'hyg': (series.get('hyg') or {}).get('value'),
            'hyg_z': hyg_z,
        },
    }


def main():
    _load_env()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching macro data...")

    series, history = _fetch_yfinance()
    print(f"  yfinance: {len(series)} series fetched")

    series = _fetch_fred(series)
    fred_keys = [k for k, v in series.items() if isinstance(v, dict) and v.get('source') == 'FRED']
    if fred_keys:
        print(f"  FRED: {fred_keys}")

    snapshot = build_snapshot(series, history)

    os.makedirs('data/macro', exist_ok=True)
    with open(SNAPSHOT_PATH, 'w') as f:
        json.dump(snapshot, f, indent=2)

    s = snapshot['summary']
    print(f"  VIX={s['vix']} (z={s['vix_z']})  10Y={s['dgs10']}%  2Y={s['dgs2']}%  "
          f"10Y-2Y={s['t10y2y']}%  yield_curve={s['yield_curve']}  risk={s['risk_regime']}")
    print(f"  Saved → {SNAPSHOT_PATH}")


if __name__ == '__main__':
    main()
