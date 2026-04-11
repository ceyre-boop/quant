"""
Sovereign Trading Intelligence — Dynamic Universe Selection
Phase 1: Data Foundation

Universe determined by Hurst regime state at signal date ONLY.
Historical performance NEVER selects the universe.
The data decides. Not the researcher.

Hurst Exponent (R/S method):
- H > 0.52  → asset eligible for MOMENTUM specialist
- H < 0.45  → asset eligible for REVERSION specialist
- 0.45–0.52 → DEAD ZONE — excluded from both specialists

The dead zone exists because we built both specialists and
discovered that BOTH get killed in this range. This is not
a theoretical threshold — it is an empirical scar.

The Hurst implementation uses the R/S (Rescaled Range) method
validated in research/hurst_diagnostic.py. The exact same
calculation is used here — no modifications, no "improvements".
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sovereign configuration mirrors
# ---------------------------------------------------------------------------
HURST_MOMENTUM_FLOOR = 0.52
HURST_REVERSION_CEILING = 0.45
HURST_WINDOW = 90  # 90-bar rolling window for Hurst calculation


def calculate_hurst(price_series: pd.Series, window: int = 90) -> pd.Series:
    """
    Hurst Exponent via R/S (Rescaled Range) analysis.
    
    This is the EXACT implementation validated in research/hurst_diagnostic.py.
    Do not modify. The thresholds (0.45, 0.52) were calibrated against this method.
    
    Interpretation:
        H < 0.5  → mean reverting
        H = 0.5  → random walk (efficient market)
        H > 0.5  → trending (persistent)
    
    Args:
        price_series: Close prices (must be > 0)
        window:       Rolling window size (default 90 bars)
    
    Returns:
        pd.Series of Hurst exponents, same index as input.
        NaN for warm-up period where window is insufficient.
    """
    
    def _hurst_rs(ts: np.ndarray) -> float:
        n = len(ts)
        if n < 30:
            return 0.5  # Insufficient data — assume random walk
        
        # Log returns for stationarity
        returns = np.diff(np.log(ts))
        n_ret = len(returns)
        
        mean = np.mean(returns)
        deviation = np.cumsum(returns - mean)
        r = np.max(deviation) - np.min(deviation)
        s = np.std(returns)
        
        if s == 0 or r == 0:
            return 0.5
        
        # H = log(R/S) / log(n)
        return np.log(r / s) / np.log(n_ret)
    
    return price_series.rolling(window).apply(_hurst_rs, raw=True)


def get_current_hurst(
    price_series: pd.Series,
    window: int = HURST_WINDOW,
) -> float:
    """
    Get the most recent Hurst reading for a price series.
    
    Returns:
        Float: current Hurst exponent.
        0.5 if insufficient data.
    """
    hurst = calculate_hurst(price_series, window=window)
    valid = hurst.dropna()
    
    if valid.empty:
        return 0.5
    
    return float(valid.iloc[-1])


def classify_regime(hurst_value: float) -> str:
    """
    Classify a Hurst reading into a regime.
    
    Returns one of: 'MOMENTUM', 'REVERSION', 'DEAD_ZONE'
    """
    if hurst_value > HURST_MOMENTUM_FLOOR:
        return 'MOMENTUM'
    elif hurst_value < HURST_REVERSION_CEILING:
        return 'REVERSION'
    else:
        return 'DEAD_ZONE'


def get_eligible_universe(
    all_assets: dict,
    date: pd.Timestamp,
    strategy: str,
    lookback_days: int = 120,
    price_data: Optional[Dict[str, pd.DataFrame]] = None,
    feed=None,
) -> List[str]:
    """
    Determine which assets are eligible for a given strategy on a given date.
    
    Universe is determined by Hurst regime state at signal date ONLY.
    Historical performance never selects the universe.
    
    Args:
        all_assets:     Dict or list of ticker symbols
        date:           Signal date (pd.Timestamp or datetime)
        strategy:       'MOMENTUM' or 'REVERSION'
        lookback_days:  Days of price history to load for Hurst (default 120)
        price_data:     Optional pre-loaded dict of {ticker: DataFrame}
                        If None, will use feed.get_bars() to load
        feed:           AlpacaFeed instance (required if price_data is None)
    
    Returns:
        List of eligible ticker strings.
        Empty list if no assets qualify (this is a valid outcome).
    """
    strategy = strategy.upper()
    if strategy not in ('MOMENTUM', 'REVERSION'):
        raise ValueError(f"Strategy must be MOMENTUM or REVERSION, got {strategy}")
    
    # Handle both dict and list inputs for all_assets
    if isinstance(all_assets, dict):
        tickers = list(all_assets.keys()) if not isinstance(
            list(all_assets.values())[0], str
        ) else list(all_assets.values())
    elif isinstance(all_assets, list):
        tickers = all_assets
    else:
        raise ValueError(f"all_assets must be dict or list, got {type(all_assets)}")
    
    # Ensure date is pandas Timestamp
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)
    
    eligible = []
    diagnostics = []
    
    for ticker in tickers:
        # Get price data up to (and including) the signal date
        if price_data and ticker in price_data:
            df = price_data[ticker]
            # Truncate to signal date — NO lookahead
            if df.index.tz is not None and date.tz is None:
                date = date.tz_localize(df.index.tz)
            elif df.index.tz is None and date.tz is not None:
                date = date.tz_localize(None)
            df = df.loc[:date]
        elif feed is not None:
            from datetime import timedelta
            start = date - timedelta(days=lookback_days)
            df = feed.get_bars(ticker, start=start, end=date, timeframe='1h')
        else:
            logger.warning(
                f"No price data for {ticker} and no feed provided"
            )
            continue
        
        if df.empty or len(df) < HURST_WINDOW + 30:
            logger.debug(
                f"{ticker}: insufficient data ({len(df)} bars, "
                f"need {HURST_WINDOW + 30})"
            )
            diagnostics.append({
                'ticker': ticker,
                'hurst': np.nan,
                'regime': 'INSUFFICIENT_DATA',
                'eligible': False,
            })
            continue
        
        # Calculate current Hurst
        hurst = get_current_hurst(df['close'], window=HURST_WINDOW)
        regime = classify_regime(hurst)
        
        is_eligible = (
            (strategy == 'MOMENTUM' and regime == 'MOMENTUM') or
            (strategy == 'REVERSION' and regime == 'REVERSION')
        )
        
        diagnostics.append({
            'ticker': ticker,
            'hurst': round(hurst, 4),
            'regime': regime,
            'eligible': is_eligible,
        })
        
        if is_eligible:
            eligible.append(ticker)
            logger.info(
                f"  ✓ {ticker}: H={hurst:.4f} → {regime} "
                f"(eligible for {strategy})"
            )
        else:
            logger.debug(
                f"  ✗ {ticker}: H={hurst:.4f} → {regime} "
                f"(excluded from {strategy})"
            )
    
    logger.info(
        f"Universe for {strategy}: {len(eligible)}/{len(tickers)} "
        f"assets eligible"
    )
    
    return eligible


def full_universe_scan(
    all_assets: list,
    date: pd.Timestamp,
    price_data: Optional[Dict[str, pd.DataFrame]] = None,
    feed=None,
) -> pd.DataFrame:
    """
    Scan the full universe and classify every asset's regime.
    
    Returns a DataFrame with columns:
    - ticker, hurst, regime, momentum_eligible, reversion_eligible
    
    This is a diagnostic tool — use get_eligible_universe() for routing.
    """
    results = []
    
    for ticker in all_assets:
        # Get price data
        if price_data and ticker in price_data:
            df = price_data[ticker]
            if df.index.tz is not None and date.tz is None:
                date_adj = date.tz_localize(df.index.tz)
            elif df.index.tz is None and date.tz is not None:
                date_adj = date.tz_localize(None)
            else:
                date_adj = date
            df = df.loc[:date_adj]
        elif feed is not None:
            from datetime import timedelta
            start = date - timedelta(days=120)
            df = feed.get_bars(ticker, start=start, end=date, timeframe='1h')
        else:
            results.append({
                'ticker': ticker,
                'hurst': np.nan,
                'regime': 'NO_DATA',
                'momentum_eligible': False,
                'reversion_eligible': False,
                'bar_count': 0,
            })
            continue
        
        if df.empty or len(df) < HURST_WINDOW + 30:
            results.append({
                'ticker': ticker,
                'hurst': np.nan,
                'regime': 'INSUFFICIENT_DATA',
                'momentum_eligible': False,
                'reversion_eligible': False,
                'bar_count': len(df),
            })
            continue
        
        hurst = get_current_hurst(df['close'], window=HURST_WINDOW)
        regime = classify_regime(hurst)
        
        results.append({
            'ticker': ticker,
            'hurst': round(hurst, 4),
            'regime': regime,
            'momentum_eligible': regime == 'MOMENTUM',
            'reversion_eligible': regime == 'REVERSION',
            'bar_count': len(df),
        })
    
    return pd.DataFrame(results)


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    """Run universe scan on the full Sovereign universe."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    from sovereign.data.feeds.alpaca_feed import AlpacaFeed, SOVEREIGN_UNIVERSE
    
    feed = AlpacaFeed()
    
    print("\n" + "=" * 70)
    print("SOVEREIGN UNIVERSE SCAN")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    print(f"Hurst window: {HURST_WINDOW} bars")
    print(f"Momentum floor:   H > {HURST_MOMENTUM_FLOOR}")
    print(f"Reversion ceiling: H < {HURST_REVERSION_CEILING}")
    print(f"Dead zone: {HURST_REVERSION_CEILING} ≤ H ≤ {HURST_MOMENTUM_FLOOR}")
    print("=" * 70 + "\n")
    
    # Load all price data first
    print("Loading price data for universe...")
    price_data = feed.get_universe_bars()
    
    # Scan
    now = pd.Timestamp.now(tz='UTC')
    scan = full_universe_scan(
        all_assets=SOVEREIGN_UNIVERSE,
        date=now,
        price_data=price_data,
    )
    
    print("\n" + "=" * 70)
    print("REGIME CLASSIFICATION")
    print("=" * 70)
    print(scan.to_string(index=False))
    
    # Summary
    momentum = scan[scan['momentum_eligible']]
    reversion = scan[scan['reversion_eligible']]
    dead = scan[scan['regime'] == 'DEAD_ZONE']
    
    print(f"\nMOMENTUM eligible ({len(momentum)}): "
          f"{list(momentum['ticker'].values)}")
    print(f"REVERSION eligible ({len(reversion)}): "
          f"{list(reversion['ticker'].values)}")
    print(f"DEAD ZONE ({len(dead)}): "
          f"{list(dead['ticker'].values)}")
    
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
