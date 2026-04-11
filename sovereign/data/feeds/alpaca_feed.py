"""
Sovereign Trading Intelligence — Alpaca OHLCV Feed
Phase 1: Data Foundation

1-hour OHLCV bars for the full 15-asset universe.
Date range: 2021-01-01 to present.
No transformations. Raw OHLCV only.
Transformations happen in features/ — never here.

Design decisions (from research history):
- T+1 open anchor means we need open prices that are NEVER adjusted after the fact
- IEX feed has known gaps during low-liquidity hours — use SIP where available
- Parquet cache prevents redundant API calls and ensures reproducibility
- Each asset cached independently — partial failures don't corrupt good data
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The 15-asset universe from SOVEREIGN_CONFIG
# ---------------------------------------------------------------------------
SOVEREIGN_UNIVERSE = [
    'SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT',
    'AMZN', 'TSLA', 'META', 'GOOGL', 'AMD',
    'TLT', 'GLD', 'HYG', 'IWM', 'XLF'
]

# Default date range — 2021-01-01 to present
DEFAULT_START = datetime(2021, 1, 1, tzinfo=timezone.utc)


class AlpacaFeed:
    """
    Clean Alpaca OHLCV feed for Sovereign.
    
    Returns raw OHLCV DataFrames with DatetimeIndex (UTC).
    Columns: open, high, low, close, volume
    No derived columns. No indicators. No transformations.
    """
    
    # Rate limit: 200 requests/minute on free tier
    REQUEST_DELAY_SECONDS = 0.35
    MAX_RETRIES = 3
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        load_dotenv()
        
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca credentials required. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            )
        
        # Lazy import — don't fail at module level if alpaca-py missing
        from alpaca.data.historical import StockHistoricalDataClient
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # Cache directory for parquet files
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'cache', 'ohlcv'
            )
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"AlpacaFeed initialized. Cache: {self.cache_dir}")
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def get_bars(
        self,
        ticker: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = '1h',
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get OHLCV bars for a single ticker.
        
        Args:
            ticker:    Stock/ETF symbol (e.g. 'SPY')
            start:     Start datetime (default: 2021-01-01 UTC)
            end:       End datetime (default: now UTC)
            timeframe: '1h' or '1d' (default '1h')
            use_cache: Whether to use/update parquet cache
            
        Returns:
            DataFrame with columns [open, high, low, close, volume]
            DatetimeIndex in UTC. Sorted ascending.
            Empty DataFrame if no data available.
        """
        ticker = ticker.upper().strip()
        start = start or DEFAULT_START
        end = end or datetime.now(timezone.utc)
        
        # Ensure timezone awareness
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        
        if use_cache:
            cached = self._load_cache(ticker, timeframe)
            if cached is not None and not cached.empty:
                # Check if cache covers the requested range
                cache_start = cached.index.min()
                cache_end = cached.index.max()
                
                if cache_start <= start and cache_end >= end - timedelta(hours=2):
                    # Cache covers the range — slice and return
                    result = cached.loc[start:end].copy()
                    logger.info(
                        f"[CACHE HIT] {ticker} {timeframe}: "
                        f"{len(result)} bars from cache"
                    )
                    return result
                
                # Cache exists but doesn't cover full range — fetch gap
                if cache_end < end - timedelta(hours=2):
                    # Fetch from cache_end forward to avoid re-fetching
                    gap_start = cache_end + timedelta(hours=1)
                    logger.info(
                        f"[CACHE UPDATE] {ticker}: fetching "
                        f"{gap_start.date()} to {end.date()}"
                    )
                    new_data = self._fetch_bars_from_api(
                        ticker, gap_start, end, timeframe
                    )
                    if new_data is not None and not new_data.empty:
                        combined = pd.concat([cached, new_data])
                        combined = combined[
                            ~combined.index.duplicated(keep='last')
                        ].sort_index()
                        self._save_cache(combined, ticker, timeframe)
                        return combined.loc[start:end].copy()
                    
                    # Gap fetch failed — return what we have
                    return cached.loc[start:end].copy()
        
        # No cache or cache disabled — full fetch
        df = self._fetch_bars_from_api(ticker, start, end, timeframe)
        
        if df is not None and not df.empty and use_cache:
            self._save_cache(df, ticker, timeframe)
        
        return df if df is not None else pd.DataFrame()
    
    def get_latest_bar(self, ticker: str) -> Optional[dict]:
        """
        Get the most recent 1h bar for a ticker.
        
        Returns:
            Dict with keys: open, high, low, close, volume, timestamp
            None if no data available.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=3)  # Buffer for weekends/holidays
        
        df = self.get_bars(ticker, start=start, end=end, timeframe='1h')
        
        if df.empty:
            return None
        
        last = df.iloc[-1]
        return {
            'open':      float(last['open']),
            'high':      float(last['high']),
            'low':       float(last['low']),
            'close':     float(last['close']),
            'volume':    int(last['volume']),
            'timestamp': df.index[-1],
        }
    
    def get_universe_bars(
        self,
        tickers: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = '1h',
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch bars for the full universe (or a subset).
        
        Returns:
            Dict mapping ticker -> DataFrame
            Tickers with no data are excluded.
        """
        tickers = tickers or SOVEREIGN_UNIVERSE
        results = {}
        
        for i, ticker in enumerate(tickers):
            logger.info(
                f"[{i+1}/{len(tickers)}] Fetching {ticker}..."
            )
            try:
                df = self.get_bars(ticker, start=start, end=end, timeframe=timeframe)
                if not df.empty:
                    results[ticker] = df
                    logger.info(f"  {ticker}: {len(df)} bars")
                else:
                    logger.warning(f"  {ticker}: NO DATA")
            except Exception as e:
                logger.error(f"  {ticker}: FAILED — {e}")
        
        return results
    
    def get_bar_counts(
        self,
        tickers: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        timeframe: str = '1h',
    ) -> pd.DataFrame:
        """
        Report bar counts per asset. Used to verify training
        universe has sufficient data (minimum 8,000 bars for Fold 1).
        """
        tickers = tickers or SOVEREIGN_UNIVERSE
        start = start or DEFAULT_START
        
        counts = []
        for ticker in tickers:
            df = self.get_bars(ticker, start=start, timeframe=timeframe)
            bar_count = len(df)
            first_date = df.index.min() if not df.empty else None
            last_date = df.index.max() if not df.empty else None
            sufficient = bar_count >= 8000
            
            counts.append({
                'ticker':     ticker,
                'bar_count':  bar_count,
                'first_bar':  first_date,
                'last_bar':   last_date,
                'sufficient': sufficient,
                'status':     'PASS' if sufficient else 'REVIEW',
            })
        
        return pd.DataFrame(counts)
    
    # ------------------------------------------------------------------
    # Internal: API fetch
    # ------------------------------------------------------------------
    
    def _fetch_bars_from_api(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch bars from Alpaca API with retry logic.
        
        Alpaca free tier limits requests to 200/minute.
        We chunk large date ranges into 6-month windows to avoid
        response size limits and timeout issues.
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        
        tf = TimeFrame.Hour if timeframe == '1h' else TimeFrame.Day
        
        # Chunk into 6-month windows for large ranges
        chunks = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=180), end)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(hours=1)
        
        all_data = []
        
        for i, (c_start, c_end) in enumerate(chunks):
            for attempt in range(self.MAX_RETRIES):
                try:
                    time.sleep(self.REQUEST_DELAY_SECONDS)
                    
                    request = StockBarsRequest(
                        symbol_or_symbols=ticker,
                        timeframe=tf,
                        start=c_start,
                        end=c_end,
                        feed=DataFeed.IEX,
                    )
                    
                    bars = self.client.get_stock_bars(request)
                    
                    if not bars or not bars.data or ticker not in bars.data:
                        logger.debug(
                            f"  {ticker} chunk {i+1}/{len(chunks)}: "
                            f"no data for {c_start.date()} to {c_end.date()}"
                        )
                        break
                    
                    rows = []
                    for bar in bars.data[ticker]:
                        rows.append({
                            'timestamp': bar.timestamp,
                            'open':      float(bar.open),
                            'high':      float(bar.high),
                            'low':       float(bar.low),
                            'close':     float(bar.close),
                            'volume':    int(bar.volume),
                        })
                    
                    if rows:
                        chunk_df = pd.DataFrame(rows)
                        chunk_df.set_index('timestamp', inplace=True)
                        all_data.append(chunk_df)
                        logger.debug(
                            f"  {ticker} chunk {i+1}/{len(chunks)}: "
                            f"{len(rows)} bars"
                        )
                    break  # Success — exit retry loop
                    
                except Exception as e:
                    if '429' in str(e) and attempt < self.MAX_RETRIES - 1:
                        wait = (attempt + 1) * 3
                        logger.warning(
                            f"  {ticker}: rate limited, "
                            f"waiting {wait}s (attempt {attempt+1})"
                        )
                        time.sleep(wait)
                    elif attempt == self.MAX_RETRIES - 1:
                        logger.error(
                            f"  {ticker}: failed after {self.MAX_RETRIES} "
                            f"attempts — {e}"
                        )
                        return None
                    else:
                        logger.warning(f"  {ticker}: {e}, retrying...")
                        time.sleep(1)
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.concat(all_data).sort_index()
        df = df[~df.index.duplicated(keep='last')]
        
        # Sanity checks on raw data
        self._validate_raw_bars(df, ticker)
        
        return df
    
    # ------------------------------------------------------------------
    # Internal: Cache
    # ------------------------------------------------------------------
    
    def _cache_path(self, ticker: str, timeframe: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker}_{timeframe}.parquet")
    
    def _load_cache(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(ticker, timeframe)
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                logger.debug(f"  Cache loaded: {ticker} ({len(df)} bars)")
                return df
            except Exception as e:
                logger.warning(f"  Cache corrupt for {ticker}: {e}")
                os.remove(path)
        return None
    
    def _save_cache(self, df: pd.DataFrame, ticker: str, timeframe: str):
        path = self._cache_path(ticker, timeframe)
        try:
            df.to_parquet(path)
            logger.debug(f"  Cache saved: {ticker} ({len(df)} bars)")
        except Exception as e:
            logger.warning(f"  Cache save failed for {ticker}: {e}")
    
    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------
    
    def _validate_raw_bars(self, df: pd.DataFrame, ticker: str):
        """
        Basic sanity checks on raw OHLCV data.
        These catch API issues, NOT lookahead — that's feed_validator's job.
        """
        if df.empty:
            return
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            neg = (df[col] <= 0).sum()
            if neg > 0:
                logger.warning(f"  {ticker}: {neg} non-positive values in {col}")
        
        # Check for high < low (impossible candles)
        impossible = (df['high'] < df['low']).sum()
        if impossible > 0:
            logger.warning(f"  {ticker}: {impossible} impossible candles (high < low)")
        
        # Check for zero volume bars (suspicious but not necessarily wrong)
        zero_vol = (df['volume'] == 0).sum()
        if zero_vol > 0:
            pct = zero_vol / len(df) * 100
            if pct > 5:
                logger.warning(
                    f"  {ticker}: {zero_vol} zero-volume bars ({pct:.1f}%)"
                )
        
        # Check for duplicate timestamps
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            logger.warning(f"  {ticker}: {dupes} duplicate timestamps")
        
        # Check OHLC relationship: open and close should be between low and high
        ohlc_violations = (
            (df['open'] > df['high']) | (df['open'] < df['low']) |
            (df['close'] > df['high']) | (df['close'] < df['low'])
        ).sum()
        if ohlc_violations > 0:
            logger.warning(
                f"  {ticker}: {ohlc_violations} OHLC relationship violations"
            )


# ======================================================================
# CLI entry point — fetches the full universe and reports bar counts
# ======================================================================

def main():
    """Fetch full universe and report bar counts."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    feed = AlpacaFeed()
    
    print("\n" + "=" * 70)
    print("SOVEREIGN DATA FOUNDATION — ALPACA FEED")
    print(f"Universe: {len(SOVEREIGN_UNIVERSE)} assets")
    print(f"Range: 2021-01-01 to present")
    print(f"Timeframe: 1h")
    print("=" * 70 + "\n")
    
    # Fetch all universe data
    universe_data = feed.get_universe_bars()
    
    # Report bar counts
    counts = feed.get_bar_counts()
    print("\n" + "=" * 70)
    print("BAR COUNT REPORT")
    print("=" * 70)
    print(counts.to_string(index=False))
    
    insufficient = counts[~counts['sufficient']]
    if not insufficient.empty:
        print(f"\n[!] {len(insufficient)} asset(s) below 8,000 bar minimum:")
        for _, row in insufficient.iterrows():
            print(f"  {row['ticker']}: {row['bar_count']} bars -- FLAGGED FOR REVIEW")
    else:
        print("\n[OK] All assets meet 8,000 bar minimum for Fold 1 training")
    
    print("=" * 70 + "\n")
    
    return counts


if __name__ == '__main__':
    main()
