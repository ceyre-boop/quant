"""
Alpaca Data Client (Warp Drive 2.0)
High-performance batch fetching and parallel IO.
"""
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

logger = logging.getLogger(__name__)

class AlpacaDataClient:
    """Optimized Alpaca Data Client with Bulk-Fetch Capabilities."""
    
    ASSET_UNIVERSE = {
        "etf_core": ["SPY", "QQQ", "IWM", "DIA", "LQD", "HYG", "GLD", "SLV", "USO"],
        "ai_semis": ["NVDA", "AMD", "SMCI", "ARM", "PLTR", "MSFT", "GOOGL", "META"],
        "mega_cap": ["AAPL", "AMZN", "TSLA"],
        "trinity": ["META", "PFE", "UNH"]
    }
    
    ALL_SYMBOLS = list(dict.fromkeys([s for group in ASSET_UNIVERSE.values() for s in group]))
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials required.")
            
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.cache_dir = Path("sovereign/cache/ohlcv")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_historical_bars(self, symbol_or_symbols, timeframe: str = '1Day', days: int = 365) -> pd.DataFrame:
        """Fetch bars using Alpaca's bulk capability or single asset logic."""
        
        # 1. Date Calibration
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        
        # 2. Timeframe Mapping
        tf_map = {
            '1Day': TimeFrame.Day,
            '1Hour': TimeFrame.Hour,
            '15Min': TimeFrame(15, TimeFrame.Minute)
        }
        tf = tf_map.get(timeframe, TimeFrame.Day)
        
        # 3. Request Generation
        request = StockBarsRequest(
            symbol_or_symbols=symbol_or_symbols,
            timeframe=tf,
            start=start,
            end=end,
            feed=DataFeed.IEX # Free tier
        )
        
        try:
            logger.info(f"[WarpDrive] Fetching {symbol_or_symbols} ({timeframe})...")
            bars = self.client.get_stock_bars(request)
            
            if not bars or not hasattr(bars, 'df') or bars.df.empty:
                return pd.DataFrame()
                
            # Alpaca returns MultiIndex (symbol, timestamp). We want friendly format.
            df = bars.df.reset_index()
            df = df.rename(columns={'timestamp': 'date'})
            return df
        except Exception as e:
            logger.error(f"Alpaca Fetch Error: {e}")
            return pd.DataFrame()

    def get_all_assets(self, timeframe: str = '1Day', days: int = 365) -> Dict[str, pd.DataFrame]:
        """The 'Harvest' — Multi-threaded Asset Acquisition."""
        symbols = self.ALL_SYMBOLS
        logger.info(f"Warp Drive: Harvesting {len(symbols)} assets in parallel groups...")
        
        # Split symbols into batches to avoid extremely large response overhead
        batch_size = 20
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        final_data = {}
        
        def fetch_batch(batch):
            df = self.get_historical_bars(batch, timeframe, days)
            if df.empty: return {}
            return {s: df[df['symbol'] == s].set_index('date') for s in batch if s in df['symbol'].unique()}

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_batch = {executor.submit(fetch_batch, b): b for b in batches}
            for future in as_completed(future_to_batch):
                batch_dict = future.result()
                final_data.update(batch_dict)
                
        logger.info(f"Harvest Complete: Captured {len(final_data)} assets.")
        return final_data

    def get_latest_price(self, symbol: str) -> Optional[float]:
        df = self.get_historical_bars(symbol, timeframe='1Day', days=3)
        if not df.empty:
            return float(df['close'].iloc[-1])
        return None
