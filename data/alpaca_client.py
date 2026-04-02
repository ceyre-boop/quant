"""
Alpaca Data Client for CLAWD Quant Trading
Supports stocks, ETFs, and futures proxies (ES=F, NQ=F)
Timeframes: 1H, 1D, 4H (via resampling)
Historical range: 3 months to 5 years
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


class AlpacaDataClient:
    """
    Alpaca Markets data client for historical and live data.
    
    Note: Alpaca doesn't directly support futures (ES1!, NQ).
    Use ES=F and NQ=F (Yahoo Finance symbols) as proxies via yfinance
    for futures data, or use this client for stock/ETF data.
    """
    
    # Symbol mappings for common futures proxies
    FUTURES_PROXIES = {
        'ES1!': 'SPY',      # E-mini S&P 500 -> SPY ETF
        'NQ': 'QQQ',        # E-mini Nasdaq -> QQQ ETF
        'YM': 'DIA',        # Dow Futures -> DIA ETF
        'RTY': 'IWM',       # Russell 2000 -> IWM ETF
        'CL': 'USO',        # Crude Oil -> USO ETF
        'GC': 'GLD',        # Gold -> GLD ETF
        'ZN': 'TLT',        # 10Y Treasuries -> TLT ETF
    }
    
    # Expanded asset universe organized by sector/theme
    ASSET_UNIVERSE = {
        "etf_core": [
            "SPY", "QQQ", "IWM", "DIA", "VTI",
            "TLT", "LQD", "HYG", "VGIT",
            "VIXY", "UVXY", "SQQQ", "SPXL",
            "GLD", "SLV", "USO", "UNG"
        ],
        "ai_semis": [
            "NVDA", "AMD", "SMCI", "ARM", "PLTR", "SOUN", "MSFT", "GOOGL", "META"
        ],
        "mega_cap": [
            "AAPL", "AMZN", "TSLA", "MSFT", "GOOGL", "META"
        ],
        "healthcare": [
            "UNH", "JNJ", "PFE", "XBI", "IBB"
        ],
        "energy": [
            "XLE", "XOM", "CVX", "OXY"
        ],
        "financials": [
            "XLF", "JPM", "BAC", "GS"
        ],
        "real_estate": [
            "VNQ", "SPG"
        ],
        "volatility_inverse": [
            "VXX", "UVXY", "SQQQ", "SPXU"
        ],
    }
    
    # Flat list for batch fetch - deduplicated
    ALL_SYMBOLS = list(dict.fromkeys(
        [s for group in ASSET_UNIVERSE.values() for s in group]
    ))
    
    # Legacy compatibility
    MAJOR_ASSETS = ALL_SYMBOLS
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, paper: bool = True):
        """
        Initialize Alpaca data client.
        
        Args:
            api_key: Alpaca API key (or from ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or from ALPACA_SECRET_KEY env var)
            paper: Use paper trading endpoints (default True)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper = paper
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars "
                "or pass to constructor."
            )
        
        # Initialize client (data works with both paper and live keys)
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = '1H',
        days: int = 90,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars for a symbol.
        
        Args:
            symbol: Stock/ETF symbol (e.g., 'SPY', 'QQQ')
            timeframe: '1H', '1D', '4H' (4H via resampling)
            days: Number of days to fetch (if start not specified)
            start: Start datetime (optional)
            end: End datetime (optional, defaults to now)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Map futures symbols to ETF proxies
        if symbol in self.FUTURES_PROXIES:
            print(f"[Alpaca] Mapping {symbol} -> {self.FUTURES_PROXIES[symbol]} (futures proxy)")
            symbol = self.FUTURES_PROXIES[symbol]
        
        # Calculate date range
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=days)
            
        # Ensure we don't request more than 5 years
        max_start = end - timedelta(days=365*5)
        if start < max_start:
            print(f"[Alpaca] Limiting to 5 years of data")
            start = max_start
            
        # Map timeframe
        if timeframe == '1H':
            tf = TimeFrame.Hour
        elif timeframe == '1D':
            tf = TimeFrame.Day
        elif timeframe == '15Min':
            tf = TimeFrame(15, TimeFrame.Minute)
        elif timeframe == '30Min':
            tf = TimeFrame(30, TimeFrame.Minute)
        else:
            tf = TimeFrame.Hour
            
        # Request data
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            feed=DataFeed.IEX  # Use IEX for free tier
        )
        
        try:
            bars = self.client.get_stock_bars(request)
            
            if not bars.data or symbol not in bars.data:
                print(f"[Alpaca] No data returned for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            data = []
            for bar in bars.data[symbol]:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                })
                
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Resample to 4H if requested
            if timeframe == '4H':
                df = self._resample_to_4h(df)
                
            print(f"[Alpaca] Fetched {len(df)} bars for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            print(f"[Alpaca] Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1H data to 4H bars"""
        return df.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    def get_multiple_assets(
        self,
        symbols: List[str],
        timeframe: str = '1H',
        days: int = 90
    ) -> dict:
        """
        Get data for multiple assets.
        
        Args:
            symbols: List of symbols
            timeframe: '1H', '4H', '1D'
            days: Number of days to fetch
            
        Returns:
            Dict mapping symbol -> DataFrame
        """
        results = {}
        for symbol in symbols:
            df = self.get_historical_bars(symbol, timeframe, days)
            if not df.empty:
                results[symbol] = df
        return results
    
    def get_major_assets(self, timeframe: str = '1H', days: int = 90) -> dict:
        """Fetch all major assets"""
        return self.get_multiple_assets(self.MAJOR_ASSETS, timeframe, days)
    
    def get_all_assets(self, timeframe: str = '1D', days: int = 365) -> dict:
        """Fetch the full expanded asset universe (57 unique symbols)"""
        print(f"[Alpaca] Fetching {len(self.ALL_SYMBOLS)} assets ({timeframe}, {days} days)...")
        return self.get_multiple_assets(self.ALL_SYMBOLS, timeframe, days)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        df = self.get_historical_bars(symbol, timeframe='1H', days=1)
        if not df.empty:
            return df['close'].iloc[-1]
        return None
    
    def get_quote(self, symbol: str) -> dict:
        """Get current quote (requires different endpoint)"""
        # Note: Real-time quotes require Market Data API subscription
        # This returns last bar close as approximation
        price = self.get_latest_price(symbol)
        return {
            'symbol': symbol,
            'price': price,
            'bid': price,  # Approximation
            'ask': price,  # Approximation
        }


def test_client():
    """Test the Alpaca client"""
    from dotenv import load_dotenv
    load_dotenv()
    
    client = AlpacaDataClient()
    
    # Test single symbol
    print("\n=== Test 1: SPY 1H (5 days) ===")
    spy = client.get_historical_bars('SPY', timeframe='1H', days=5)
    print(spy.head())
    print(f"Total bars: {len(spy)}")
    
    # Test futures proxy
    print("\n=== Test 2: ES1! (maps to SPY) ===")
    es = client.get_historical_bars('ES1!', timeframe='1H', days=5)
    print(f"Total bars: {len(es)}")
    
    # Test 4H resampling
    print("\n=== Test 3: QQQ 4H (30 days) ===")
    qqq = client.get_historical_bars('QQQ', timeframe='4H', days=30)
    print(qqq.head())
    print(f"Total bars: {len(qqq)}")
    
    # Test multiple assets
    print("\n=== Test 4: Multiple assets ===")
    data = client.get_multiple_assets(['SPY', 'QQQ', 'IWM'], timeframe='1D', days=30)
    for sym, df in data.items():
        print(f"{sym}: {len(df)} bars")
    
    # Test expanded universe (57 symbols)
    print(f"\n=== Test 5: Full Asset Universe ({len(client.ALL_SYMBOLS)} symbols, 90 days) ===")
    print("Groups:", list(client.ASSET_UNIVERSE.keys()))
    all_data = client.get_all_assets(timeframe='1D', days=90)
    print(f"Successfully fetched: {len(all_data)}/{len(client.ALL_SYMBOLS)} assets")
    print(f"Total bars: {sum(len(df) for df in all_data.values())}")


if __name__ == '__main__':
    test_client()
