from dotenv import load_dotenv
import os
load_dotenv()

from datetime import datetime, timedelta, timezone
from data.alpaca_client import AlpacaDataClient
import pandas as pd

def diagnose():
    alpaca = AlpacaDataClient()
    symbol = 'SPY'
    start = datetime(2025, 4, 1, tzinfo=timezone.utc)
    end = datetime(2025, 4, 8, tzinfo=timezone.utc)
    
    print(f"Fetching {symbol} from {start} to {end}...")
    df = alpaca.get_historical_bars(symbol, '1H', start=start, end=end)
    
    if df is not None:
        print(f"Success! Rows: {len(df)}")
        print(df.head())
    else:
        print("Failure: Received None")

if __name__ == "__main__":
    diagnose()
