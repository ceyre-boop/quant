"""Quick test of expanded asset universe"""

from dotenv import load_dotenv

load_dotenv()
from data.alpaca_client import AlpacaDataClient
import time

c = AlpacaDataClient()

print(f"Asset Universe Groups: {list(c.ASSET_UNIVERSE.keys())}")
print(f"Total unique symbols: {len(c.ALL_SYMBOLS)}")
print(f"Symbols: {c.ALL_SYMBOLS}")
print()

# Test fetch 1 year of daily data for all 57 symbols
print("Fetching 365 days of 1D data for all assets...")
results = c.get_all_assets(timeframe="1D", days=365)

print(f"\nSuccess: {len(results)}/{len(c.ALL_SYMBOLS)} assets")
print(f"Total bars: {sum(len(df) for df in results.values())}")

# Show sample
for sym in ["SPY", "QQQ", "NVDA", "XLE"]:
    if sym in results:
        df = results[sym]
        print(f"{sym}: {len(df)} bars | {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
