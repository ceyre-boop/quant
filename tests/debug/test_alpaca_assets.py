"""Test fetching 3 months of data for all major assets"""

from dotenv import load_dotenv

load_dotenv()
from data.alpaca_client import AlpacaDataClient
import time

c = AlpacaDataClient()
assets = c.MAJOR_ASSETS
print(f"Testing 3 months of data for {len(assets)} major assets...")
print("=" * 60)

results = {}
errors = []
for sym in assets:
    try:
        df = c.get_historical_bars(sym, "1D", days=90)
        if not df.empty:
            results[sym] = len(df)
            start_date = df.index[0].strftime("%Y-%m-%d")
            end_date = df.index[-1].strftime("%Y-%m-%d")
            print(f"{sym}: {len(df)} bars | {start_date} to {end_date}")
        else:
            print(f"{sym}: NO DATA")
            errors.append(sym)
        time.sleep(0.2)
    except Exception as e:
        print(f"{sym}: ERROR - {e}")
        errors.append(sym)

print("=" * 60)
print(f"Success: {len(results)}/{len(assets)} assets")
print(f"Total bars: {sum(results.values())}")
if errors:
    print(f"Errors: {errors}")
