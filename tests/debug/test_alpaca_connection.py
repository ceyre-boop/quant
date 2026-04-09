# Test Alpaca connection
import os
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

print("Testing Alpaca connection...")
print(f"Key: {os.getenv('ALPACA_API_KEY')[:20]}...")
print(f"URL: {os.getenv('ALPACA_BASE_URL')}")
print()

# Test data client
from data.alpaca_client import AlpacaDataClient

client = AlpacaDataClient()

df = client.get_historical_bars("SPY", timeframe="1D", days=5)
print(f"SPY data: {len(df)} bars")
print(df.tail(2))
print()

# Test trading client
from alpaca.trading.client import TradingClient

trading = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)
account = trading.get_account()
print(f"Account ID: {account.id}")
print(f"Equity: ${account.equity}")
print(f"Buying Power: ${account.buying_power}")
print(f"Status: {account.status}")
print()
print("SUCCESS: Alpaca paper trading connected!")
