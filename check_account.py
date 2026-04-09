# Check Alpaca positions and orders
import os
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient

client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

print("=" * 60)
print("ALPACA PAPER ACCOUNT STATUS")
print("=" * 60)

account = client.get_account()
print(f"Account ID: {account.id}")
print(f"Status: {account.status}")
print(f"Equity: ${float(account.equity):,.2f}")
print(f"Buying Power: ${float(account.buying_power):,.2f}")
print(f"Cash: ${float(account.cash):,.2f}")
print(f"Daytrade Count: {account.daytrade_count}")
print()

print("=" * 60)
print("OPEN POSITIONS")
print("=" * 60)
positions = client.get_all_positions()
if positions:
    for p in positions:
        print(f"{p.symbol}: {p.qty} shares @ ${float(p.avg_entry_price):.2f}")
        print(f"  Market: ${float(p.market_value):,.2f} | P&L: ${float(p.unrealized_pl):,.2f}")
else:
    print("No open positions")
print()

print("=" * 60)
print("RECENT ORDERS")
print("=" * 60)
from alpaca.trading.requests import GetOrdersRequest

orders_request = GetOrdersRequest(status="all", limit=5)
orders = client.get_orders(filter=orders_request)
for o in orders:
    print(f"{o.symbol} | {o.side.value} {o.qty} | {o.status.value}")
    print(f"  Order ID: {o.id}")
    print(f"  Submitted: {o.submitted_at}")
    print()
