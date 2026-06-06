#!/usr/bin/env python3
"""
IB Gateway connection test. Run this before Monday to confirm everything works.

Requirements:
  - IB Gateway running (or TWS)
  - API connections enabled: Edit > Global Configuration > API > Settings
  - Socket port 4002 (paper) checked

Usage:
    python3.13 scripts/futures_connect_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.futures.ib_bridge import IBBridge

def main():
    print("Testing IB Gateway connection...")
    print("(Make sure IB Gateway is running and API is enabled)\n")

    try:
        with IBBridge() as bridge:
            print("\n── Account ─────────────────────────────")
            summary = bridge.account_summary()
            for k, v in summary.items():
                print(f"  {k}: ${v:,.2f}")

            print("\n── Open Positions ──────────────────────")
            pos = bridge.positions()
            if pos:
                for p in pos:
                    print(f"  {p['symbol']} {p['expiry']}: {p['position']} @ {p['avg_cost']:.2f}")
            else:
                print("  (none)")

            print("\n── MES Front Month ─────────────────────")
            try:
                mes = bridge.mes_contract()
                price = bridge.last_price(mes)
                print(f"  {mes.localSymbol} expires {mes.lastTradeDateOrContractMonth}")
                print(f"  Last price: {price}")
            except Exception as e:
                print(f"  MES lookup failed (market may be closed): {e}")

            print("\n── MNQ Front Month ─────────────────────")
            try:
                mnq = bridge.mnq_contract()
                price = bridge.last_price(mnq)
                print(f"  {mnq.localSymbol} expires {mnq.lastTradeDateOrContractMonth}")
                print(f"  Last price: {price}")
            except Exception as e:
                print(f"  MNQ lookup failed (market may be closed): {e}")

            print("\n✓ Connection test passed. Ready for Monday.")

    except ConnectionRefusedError:
        print("\n✗ Connection refused.")
        print("  → Is IB Gateway running?")
        print("  → Is API enabled? (Edit > Global Configuration > API > Settings)")
        print("  → Is port 4002 correct? (check IB_PORT in .env)")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Connection failed: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
