"""
scripts/build_market_memory.py
===============================
One-time script: populate the trading memory database from yfinance.

Run this once to initialise memory from 20 historical events.
After that, the orchestrator's on_trade_close() auto-learns new events.

Usage:
    python3 scripts/build_market_memory.py
    python3 scripts/build_market_memory.py --rebuild   # force re-fetch all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true',
                        help='Force re-fetch all patterns even if already stored')
    args = parser.parse_args()

    from sovereign.risk.market_memory import MarketMemory, HISTORICAL_EVENTS
    mem = MarketMemory()

    print(f"Trading Memory — current state: {mem.describe()}")
    print(f"Events to process: {len(HISTORICAL_EVENTS)}")
    print()

    built = mem.build_from_history(force_rebuild=args.rebuild)

    print()
    print(f"Build complete: {built} new patterns added")
    print(f"Total patterns in memory: {mem.n_patterns}")
    print()
    print("Stored events:")
    for name in sorted(mem.pattern_names):
        print(f"  {name}")

    # Quick sanity test: extract current features using recent SPY and compare
    print()
    print("Running self-test comparison (requires yfinance)...")
    try:
        import yfinance as yf
        import numpy as np
        spy = yf.download('SPY', period='2y', progress=False, auto_adjust=True)
        vix = yf.download('^VIX', period='2y', progress=False, auto_adjust=True)

        if len(spy) >= 200:
            spy_arr = spy['Close'].values.astype(float).squeeze()
            vix_arr = vix['Close'].values.astype(float).squeeze() if len(vix) >= 30 else None
            result = mem.compare(spy_arr, vix_arr)

            print(f"Current threat level: {result.threat_level}")
            print(f"Composite score:      {result.threat_score:.4f}")
            print(f"Size modifier:        {result.size_modifier:.2f}×")
            print()
            print("Top matches:")
            for m in result.top_matches:
                bar = '█' * int(m.similarity * 20)
                print(f"  {m.event_name:<30} {bar:<20} {m.similarity:.3f}")
        else:
            print("Not enough SPY data for comparison")
    except ImportError:
        print("yfinance not available — install with: pip3 install yfinance")
    except Exception as e:
        print(f"Comparison test error: {e}")

if __name__ == '__main__':
    main()
