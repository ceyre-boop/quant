"""
Run ClawdBrain - PRACTICAL MODE (bypasses Chi2 for initial trading)

This runs with Chi2 disabled initially to build trade history,
then re-enables it once we have statistical validation.

Usage:
    python run_clawd_practical.py [--live] [--symbols SPY,QQQ]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clawd_brain import ClawdBrain
from ai_trading_bridge import AITradingBridge


def main():
    parser = argparse.ArgumentParser(description="ClawdBrain Practical Trading")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,IWM,NVDA,AAPL,TSLA",
        help="Symbols to trade",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.52,
        help="Confidence threshold (default: 0.52, lower = more trades)",
    )
    parser.add_argument(
        "--use-chi2",
        action="store_true",
        help="Enable Chi-Squared gate (default: OFF for initial trading)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" CLAWD BRAIN - PRACTICAL TRADING MODE")
    print("=" * 70)
    print(f" Mode: {args.mode.upper()}")
    print(f" Symbols: {args.symbols}")
    print(f" Confidence: {args.confidence}")
    print(f" Chi-Squared Gate: {'ON' if args.use_chi2 else 'OFF (building history)'}")
    print("=" * 70)
    print()

    if args.mode == "live":
        print("⚠️  LIVE TRADING MODE")
        confirm = input("Type 'LIVE' to trade with real money: ")
        if confirm != "LIVE":
            print("Aborted.")
            return

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    print("[Setup] Initializing ClawdBrain...")
    brain = ClawdBrain(
        kelly_fraction=0.25,
        max_position_pct=0.10,
        min_confidence=args.confidence,
        paper=(args.mode == "paper"),
        use_chi2=args.use_chi2,  # Disabled by default for initial trading
    )

    print("[Setup] Connecting to Alpaca...")
    bridge = AITradingBridge(
        brain=brain,
        symbols=symbols,
        timeframe="1D",
        lookback_days=60,
        paper=(args.mode == "paper"),
        min_confidence=args.confidence,
    )

    print("[Setup] Ready!")
    print()

    print("[RUNNING] Trading cycle...")
    result = bridge.run_cycle()

    print("\n" + "=" * 70)
    print(" RESULT")
    print("=" * 70)
    print(f" Symbols: {result['symbols_data']}")
    print(f" Signals: {result['signals']}")
    print(f" Trades Executed: {result['executed']}")

    if result["executed"] > 0:
        print("\n[✓] TRADES EXECUTED!")
        print(f"    Check your Alpaca dashboard for orders.")
        if not args.use_chi2:
            print(f"\n    Tip: Run with --use-chi2 after 30+ trades for validation.")
    else:
        print("\n[○] No trades this cycle.")
        print("    (Model confidence too low or gates blocked)")

    print("=" * 70)


if __name__ == "__main__":
    main()
