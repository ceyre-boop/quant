#!/usr/bin/env python3
"""
FVG Express — rapid G2b execution validation trades

Scans all forex pairs for unmitigated Fair Value Gaps (no full ICT setup required)
and places a small paper trade through OANDA for each one found. Purpose: accumulate
G2b execution validation trades quickly without waiting for Grade A signals.

Risk per trade: 0.25% of account (vs 0.75% for live Grade A trades)
TP target: 2R (stop * 2)
Max trades: configurable (default 5)

Usage:
    python3 scripts/run_fvg_express.py
    python3 scripts/run_fvg_express.py --max-trades 3 --dry-run

WARNING: Places real OANDA practice account trades. Requires OANDA_API_KEY +
OANDA_ACCOUNT_ID in .env.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

PAIRS_YF = ["GBPUSD=X", "EURUSD=X", "AUDUSD=X", "AUDNZD=X", "USDJPY=X"]
FVG_LOOKBACK = 40       # bars to scan for FVGs
FVG_MIN_BODY = 0.0001   # minimum gap size (in price) to qualify
RISK_PCT = 0.0025       # 0.25% per express trade
ACCOUNT_SIZE = 100_000
TP_R = 2.0              # take-profit in R multiples


def find_fvg(df: pd.DataFrame, pair: str) -> Optional[dict]:
    """Scan last FVG_LOOKBACK bars for the most recent unmitigated 3-bar FVG."""
    if len(df) < FVG_LOOKBACK + 2:
        return None

    recent = df.tail(FVG_LOOKBACK + 2).reset_index()
    best = None

    for i in range(1, len(recent) - 1):
        prev_bar = recent.iloc[i - 1]
        next_bar = recent.iloc[i + 1]
        cur_bar  = recent.iloc[i]

        # Bullish FVG: gap between prev high and next low (price shot up through gap)
        if next_bar["Low"] > prev_bar["High"] + FVG_MIN_BODY:
            gap_high = float(next_bar["Low"])
            gap_low  = float(prev_bar["High"])
            midpoint = (gap_high + gap_low) / 2
            # Check if price has since returned to mitigate (touched) the FVG
            post = recent.iloc[i + 1:]
            mitigated = any(row["Low"] <= gap_high for _, row in post.iterrows())
            if not mitigated:
                best = {
                    "direction": "LONG",
                    "entry": midpoint,
                    "stop": gap_low - (gap_high - gap_low) * 0.1,  # slight buffer below gap
                    "tp1": midpoint + (midpoint - (gap_low - (gap_high - gap_low) * 0.1)) * TP_R,
                    "gap_high": gap_high,
                    "gap_low": gap_low,
                    "bar_idx": i,
                }
                # Take the most recent unmitigated FVG
        # Bearish FVG: gap between prev low and next high (price shot down through gap)
        elif prev_bar["Low"] > next_bar["High"] + FVG_MIN_BODY:
            gap_high = float(prev_bar["Low"])
            gap_low  = float(next_bar["High"])
            midpoint = (gap_high + gap_low) / 2
            post = recent.iloc[i + 1:]
            mitigated = any(row["High"] >= gap_low for _, row in post.iterrows())
            if not mitigated:
                best = {
                    "direction": "SHORT",
                    "entry": midpoint,
                    "stop": gap_high + (gap_high - gap_low) * 0.1,
                    "tp1": midpoint - (gap_high + (gap_high - gap_low) * 0.1 - midpoint) * TP_R,
                    "gap_high": gap_high,
                    "gap_low": gap_low,
                    "bar_idx": i,
                }

    return best


def calc_units(entry: float, stop: float, pair: str) -> int:
    risk_dollars = ACCOUNT_SIZE * RISK_PCT  # $250
    pip_size = 0.01 if "JPY" in pair else 0.0001
    stop_pips = abs(entry - stop) / pip_size
    if stop_pips < 1:
        return 0
    # Approximate: $10 per pip per 100k units for major pairs
    pip_value_per_100k = 10.0 if "JPY" not in pair else 9.5
    units_100k = risk_dollars / (stop_pips * pip_value_per_100k)
    return max(1000, int(units_100k * 100_000))


def yf_to_oanda(pair: str) -> str:
    """Convert yfinance format (GBPUSD=X) to OANDA format (GBP_USD)."""
    base = pair.replace("=X", "").replace("/", "")
    return base[:3] + "_" + base[3:]


def main() -> None:
    parser = argparse.ArgumentParser(description="FVG Express — rapid G2b trades")
    parser.add_argument("--max-trades", type=int, default=5,
                        help="Maximum trades to place (default 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print signals without placing trades")
    args = parser.parse_args()

    print("FVG Express — G2b Execution Validation Trades")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE (OANDA practice)'}")
    print(f"Max trades: {args.max_trades} | Risk per trade: {RISK_PCT:.2%} | TP: {TP_R}R\n")

    print("Downloading 5d 1h data…")
    signals_found = []
    for pair in PAIRS_YF:
        try:
            df = yf.download(pair, period="10d", interval="1h",
                             progress=False, auto_adjust=True)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.capitalize)
            df.index = pd.to_datetime(df.index, utc=True)
            fvg = find_fvg(df, pair)
            if fvg:
                fvg["pair_yf"] = pair
                fvg["pair_oanda"] = yf_to_oanda(pair)
                fvg["current_price"] = float(df["Close"].iloc[-1])
                signals_found.append(fvg)
                print(f"  {pair}: {fvg['direction']} FVG @ "
                      f"{fvg['gap_low']:.5f}–{fvg['gap_high']:.5f} "
                      f"(entry {fvg['entry']:.5f})")
            else:
                print(f"  {pair}: no unmitigated FVG")
        except Exception as exc:
            print(f"  {pair}: error — {exc}")

    if not signals_found:
        print("\nNo FVGs found. Try again in a few hours or reduce FVG_MIN_BODY threshold.")
        return

    print(f"\nFound {len(signals_found)} FVG(s). "
          f"Placing up to {args.max_trades}{'(DRY RUN)' if args.dry_run else ''}…\n")

    if not args.dry_run:
        try:
            from sovereign.execution.oanda_bridge import OandaBridge
            bridge = OandaBridge()
        except EnvironmentError as e:
            print(f"ERROR: {e}")
            print("Set OANDA_API_KEY and OANDA_ACCOUNT_ID in .env")
            sys.exit(1)

    placed = 0
    for sig in signals_found[:args.max_trades]:
        units = calc_units(sig["entry"], sig["stop"], sig["pair_yf"])
        if units == 0:
            print(f"  {sig['pair_oanda']}: skipped — stop too tight for sizing")
            continue

        risk_dollars = abs(sig["entry"] - sig["stop"]) * units
        r_label = f"entry={sig['entry']:.5f} stop={sig['stop']:.5f} tp={sig['tp1']:.5f}"

        if args.dry_run:
            print(f"  [DRY RUN] {sig['pair_oanda']} {sig['direction']} "
                  f"{units} units | {r_label}")
            placed += 1
            continue

        try:
            result = bridge.place_trade(
                pair=sig["pair_oanda"],
                direction=sig["direction"],
                units=units,
                stop_price=sig["stop"],
                tp1_price=sig["tp1"],
            )
            status = result.get("status", "UNKNOWN")
            if status == "FILLED":
                print(f"  ✓ {sig['pair_oanda']} {sig['direction']} "
                      f"{units}u | {r_label}")
                placed += 1
            else:
                print(f"  ✗ {sig['pair_oanda']}: {status} — {result.get('message','')}")
        except Exception as exc:
            print(f"  ERROR {sig['pair_oanda']}: {exc}")

    print(f"\nPlaced {placed} FVG Express trade(s).")
    if placed > 0 and not args.dry_run:
        print("  Run sync_dashboard_data.py after trades close to update G2b count.")


if __name__ == "__main__":
    main()
