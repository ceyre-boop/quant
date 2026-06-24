#!/usr/bin/env python3
"""
scripts/thetadata_probe.py — STEP 1: what does the ThetaData tier actually serve?

Honest connectivity + coverage probe. ThetaData is an OPTIONS/equities/indices vendor
(no spot forex). The existing integration (sovereign/research/vrp/) is options-only and
on the current tier even stock history returns 403. This probe confirms that against the
live ThetaTerminal so we don't build on a wrong assumption.

    python3 scripts/thetadata_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print("=" * 60)
    print("  THETADATA COVERAGE PROBE")
    print("=" * 60)
    try:
        from sovereign.research.vrp.data_loader import ThetaDataLoader
    except Exception as exc:  # noqa: BLE001
        print(f"  cannot import ThetaDataLoader: {type(exc).__name__}: {exc}")
        return 1

    loader = ThetaDataLoader()
    print(f"  base_url: {getattr(loader, 'base_url', '?')}  (ThetaTerminal must be running)")

    # (a) options access — the thing this tier is for
    try:
        exps = loader.list_expirations("SPY")
        print(f"  ✅ OPTIONS (SPY expirations): {len(exps)} returned — options access OK")
        try:
            print(f"     earliest available: {loader.earliest_available_date('SPY')}")
        except Exception as exc:  # noqa: BLE001
            print(f"     earliest_available_date failed: {type(exc).__name__}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ OPTIONS list failed: {type(exc).__name__}: {str(exc)[:160]}")
        print("     → ThetaTerminal not running, or not logged in. Start it and retry.")

    # (b) stock underlying — expected 403 on the Options VALUE tier
    try:
        loader.get_underlying_close("SPY", "2025-01-02")
        print("  ✅ STOCK underlying EOD served on this tier (unexpected — usable for equities)")
    except NotImplementedError:
        print("  ℹ️ STOCK underlying: get_underlying_close is NotImplemented in the loader "
              "(VRP pulls the underlying from yfinance) — tier serves OPTIONS, not stock OHLCV")
    except Exception as exc:  # noqa: BLE001
        print(f"  ℹ️ STOCK underlying: {type(exc).__name__} (expected 403 on Options VALUE tier)")

    print()
    print("  VERDICT:")
    print("   • ThetaData = options/equities vendor. NO spot forex (EURUSD etc.).")
    print("   • Current tier ≈ options chains only; clean stock OHLCV is NOT available here.")
    print("   • For a CLEAN equity-index discovery, use the on-disk NQ parquet")
    print("     (data/es_nq/nq_daily.parquet) → `discover.py --track equity --source parquet`.")
    print("   • For a clean FOREX re-test (the original question), use OANDA or Dukascopy — not ThetaData.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
