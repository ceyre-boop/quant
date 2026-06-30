"""
scripts/build_market_memory.py
===============================
Refresh the FRED macro cache and rebuild the 23-feature "market memory"
fingerprint from current market data.

History: this script used to drive a `MarketMemory` class (with
`build_from_history()` / `HISTORICAL_EVENTS`) that was refactored away. The
surviving market-fingerprint extractor now lives in
`sovereign.risk.market_memory.extract_features`, and the historical-pattern
store it feeds is `sovereign.risk.alexandrian_library.AlexandrianLibrary`
(populated by scripts/build_alexandrian_library.py). This script now:

  1. Refreshes the FRED economic cache (data/macro/fred_economic_latest.json)
  2. Rebuilds the current market-memory feature vector via extract_features()
  3. Self-tests it against the live Alexandrian Library (threat read)

Usage:
    python3 scripts/build_market_memory.py
    python3 scripts/build_market_memory.py --rebuild   # force FRED re-fetch even if cached today
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def refresh_fred_cache(force: bool) -> None:
    """Step 1 — refresh the FRED economic snapshot cache (best-effort)."""
    latest = ROOT / "data" / "macro" / "fred_economic_latest.json"
    if latest.exists() and not force:
        try:
            cached_date = json.loads(latest.read_text()).get("date")
        except Exception:
            cached_date = None
        if cached_date == date.today().isoformat():
            print(f"FRED cache already fresh for {cached_date} (use --rebuild to force).")
            return
    try:
        from scripts.fetch_fred_economic import fetch_economic_snapshot
        snap = fetch_economic_snapshot()
        ok = snap.get("provenance", {}).get("verified", False)
        print(f"FRED cache refreshed: {snap['date']} | verified={ok} | "
              f"series={len(snap.get('metrics', {}))}")
    except Exception as e:
        print(f"FRED cache refresh skipped ({e}).")


def fetch_prices():
    """Download SPY / VIX / gold history for the feature rebuild."""
    import yfinance as yf

    spy = yf.download('SPY', period='2y', progress=False, auto_adjust=True)
    vix = yf.download('^VIX', period='2y', progress=False, auto_adjust=True)
    gld = yf.download('GLD', period='2y', progress=False, auto_adjust=True)

    spy_arr = spy['Close'].values.astype(float).squeeze() if len(spy) else None
    vix_arr = vix['Close'].values.astype(float).squeeze() if len(vix) >= 30 else None
    gold_arr = gld['Close'].values.astype(float).squeeze() if len(gld) >= 30 else None
    return spy_arr, vix_arr, gold_arr


def main():
    parser = argparse.ArgumentParser(description="Refresh FRED cache + rebuild market memory features")
    parser.add_argument('--rebuild', action='store_true',
                        help='Force a fresh FRED re-fetch even if the cache is current for today')
    args = parser.parse_args()

    from sovereign.risk.market_memory import extract_features, FEATURE_NAMES

    # ── Step 1: refresh FRED cache ─────────────────────────────────────────── #
    print("── Refreshing FRED economic cache ──")
    refresh_fred_cache(force=args.rebuild)
    print()

    # ── Step 2: rebuild market-memory features ─────────────────────────────── #
    print("── Rebuilding market-memory features ──")
    try:
        spy_arr, vix_arr, gold_arr = fetch_prices()
    except ImportError:
        print("yfinance not available — install with: pip3 install yfinance")
        return
    except Exception as e:
        print(f"Price fetch failed: {e}")
        return

    if spy_arr is None or len(spy_arr) < 200:
        print("Not enough SPY data to rebuild features (need 200+ bars).")
        return

    feats = extract_features(spy_arr, vix_prices=vix_arr, gold_prices=gold_arr)
    print(f"Extracted {len(feats)}-feature fingerprint "
          f"(SPY={len(spy_arr)} bars, "
          f"VIX={'live' if vix_arr is not None else 'default'}, "
          f"gold={'live' if gold_arr is not None else 'default'}):")
    for name, val in zip(FEATURE_NAMES, feats):
        print(f"  {name:<22} {val:+.4f}")
    print()

    # ── Step 3: self-test against the live Alexandrian Library ─────────────── #
    print("── Self-test: comparing current conditions to market memory ──")
    try:
        from sovereign.risk.alexandrian_library import AlexandrianLibrary
        lib = AlexandrianLibrary()
        if lib.n_patterns == 0:
            print("Library empty — run scripts/build_alexandrian_library.py to populate.")
            return
        print(lib.describe())
        insight = lib.query(spy_arr, vix_arr, gold_arr)
        print(f"Primary regime:  {insight.primary_regime}")
        print(f"Threat level:    {insight.threat_level}")
        print(f"Threat score:    {insight.threat_score:.4f}")
        print(f"Size modifier:   {insight.size_modifier:.2f}×")
        print()
        print("Top matches:")
        for m in insight.top_matches:
            bar = '█' * int(m.similarity * 20)
            print(f"  {m.label:<32} {bar:<20} {m.similarity:.3f}")
    except Exception as e:
        print(f"Self-test comparison skipped ({e}).")


if __name__ == '__main__':
    main()
