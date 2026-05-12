"""
scripts/build_alexandrian_library.py
=====================================
One-time script: populate the Alexandrian Library from yfinance.

Builds all 10 volumes from historical data. Run once to initialise.
After that, the orchestrator's on_trade_close() auto-learns new events.

Usage:
    python3 scripts/build_alexandrian_library.py
    python3 scripts/build_alexandrian_library.py --rebuild   # force re-fetch all
    python3 scripts/build_alexandrian_library.py --query     # query current state
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true',
                        help='Force re-fetch all entries even if already stored')
    parser.add_argument('--query', action='store_true',
                        help='Query current conditions against the library')
    args = parser.parse_args()

    from sovereign.risk.alexandrian_library import AlexandrianLibrary, ALL_ENTRIES
    lib = AlexandrianLibrary()

    vol_summary = lib.volume_summary()
    print("Alexandrian Library — current state:")
    for vol, count in sorted(vol_summary.items()):
        print(f"  {vol:<40} {count} patterns")
    print()

    print(f"Total entries across all volumes: {len(ALL_ENTRIES)}")
    print()

    built = lib.build_from_history(force_rebuild=args.rebuild)
    print(f"Build complete: {built} new patterns added")
    vol_summary_after = lib.volume_summary()
    total_after = sum(vol_summary_after.values())
    print(f"Total patterns in library: {total_after}")
    print()

    print("Patterns by volume:")
    for vol, count in sorted(vol_summary_after.items()):
        print(f"  {vol:<40} {count}")

    if args.query:
        print()
        print("Querying current market conditions...")
        try:
            import yfinance as yf
            import numpy as np
            spy = yf.download('SPY', period='3y', progress=False, auto_adjust=True)
            vix = yf.download('^VIX', period='3y', progress=False, auto_adjust=True)

            if len(spy) >= 200:
                spy_arr = spy['Close'].values.astype(float).squeeze()
                vix_arr = vix['Close'].values.astype(float).squeeze() if len(vix) >= 30 else None
                insight = lib.query(spy_arr, vix_arr)

                print(f"\nPrimary regime:    {insight.primary_regime}")
                print(f"Primary volume:    {insight.primary_volume}")
                print(f"Threat score:      {insight.threat_score:.4f}")
                print(f"Size modifier:     {insight.size_modifier:.2f}×")
                print(f"Converging signal: {insight.converging_signal}")
                print()
                print(f"Advisory: {insight.advisory}")
                print()
                print(f"Action summary: {insight.action_summary}")
                print()
                print("Volume matches:")
                for vm in insight.volume_matches:
                    bar = '█' * int(vm.similarity * 20)
                    vol = str(vm.volume).split('.')[-1] if hasattr(vm, 'volume') else str(getattr(vm, 'volume_type', '?'))
                    name = getattr(vm, 'label', getattr(vm, 'entry_name', getattr(vm, 'entry_id', '?')))
                    print(f"  {vol:<35} {bar:<20} {vm.similarity:.3f}  [{name}]")
                print()
                print("Top matches across all volumes:")
                for m in insight.top_matches[:8]:
                    bar = '█' * int(m.similarity * 20)
                    vol = str(m.volume).split('.')[-1] if hasattr(m, 'volume') else str(getattr(m, 'volume_type', '?'))
                    name = getattr(m, 'label', getattr(m, 'entry_name', getattr(m, 'entry_id', '?')))
                    print(f"  {name:<35} {bar:<20} {m.similarity:.3f}  {vol}")
            else:
                print("Not enough SPY data for query")
        except ImportError:
            print("yfinance not available — install with: pip3 install yfinance")
        except Exception as e:
            print(f"Query error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
