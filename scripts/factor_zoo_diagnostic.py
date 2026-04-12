"""
Phase 0 — Factor Zoo Diagnostic
Runs Bonferroni-corrected ICIR scan across 6 assets at hourly + daily resolution.
Decision tree outcomes logged to logs/factor_zoo_diagnostic_{date}.json
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sovereign.data.feeds.alpaca_feed import AlpacaFeed
from sovereign.features.factor_zoo import FactorZooScanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIAGNOSTIC_UNIVERSE = ["NVDA", "TSLA", "AMD", "META", "TLT", "SPY"]
TRAIN_START = "2022-01-01"
TRAIN_END = "2024-01-01"
OOS_START = "2025-01-01"
OOS_END = "2026-01-01"
ICIR_THRESHOLD = 0.30
ICIR_FALLBACK = 0.15
MIN_ROBUST_FEATURES = 8


def run_scan_for_asset(feed: AlpacaFeed, symbol: str, resolution: str) -> dict:
    """Run IS + OOS factor zoo scan for one symbol at one resolution."""
    start_dt = datetime(2021, 1, 1, tzinfo=timezone.utc)
    timeframe = "1Day" if resolution == "1d" else "1Hour"

    df = feed.get_bars(symbol, start=start_dt, timeframe=timeframe, use_cache=True)
    if df.empty or len(df) < 500:
        logger.warning(f"  {symbol}@{resolution}: insufficient data ({len(df)} bars)")
        return {"symbol": symbol, "resolution": resolution, "n_robust": 0, "features": []}

    logger.info(f"  {symbol}@{resolution}: {len(df)} bars loaded")

    scanner = FactorZooScanner()

    # IS scan
    df_is = df.loc[TRAIN_START:TRAIN_END]
    is_result = {}
    oos_result = {}

    if len(df_is) >= 200:
        feat_is = scanner.build_feature_matrix(df_is)
        res_is = scanner.scan(feat_is)
        is_passed = set(res_is[res_is["is_real"]]["feature"].tolist())
        is_icirs = dict(zip(res_is["feature"], res_is["icir"].round(4)))
        is_pvals = dict(zip(res_is["feature"], res_is["p_value"].round(6)))
        is_result = {"passed": list(is_passed), "icirs": is_icirs, "p_values": is_pvals}
    else:
        logger.warning(f"  {symbol}@{resolution}: IS period too short ({len(df_is)} bars)")
        is_passed = set()

    # OOS scan
    df_oos = df.loc[OOS_START:OOS_END]
    if len(df_oos) >= 100:
        feat_oos = scanner.build_feature_matrix(df_oos)
        res_oos = scanner.scan(feat_oos)
        oos_passed = set(res_oos[res_oos["is_real"]]["feature"].tolist())
        oos_icirs = dict(zip(res_oos["feature"], res_oos["icir"].round(4)))
        oos_result = {"passed": list(oos_passed), "icirs": oos_icirs}
    else:
        logger.warning(f"  {symbol}@{resolution}: OOS period too short ({len(df_oos)} bars)")
        oos_passed = set()

    # Robust = IS ∩ OOS
    robust = list(is_passed & oos_passed)

    return {
        "symbol": symbol,
        "resolution": resolution,
        "n_robust": len(robust),
        "robust_features": robust,
        "is": is_result,
        "oos": oos_result,
    }


def run_diagnostic():
    feed = AlpacaFeed()
    results = {}
    all_summary = []

    for symbol in DIAGNOSTIC_UNIVERSE:
        results[symbol] = {}
        for resolution in ["1h", "1d"]:
            logger.info(f"Scanning {symbol} @ {resolution}...")
            r = run_scan_for_asset(feed, symbol, resolution)
            results[symbol][resolution] = r
            all_summary.append({
                "symbol": symbol,
                "resolution": resolution,
                "n_robust": r["n_robust"],
                "features": r.get("robust_features", []),
            })
            logger.info(f"  → {r['n_robust']} robust features: {r.get('robust_features', [])}")

    # Decision tree
    logger.info("\n" + "=" * 60)
    logger.info("FACTOR ZOO DIAGNOSTIC — DECISION TREE")
    logger.info("=" * 60)

    hourly_robust = {}
    daily_robust = {}
    all_robust_features = set()

    for symbol in DIAGNOSTIC_UNIVERSE:
        h = results[symbol].get("1h", {})
        d = results[symbol].get("1d", {})
        hourly_robust[symbol] = h.get("robust_features", [])
        daily_robust[symbol] = d.get("robust_features", [])
        all_robust_features.update(h.get("robust_features", []))
        all_robust_features.update(d.get("robust_features", []))

    total_hourly = sum(len(v) for v in hourly_robust.values())
    total_daily = sum(len(v) for v in daily_robust.values())
    hourly_across_universe = {f for feats in hourly_robust.values() for f in feats}
    daily_across_universe = {f for feats in daily_robust.values() for f in feats}

    decision = {}

    if len(daily_across_universe) >= MIN_ROBUST_FEATURES:
        decision["resolution"] = "1d"
        decision["icir_threshold"] = ICIR_THRESHOLD
        decision["action"] = "PROCEED — use daily resolution for signal generation. Hourly = execution only."
        decision["robust_features"] = list(daily_across_universe)
        logger.info(f"OUTCOME: Daily bars → {len(daily_across_universe)} robust features across universe.")
        logger.info(f"ACTION: {decision['action']}")

    elif len(hourly_across_universe) >= MIN_ROBUST_FEATURES:
        # Check if SPY is the drag
        non_spy_hourly = {f for sym, feats in hourly_robust.items() for f in feats if sym != "SPY"}
        if len(non_spy_hourly) >= MIN_ROBUST_FEATURES:
            decision["resolution"] = "1h"
            decision["icir_threshold"] = ICIR_THRESHOLD
            decision["action"] = "PROCEED — hourly works on volatile stocks. Exclude SPY from signal generation."
            decision["robust_features"] = list(non_spy_hourly)
            logger.info(f"OUTCOME: Hourly bars (ex-SPY) → {len(non_spy_hourly)} robust features.")
            logger.info(f"ACTION: {decision['action']}")
        else:
            decision["resolution"] = "1h"
            decision["icir_threshold"] = ICIR_THRESHOLD
            decision["action"] = "PROCEED — hourly features valid."
            decision["robust_features"] = list(hourly_across_universe)

    else:
        # Fallback: try ICIR=0.15
        logger.info(f"WARNING: Fewer than {MIN_ROBUST_FEATURES} features at ICIR≥0.30. Trying ICIR≥0.15 fallback...")
        # Re-check raw ICIR values from summary
        fallback_features = set()
        for symbol in DIAGNOSTIC_UNIVERSE:
            for resolution in ["1h", "1d"]:
                r = results[symbol].get(resolution, {})
                is_icirs = r.get("is", {}).get("icirs", {})
                oos_icirs = r.get("oos", {}).get("icirs", {})
                for feat in is_icirs:
                    if abs(is_icirs.get(feat, 0)) >= ICIR_FALLBACK and abs(oos_icirs.get(feat, 0)) >= ICIR_FALLBACK:
                        fallback_features.add(feat)

        if len(fallback_features) >= MIN_ROBUST_FEATURES:
            decision["resolution"] = "1d"
            decision["icir_threshold"] = ICIR_FALLBACK
            decision["action"] = f"PROCEED with lowered ICIR threshold ({ICIR_FALLBACK}). Document in config."
            decision["robust_features"] = list(fallback_features)
            logger.info(f"OUTCOME: {len(fallback_features)} features pass at ICIR≥{ICIR_FALLBACK}.")
        else:
            decision["resolution"] = "UNKNOWN"
            decision["icir_threshold"] = None
            decision["action"] = "STOP — fundamental feature redesign required. Do not proceed to Phase 3."
            decision["robust_features"] = []
            logger.error("OUTCOME: ZERO robust features at any resolution or threshold.")
            logger.error("ACTION: STOP. Report to operator. Do not build Phase 3 specialists.")

    # Save results
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = log_dir / f"factor_zoo_diagnostic_{date_str}.json"

    output = {
        "run_date": date_str,
        "config": {
            "universe": DIAGNOSTIC_UNIVERSE,
            "train": f"{TRAIN_START} to {TRAIN_END}",
            "oos": f"{OOS_START} to {OOS_END}",
            "icir_threshold_original": ICIR_THRESHOLD,
        },
        "per_asset_results": results,
        "summary": all_summary,
        "decision": decision,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved → {output_path}")
    logger.info(f"\nFINAL DECISION: {decision['action']}")

    if decision["robust_features"]:
        logger.info(f"Robust features to enter config: {decision['robust_features']}")
        logger.info(f"Resolution to use: {decision['resolution']}")
        logger.info(f"ICIR threshold to use: {decision['icir_threshold']}")

    return decision


if __name__ == "__main__":
    run_diagnostic()
