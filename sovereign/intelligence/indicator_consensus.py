"""
sovereign/intelligence/indicator_consensus.py

Live indicator consensus retrieval — called by pulse_check every 2h.
Loads green_conditions.json ONCE at module import (cached).
Runs all 30 indicators on a live OHLCV DataFrame and matches the result
against historical memory. No recomputation of history.

Public API:
    score_indicator_consensus(pair, ohlcv_df) → IndicatorConsensus
    validate_on_holdout(start, end) → dict  — holdout overfitting check
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from sovereign.intelligence.indicator_library import (
    INDICATOR_NAMES,
    IndicatorState,
    compute_all_indicators,
)

log = logging.getLogger("sovereign.indicator_consensus")

ROOT = Path(__file__).resolve().parents[2]
_GREEN_PATH  = ROOT / "data" / "indicators" / "green_conditions.json"
_MEMORY_PATH = ROOT / "data" / "indicators" / "oracle_indicator_memory.json"

# Module-level cache — loaded once per process
_GREEN_CONDITIONS: Optional[dict] = None


def _load_green_conditions() -> dict:
    global _GREEN_CONDITIONS
    if _GREEN_CONDITIONS is not None:
        return _GREEN_CONDITIONS
    if not _GREEN_PATH.exists():
        log.info("green_conditions.json not found — run build_indicator_ontology.py first")
        _GREEN_CONDITIONS = {}
        return _GREEN_CONDITIONS
    try:
        _GREEN_CONDITIONS = json.loads(_GREEN_PATH.read_text())
    except Exception as exc:
        log.warning("Failed to load green_conditions.json: %s", exc)
        _GREEN_CONDITIONS = {}
    return _GREEN_CONDITIONS


# ─── Result type ─────────────────────────────────────────────────────────────

@dataclass
class IndicatorConsensus:
    pair: str
    bullish_count: int             # out of 30
    bearish_count: int
    neutral_count: int
    matching_green_long: list[dict]   # green conditions matched by current state
    matching_green_short: list[dict]
    direction: str                 # LONG | SHORT | NEUTRAL | FLAT
    conviction: float              # 0.0–1.0
    historical_hit_rate: float     # best matching green condition hit_rate (0 if no match)
    snapshot: dict[str, int]       # {indicator_name: state} for all 30
    top_bullish: list[str] = field(default_factory=list)   # names of BULLISH indicators
    top_bearish: list[str] = field(default_factory=list)   # names of BEARISH indicators


# ─── Core scorer ─────────────────────────────────────────────────────────────

def score_indicator_consensus(pair: str, ohlcv_df: pd.DataFrame) -> IndicatorConsensus:
    """
    Run all 30 indicators on ohlcv_df, match against historical green conditions,
    return consensus. Safe — never raises; returns a FLAT consensus on any error.
    """
    try:
        return _score(pair, ohlcv_df)
    except Exception as exc:
        log.warning("score_indicator_consensus(%s) failed: %s", pair, exc)
        return _flat_consensus(pair)


def _score(pair: str, ohlcv_df: pd.DataFrame) -> IndicatorConsensus:
    # Run all 30 indicators on the full OHLCV (uses last bar's state)
    indicators: dict[str, IndicatorState] = compute_all_indicators(ohlcv_df)

    snapshot = {name: indicators[name].state for name in INDICATOR_NAMES if name in indicators}
    bullish = [n for n, s in snapshot.items() if s == 1]
    bearish = [n for n, s in snapshot.items() if s == -1]
    neutral = [n for n, s in snapshot.items() if s == 0]

    n_bull = len(bullish)
    n_bear = len(bearish)
    n_neut = len(neutral)
    total  = max(n_bull + n_bear + n_neut, 1)

    # Match against stored green conditions
    green = _load_green_conditions()
    pair_green = green.get(pair, {"best_long": [], "best_short": []})

    matching_long  = _match_conditions(snapshot, pair_green.get("best_long", []),  direction=1)
    matching_short = _match_conditions(snapshot, pair_green.get("best_short", []), direction=-1)

    # Direction + conviction
    bull_ratio = n_bull / total
    bear_ratio = n_bear / total
    spread = bull_ratio - bear_ratio

    if n_bull >= 18:
        direction = "LONG"
    elif n_bear >= 18:
        direction = "SHORT"
    elif spread > 0.15:
        direction = "LONG"
    elif spread < -0.15:
        direction = "SHORT"
    elif abs(spread) <= 0.05:
        direction = "FLAT"
    else:
        direction = "NEUTRAL"

    conviction = min(abs(spread) * 2.5, 1.0)

    # Historical hit rate — best matching green condition
    all_matches = matching_long + matching_short
    hist_hr = max((c["hit_rate"] for c in all_matches), default=0.0)

    return IndicatorConsensus(
        pair=pair,
        bullish_count=n_bull,
        bearish_count=n_bear,
        neutral_count=n_neut,
        matching_green_long=matching_long,
        matching_green_short=matching_short,
        direction=direction,
        conviction=round(conviction, 3),
        historical_hit_rate=round(hist_hr, 4),
        snapshot=snapshot,
        top_bullish=bullish[:10],
        top_bearish=bearish[:10],
    )


def _match_conditions(snapshot: dict[str, int], conditions: list[dict], direction: int) -> list[dict]:
    matched = []
    for cond in conditions:
        inds = cond.get("indicators", [])
        if all(snapshot.get(ind) == direction for ind in inds):
            matched.append(cond)
    return matched


def _flat_consensus(pair: str) -> IndicatorConsensus:
    return IndicatorConsensus(
        pair=pair,
        bullish_count=0,
        bearish_count=0,
        neutral_count=len(INDICATOR_NAMES),
        matching_green_long=[],
        matching_green_short=[],
        direction="FLAT",
        conviction=0.0,
        historical_hit_rate=0.0,
        snapshot={name: 0 for name in INDICATOR_NAMES},
    )


# ─── Holdout validation ───────────────────────────────────────────────────────

_MIN_HOLDOUT_N = 10   # combos with fewer holdout samples → INSUFFICIENT_DATA
_OVERFIT_DELTA = 0.20  # train-holdout gap > 20% → OVERFIT


@dataclass
class HoldoutResult:
    indicators: list[str]
    direction: str          # LONG | SHORT
    train_hit_rate: float
    holdout_hit_rate: Optional[float]
    delta: Optional[float]  # train - holdout (positive = optimistic on train)
    n_holdout: int
    verdict: str            # REAL_SIGNAL | WEAK_SIGNAL | OVERFIT | INSUFFICIENT_DATA


def validate_on_holdout(
    start: str = "2024-01-01",
    end: str = "2024-12-31",
) -> dict[str, list[HoldoutResult]]:
    """
    Test every green condition against an out-of-sample holdout window.

    The green conditions in green_conditions.json were found on the FULL 2015-2024 dataset.
    This function re-tests each combo on the holdout slice only to detect overfitting.

    Verdicts:
        REAL_SIGNAL      — delta < 10%  (holds up on unseen data)
        WEAK_SIGNAL      — delta 10-20% (some decay, still tradeable)
        OVERFIT          — delta > 20%  (train bias, do not trade)
        INSUFFICIENT_DATA — n_holdout < 10 (can't tell, need more bars)
    """
    hist_path = ROOT / "data" / "indicators" / "history.parquet"
    if not hist_path.exists():
        raise FileNotFoundError("history.parquet not found — run build_indicator_ontology.py first")

    hist = pd.read_parquet(hist_path)
    hist["date"] = pd.to_datetime(hist["date"])

    holdout_full = hist[
        (hist["date"] >= start) & (hist["date"] <= end)
    ].dropna(subset=["fwd_10d"]).copy()

    green = _load_green_conditions()
    results: dict[str, list[HoldoutResult]] = {}

    for pair, conditions in green.items():
        g = holdout_full[holdout_full["pair"] == pair]
        pair_results: list[HoldoutResult] = []

        for dir_key, sign in [("best_long", 1), ("best_short", -1)]:
            direction_label = "LONG" if sign == 1 else "SHORT"
            fwd = g["fwd_10d"].astype(float)

            for cond in conditions.get(dir_key, []):
                inds = cond["indicators"]
                ca, cb, cc = f"state_{inds[0]}", f"state_{inds[1]}", f"state_{inds[2]}"

                if ca not in g.columns or cb not in g.columns or cc not in g.columns:
                    continue

                mask = (g[ca] == sign) & (g[cb] == sign) & (g[cc] == sign)
                n_holdout = int(mask.sum())

                if n_holdout == 0:
                    holdout_hr = None
                    delta = None
                    verdict = "INSUFFICIENT_DATA"
                else:
                    subset = fwd[mask]
                    holdout_hr = float((subset > 0).mean() if sign == 1 else (subset < 0).mean())
                    delta = round(cond["hit_rate"] - holdout_hr, 4)

                    if n_holdout < _MIN_HOLDOUT_N:
                        verdict = "INSUFFICIENT_DATA"
                    elif abs(delta) < 0.10:
                        verdict = "REAL_SIGNAL"
                    elif abs(delta) < _OVERFIT_DELTA:
                        verdict = "WEAK_SIGNAL"
                    else:
                        verdict = "OVERFIT"

                pair_results.append(HoldoutResult(
                    indicators=inds,
                    direction=direction_label,
                    train_hit_rate=cond["hit_rate"],
                    holdout_hit_rate=round(holdout_hr, 4) if holdout_hr is not None else None,
                    delta=delta,
                    n_holdout=n_holdout,
                    verdict=verdict,
                ))

        results[pair] = pair_results

    return results


def print_holdout_report(results: dict[str, list[HoldoutResult]]) -> None:
    verdicts_all: list[str] = []
    print(f"\n{'='*70}")
    print("INDICATOR GREEN CONDITIONS — HOLDOUT VALIDATION")
    print(f"{'='*70}")
    print(f"{'PAIR':<8} {'DIRECTION':<6} {'INDICATORS':<40} {'TRAIN':>6} {'HOLD':>6} {'DELTA':>7} {'N':>4}  VERDICT")
    print("-" * 90)

    for pair, pair_results in sorted(results.items()):
        for r in pair_results:
            ind_str = "+".join(r.indicators)
            hold_str = f"{r.holdout_hit_rate:.0%}" if r.holdout_hit_rate is not None else "  n/a"
            delta_str = f"{r.delta:+.0%}" if r.delta is not None else "   n/a"
            v_icon = {"REAL_SIGNAL": "✓", "WEAK_SIGNAL": "~", "OVERFIT": "✗", "INSUFFICIENT_DATA": "?"}.get(r.verdict, " ")
            print(
                f"{pair:<8} {r.direction:<6} {ind_str:<40} "
                f"{r.train_hit_rate:.0%}{'':<1} {hold_str:<6} {delta_str:<7} {r.n_holdout:>4}  {v_icon} {r.verdict}"
            )
            verdicts_all.append(r.verdict)

    print()
    from collections import Counter
    vc = Counter(verdicts_all)
    total = len(verdicts_all)
    print(f"Summary: {vc.get('REAL_SIGNAL',0)} REAL | {vc.get('WEAK_SIGNAL',0)} WEAK | "
          f"{vc.get('OVERFIT',0)} OVERFIT | {vc.get('INSUFFICIENT_DATA',0)} INSUFFICIENT_DATA  "
          f"(of {total} conditions)")
    real_pct = vc.get("REAL_SIGNAL", 0) / max(total, 1)
    print(f"Signal quality: {real_pct:.0%} of green conditions hold up on holdout data")
    if vc.get("OVERFIT", 0) / max(total, 1) > 0.40:
        print("\n⚠ WARNING: >40% overfit — thresholds too loose. Raise MIN_HIT_RATE before live use.")
    elif vc.get("REAL_SIGNAL", 0) / max(total, 1) > 0.50:
        print("\n✓ CONCLUSION: >50% REAL_SIGNAL — green conditions are statistically robust.")
    else:
        print("\n~ CONCLUSION: Mixed signal quality — use REAL_SIGNAL combos only, discard OVERFIT.")


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import yfinance as yf

    parser = argparse.ArgumentParser(description="Indicator consensus — live score or holdout validation")
    parser.add_argument("--validate-holdout", action="store_true", help="Run green condition holdout validation")
    parser.add_argument("--holdout-start", default="2024-01-01")
    parser.add_argument("--holdout-end",   default="2024-12-31")
    parser.add_argument("--pair", default="GBPUSD", help="Pair for live score (default: GBPUSD)")
    args = parser.parse_args()

    if args.validate_holdout:
        results = validate_on_holdout(args.holdout_start, args.holdout_end)
        print_holdout_report(results)
    else:
        pair = args.pair
        ticker = pair if "=X" in pair else f"{pair}=X"
        df = yf.Ticker(ticker).history(period="90d", interval="1d", auto_adjust=True)
        c = score_indicator_consensus(pair, df)
        print(f"{pair}: {c.bullish_count}/30 bull | {c.bearish_count}/30 bear | dir={c.direction} | conv={c.conviction:.0%}")
        print(f"  Historical hit rate: {c.historical_hit_rate:.0%}")
        print(f"  Matching green long:  {len(c.matching_green_long)}")
        print(f"  Matching green short: {len(c.matching_green_short)}")
        print(f"  Top bullish:  {c.top_bullish[:5]}")
        print(f"  Top bearish:  {c.top_bearish[:5]}")
