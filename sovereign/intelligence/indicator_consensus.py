"""
sovereign/intelligence/indicator_consensus.py

Live indicator consensus retrieval — called by pulse_check every 2h.
Loads green_conditions.json ONCE at module import (cached).
Runs all 30 indicators on a live OHLCV DataFrame and matches the result
against historical memory. No recomputation of history.

Public API:
    score_indicator_consensus(pair, ohlcv_df) → IndicatorConsensus
"""
from __future__ import annotations

import json
import logging
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


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    pair = "GBPUSD"
    df = yf.Ticker("GBPUSD=X").history(period="90d", interval="1d", auto_adjust=True)
    c = score_indicator_consensus(pair, df)
    print(f"{pair}: {c.bullish_count}/30 bull | {c.bearish_count}/30 bear | dir={c.direction} | conv={c.conviction:.0%}")
    print(f"  Historical hit rate: {c.historical_hit_rate:.0%}")
    print(f"  Matching green long:  {len(c.matching_green_long)}")
    print(f"  Matching green short: {len(c.matching_green_short)}")
    print(f"  Top bullish:  {c.top_bullish[:5]}")
    print(f"  Top bearish:  {c.top_bearish[:5]}")
