"""Unit tests for sovereign/forex/entry_engine.py"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from sovereign.forex.macro_engine import ForexSignal
from sovereign.forex.ict_engine import ICTAnalysis
from sovereign.forex.entry_engine import ForexEntryEngine, ForexEntrySignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_macro_signal(direction: str = "LONG", conviction: float = 0.75) -> ForexSignal:
    return ForexSignal(
        pair="EURUSD=X",
        direction=direction,
        conviction=conviction,
        hold_period_estimate=45,
        primary_driver="rate_diff_momentum",
        rate_differential=1.5,
        irp_z=-1.8,
        ppp_z=-1.0,
        cycle_divergence=0.6,
        hurst=0.6,
        spot=1.12,
        base_cycle="MID_EXP",
        quote_cycle="CONTRACTION",
    )


def _make_ict_analysis(
    price: float = 1.1200,
    atr: float = 0.005,
    trend: str = "BULLISH",
) -> ICTAnalysis:
    from sovereign.forex.ict_engine import (
        MarketStructure,
        OrderBlock,
        FVG,
        LiquiditySweep,
    )

    ms = MarketStructure(
        trend=trend,
        last_choch="BULLISH",
        last_bos="BULLISH",
    )
    ob = OrderBlock(
        kind="BULLISH",
        high=price * 1.002,
        low=price * 0.998,
        midpoint=price,
        formed_at=pd.Timestamp("2024-01-01"),
    )
    fvg = FVG(
        kind="BULLISH",
        top=price * 1.001,
        bottom=price * 0.999,
        midpoint=price,
        formed_at=pd.Timestamp("2024-01-01"),
    )
    sweep = LiquiditySweep(
        direction="BULLISH_SWEEP",
        swept_level=price * 0.995,
        sweep_candle_low=price * 0.994,
        sweep_candle_high=price * 0.996,
        formed_at=pd.Timestamp("2024-01-01"),
    )
    return ICTAnalysis(
        pair="EURUSD=X",
        as_of=pd.Timestamp("2024-01-01"),
        market_structure=ms,
        active_fvgs=[fvg],
        active_obs=[ob],
        recent_sweeps=[sweep],
        in_kill_zone=True,
        kill_zone_name="London Open",
        in_ny_lunch=False,
        current_price=price,
        atr_daily=atr,
        nearest_bullish_ob=ob,
        nearest_bearish_ob=None,
        nearest_bullish_fvg=fvg,
        nearest_bearish_fvg=None,
    )


def _make_prices(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(1.10, 1.15, n)
    return pd.DataFrame({
        "Open": close * 0.9998,
        "High": close * 1.002,
        "Low": close * 0.998,
        "Close": close,
        "Volume": np.zeros(n),
    }, index=idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_evaluate_returns_no_trade_when_macro_neutral():
    engine = ForexEntryEngine()
    neutral_sig = _make_macro_signal(direction="NEUTRAL", conviction=0.10)

    with patch.object(engine._macro, "score_pair", return_value=neutral_sig), \
         patch.object(engine._ict, "analyse", return_value=_make_ict_analysis()):
        result = engine.evaluate("EURUSD=X", daily_df=_make_prices())

    assert result is not None
    assert result.direction == "NO_TRADE"
    assert not result.is_tradeable


def test_evaluate_returns_no_trade_below_conviction():
    engine = ForexEntryEngine()
    low_conv = _make_macro_signal(direction="LONG", conviction=0.20)

    with patch.object(engine._macro, "score_pair", return_value=low_conv), \
         patch.object(engine._ict, "analyse", return_value=_make_ict_analysis()):
        result = engine.evaluate("EURUSD=X", daily_df=_make_prices())

    assert result is not None
    assert result.direction == "NO_TRADE"


def test_evaluate_long_signal_is_tradeable():
    engine = ForexEntryEngine()
    macro_sig = _make_macro_signal(direction="LONG", conviction=0.80)
    ict = _make_ict_analysis()

    with patch.object(engine._macro, "score_pair", return_value=macro_sig), \
         patch.object(engine._ict, "analyse", return_value=ict), \
         patch.object(engine._commodity, "score_pair", return_value=None):
        result = engine.evaluate("EURUSD=X", daily_df=_make_prices())

    assert result is not None
    # Score should be ≥4 given bullish structure + OB + FVG + sweep + kill zone + CHOCH
    assert result.score >= 4
    assert result.direction == "LONG"
    assert result.is_tradeable


def test_evaluate_unknown_pair_returns_none():
    engine = ForexEntryEngine()
    with patch.object(engine._macro, "score_pair", return_value=None):
        result = engine.evaluate("UNKNOWN=X", daily_df=_make_prices())
    assert result is None
