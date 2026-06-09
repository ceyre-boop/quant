"""Decision engine + reasoning tests — learning-mode behaviour, telemetry, null-safety.

Sandbox-local; no network. Builds deterministic synthetic 1-min RTH bars.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sovereign.futures import scalp_strategy as strat
from sovereign.futures import reasoning as rsn
from sovereign.futures.decision_engine import evaluate_entry, EntryDecision


def _bars(closes, vol=1000.0):
    """Build an OHLCV frame with a UTC DatetimeIndex from a close series."""
    n = len(closes)
    idx = pd.date_range("2026-06-08 13:30", periods=n, freq="1min", tz="UTC")
    c = np.array(closes, float)
    return pd.DataFrame({"Open": c, "High": c + 0.5, "Low": c - 0.5, "Close": c,
                         "Volume": np.full(n, vol)}, index=idx)


MIDDAY = datetime(2026, 6, 8, 17, 0, tzinfo=timezone.utc)   # 13:00 ET → MIDDAY (strict blocks)
OPEN_ET = datetime(2026, 6, 8, 14, 0, tzinfo=timezone.utc)  # 10:00 ET → OPEN window


def test_orb_fires_in_learning_blocks_in_strict():
    bars = _bars([100.0] * 40 + [110.0])     # breaks above orb_high
    bias = {"bias": "LONG", "conviction": 2, "key_levels": {"overnight_low": 95.0}}
    orb = (105.0, 99.0)
    strict = evaluate_entry(bars, bias=bias, ts=MIDDAY, instrument="MES",
                            orb_levels=orb, learning_mode=False)
    learn = evaluate_entry(bars, bias=bias, ts=MIDDAY, instrument="MES",
                           orb_levels=orb, learning_mode=True)
    assert strict is None                                   # midday window blocks strict
    assert isinstance(learn, EntryDecision)
    assert learn.setup_type == "ORB" and learn.direction == "LONG"
    assert "session_window" in learn.would_have_blocked     # records the bypassed strict gate
    assert learn.learning_mode is True


def test_no_false_positive_when_no_setup():
    bars = _bars([100.0] * 41)               # flat: no ORB break, no cross
    bias = {"bias": "LONG", "conviction": 1, "key_levels": {}}
    for lm in (False, True):
        d = evaluate_entry(bars, bias=bias, ts=OPEN_ET, instrument="MES",
                           orb_levels=(105.0, 99.0), learning_mode=lm)
        assert d is None                     # price inside the range → nothing fires


def test_null_safety_thin_data():
    # too few bars for indicators → None, no crash
    assert evaluate_entry(_bars([100.0, 100.5]), bias={"bias": "LONG"}, ts=OPEN_ET,
                          instrument="MES", learning_mode=True) is None


def test_telemetry_populated():
    bars = _bars([100.0] * 40 + [110.0])
    d = evaluate_entry(bars, bias={"bias": "LONG", "conviction": 2, "key_levels": {}},
                       ts=OPEN_ET, instrument="MES", orb_levels=(105.0, 99.0), learning_mode=True)
    assert d.expected_r > 0 and d.confidence in ("HIGH", "MEDIUM", "LOW")
    assert d.time_gate in ("OPEN", "CLOSE", "MIDDAY")
    assert "vwap" in d.key_levels


def test_entry_reasoning_block():
    bars = _bars([100.0] * 40 + [110.0])
    d = evaluate_entry(bars, bias={"bias": "LONG", "conviction": 2, "key_levels": {}},
                       ts=OPEN_ET, instrument="MES", orb_levels=(105.0, 99.0), learning_mode=True)
    block = rsn.entry_reasoning(d, {"bias": "LONG", "conviction": 2})
    assert block["setup_type"] == "ORB"
    assert "ORB" in block["why_this_direction"]
    for k in ("confluence_score", "cvd_quality", "confidence", "expected_r", "would_have_blocked"):
        assert k in block


def test_exit_attribution_win_and_loss():
    rec = {"reasoning": {"setup_type": "ORB", "cvd_confirmed": False, "expected_target": 110}}
    loss = rsn.exit_attribution(rec, {"exit_type": "STOP_LOSS", "exit_price": 99, "r_realized": -1.0})
    assert loss["post_trade_hypothesis"] and "CVD" in loss["post_trade_hypothesis"]
    win = rsn.exit_attribution({"reasoning": {"setup_type": "ORB", "cvd_confirmed": True}},
                               {"exit_type": "TARGET", "exit_price": 110, "r_realized": 1.8})
    assert win["r_realized"] == 1.8 and "win" in win["post_trade_hypothesis"].lower()


def test_exit_attribution_null_safe():
    # missing reasoning entirely → no crash, still produces a hypothesis
    out = rsn.exit_attribution({}, {"exit_type": "TIME", "r_realized": 0.0})
    assert "post_trade_hypothesis" in out
