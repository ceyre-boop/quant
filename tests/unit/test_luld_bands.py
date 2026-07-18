"""Tiered LULD band table + the regression that states the bug fix.

Reg NMS Plan to Address Extraordinary Volatility:
    Tier 1, px >= $3.00 :  5%      Tier 2, px >= $3.00 : 10%
    $0.75 <= px < $3.00 : 20%      px < $0.75 : lesser of 75% or $0.15
Bands DOUBLE during 09:30-09:45 and 15:35-16:00.
"""
from datetime import time as dtime

import numpy as np
import pandas as pd
import pytest

from backtester.luld import (halt_flags, is_doubled, luld_band, reference_price)
from execution.halts import legacy_halt_flags

MID = dtime(11, 0)      # ordinary mid-session, no doubling
OPEN_ = dtime(9, 35)    # inside the opening doubling window
CLOSE_ = dtime(15, 40)  # inside the closing doubling window


@pytest.mark.parametrize("price,t,tier,expected", [
    # Tier 2 (the default for every gapper in this universe)
    (5.00, MID, 2, 0.10),
    (5.00, OPEN_, 2, 0.20),
    (5.00, CLOSE_, 2, 0.20),
    # Tier 1
    (5.00, MID, 1, 0.05),
    (5.00, OPEN_, 1, 0.10),
    # $0.75 - $3.00
    (2.00, MID, 2, 0.20),
    (2.00, OPEN_, 2, 0.40),
    # Sub-$0.75: lesser of 75% or $0.15/px
    (0.50, MID, 2, 0.30),      # 0.15/0.50 = 0.30 < 0.75
    (0.50, OPEN_, 2, 0.60),
    (0.10, MID, 2, 0.75),      # 0.15/0.10 = 1.50 -> capped at 0.75
])
def test_band_table(price, t, tier, expected):
    assert luld_band(price, t, tier=tier) == pytest.approx(expected)


@pytest.mark.parametrize("price,expected", [
    (3.00, 0.10),    # boundary: >= $3.00 takes the tier band
    (2.999, 0.20),
    (0.75, 0.20),    # boundary: >= $0.75 takes the mid band
    (0.749, pytest.approx(0.15 / 0.749)),
])
def test_price_boundaries(price, expected):
    assert luld_band(price, MID, tier=2) == pytest.approx(expected)


@pytest.mark.parametrize("t,doubled", [
    (dtime(9, 29), False),
    (dtime(9, 30), True),
    (dtime(9, 44), True),
    (dtime(9, 45), False),     # exclusive end
    (dtime(15, 34), False),
    (dtime(15, 35), True),
    (dtime(15, 59), True),
])
def test_doubling_windows(t, doubled):
    assert is_doubled(t) is doubled


def test_reference_price_is_trailing_mean():
    closes = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    assert reference_price(closes, 0) == 10.0
    assert reference_price(closes, 3, lookback=3) == pytest.approx(11.0)


def _frame(rows):
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])


def test_normal_gapper_open_is_not_a_halt_under_new_rule():
    """THE REGRESSION THAT STATES THE FIX.

    A 12% move inside the opening minute of a $5 microcap is ordinary volatility.
    The old flat-10% rule called it a halt and charged HALT_RESUME_SLIP=0.02
    against it, biasing every backtest pessimistic. The tiered rule (20% at the
    open for a Tier 2 name) correctly leaves it alone.
    """
    bars = _frame([
        ["09:30", 5.00, 5.70, 4.95, 5.60, 1_000_000],   # +12% intrabar
        ["09:31", 5.60, 5.70, 5.50, 5.65, 500_000],
        ["09:32", 5.65, 5.75, 5.60, 5.70, 400_000],
    ])
    assert bool(legacy_halt_flags(bars)[0]) is True     # old rule: false positive
    assert bool(halt_flags(bars)[0]) is False           # new rule: correct


def test_genuine_tape_gap_is_a_halt_under_both_rules():
    """A missing RTH minute is real halt evidence and must survive the fix."""
    bars = _frame([
        ["09:30", 5.00, 5.10, 4.95, 5.05, 1_000_000],
        ["09:34", 5.05, 5.15, 5.00, 5.10, 800_000],     # 4-minute gap
        ["09:35", 5.10, 5.20, 5.05, 5.15, 300_000],
    ])
    assert bool(legacy_halt_flags(bars)[1]) is True
    assert bool(halt_flags(bars)[1]) is True


def test_extreme_excursion_still_flags_under_new_rule():
    """The tiered rule is not simply permissive — a 60% minute on a $5 name
    exceeds even the doubled 20% band and is still flagged."""
    bars = _frame([
        ["09:30", 5.00, 5.10, 4.95, 5.05, 1_000_000],
        ["09:31", 5.05, 8.20, 5.00, 8.10, 900_000],     # +60% intrabar
        ["09:32", 8.10, 8.30, 8.00, 8.20, 400_000],
    ])
    assert bool(halt_flags(bars)[1]) is True
