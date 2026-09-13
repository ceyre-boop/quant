"""Break-even lemma from the measured spread model (TICK-039). For a rule with R round trips per
trade-unit and the measured half-spread s(price, range, $vol), the net mean return is
    net = gross − 2·R·s − slip
so the minimum gross mean return per trade for net > 0 is 2·R·s + slip. Stated symbolically and
evaluated at the rule's declared typical (price, bar_range, minute $vol)."""
from __future__ import annotations

from dataclasses import dataclass

from backtester.realistic_fills import _half_spread_measured, MEASURED_SPREAD


@dataclass
class CostLemma:
    half_spread: float
    round_trips: float
    slip: float
    min_gross_per_trade: float
    statement: str


def break_even(price: float, bar_range_pct: float, minute_dollar_vol: float, round_trips: float = 1.0, slip: float = 0.0) -> CostLemma:
    s = _half_spread_measured(price, bar_range_pct, minute_dollar_vol)
    mg = 2 * round_trips * s + slip
    return CostLemma(s, round_trips, slip, mg,
                     f"net = gross − 2·{round_trips}·s − {slip}; s = clip(exp({MEASURED_SPREAD['intercept']} + {MEASURED_SPREAD['log_price']}·ln P "
                     f"+ {MEASURED_SPREAD['log_dollar_vol']}·ln $V + {MEASURED_SPREAD['log_bar_range']}·ln range), {MEASURED_SPREAD['floor']}, {MEASURED_SPREAD['cap']}) "
                     f"= {s:.5f} at P={price}, range={bar_range_pct}, $V={minute_dollar_vol:.0f} → gross must exceed {mg*100:.3f}% per trade")
