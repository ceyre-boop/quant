"""sovereign/training/state_space.py — the canonical state vector S(t) (spec §4.1).

ONE builder function, ONE fixed dimension order. Everything downstream (the gym in
trading_env.py, the value scorer, any future policy) consumes S(t) through
`build_state()` — never hand-rolls its own feature ordering. STATE_DIMS is the
single source of truth for "what is dimension i"; a test asserts it can never
silently drift (test_state_space.py).

Freeze-safe: reads nothing, mutates nothing, imports nothing from ict/ or the
frozen execution path. Pure feature transform.
"""
from __future__ import annotations

import numpy as np

# Locked order — 8 dims, exactly this order, forever. Changing this tuple is a
# breaking schema change and must bump a version marker anywhere S(t) is persisted.
STATE_DIMS: tuple[str, ...] = (
    "hold_frac",          # hold_count / hold_limit, clipped to [0, 1]
    "excursion_pct",      # direction * (close - entry_price) / entry_price
    "atr_pct",            # today's ATR as a % of price
    "rsi_14",             # 14-period RSI, [0, 100]
    "carry_alignment",    # +1 aligned, -1 opposed, 0 no active carry signal
    "rate_diff_z",        # interest-rate-differential z-score (IRP), 0.0 if unavailable
    "cot_z",              # COT positioning z-score, 0.0 if unavailable
    "drawdown_from_peak", # direction-adjusted retracement from best excursion since entry
)
NUM_STATE_DIMS = len(STATE_DIMS)
assert NUM_STATE_DIMS == 8, "spec locks S(t) at exactly 8 dimensions"

# Fields build_state() requires on the input record. hold_count/hold_limit combine
# into hold_frac; rate_diff_z/cot_z are optional (external-data-dependent) and
# default to 0.0 when absent or None — everything else is required.
_REQUIRED_FIELDS = (
    "hold_count", "hold_limit", "excursion_pct", "atr_pct", "rsi_14",
    "carry_alignment", "drawdown_from_peak",
)
_OPTIONAL_FIELDS = ("rate_diff_z", "cot_z")


def _hold_frac(hold_count: float, hold_limit: float) -> float:
    if hold_limit <= 0:
        return 1.0
    return float(np.clip(hold_count / hold_limit, 0.0, 1.0))


def build_state(record: dict) -> np.ndarray:
    """Build S(t) from any trade/bar record exposing the required fields.

    `record` is a plain dict (a decision-log entry, a backtest bar row, a live
    position snapshot) — build_state() only cares that the keys exist, not where
    they came from. Missing required keys raise KeyError (no silent mocking);
    missing optional keys (rate_diff_z, cot_z) default to 0.0.

    Returns a float64 array of shape (8,) in STATE_DIMS order.
    """
    missing = [f for f in _REQUIRED_FIELDS if f not in record]
    if missing:
        raise KeyError(f"build_state: record missing required field(s): {missing}")

    hold_frac = _hold_frac(float(record["hold_count"]), float(record["hold_limit"]))
    rate_diff_z = record.get("rate_diff_z")
    cot_z = record.get("cot_z")

    values = {
        "hold_frac": hold_frac,
        "excursion_pct": float(record["excursion_pct"]),
        "atr_pct": float(record["atr_pct"]),
        "rsi_14": float(record["rsi_14"]),
        "carry_alignment": float(record["carry_alignment"]),
        "rate_diff_z": float(rate_diff_z) if rate_diff_z is not None else 0.0,
        "cot_z": float(cot_z) if cot_z is not None else 0.0,
        "drawdown_from_peak": float(record["drawdown_from_peak"]),
    }
    return np.array([values[name] for name in STATE_DIMS], dtype=np.float64)


def carry_alignment(direction: int, carry_signal: int) -> float:
    """+1.0 if the trade direction matches the active carry signal, -1.0 if it
    opposes it, 0.0 if there is no active carry signal (signal == 0)."""
    if carry_signal == 0:
        return 0.0
    return 1.0 if int(direction) == int(carry_signal) else -1.0


def drawdown_from_peak(direction: int, best_price: float, close_now: float) -> float:
    """Direction-adjusted retracement from the best excursion seen since entry, as
    a fraction of best_price. 0.0 at the peak; grows as price gives back gains."""
    if best_price == 0:
        return 0.0
    if direction >= 0:
        return float(max(0.0, (best_price - close_now) / best_price))
    return float(max(0.0, (close_now - best_price) / best_price))
