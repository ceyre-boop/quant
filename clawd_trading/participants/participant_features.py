"""
Participant Feature Extraction for Clawd Trading

Extracts microstructure features from market data for participant classification.
Integrates with Layer 1 feature builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ParticipantFeatureVector:
    """Microstructure features for participant classification."""

    orderflow_velocity: float
    sweep_intensity: float
    absorption_ratio: float
    spread_pressure: float
    liquidity_removal_rate: float
    volatility_reaction: float
    time_of_day_bias: str
    news_window_behavior: str
    metadata: Dict[str, Any]


def extract_participant_features(
    tick_data: Dict[str, Any], time_of_day: str = "all_day", news_window: str = "none"
) -> ParticipantFeatureVector:
    """
    Extract participant features from tick/quote data.

    Args:
        tick_data: Dictionary with bid/ask/volume data
        time_of_day: 'open', 'mid', 'close', or 'all_day'
        news_window: 'pre', 'during', 'post', or 'none'

    Returns:
        ParticipantFeatureVector for classification
    """
    # Extract from tick data
    bids = tick_data.get("bids", [])
    asks = tick_data.get("asks", [])
    trades = tick_data.get("trades", [])

    # Orderflow velocity: aggressive orders per second
    aggressive_orders = len([t for t in trades if t.get("aggressive", False)])
    time_window = tick_data.get("time_window", 1.0)
    orderflow_velocity = aggressive_orders / max(time_window, 1e-9)

    # Sweep intensity: large market orders hitting multiple levels
    sweep_events = tick_data.get("sweep_events", [])
    sweep_intensity = sum(e.get("size", 0) for e in sweep_events) / max(time_window, 1e-9)

    # Absorption ratio: passive fills / aggressive volume
    passive_fill_volume = sum(t.get("size", 0) for t in trades if not t.get("aggressive", False))
    aggressive_volume = sum(t.get("size", 0) for t in trades if t.get("aggressive", False))
    absorption_ratio = passive_fill_volume / max(aggressive_volume, 1e-9)

    # Spread pressure
    bid_pressure = sum(b.get("size", 0) for b in bids[:3])
    ask_pressure = sum(a.get("size", 0) for a in asks[:3])
    spread_pressure = (bid_pressure - ask_pressure) / max(abs(bid_pressure) + abs(ask_pressure), 1e-9)

    # Liquidity removal rate
    book_depletion = tick_data.get("book_depletion", 0.0)
    liquidity_removal_rate = book_depletion / max(time_window, 1e-9)

    # Volatility reaction
    short_vol = tick_data.get("short_horizon_vol", 0.0)
    baseline_vol = tick_data.get("baseline_vol", 1e-9)
    volatility_reaction = short_vol / baseline_vol

    return ParticipantFeatureVector(
        orderflow_velocity=float(orderflow_velocity),
        sweep_intensity=float(sweep_intensity),
        absorption_ratio=float(absorption_ratio),
        spread_pressure=float(spread_pressure),
        liquidity_removal_rate=float(liquidity_removal_rate),
        volatility_reaction=float(volatility_reaction),
        time_of_day_bias=(time_of_day if time_of_day in {"open", "mid", "close", "all_day"} else "all_day"),
        news_window_behavior=(news_window if news_window in {"pre", "during", "post", "none"} else "none"),
        metadata={"source": "clawd_trading", "ticks_processed": len(trades)},
    )


def extract_from_layer1_context(
    layer1_output: Dict[str, Any],
) -> ParticipantFeatureVector:
    """
    Extract features from Layer 1 output (Hard Constraints results).

    This integrates with your existing Layer 1 system.
    """
    # Map Layer 1 outputs to participant features
    tick_data = {
        "bids": layer1_output.get("bids", []),
        "asks": layer1_output.get("asks", []),
        "trades": layer1_output.get("trades", []),
        "time_window": layer1_output.get("time_window", 1.0),
        "book_depletion": layer1_output.get("book_depletion", 0.0),
        "short_horizon_vol": layer1_output.get("volatility", 0.0),
        "baseline_vol": layer1_output.get("baseline_volatility", 1e-9),
    }

    time_of_day = layer1_output.get("session", "all_day")
    news_window = layer1_output.get("news_window", "none")

    return extract_participant_features(tick_data, time_of_day, news_window)
