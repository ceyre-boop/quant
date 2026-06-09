"""Futures sandbox config loader.

Sandbox-local: reads config/futures_params.yml ONLY. No forex/ICT/intelligence imports.
Every threshold in the futures sandbox comes from here so what you backtest is what
you trade, and changes are made in one logged place (CLAUDE.md rule #4).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "config" / "futures_params.yml"


@lru_cache(maxsize=1)
def futures_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def contract_spec(instrument: str) -> dict:
    """{'underlying','dollars_per_point','tick'} for MES/MNQ (or its ES/NQ alias)."""
    p = futures_params()
    inst = {"ES": "MES", "NQ": "MNQ"}.get((instrument or "").upper(), (instrument or "").upper())
    return p["contracts"][inst]


def tick_value_usd(instrument: str) -> float:
    """Dollar value of one tick (MES $1.25, MNQ $0.50)."""
    spec = contract_spec(instrument)
    return round(spec["tick"] * spec["dollars_per_point"], 4)


def round_turn_cost_usd(instrument: str, n_contracts: int = 1) -> float:
    """Dollar cost of one round trip: slippage (both sides) + commission, per spec.

    Independent of price (slippage is in ticks), so it's the honest floor a scalp
    must clear. Mirrors futures_backtester._cost_fraction's components in $ terms.
    """
    p = futures_params()
    spec = contract_spec(instrument)
    c = p["costs"]
    slip_pts = c["slippage_ticks_per_side"] * spec["tick"] * 2          # entry + exit
    slip_usd = slip_pts * spec["dollars_per_point"]
    return round((slip_usd + c["commission_per_round_turn_usd"]) * n_contracts, 4)
