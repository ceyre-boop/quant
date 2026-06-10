"""ES/NQ config loader.

Sandbox-local: reads config/es_nq_params.yml ONLY. No forex/ICT/intelligence imports.
Every threshold in the es_nq system comes from here so what you backtest is what you
trade, and changes are made in one logged place (CLAUDE.md rule #4).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "config" / "es_nq_params.yml"
DATA_DIR = ROOT / "data" / "es_nq"
RESEARCH_DIR = ROOT / "data" / "research"


@lru_cache(maxsize=1)
def es_nq_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def contract_spec(instrument: str) -> dict:
    """{'underlying','dollars_per_point','tick'} for MES/MNQ (or its ES/NQ alias)."""
    p = es_nq_params()
    inst = {"ES": "MES", "NQ": "MNQ"}.get((instrument or "").upper(), (instrument or "").upper())
    return p["contracts"][inst]


def tick_value_usd(instrument: str) -> float:
    """Dollar value of one tick (MES $1.25, MNQ $0.50)."""
    spec = contract_spec(instrument)
    return round(spec["tick"] * spec["dollars_per_point"], 4)


def round_turn_cost_usd(instrument: str, n_contracts: int = 1, *, stop_fill: bool = False) -> float:
    """Dollar cost of one round trip under the brief's cost model.

    commission $0.35/side + slippage 0.25 ticks on entry and 0.25 (target) or
    0.5 (stop) ticks on exit. `stop_fill=True` uses the stop-exit slippage.
    """
    p = es_nq_params()
    spec = contract_spec(instrument)
    c = p["costs"]
    exit_slip_ticks = c["slippage_ticks_stop"] if stop_fill else c["slippage_ticks_entry"]
    slip_pts = (c["slippage_ticks_entry"] + exit_slip_ticks) * spec["tick"]
    slip_usd = slip_pts * spec["dollars_per_point"]
    comm_usd = 2.0 * c["commission_per_side_usd"]
    return round((slip_usd + comm_usd) * n_contracts, 4)


def legacy_round_turn_cost_usd(instrument: str, n_contracts: int = 1) -> float:
    """Harsher sandbox cost model (1 tick/side + $0.74 RT) — robustness reporting only."""
    p = es_nq_params()
    spec = contract_spec(instrument)
    c = p["costs"]
    slip_pts = c["legacy_slippage_ticks_per_side"] * spec["tick"] * 2
    slip_usd = slip_pts * spec["dollars_per_point"]
    return round((slip_usd + c["legacy_commission_per_round_turn_usd"]) * n_contracts, 4)
