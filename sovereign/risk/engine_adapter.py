"""Adapter — wire the live paths (DecisionChain, forex scan) to the risk engine.

The engine is the SOLE sizing authority. This assembles a RiskState from live system data and
returns the engine's RiskDecision. Every external read is guarded with a safe default; the engine
fails safe (halt/zero), never over-sizes. Use `.final_risk_pct` as the authoritative risk.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from sovereign.risk.config.loader import load_risk_config
from sovereign.risk.risk_engine import decide
from sovereign.risk.risk_state import RiskState, Signal

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _pooled_edge_stats() -> dict:
    """One strategy 'forex_macro' from the real v015 pool (n=103) for Kelly."""
    try:
        import numpy as np
        from sovereign.risk.monte_carlo_prop import load_pool
        pnls, _, _ = load_pool()
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        p = len(wins) / len(pnls)
        b = (wins.mean() / abs(losses.mean())) if len(losses) and losses.mean() != 0 else 2.0
        return {"forex_macro": {"win_rate": float(p), "payoff": float(b), "n_trades": int(len(pnls))}}
    except Exception:
        return {}


def _read_peak(default: float) -> float:
    try:
        d = json.loads((ROOT / "data" / "agent" / "equity_peak.json").read_text())
        return float(d.get("peak_equity") or d.get("peak") or default)
    except Exception:
        return default


def _read_mc_breach():
    try:
        d = json.loads((ROOT / "data" / "risk" / "prop_monte_carlo.json").read_text())
        return float(d["horizons"]["90"]["p_fail"])     # conservative breach proxy
    except Exception:
        return None


def grade_from_risk(risk_pct: float) -> str:
    """Map a path's intended risk_pct onto a grade base (preserves conviction as the base)."""
    if risk_pct >= 0.009:
        return "A+"
    if risk_pct >= 0.006:
        return "A"
    if risk_pct >= 0.0035:
        return "B"
    return "C"


def size(pair, direction, entry, stop, *, grade="B", equity=None, point_value=1.0,
         open_positions=None, threat_score=0.0, regime="UNKNOWN", health_ok=True, cfg=None,
         notes=None):
    cfg = cfg or load_risk_config()
    start = float(cfg["prop"]["account_size"])
    equity = float(equity) if equity else start
    peak = _read_peak(max(equity, start))
    dd_s, dd_t = RiskState.derive_drawdowns(equity, peak, start)
    state = RiskState(
        equity=equity, peak_equity=peak, starting_balance=start,
        daily_realized_pnl=0.0, daily_open_pnl=0.0, open_positions=open_positions or [],
        drawdown_static=dd_s, drawdown_trailing=dd_t, regime=regime, threat_score=threat_score,
        edge_stats=_pooled_edge_stats(), mc_breach_prob=_read_mc_breach(), health_ok=health_ok)
    d = 1 if str(direction).upper() == "LONG" else -1
    sig = Signal(pair, d, float(entry), float(stop), grade, strategy="forex_macro", point_value=point_value,
                 notes=notes or {})
    return decide(sig, state, cfg)
