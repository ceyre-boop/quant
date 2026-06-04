"""Layer 2 — KELLY CEILING. Quarter-Kelly from live edge stats, hard-guarded.

Reuses sovereign/risk/kelly_engine.py (fractional_kelly + hoeffding_win_rate). Kelly is
hypersensitive, so:
  • n_trades < min_sample  → fixed_fractional_floor (don't trust Kelly on thin data)
  • full_kelly <= 0 (no edge) → 0 (don't trade a non-edge)
  • clamp to [0, hard_cap] so a bad estimate can't explode
Missing edge stats → fixed_fractional_floor (a known thin-data condition; only ever REDUCES vs base).
Returns an absolute risk_pct ceiling (binds via min()).
"""
from __future__ import annotations

from sovereign.risk.kelly_engine import fractional_kelly, hoeffding_win_rate


def ceiling(signal, state, cfg) -> float:
    k = cfg["kelly"]
    floor = k["fixed_fractional_floor"]
    hard_cap = k["hard_cap"]

    stats = (state.edge_stats or {}).get(signal.strategy)
    if not stats:
        return floor                                   # no stats → thin-data floor

    n = int(stats.get("n_trades", 0))
    p = float(stats.get("win_rate", 0.0))
    b = float(stats.get("payoff", 0.0))                # avg_win_R / avg_loss_R

    if n < k["min_sample"]:
        return floor                                   # distrust Kelly on < min_sample trades
    if b <= 0:
        return 0.0

    p_adj = hoeffding_win_rate(p, n)                    # conservative lower-bound win rate
    full_kelly = (p_adj * b - (1.0 - p_adj)) / b
    if full_kelly <= 0:
        return 0.0                                     # no edge → don't trade

    frac = fractional_kelly(p_adj, b, 1.0, fraction=k["fraction"], floor=0.0, ceiling=hard_cap)
    return max(0.0, min(frac, hard_cap))
