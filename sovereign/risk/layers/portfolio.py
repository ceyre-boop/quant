"""Layer 6 — PORTFOLIO / CVaR CEILING. Two sub-caps; take the lower.

(a) Correlation heat: existing open risk weighted by stressed correlation (ρ can spike toward 1
    under stress); cap new risk so total correlated heat <= max_portfolio_heat.
(b) CVaR (EMPIRICAL, never normal): bootstrap the real v015 return pool, estimate the tail loss
    per unit risk, and cap new risk so portfolio CVaR <= cvar_limit.

Falls back to the correlation-heat cap alone if the return pool is unavailable (noted by returning
+inf for the CVaR sub-cap). Returns an absolute risk_pct ceiling (binds via min()).
"""
from __future__ import annotations

import math


def _heat_ceiling(signal, state, pcfg) -> float:
    max_heat = pcfg["max_portfolio_heat"]
    stress = pcfg["correlation_stress_mult"]
    unknown = pcfg.get("unknown_correlation", 1.0)
    corr = state.correlation_matrix or {}
    existing_weighted = 0.0
    for pos in state.open_positions:
        rho = corr.get((signal.instrument, pos.instrument),
                       corr.get((pos.instrument, signal.instrument), unknown))
        stressed = min(1.0, abs(float(rho)) * stress)
        existing_weighted += float(pos.risk_pct_at_entry) * stressed
    return max(0.0, max_heat - existing_weighted)


def _cvar_ceiling(signal, state, pcfg) -> float:
    try:
        from sovereign.risk.monte_carlo_prop import load_pool
        import numpy as np
        pnls, _, _ = load_pool()
    except (Exception, SystemExit):
        return math.inf                          # no return history → CVaR deferred (heat-only)

    alpha = pcfg["cvar_alpha"]
    q = np.percentile(pnls, alpha * 100.0)
    tail = pnls[pnls <= q]
    cvar_unit = -float(tail.mean()) if len(tail) else 0.0   # positive expected tail loss
    embedded = pcfg.get("cvar_embedded_risk", 0.0075)
    if cvar_unit <= 0 or embedded <= 0:
        return math.inf
    cvar_per_risk = cvar_unit / embedded         # tail loss per unit of risk_pct (R-space)
    existing_risk = sum(float(p.risk_pct_at_entry) for p in state.open_positions)
    budget = pcfg["cvar_limit"] / cvar_per_risk - existing_risk
    return max(0.0, budget)


def ceiling(signal, state, cfg) -> float:
    pcfg = cfg["portfolio"]
    heat_c = _heat_ceiling(signal, state, pcfg)
    if not pcfg.get("enabled", True):
        return heat_c
    return min(heat_c, _cvar_ceiling(signal, state, pcfg))
