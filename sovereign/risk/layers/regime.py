"""Layer 5 — REGIME MODULATION. Maps AlexandrianLibrary threat_score -> factor in [0,1].

Favorable regime + low threat → 1.0; rising threat → progressively lower (config table). This is
where the system sizes down in the regimes where the edge historically fails. Never > 1.0.
"""
from __future__ import annotations


def factor(signal, state, cfg) -> float:
    r = cfg["regime"]
    threat = state.threat_score
    if threat is None:
        return r["default_factor"]
    table = r["threat_table"]                 # ascending threat, non-increasing factor
    f = table[0][1]
    for thr, fac in table:
        if threat >= thr:
            f = fac
        else:
            break
    return max(r["factor_floor"], min(1.0, f))
