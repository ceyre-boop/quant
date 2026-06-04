"""Layer 1 — BASE SIZE. Grade-scaled conviction base (the ONLY layer that SETS risk).

Grade levels live in risk_config.yaml (mirror ict/micro_risk._GRADE_RISK; a parity test enforces
no drift). Every other layer can only reduce this. Hard-clamped to the configured ceiling.
FAIL LOUD on an unknown grade — never silently default to a larger size.
"""
from __future__ import annotations


def base_size(signal, cfg) -> float:
    b = cfg["base"]
    grade = signal.grade
    levels = b["grade_risk"]
    if grade not in levels:
        raise ValueError(f"Layer 1 base_size: unknown grade {grade!r}; "
                         f"expected one of {sorted(levels)}. Refusing to size.")
    return min(float(levels[grade]), float(b["ceiling"]))
