"""Risk engine layers. Each is a PURE function (signal, state, config) -> float.

  gates      : 0.0 (halt) | math.inf (no constraint)
  modulators : factor in [0, 1]   (volatility, drawdown, regime)  — compound multiplicatively
  base+ceil  : absolute risk_pct >= 0  (base, kelly, portfolio, prop) — bind via min()

No layer reaches into global state; everything comes via RiskState. Independently testable.
"""
