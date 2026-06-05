"""Risk engine orchestrator — the SOLE sizing authority.

decide(signal, state, cfg) composes the cascade:
    desired = base × vol_f × dd_f × regime_f          (modulators COMPOUND)
    capped  = min(desired, kelly, portfolio, prop)     (ceilings bind via min)
    final   = 0 if any hard gate fires else capped

INVARIANT: final_risk <= base_risk and final_risk <= every ceiling, always. Every operation can
only REDUCE risk. Modulators are clamped to [0,1] defensively; a layer that's not built yet is
identity (factor 1.0 / ceiling +inf); a layer that EXISTS but errors raises (fail loud, never a
silent risk-increasing default). The engine NEVER executes — it sizes and constrains only.
"""
from __future__ import annotations

import json
import math
from importlib import import_module
from pathlib import Path

from sovereign.risk.config.loader import load_risk_config
from sovereign.risk.layers.base_size import base_size
from sovereign.risk.layers.gates import run_gates
from sovereign.risk.layers.prop import prop_ceiling
from sovereign.risk.risk_state import RiskDecision

ROOT = Path(__file__).resolve().parents[2]

# Layers added in later commits — identity until present.
_MODULATORS = ("volatility", "drawdown", "regime")   # factor() in [0,1], compound
_CEILINGS = ("kelly", "portfolio")                   # ceiling() absolute risk_pct, min()


def _modulator(name, signal, state, cfg) -> float:
    try:
        mod = import_module(f"sovereign.risk.layers.{name}")
    except ImportError:
        return 1.0                                   # not built yet → identity
    f = float(mod.factor(signal, state, cfg))        # errors here propagate (fail loud)
    if not (0.0 <= f <= 1.0):
        return max(0.0, min(1.0, f))                 # clamp: a modulator can never amplify
    return f


def _ceiling(name, signal, state, cfg) -> float:
    try:
        mod = import_module(f"sovereign.risk.layers.{name}")
    except ImportError:
        return math.inf                              # not built yet → no constraint
    c = float(mod.ceiling(signal, state, cfg))       # errors propagate (fail loud)
    return max(0.0, c)


def _safe_append(record: dict, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass  # logging must never block a sizing decision


def _convert_size(equity, final_risk, signal, cfg) -> float:
    """risk_pct -> tradable units via (equity·risk)/(stop_distance·point_value). Round DOWN."""
    stop_distance = signal.stop_distance
    pv = signal.point_value or cfg["sizing"]["default_point_value"]
    if stop_distance <= 0 or pv <= 0:
        return 0.0
    raw = (equity * final_risk) / (stop_distance * pv)
    gran = cfg["sizing"]["lot_granularity"]
    return math.floor(raw / gran) * gran             # never round up


def decide(signal, state, cfg=None) -> RiskDecision:
    cfg = cfg or load_risk_config()

    # ── Layer 0: hard gates ──────────────────────────────────────────────────
    halt_reason = run_gates(signal, state, cfg)

    # ── Layer 1: base ────────────────────────────────────────────────────────
    base = base_size(signal, cfg)

    # ── Layers 3/4/5: modulators (compound) ──────────────────────────────────
    vol_f = _modulator("volatility", signal, state, cfg)
    dd_f = _modulator("drawdown", signal, state, cfg)
    regime_f = _modulator("regime", signal, state, cfg)
    modulated = base * vol_f * dd_f * regime_f

    # ── Layers 2/6/7: ceilings (min) ─────────────────────────────────────────
    kelly_c = _ceiling("kelly", signal, state, cfg)
    portfolio_c = _ceiling("portfolio", signal, state, cfg)
    prop_c = prop_ceiling(signal, state, cfg)

    capped = min(modulated, kelly_c, portfolio_c, prop_c)
    final_risk = 0.0 if halt_reason else max(0.0, capped)

    # ── binding constraint ───────────────────────────────────────────────────
    candidates = {"base_modulated": modulated, "kelly": kelly_c,
                  "portfolio": portfolio_c, "prop": prop_c}
    if halt_reason:
        binding = f"halt:{halt_reason.split(':')[0]}"
    else:
        binding = min(candidates, key=lambda k: candidates[k])

    final_size = 0.0 if halt_reason else _convert_size(state.equity, final_risk, signal, cfg)

    modulators = {"vol": round(vol_f, 4), "drawdown": round(dd_f, 4), "regime": round(regime_f, 4)}
    # NOTE: invariant-critical fields are stored at FULL precision (no rounding) so the
    # invariant (final <= base, final <= every ceiling) holds exactly on the reported numbers.
    layer_budgets = {
        "base": base,
        "modulated": modulated,
        "kelly_ceiling": kelly_c,
        "portfolio_ceiling": portfolio_c,
        "prop_ceiling": prop_c,
        "capped": capped,
        "final": final_risk,
    }

    if halt_reason:
        reasoning = f"HALT — {halt_reason}. final_risk=0, size=0."
    else:
        reasoning = (f"base {base:.3%} (grade {signal.grade}) × vol {vol_f:.2f} × dd {dd_f:.2f} × "
                     f"regime {regime_f:.2f} = {modulated:.3%}; "
                     f"{binding} cap binding → final {final_risk:.3%}, size {final_size:g}.")

    decision = RiskDecision(
        final_size=final_size, final_risk_pct=final_risk, base_risk_pct=base,
        binding_constraint=binding, layer_budgets=layer_budgets, modulators=modulators,
        halt=bool(halt_reason), halt_reason=halt_reason, reasoning=reasoning,
        instrument=signal.instrument, strategy=signal.strategy,
    )

    if cfg["audit"].get("enabled", True):
        _safe_append({
        "timestamp": decision.timestamp, "instrument": decision.instrument,
        "strategy": decision.strategy, "grade": signal.grade,
        "final_risk_pct": decision.final_risk_pct, "base_risk_pct": decision.base_risk_pct,
        "final_size": decision.final_size, "binding_constraint": decision.binding_constraint,
        "halt": decision.halt, "halt_reason": decision.halt_reason,
        "layer_budgets": decision.layer_budgets, "modulators": decision.modulators,
        "reasoning": decision.reasoning,
        "notes": signal.notes,
    }, ROOT / cfg["audit"]["decisions_log"])

    return decision


class RiskEngine:
    """Thin OO wrapper for call sites that prefer an object."""
    def __init__(self, cfg=None):
        self.cfg = cfg or load_risk_config()

    def decide(self, signal, state):
        return decide(signal, state, self.cfg)
