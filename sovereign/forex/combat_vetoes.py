"""Combat vetoes — ANALYSIS INSTRUMENT ONLY. NOT a validated production gate (config enabled=false).

⚠️  Honest replay (scripts/validate_combat_vetoes.py) showed these 4 vetoes, applied as blanket
    force-skips, are NET -143R — they forgo more winners than losses they avoid. The combat-rules
    forensic's "-273R recoverable" counted only the LOSING tail of each condition (survivorship bias).
    See COMBAT_VETOES_FINDING.md. This module is retained to REPRODUCE that analysis, not to gate
    live trades. Do not wire it into forex_live_scan without a proper net-expectancy re-validation.

Implements the four pre-trade conditions exactly as the forensic classifier defines them
(trade_forensics._classify_failure), for use by the replay/validation tooling:

  C-001 MACRO_AGAINST   real_rate_diff sign opposes direction (deadband 0.2)   -158.37R
  C-005 RATE_SIGNAL_WEAK |real_rate_diff| < 0.5                                  -29.97R
  C-006 VOLATILITY_FLOOR atr_14d_pct < 0.006                                     -18.75R
  C-003 COUNTER_MOMENTUM momentum_63d * direction < -0.01                        -55.21R

Pure function — independently testable. `direction` is +1 long / -1 short. A veto input of None
means "can't assess this rule" (that single rule is skipped, never silently passes the others).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

_CONFIG_PATH = Path(__file__).parent / "config" / "combat_vetoes.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"combat_vetoes.yaml missing at {_CONFIG_PATH}")
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("combat_vetoes.yaml is empty")
    return cfg


@dataclass(frozen=True)
class VetoHit:
    rule_id: str
    reason: str


def _enabled(cfg: dict, rule_id: str) -> bool:
    return cfg.get("enabled", True) and cfg.get("rules", {}).get(rule_id, {}).get("enabled", False)


def evaluate(real_rate_diff: Optional[float], momentum_63d: Optional[float],
             atr_14d_pct: Optional[float], direction: int, cfg: Optional[dict] = None) -> list:
    """Return the list of VetoHits a setup triggers. Empty list = passes the gate."""
    cfg = cfg or load_config()
    if not cfg.get("enabled", True):
        return []
    rules = cfg["rules"]
    hits = []

    # C-001 MACRO_AGAINST — real rate differential favors the opposite direction.
    if _enabled(cfg, "C-001") and real_rate_diff is not None:
        deadband = rules["C-001"]["deadband"]
        macro_sign = int(math.copysign(1, real_rate_diff)) if abs(real_rate_diff) > deadband else 0
        if macro_sign * direction == -1:
            hits.append(VetoHit("C-001", f"MACRO_AGAINST (real_rate_diff={real_rate_diff:+.3f} vs dir {direction:+d})"))

    # C-005 RATE_SIGNAL_WEAK — rate differential too narrow to sustain the macro edge.
    if _enabled(cfg, "C-005") and real_rate_diff is not None:
        if abs(real_rate_diff) < rules["C-005"]["weak_rate"]:
            hits.append(VetoHit("C-005", f"RATE_SIGNAL_WEAK (|real_rate_diff|={abs(real_rate_diff):.3f} < {rules['C-005']['weak_rate']})"))

    # C-006 VOLATILITY_FLOOR — ATR compressed; can't deliver the range to hit targets.
    if _enabled(cfg, "C-006") and atr_14d_pct is not None:
        if atr_14d_pct < rules["C-006"]["atr_floor"]:
            hits.append(VetoHit("C-006", f"VOLATILITY_FLOOR (atr_14d_pct={atr_14d_pct:.4f} < {rules['C-006']['atr_floor']})"))

    # C-003 COUNTER_MOMENTUM — 63d price momentum opposes the trade by >1%.
    if _enabled(cfg, "C-003") and momentum_63d is not None:
        if momentum_63d * direction < -rules["C-003"]["momentum_opp"]:
            hits.append(VetoHit("C-003", f"COUNTER_MOMENTUM (momentum_63d={momentum_63d:+.4f} vs dir {direction:+d})"))

    return hits


def vetoed(real_rate_diff, momentum_63d, atr_14d_pct, direction, cfg=None) -> bool:
    return len(evaluate(real_rate_diff, momentum_63d, atr_14d_pct, direction, cfg)) > 0
