"""
Forex Turtle-compliance guardrails and scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class ForexComplianceConfig:
    strict_mode: bool = False
    rule_set_version: str = "turtle-v1"
    use_atr_stops: bool = True
    use_trailing_exits: bool = True
    use_hard_tp_ladders: bool = False
    use_donchian_entries: bool = True
    use_donchian_exits: bool = True
    max_risk_per_trade_pct: float = 0.01
    max_shared_jpy_positions: int = 2
    allow_pyramiding: bool = True
    max_pyramid_units: int = 4
    use_macro_overlay: bool = False
    min_live_score: int = 100

    def validate_startup(self) -> None:
        if not self.strict_mode:
            return
        if not self.use_atr_stops:
            raise ValueError("Strict mode requires ATR-based stops.")
        if self.max_risk_per_trade_pct > 0.01:
            raise ValueError("Strict mode requires max 1% risk per trade.")
        if self.use_hard_tp_ladders:
            raise ValueError("Strict mode forbids hard TP ladders as mandatory exits.")
        if not self.use_donchian_entries or not self.use_donchian_exits:
            raise ValueError("Strict mode requires Donchian entry/exit mechanics.")
        if self.max_shared_jpy_positions < 1:
            raise ValueError("Correlation cap must be at least 1.")
        if self.allow_pyramiding and self.max_pyramid_units < 1:
            raise ValueError("Pyramiding requires positive unit caps.")


def score_compliance(config: ForexComplianceConfig) -> Dict[str, Any]:
    checks = {
        "strict_mode_enabled": config.strict_mode,
        "atr_stops": config.use_atr_stops,
        "trailing_exits": config.use_trailing_exits,
        "no_hard_tp_ladders": not config.use_hard_tp_ladders,
        "risk_cap_1pct": config.max_risk_per_trade_pct <= 0.01,
        "donchian_entries": config.use_donchian_entries,
        "donchian_exits": config.use_donchian_exits,
        "correlation_cap": config.max_shared_jpy_positions >= 1,
        "pyramiding_unit_cap": (not config.allow_pyramiding) or (1 <= config.max_pyramid_units <= 4),
        "macro_overlay_optional": isinstance(config.use_macro_overlay, bool),
    }
    passed = sum(1 for v in checks.values() if v)
    score = int(round((passed / len(checks)) * 100))
    return {
        "rule_set_version": config.rule_set_version,
        "strict_mode": config.strict_mode,
        "score": score,
        "status": "pass" if score >= config.min_live_score else "fail",
        "checks": checks,
    }


def block_live_mode_if_needed(mode: str, report: Dict[str, Any]) -> None:
    if mode == "live" and report.get("status") != "pass":
        raise RuntimeError(
            f"Live mode blocked by forex compliance gate (score={report.get('score')})."
        )
