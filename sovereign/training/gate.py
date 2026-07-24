"""Ignition gate for the self-play training loop.

This is the constitutional refusal, mirroring sovereign/autonomous/research_factory.py
+ config/autonomous.yml::live. Training a trading policy is model training, gated by
RISK_CONSTITUTION Art. 6 behind a CONFIRMED ledger verdict. The gate opens ONLY when
BOTH ignition flags in config/training.yml are true AND the corroborating runtime
guards pass. Config flags are necessary but not sufficient: the runtime guards can
only ADD refusal reasons, never open a gate the data doesn't support. This makes the
runner physically incapable of igniting a real cycle before TICK-024 + HYP-071-net.

Nothing here trades or mutates state — it reads config + the value board and returns
a verdict. The runner (scripts/sovereign_train.py) prints it loudly at startup.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "training.yml"


@dataclass
class GateStatus:
    """Result of evaluating the ignition gate. `open` is the ONLY thing the runner
    may use to decide whether a real cycle is permitted."""
    open: bool
    reasons: list[str] = field(default_factory=list)   # why the gate is CLOSED
    checks: dict = field(default_factory=dict)          # individual check results

    @property
    def mode(self) -> str:
        return "LIVE" if self.open else "SCAFFOLD/DRY"


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        # No silent mocking (CLAUDE.md): missing config is a hard stop, not a default.
        raise FileNotFoundError(
            f"training config not found: {path}. Cannot evaluate ignition gate."
        )
    return yaml.safe_load(path.read_text()) or {}


def _board_is_net(cfg: dict) -> tuple[bool, str]:
    """Corroborating runtime guard: is the HYP-071 value board computed on NET
    returns? If it still carries the gross-returns marker, the board is gross and
    the gate stays closed regardless of config flags. Returns (is_net, detail)."""
    vf = cfg.get("value_function", {})
    board_path = ROOT / vf.get("board_path", "")
    marker = vf.get("gross_marker_key", "gross_R_caveat")
    if not board_path.exists():
        return False, f"value board missing: {board_path}"
    try:
        board = json.loads(board_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"value board unreadable: {exc}"
    summary = board.get("summary", {})
    if summary.get(marker):
        return False, f"board carries gross marker '{marker}' (returns are GROSS, not net)"
    return True, "board carries no gross marker"


def evaluate_gate(config_path: Path | None = None) -> GateStatus:
    """Evaluate the ignition gate. Gate opens ONLY if every condition below holds."""
    cfg = load_config(config_path)
    ign = cfg.get("ignition", {})

    tick_024 = bool(ign.get("tick_024_carry_fix_landed", False))
    hyp_071 = bool(ign.get("hyp_071_net_confirmed", False))
    board_net, board_detail = _board_is_net(cfg)

    checks = {
        "tick_024_carry_fix_landed": tick_024,
        "hyp_071_net_confirmed": hyp_071,
        "value_board_is_net": board_net,
    }

    reasons: list[str] = []
    if not tick_024:
        reasons.append(
            "BLOCKER 8.1: TICK-024 carry-cost fix not landed "
            "(config ignition.tick_024_carry_fix_landed=false)"
        )
    if not hyp_071:
        reasons.append(
            "BLOCKER 8.2: HYP-071 net recompute not CONFIRMED "
            "(config ignition.hyp_071_net_confirmed=false)"
        )
    if not board_net:
        reasons.append(f"NET-RETURN GUARD: {board_detail}")

    is_open = tick_024 and hyp_071 and board_net
    return GateStatus(open=is_open, reasons=reasons, checks=checks)


def render_gate_banner(status: GateStatus) -> str:
    """Human-facing gate banner printed loudly at runner startup."""
    bar = "=" * 68
    icon = "\U0001F513 OPEN" if status.open else "\U0001F512 CLOSED"
    lines = [
        bar,
        f"  IGNITION GATE: {icon}   →   MODE: {status.mode}",
        bar,
    ]
    for name, ok in status.checks.items():
        lines.append(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    if not status.open:
        lines.append("  Gate held CLOSED. Running SCAFFOLD/DRY: full pipeline, NO")
        lines.append("  production policy trained, NO parameter update written.")
        lines.append("  Refusal reasons:")
        for r in status.reasons:
            lines.append(f"    - {r}")
    lines.append(bar)
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_gate_banner(evaluate_gate()))
