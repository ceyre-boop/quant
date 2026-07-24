"""Ignition gate for the self-play training loop.

This is the constitutional refusal, mirroring sovereign/autonomous/research_factory.py
+ config/autonomous.yml::live. Training a trading policy is model training, gated by
RISK_CONSTITUTION Art. 6 behind a CONFIRMED ledger verdict. The gate opens ONLY when
BOTH ignition flags in config/training.yml are true AND the corroborating runtime
guards pass. Config flags are necessary but not sufficient: the runtime guards can
only ADD refusal reasons, never open a gate the data doesn't support. This makes the
runner physically incapable of igniting a real cycle before TICK-024 + HYP-071-net.

HYP-071 REVIVAL GUARD (added post HYP-071-GOVFLAG, ledger 2026-07-24): HYP-071 is
adjudicated METRIC_ARTIFACT (2026-06-30, hash-locked prereg 3d500bda…, addendum
c1fab807…) — the flaw is structural to the value metric itself (EXIT_NOW dominance
is a forecast-variance artifact, not an edge), not to the cost model. A recompute on
NET returns does not touch that flaw, so `hyp_071_net_confirmed=true` alone must
never be trusted to mean HYP-071 is safe to train on — the 2026-07-23 recompute
(2be5726) proved a mere rerun can flip a plausible-looking number without curing the
underlying artifact (see HYP-071-GOVFLAG). The gate additionally requires a FRESH
pre-registration (a prereg hash distinct from the original locked v1/v2/addendum
hashes) backing a NEW ledger adjudication that flips HYP-071's status to CONFIRMED,
dated after the original verdict. Reusing the killed prereg, or the METRIC_ARTIFACT /
GOVFLAG entries themselves, does NOT satisfy this — the gate fails closed and loud.

Nothing here trades or mutates state — it reads config + the value board + the
hypothesis ledger (read-only) and returns a verdict. The runner
(scripts/sovereign_train.py) prints it loudly at startup.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "training.yml"
DEFAULT_LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"

HYP071_ID = "HYP-071"
# The original 2026-06-30 METRIC_ARTIFACT verdict date. Any revival adjudication
# must be dated strictly after this.
HYP071_ORIGINAL_VERDICT_DATE = "2026-06-30"
# The hash-locked prereg family behind the killed verdict (data/research/preregister/
# HYP-071_tabular_exit_value.v1.yaml, .yaml [v2], HYP-071_interpretation_notes.yaml).
# A revival must NOT reuse any of these — that's a rerun, not a fresh registration.
HYP071_LOCKED_PREREG_HASHES = frozenset({
    "c4f29ac387669fc77ac33f1d2570042898d8f81bc0409e1fd0e7d57ba9a41546",  # v1 prereg
    "3d500bda3249c4615698ce311a7cbad41a35600a23abd2a4ea4526416eac06a4",  # v2 prereg
    "c1fab80730f1ebf3af7c35e4bbd8fc80e2bafd86419fc0125acc140b414d806f",  # interp addendum
})


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


def _hyp071_revival_confirmed(cfg: dict) -> tuple[bool, str]:
    """Corroborating runtime guard: does the hypothesis ledger record an actual
    revival of HYP-071 (fresh prereg + new CONFIRMED adjudication), as opposed to
    the killed METRIC_ARTIFACT verdict standing plus an unprereg'd recompute?

    Fail-closed: any missing file, unreadable/malformed JSON, or absence of a
    qualifying entry returns (False, <reason>). Never raises — the gate itself
    decides whether that closes ignition.
    """
    ign = cfg.get("ignition", {})
    ledger_path_cfg = ign.get("hypothesis_ledger_path")
    ledger_path = Path(ledger_path_cfg) if ledger_path_cfg else DEFAULT_LEDGER
    if not ledger_path.is_absolute():
        ledger_path = ROOT / ledger_path

    if not ledger_path.exists():
        return False, f"hypothesis ledger missing: {ledger_path}"
    try:
        ledger = json.loads(ledger_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"hypothesis ledger unreadable: {exc}"
    if not isinstance(ledger, list):
        return False, "hypothesis ledger malformed (expected a list of entries)"

    reused_locked_prereg = False
    for entry in ledger:
        if not isinstance(entry, dict) or entry.get("id") != HYP071_ID:
            continue
        if str(entry.get("status", "")).upper() != "CONFIRMED":
            continue
        result = entry.get("result", {})
        if not isinstance(result, dict):
            result = {}
        prereg_hash = result.get("prereg_hash") or entry.get("prereg_hash")
        if not prereg_hash:
            continue  # CONFIRMED with no traceable prereg hash — not qualifying
        if prereg_hash in HYP071_LOCKED_PREREG_HASHES:
            reused_locked_prereg = True
            continue  # rerun on the killed prereg, not a fresh registration
        date_tested = entry.get("date_tested") or entry.get("date")
        if not date_tested or str(date_tested) <= HYP071_ORIGINAL_VERDICT_DATE:
            continue  # not dated after the original METRIC_ARTIFACT verdict
        return True, (
            f"fresh CONFIRMED adjudication found "
            f"(prereg_hash={prereg_hash[:12]}…, date={date_tested})"
        )

    if reused_locked_prereg:
        return False, (
            "HYP-071 CONFIRMED entry found but its prereg_hash reuses the "
            "original locked/killed prereg — a rerun is not a fresh registration"
        )
    return False, (
        "HYP-071 METRIC_ARTIFACT verdict stands; recompute is not revival — "
        "fresh prereg + adjudication required"
    )


def evaluate_gate(config_path: Path | None = None) -> GateStatus:
    """Evaluate the ignition gate. Gate opens ONLY if every condition below holds."""
    cfg = load_config(config_path)
    ign = cfg.get("ignition", {})

    tick_024 = bool(ign.get("tick_024_carry_fix_landed", False))
    hyp_071 = bool(ign.get("hyp_071_net_confirmed", False))
    board_net, board_detail = _board_is_net(cfg)
    hyp_071_revival, hyp_071_revival_detail = _hyp071_revival_confirmed(cfg)

    checks = {
        "tick_024_carry_fix_landed": tick_024,
        "hyp_071_net_confirmed": hyp_071,
        "value_board_is_net": board_net,
        "hyp_071_revival_confirmed": hyp_071_revival,
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
    if not hyp_071_revival:
        reasons.append(f"HYP-071 REVIVAL GUARD: {hyp_071_revival_detail}")

    is_open = tick_024 and hyp_071 and board_net and hyp_071_revival
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
