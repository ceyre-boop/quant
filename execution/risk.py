"""Layer 5 — the ratified risk constitution, as enforceable code.

WHAT IS AND IS NOT IN HERE
--------------------------
`RISK_CONSTITUTION.md` is RATIFIED (`:7`) and legislates exactly FIVE numbers:

    Art. 1  per-trade worst-case loss at stop     0.75%   (:19)
    Art. 2  carry-complex combined open heat      2.5%    (:28)
    Art. 3  halve all new sizing                  3.5% DD (:34)
    Art. 3  halt new entries                      5%   DD (:35)
    Art. 3  flatten predictive layer              6.5% DD (:35)

Those five are implemented here. **Nothing else is.**

Daily-loss limits, consecutive-loss halts and VIX gates appear NOWHERE in the
constitution. Article 5 (`:49-56`) says there is no override and that amendments
require the prose and the YAML twin to change in the same commit. Inventing
thresholds for unlegislated gates would be exactly the override Article 5
forbids, so they are written up in `docs/proposed_amendment_art7-9.md` for
ratification instead.

The VIX gate in particular must not be quietly reintroduced: HYP-044's VIX gate
was tested and ROLLED BACK as `REJECTED_OOS` (p=0.50, delta approximately 0).
`CLAUDE.md:134` still carries a stale example string that reads like an
instruction to wire it — that is drift, not a requirement.

WHY THIS FILE DOES NOT IMPORT THE YAML TWIN
--------------------------------------------
`tests/test_risk_constitution.py:167-174` (`test_yaml_not_wired_into_live_config_loader`)
deliberately forbids live code from importing `config/risk_constitution.yaml`.
The twin is a tripwire artifact, not a runtime config: if code read it, editing
the YAML would silently change behaviour and the drift test would have nothing
independent to compare against. So the constants live here, and
`test_constitution_drift` compares them to the YAML from the test side.

THE GAP THIS CLOSES
-------------------
Before this module, the 3.5/5/6.5 ladder was implemented NOWHERE. The engine's
effective flatten came from `sovereign/risk/config/risk_config.yaml:16-17`
(0.08 - 0.005 = 7.5%), i.e. ABOVE the constitutional 6.5% — the same class of
error as the draft v0.1.0 ladder whose 8.5% flatten sat above the 8% prop halt,
described in the constitution itself as "a decorative emergency brake" (`:36-41`).

SCOPE: PAPER ONLY
-----------------
This gates simulated fills from `execution/harness.py`. It does NOT touch the
live forex execution path, which is under the standing shadow freeze until
~2026-07-28 (`CLAUDE.md` standing constraints). Grade ladders elsewhere in the
repo legitimately exceed 0.75% until that reconciliation lands; this module
reports such a breach for paper fills rather than clamping live behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ── The ratified five. Do not edit without an Article 5 amendment. ────────────
PER_TRADE_CAP = 0.0075      # Art. 1  RISK_CONSTITUTION.md:19
CARRY_HEAT_CAP = 0.025      # Art. 2  :28
DD_HALVE = 0.035            # Art. 3  :34
DD_HALT = 0.050             # Art. 3  :35
DD_FLATTEN = 0.065          # Art. 3  :35

#: The external prop line the ladder is anchored beneath. Rationale only —
#: deliberately unbolded in the constitution (`:36-37`), not a binding cap here.
PROP_TRAILING_REFERENCE = 0.08

#: Art. 2 applies to the carry complex specifically, not to all open risk.
CARRY_COMPLEX = frozenset({"GBPUSD", "EURUSD", "AUDUSD", "AUDNZD", "USDJPY",
                           "GBP_USD", "EUR_USD", "AUD_USD", "AUD_NZD", "USD_JPY"})


class Action(str, Enum):
    ALLOW = "ALLOW"
    HALVE = "HALVE"          # >= 3.5% DD — new sizing halved
    BLOCK = "BLOCK"          # >= 5% DD — no new entries
    FLATTEN = "FLATTEN"      # >= 6.5% DD — predictive layer flattened


@dataclass
class AccountState:
    """Everything the ladder needs. Drawdown is peak-to-trough at ACCOUNT level
    (`RISK_CONSTITUTION.md:33`), not per-strategy."""
    equity: float
    peak_equity: float
    open_carry_risk: float = 0.0      # fraction of equity at risk in carry pairs
    open_positions: int = 0

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)


@dataclass
class RiskDecision:
    allowed: bool
    action: Action
    size_mult: float
    breached: list[str] = field(default_factory=list)
    reason: str = ""
    drawdown: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"allowed": self.allowed, "action": self.action.value,
                "size_mult": self.size_mult, "breached": list(self.breached),
                "reason": self.reason, "drawdown": round(self.drawdown, 6),
                "detail": self.detail}


def ladder_action(drawdown: float) -> tuple[Action, float]:
    """Article 3, evaluated worst-first.

    Boundaries are inclusive: at exactly 3.5% the halve is in force. The
    constitution states thresholds as the level at which the action applies, and
    an exclusive reading would let a position sit precisely on a limit untouched.
    """
    if drawdown >= DD_FLATTEN:
        return Action.FLATTEN, 0.0
    if drawdown >= DD_HALT:
        return Action.BLOCK, 0.0
    if drawdown >= DD_HALVE:
        return Action.HALVE, 0.5
    return Action.ALLOW, 1.0


def is_carry(symbol: str) -> bool:
    return symbol.replace("/", "").replace("_", "").upper() in {
        s.replace("_", "") for s in CARRY_COMPLEX}


def check(*, symbol: str, risk_fraction: float, state: AccountState) -> RiskDecision:
    """Gate one prospective fill against the ratified five.

    `risk_fraction` is worst-case loss at stop as a fraction of equity — the
    quantity Article 1 actually caps. Passing a notional fraction here would
    silently under-report risk.
    """
    breached: list[str] = []
    dd = state.drawdown
    action, mult = ladder_action(dd)
    if action is Action.HALVE:
        breached.append(f"ART3_HALVE dd={dd:.4f}>={DD_HALVE}")
    elif action is Action.BLOCK:
        breached.append(f"ART3_HALT dd={dd:.4f}>={DD_HALT}")
    elif action is Action.FLATTEN:
        breached.append(f"ART3_FLATTEN dd={dd:.4f}>={DD_FLATTEN}")

    effective = risk_fraction * mult

    # Art. 1 — the cap applies to the size that would actually be taken.
    if effective > PER_TRADE_CAP:
        breached.append(
            f"ART1_PER_TRADE {effective:.5f}>{PER_TRADE_CAP} — "
            f"'No grade, conviction score, or Kelly output raises it'")

    # Art. 2 — carry complex aggregate.
    if is_carry(symbol):
        projected = state.open_carry_risk + effective
        if projected > CARRY_HEAT_CAP:
            breached.append(
                f"ART2_CARRY_HEAT {projected:.5f}>{CARRY_HEAT_CAP}")

    art1_breach = any(b.startswith("ART1") for b in breached)
    art2_breach = any(b.startswith("ART2") for b in breached)
    allowed = action in (Action.ALLOW, Action.HALVE) and not art1_breach and not art2_breach

    if not breached:
        reason = "within all ratified limits"
    elif action is Action.FLATTEN:
        reason = "Article 3 flatten — predictive layer flattened"
    elif action is Action.BLOCK:
        reason = "Article 3 halt — no new entries"
    elif art1_breach:
        reason = "Article 1 per-trade cap exceeded"
    elif art2_breach:
        reason = "Article 2 carry-complex heat exceeded"
    else:
        reason = "Article 3 halve — new sizing halved"

    return RiskDecision(
        allowed=allowed, action=action, size_mult=mult if allowed else 0.0,
        breached=breached, reason=reason, drawdown=dd,
        detail={"requested_risk": risk_fraction, "effective_risk": effective,
                "is_carry": is_carry(symbol), "symbol": symbol},
    )


def constants() -> dict[str, float]:
    """For the drift test to compare against the YAML twin from the test side."""
    return {"hard_cap_frac": PER_TRADE_CAP, "carry_heat_cap_frac": CARRY_HEAT_CAP,
            "halve": DD_HALVE, "halt": DD_HALT, "flatten": DD_FLATTEN}
