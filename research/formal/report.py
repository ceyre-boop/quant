"""attack(rule, doc, ...) -> AttackReport. The prereg helper refuses to seal without one whose causal=True."""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from datetime import time as dtime

from research.formal.causality import prove
from research.formal.costs import break_even
from research.formal.dsl import Rule
from research.formal.mechanics import max_loss_before_halt
from research.formal.multiplicity import expected_n_trials


@dataclass
class AttackReport:
    rule: str
    causal: bool
    leaks: list
    n_leaves: int
    causality_proof: str
    halt_bound: dict | None
    cost_lemma: dict | None
    n_trials_declared: int
    n_trials_floor: int
    multiplicity_ok: bool


def attack(rule: Rule, doc: dict | None = None, *, entry_price: float | None = None, entry_clock: str | None = None, tier: int = 2,
           typical_price: float | None = None, bar_range_pct: float | None = None, minute_dollar_vol: float | None = None,
           round_trips: float = 1.0, slip: float = 0.0) -> AttackReport:
    c = prove(rule)
    hb = None
    if entry_price is not None and entry_clock is not None:
        h, m = entry_clock.split(":"); hb = asdict(max_loss_before_halt(entry_price, dtime(int(h), int(m)), tier))
    cl = None
    if typical_price is not None:
        cl = asdict(break_even(typical_price, bar_range_pct or 0.01, minute_dollar_vol or 1e5, round_trips, slip))
    decl, floor = expected_n_trials(doc or {})
    return AttackReport(rule.name, c.causal, c.leaks, c.n_leaves, c.proof, hb, cl, decl, floor, decl >= floor)


def to_dict(r: AttackReport) -> dict:
    return asdict(r)


if __name__ == "__main__":
    print("use: from research.formal.report import attack", file=sys.stderr)
