"""Look-ahead prover. For every leaf in a rule's features and universe predicate, assert with Z3 that
the data used is strictly before the decision bar (offset ≤ −1), or — for same-bar clock-stamped
values — strictly before the decision clock. We ask Z3 for a model of the NEGATION (some leaf with
offset ≥ 0, or a clock stamp ≥ decision clock); UNSAT ⇒ causality proven; SAT ⇒ the model IS the leak.

The "same-bar decision" convention is the strict one used by every sealed test here: a feature at t
may use bars ≤ t−1 only. A rule that wants to use the decision bar's own close must declare it via
entry_offset ≥ 1 and features on lag(...,0) — which this prover rejects, on purpose."""
from __future__ import annotations

from dataclasses import dataclass

import z3

from research.formal.dsl import Rule, leaves, Leaf


def _clock_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":"); return int(h) * 60 + int(m)


@dataclass
class CausalityResult:
    causal: bool
    leaks: list[dict]
    n_leaves: int
    proof: str


def prove(rule: Rule) -> CausalityResult:
    exprs = dict(rule.features)
    if rule.universe is not None:
        exprs["__universe__"] = rule.universe
    all_leaves: list[tuple[str, Leaf]] = [(k, l) for k, e in exprs.items() for l in leaves(e)]
    s = z3.Solver()
    dec = z3.Int("t_decision")
    dclock = _clock_minutes(rule.decision_clock) if rule.decision_clock else None
    leak_flags = []
    for i, (name, l) in enumerate(all_leaves):
        used_end = z3.Int(f"end_{i}"); s.add(used_end == dec + l.end)
        cond = used_end >= dec                                   # uses the decision bar or later
        if l.clock is not None and dclock is not None and l.end == 0:
            # same-day clock-stamped value: leak iff its clock is at/after the decision clock
            cond = z3.BoolVal(_clock_minutes(l.clock) >= dclock)
        elif l.clock is not None and dclock is not None and l.end < 0:
            cond = z3.BoolVal(False)                             # previous day's stamped value: fine
        flag = z3.Bool(f"leak_{i}"); s.add(flag == cond); leak_flags.append((flag, name, l))
    s.add(z3.Or([f for f, _, _ in leak_flags]) if leak_flags else z3.BoolVal(False))
    r = s.check()
    if r == z3.unsat:
        return CausalityResult(True, [], len(all_leaves), f"UNSAT: no leaf of {len(all_leaves)} can reach t_decision; every dependency is ≤ t−1 (or before the decision clock).")
    m = s.model(); leaks = []
    for f, name, l in leak_flags:
        if z3.is_true(m.eval(f, model_completion=True)):
            leaks.append({"feature": name, "series": l.series, "start": l.start, "end": l.end, "clock": l.clock})
    return CausalityResult(False, leaks, len(all_leaves), f"SAT: {len(leaks)} leaf/leaves use data at or after the decision — see leaks.")
