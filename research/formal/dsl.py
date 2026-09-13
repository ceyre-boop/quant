"""A tiny feature DSL so that a rule's data dependencies are DECLARED, not inferred from pandas.

A rule is a dict of named expressions over series. Every leaf is a series reference with an explicit
time offset relative to the decision bar t (offset 0 = the decision bar itself, −1 = previous bar,
+k = the future). Windows declare [start, end] offsets. The prover reasons over these offsets only —
which is exactly the information the three look-ahead refutations lacked.

    lag("close", 1)                      -> close at t-1
    win("volume", start=-20, end=-1)     -> volume over t-20..t-1
    win("volume", start=-5, end=0)       -> INCLUDES the decision bar -> not causal for a decision at t
    at("gain", offset=0, clock="10:30")  -> a value stamped at 10:30 of day t (HYP-105's universe leak,
                                            if the entry clock is 09:31)
    cs(expr)                             -> cross-sectional transform (rank/z), inherits the expr's offsets
    fn(name, *exprs)                     -> arithmetic/logic over exprs, inherits the union of offsets
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Leaf:
    series: str
    start: int            # bar offset relative to decision bar t (negative = past)
    end: int
    clock: str | None = None   # optional intraday clock stamp "HH:MM" for daily-indexed values


@dataclass(frozen=True)
class Node:
    op: str
    args: tuple = field(default_factory=tuple)


Expr = Leaf | Node


def lag(series: str, k: int = 1) -> Leaf:      return Leaf(series, -k, -k)
def now(series: str) -> Leaf:                  return Leaf(series, 0, 0)
def win(series: str, start: int, end: int) -> Leaf: return Leaf(series, start, end)
def at(series: str, offset: int = 0, clock: str | None = None) -> Leaf: return Leaf(series, offset, offset, clock)
def cs(expr: Expr) -> Node:                    return Node("cs", (expr,))
def fn(name: str, *exprs: Expr) -> Node:       return Node(name, tuple(exprs))


def leaves(expr: Expr) -> list[Leaf]:
    if isinstance(expr, Leaf):
        return [expr]
    out = []
    for a in expr.args:
        out += leaves(a)
    return out


@dataclass
class Rule:
    """What the prover attacks: features, an optional universe predicate, the decision bar's clock
    (for intraday rules) and how many bars after the decision the position is entered."""
    name: str
    features: dict[str, Expr]
    universe: Expr | None = None
    decision_clock: str | None = None       # "09:31" etc.; None for daily rules
    entry_offset: int = 1                   # bars after decision at which the trade is entered
    hold_bars: int = 1
