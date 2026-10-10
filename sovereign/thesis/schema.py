"""Thesis record schema + canonical hash (same recipe as research/hyp111/prereg.py)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

STATUSES = ("DRAFT", "SEALED", "LIVE", "INVALIDATED", "CLOSED")
SOURCES = ("ai-proposed", "retro-registered", "operator")

# Fields frozen by `seal`. Everything else may change after sealing.
SEALED_FIELDS = (
    "id", "title", "thesis", "horizon_months", "basket", "entry_date", "entry_prices",
    "invalidation", "benchmark", "sizing", "source",
)


@dataclass
class Leg:
    ticker: str
    weight: float


@dataclass
class Invalidation:
    rule: str            # human sentence
    metric: str          # "basket_drawdown" | "manual"
    threshold: float | None = None


@dataclass
class Thesis:
    id: str
    title: str
    thesis: str
    horizon_months: int
    basket: list[dict[str, Any]]
    entry_date: str
    invalidation: dict[str, Any]
    entry_prices: dict[str, float] = field(default_factory=dict)
    benchmark: str = "SPY"
    sizing: dict[str, Any] = field(default_factory=dict)
    source: str = "ai-proposed"
    status: str = "DRAFT"
    sealed_at: str | None = None
    hash_lock: str | None = None
    scorecards: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Thesis":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def sealed_body(d: dict[str, Any]) -> dict[str, Any]:
    return {k: d[k] for k in SEALED_FIELDS if k in d}


def canonical_hash(d: dict[str, Any]) -> str:
    """sha256 over the sealed fields only, key-sorted, no whitespace."""
    body = sealed_body(d)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate(d: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if not str(d.get("id", "")).startswith("THS-"):
        errs.append("id must start with THS-")
    if not d.get("thesis"):
        errs.append("thesis text required")
    if int(d.get("horizon_months", 0)) <= 0:
        errs.append("horizon_months must be > 0")
    basket = d.get("basket") or []
    if not basket:
        errs.append("basket must have at least one leg")
    w = sum(float(l.get("weight", 0)) for l in basket)
    if basket and abs(w - 1.0) > 1e-6:
        errs.append(f"weights must sum to 1.0 (got {w:.6f})")
    inv = d.get("invalidation") or {}
    if not inv.get("rule") or not inv.get("metric"):
        errs.append("invalidation.rule and invalidation.metric required (no exit rule, no thesis)")
    if d.get("source") not in SOURCES:
        errs.append(f"source must be one of {SOURCES}")
    if d.get("status", "DRAFT") not in STATUSES:
        errs.append(f"status must be one of {STATUSES}")
    return errs
