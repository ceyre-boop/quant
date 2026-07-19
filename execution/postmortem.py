"""Post-mortem log — structured records of what happened, and nothing more.

WHAT THIS IS
------------
One row per closed signal: what the frozen filter saw, what the morning context
looked like, what the risk layer said, and what actually happened. It is the
honest way to accumulate the dataset that a future learner would need.

WHAT THIS IS NOT — READ BEFORE USING IT AS TRAINING DATA
---------------------------------------------------------
**This file is not a training set and must not be treated as one yet.**

There is no inference here. No label is assigned beyond the mechanical outcome, no
"why did this lose" narrative is generated, and nothing is fitted. That restraint
is deliberate:

1. **Sample size.** As of 2026-07-18 the repo holds fewer than 34 feature-complete
   live labels. The 3,460 ICT "closed" decision records are BACKTEST REPLAY —
   `decisions_2026_06.jsonl` line 1 carries `entry_timestamp: 2023-08-18` while
   sitting in the June file — and must never be mixed in as live evidence.

2. **The base rate poisons narrative labels.** Live results are 3W/24L. An
   LLM-generated explanation of why each loss occurred would be fitting stories to
   a loss streak, and every story would be plausible.

3. **HYP-090.** Adaptive parameter selection was tested across 17,325 cells:

       A0 static (do nothing)             Sharpe 0.9478
       A2_W365 (best adaptive arm)        Sharpe 0.4343
       A3 random-selection placebo, p95   Sharpe 0.9115

   Every arm lost to random selection AND to doing nothing. Its own report:
   "beating A0 while not beating A3 is the in-sample-inflation signature, not an
   edge." `research/yield_frontier/OPTIMIZATION_PROGRAM.md:12` records it as
   BANNED. Fifty post-mortem rows is a far smaller sample than 17,325 cells.

MINIMUM BEFORE THIS BECOMES A DATASET
--------------------------------------
- `MIN_ROWS_FOR_ANY_MODELLING` feature-complete LIVE rows (not backtest replay), and
- a preregistration written before the rows are examined, and
- for any adaptive proposal, clearing the A3 random-selection placebo — not merely
  beating a static baseline.

Until all three hold, this is a ledger. `readiness()` reports the gap so nobody has
to guess.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from execution.context import Status
from sovereign.autonomous._common import append_jsonl
from sovereign.utils.timestamps import canonical_timestamp

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "postmortem"
FILL_DIR = ROOT / "data" / "execution"
CONTEXT_DIR = ROOT / "data" / "context"
UTC = timezone.utc

#: Below this many feature-complete LIVE rows, no modelling of any kind.
#: Derived from the drift power analysis (n≈177 for 80% power on a 10-point
#: effect), not from convenience.
MIN_ROWS_FOR_ANY_MODELLING = 177


@dataclass
class PostMortem:
    """One closed signal, fully described. No derived labels beyond the outcome."""
    signal_id: str
    date: str
    ticker: str
    hypothesis: str
    side: str

    # what the frozen filter saw
    filter_features: dict[str, Any] = field(default_factory=dict)
    filter_decision: str = ""
    filter_reason: str = ""

    # what actually happened
    outcome: str = ""                  # WIN | LOSS | FLAT | SKIPPED
    net_return: float | None = None
    gross_return: float | None = None
    spread_cost: float | None = None
    backtest_expected_return: float | None = None

    # the conditions that held around it
    context_health: dict[str, Any] = field(default_factory=dict)
    source_status: dict[str, str] = field(default_factory=dict)
    risk_action: str = ""
    risk_breached: list[str] = field(default_factory=list)

    # provenance
    is_live: bool = False              # broker-confirmed, not replay or backfill
    frozen_hash: str = ""
    recorded_at: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["recorded_at"] = d["recorded_at"] or canonical_timestamp()
        return d

    @property
    def feature_complete(self) -> bool:
        """Usable for future analysis: real features and a real outcome."""
        return bool(self.filter_features and self.outcome in ("WIN", "LOSS", "FLAT")
                    and self.net_return is not None)


def classify(net_return: float | None, signal_type: str) -> str:
    """Mechanical outcome only. No interpretation, no narrative."""
    if signal_type.startswith("SKIP_"):
        return "SKIPPED"
    if net_return is None:
        return "UNKNOWN"
    if net_return > 0:
        return "WIN"
    if net_return < 0:
        return "LOSS"
    return "FLAT"


def build(day: date | str, *, fill_dir: Path | None = None,
          ctx_dir: Path | None = None, is_live: bool = False) -> list[PostMortem]:
    """Build post-mortem rows for one session from the fill log and context."""
    fill_dir = fill_dir or FILL_DIR
    ctx_dir = ctx_dir or CONTEXT_DIR

    fills_path = fill_dir / "fill_log.jsonl"
    if not fills_path.exists():
        return []

    ctx_path = ctx_dir / f"morning_context_{day}.json"
    ctx = json.loads(ctx_path.read_text()) if ctx_path.exists() else {}
    health = ctx.get("health", {})
    source_status = {n: f["status"] for n, f in (ctx.get("fields") or {}).items()}

    rows: list[PostMortem] = []
    for line in fills_path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("date") != str(day):
            continue
        rows.append(PostMortem(
            signal_id=r.get("signal_id", ""), date=str(day),
            ticker=r.get("ticker", ""), hypothesis=r.get("hypothesis", ""),
            side=r.get("signal_type", ""),
            filter_decision="GO" if not str(r.get("signal_type", "")).startswith("SKIP_") else "SKIP",
            filter_reason=r.get("reason", ""),
            outcome=classify(r.get("net_return"), str(r.get("signal_type", ""))),
            net_return=r.get("net_return"), gross_return=r.get("gross_return"),
            spread_cost=r.get("spread_cost"),
            backtest_expected_return=r.get("backtest_expected_return"),
            context_health=health, source_status=source_status,
            risk_action=r.get("risk_action", ""),
            risk_breached=list(r.get("risk_breached") or []),
            is_live=is_live, frozen_hash=r.get("frozen_hash", ""),
        ))
    return rows


def record(rows: list[PostMortem], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUT_DIR
    p = out_dir / "postmortem.jsonl"
    for r in rows:
        append_jsonl(p, r.to_json())
    return p


def load_all(out_dir: Path | None = None) -> list[dict]:
    p = (out_dir or OUT_DIR) / "postmortem.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def readiness(out_dir: Path | None = None) -> dict:
    """How far this ledger is from being a usable dataset.

    Counts LIVE feature-complete rows only. Replay and backfill rows are excluded
    from the readiness number even though they are stored — mixing them is how a
    backtest corpus gets mistaken for live evidence.
    """
    rows = load_all(out_dir)
    live_complete = [r for r in rows
                     if r.get("is_live")
                     and r.get("filter_features") is not None
                     and r.get("outcome") in ("WIN", "LOSS", "FLAT")
                     and r.get("net_return") is not None]
    n = len(live_complete)
    return {
        "n_rows_total": len(rows),
        "n_live_feature_complete": n,
        "minimum_required": MIN_ROWS_FOR_ANY_MODELLING,
        "shortfall": max(0, MIN_ROWS_FOR_ANY_MODELLING - n),
        "ready_for_modelling": False,   # never True from code alone — see note
        "note": (
            "ready_for_modelling is hard-coded False. Reaching the row count is "
            "NECESSARY but NOT SUFFICIENT: a preregistration must be written "
            "before the rows are examined, and any adaptive proposal must clear "
            "the HYP-090 A3 random-selection placebo, not merely beat a static "
            "baseline. This flag is not something code should be able to flip."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Post-mortem ledger (records only)")
    ap.add_argument("--day", default=None)
    ap.add_argument("--fills", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--live", action="store_true",
                    help="mark rows as broker-confirmed live (not replay/backfill)")
    ap.add_argument("--readiness", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else None
    if args.readiness:
        print(json.dumps(readiness(out_dir), indent=2))
        return 0

    day = date.fromisoformat(args.day) if args.day else datetime.now(UTC).date()
    rows = build(day, fill_dir=Path(args.fills) if args.fills else None,
                 is_live=args.live)
    if not rows:
        print(f"{day}: nothing to record")
        return 0
    p = record(rows, out_dir)
    by = {}
    for r in rows:
        by[r.outcome] = by.get(r.outcome, 0) + 1
    print(f"{day}: recorded {len(rows)} row(s) -> {p}")
    print(f"  outcomes: {by}")
    rd = readiness(out_dir)
    print(f"  live feature-complete: {rd['n_live_feature_complete']} / "
          f"{rd['minimum_required']} (shortfall {rd['shortfall']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
