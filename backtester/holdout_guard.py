"""Central holdout registry + date-range guard for the backtester stack.

Before this module, holdout boundaries were per-script constants: megascan.py and
megascan_v2.py each hardcoded "2025-07-17", scripts/edge_factory_worker.py used
"2025-01-01", scripts/codify_holdout_results.py used "2024-01-01", and
research/yield_frontier/holdout_guard.py fenced a fourth window. Nothing stopped
`engine.run()` or `daily_engine.backtest_daily()` from being handed holdout dates
during a mining pass — the split was convention, not enforcement.

This module makes the boundary a checked precondition. Datasets are registered
here once; miners get an exception, not a silent contamination.

Policy
------
Mining/optimisation code may touch dates strictly BEFORE the dataset's
holdout_start. A run that reaches into the holdout must be a sanctioned,
pre-registered verdict run and must opt in explicitly, either by

    ALLOW_HOLDOUT_ACCESS=1 python -m backtester.hyp104_holdout

or, for a runner that already carries its own prereg-hash guard, in-process:

    from backtester import holdout_guard
    holdout_guard.sanction("HYP-104 prereg a984372 verified")

Both paths are recorded to data/agent/holdout_access_log.jsonl so that "the
holdout was only touched once" is an auditable claim rather than a memory.

The physical-absence fence in research/yield_frontier/holdout_guard.py is the
stronger control and stays authoritative for the yield-frontier program; this
module is the stack-wide backstop for datasets that live on disk in full.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ACCESS_LOG = REPO / "data/agent/holdout_access_log.jsonl"

# dataset -> holdout start (inclusive). Everything on/after this date is sealed.
# Sourced from the constants that were previously scattered across the callers;
# changing a value here is a parameter change and needs a param_change_log entry
# per CLAUDE.md non-negotiable #4.
HOLDOUT_REGISTRY: dict[str, str] = {
    # backtester/megascan.py + megascan_v2.py: last 12 months reserved.
    "equities_daily": "2025-07-17",
    # backtester/engine.py gapper events (HYP-093/104/105 family).
    "gapper_intraday": "2025-07-17",
    # research/yield_frontier/holdout_guard.py EQUITIES_HOLDOUT lower bound.
    "yield_frontier_equities": "2024-07-01",
    # research/fvg_corridor/core.py + yield_frontier NQ_MINING_END.
    "nq_futures": "2024-07-01",
    # research/yield_frontier/holdout_guard.py OPTIONS_QUOTE_CUTOFF + 1 day.
    "options_chains": "2023-10-01",
}

DEFAULT_DATASET = "equities_daily"

_SANCTION: str | None = None


class HoldoutViolation(RuntimeError):
    """Raised when a non-sanctioned run reaches into a sealed holdout window."""


def holdout_start(dataset: str = DEFAULT_DATASET) -> str:
    """Holdout start for `dataset`; env override HOLDOUT_START_<DATASET> wins."""
    env = os.getenv(f"HOLDOUT_START_{dataset.upper()}")
    if env:
        return env
    if dataset not in HOLDOUT_REGISTRY:
        raise KeyError(
            f"unregistered dataset {dataset!r} — add it to HOLDOUT_REGISTRY "
            f"rather than passing an ad-hoc date. Known: "
            f"{sorted(HOLDOUT_REGISTRY)}")
    return HOLDOUT_REGISTRY[dataset]


def sanction(reason: str) -> None:
    """Authorise holdout access for the rest of this process.

    Call ONLY from a runner that has already verified its pre-registration
    (committed + hash-matched). `reason` is written to the access log.
    """
    global _SANCTION
    _SANCTION = reason


def revoke() -> None:
    """Drop an in-process sanction (used by tests)."""
    global _SANCTION
    _SANCTION = None


def _sanction_reason() -> str | None:
    if _SANCTION:
        return _SANCTION
    if os.getenv("ALLOW_HOLDOUT_ACCESS"):
        return f"env ALLOW_HOLDOUT_ACCESS={os.environ['ALLOW_HOLDOUT_ACCESS']}"
    return None


def _log_access(context: str, start: str, end: str, dataset: str,
                reason: str) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "context": context,
           "dataset": dataset, "start": start, "end": end, "reason": reason}
    try:
        ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ACCESS_LOG.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        # Never let audit-logging failure abort a sanctioned verdict run.
        pass


def validate_date_range(start: str | None, end: str | None,
                        context: str = "backtest",
                        dataset: str = DEFAULT_DATASET) -> None:
    """Raise HoldoutViolation if [start, end] touches the sealed holdout.

    `end` is treated as inclusive — an unbounded sentinel ("9999", None) counts
    as touching the holdout, because an unbounded scan is exactly how holdout
    data leaks into a mining run.
    """
    hs = holdout_start(dataset)
    end_eff = "9999" if end is None else str(end)
    start_eff = "0000" if start is None else str(start)
    if end_eff < hs:
        return
    reason = _sanction_reason()
    if reason:
        _log_access(context, start_eff, end_eff, dataset, reason)
        return
    raise HoldoutViolation(
        f"{context}: date range {start_eff}→{end_eff} touches the "
        f"{dataset} holdout (>= {hs}). Mining runs must end before {hs}. "
        f"Set ALLOW_HOLDOUT_ACCESS=1 (or call holdout_guard.sanction()) only "
        f"for a pre-registered final verdict run — every such access is "
        f"recorded to {ACCESS_LOG.relative_to(REPO)}.")


def is_mining_safe(start: str | None, end: str | None,
                   dataset: str = DEFAULT_DATASET) -> bool:
    """Non-raising predicate form, for callers that want to branch."""
    try:
        validate_date_range(start, end, dataset=dataset)
        return True
    except HoldoutViolation:
        return False
