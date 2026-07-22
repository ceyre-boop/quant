#!/usr/bin/env python3
"""build_system_regime.py — the writer for the system regime contract.

Reads each strategy's EXISTING regime signal (JSON files, never imports), runs
the small per-strategy classifiers in alta_platform/regime_contract.py, assembles one
canonical contract, and writes:

    data/agent/system_regime_state.json

This is the "mirror" pattern (scripts/obsidian_sync.py) applied to regime: read
the scattered per-strategy signals, write one honest unified contract, everyone
reads it via alta_platform.regime_client.get_regime().

DISCIPLINE (non-negotiable):
  * Never raises. A crashed writer that leaves a stale contract claiming
    freshness is worse than one that writes DEGRADED with a reason.
  * Every section carries a status (OK | STALE | UNAVAILABLE) + reason.
  * A stale/missing source downgrades that strategy to STAND_ASIDE. We NEVER
    emit a favorable verdict from missing data.
  * No hardcoded thresholds — all read from config/parameters.yml. New keys were
    logged to data/agent/param_change_log.jsonl before use.

Isolation: this script + alta_platform/ import neither ict/ nor sovereign/. It only
reads their JSON outputs. The ict/<->sovereign wall is never touched.

Usage:
    python3 scripts/build_system_regime.py            # write the contract
    python3 scripts/build_system_regime.py --dry-run  # print, write nothing
    python3 scripts/build_system_regime.py --check     # exit 1 if degraded
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from alta_platform import regime_contract as rc  # noqa: E402  (after sys.path setup)

DATA = REPO / "data"
CONFIG = REPO / "config" / "parameters.yml"
OUT = DATA / "agent" / "system_regime_state.json"

OK = rc.OK
STALE = rc.STALE
UNAVAILABLE = rc.UNAVAILABLE


def _freshness(config: dict, key: str, default: float) -> float:
    try:
        return float(config["platform"]["regime"]["freshness_hours"][key])
    except (KeyError, TypeError, ValueError):
        return default


def build() -> dict:
    now = datetime.now(timezone.utc)
    try:
        config = rc.load_yaml(CONFIG)
    except Exception as exc:  # config unreadable — still emit a DEGRADED contract
        config = {}
        config_error = f"config unreadable: {exc}"
    else:
        config_error = None

    sections = {
        "carry": rc.classify_carry(config, DATA, _freshness(config, "carry", 6.0)),
        "es_nq": rc.classify_es_nq(config, DATA, _freshness(config, "es_nq", 12.0)),
        "macro": rc.classify_macro(config, DATA, _freshness(config, "macro", 12.0)),
    }

    portfolio = build_portfolio(config)

    # Top-level status: OK only if every non-INFO section is OK. STALE if any
    # section stale; DEGRADED if any unavailable or config failed.
    statuses = {name: s.status for name, s in sections.items()}
    if config_error or UNAVAILABLE in statuses.values() or portfolio.get("status") == UNAVAILABLE:
        top = "DEGRADED"
    elif STALE in statuses.values():
        top = "STALE"
    else:
        top = "OK"

    reason_bits = [f"{n}:{s}" for n, s in statuses.items()]
    reason_bits.append(f"portfolio:{portfolio.get('status')}")
    if config_error:
        reason_bits.append(config_error)

    return {
        "generated_at": now.isoformat(),
        "status": top,
        "status_reason": "; ".join(reason_bits),
        "strategies": {name: s.to_dict() for name, s in sections.items()},
        "portfolio": portfolio,
        "_meta": {
            "writer": "scripts/build_system_regime.py",
            "note": "Neutral alta_platform/ layer — reads JSON outputs, imports neither ict/ nor sovereign/.",
        },
    }


def build_portfolio(config: dict) -> dict:
    """Portfolio section — STUB pending a unified position-by-cluster ledger.

    There is no unified open-exposure-by-cluster ledger on disk today, so we do
    NOT fabricate a confident zero. Exposure + the drawdown breaker are labelled
    UNAVAILABLE with a reason. Caps and the daily-loss limit ARE read from config
    (never hardcoded) so the shape is real and the follow-on wiring is a drop-in.
    """
    hc = config.get("hard_constraints", {})
    daily_loss_limit = hc.get("max_daily_loss_pct")  # 0.02, from config
    max_positions = hc.get("max_concurrent_positions")

    return {
        "status": UNAVAILABLE,
        "reason": (
            "no unified position-by-cluster ledger on disk; open exposure and the "
            "drawdown breaker cannot be read. Caps/limits shown are from config."
        ),
        "open_exposure_by_cluster": None,       # UNAVAILABLE, not a confident {}=0
        "cluster_caps": None,                    # no cluster-cap config key exists yet
        "max_concurrent_positions": max_positions,
        "daily_pnl_pct": None,                   # UNAVAILABLE (no attributed daily P&L feed)
        "daily_drawdown_limit_pct": daily_loss_limit,
        "drawdown_breaker_tripped": None,        # None = UNAVAILABLE, NOT a confident False
        "drawdown_breaker_status": UNAVAILABLE,
        "source": "config/parameters.yml::hard_constraints (caps only); no position ledger",
    }


def write(contract: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(contract, fh, indent=2)
        fh.write("\n")
    tmp.replace(OUT)  # atomic-ish: never leave a half-written contract


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the system regime contract.")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    ap.add_argument("--check", action="store_true", help="exit 1 if contract is degraded")
    args = ap.parse_args(argv)

    try:
        contract = build()
    except Exception:  # never raise out of the writer
        contract = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "DEGRADED",
            "status_reason": "writer crashed: " + traceback.format_exc().splitlines()[-1],
            "strategies": {},
            "portfolio": {"status": UNAVAILABLE, "reason": "writer crashed"},
        }

    if args.dry_run:
        print(json.dumps(contract, indent=2))
    else:
        write(contract)
        print(f"wrote {OUT.relative_to(REPO)} — status {contract['status']}: {contract['status_reason']}")

    if args.check and contract.get("status") != OK:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
