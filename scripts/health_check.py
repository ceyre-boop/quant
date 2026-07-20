#!/usr/bin/env python3
"""
"Green but empty" job health check → data/health/system_health.json
===================================================================

The failure mode this exists to catch: a scheduled job exits 0, rewrites its
output file with a fresh mtime, and the payload is empty. Every freshness-based
monitor stays GREEN while the system is actually blind. Freshness alone is not
health — you need freshness AND substance.

Each registered job declares:
  path         — the output file it is supposed to write
  max_age_min  — how stale that file may get before it counts as STALE
  substance    — a predicate over the parsed payload; False means "green but empty"
  optional     — a missing file is INFO, not RED (job may not be scheduled here)

Verdicts per job:
  OK       — file fresh and substantive
  EMPTY    — file fresh but the payload has no content  ← the bug this hunts
  STALE    — file older than max_age_min (job stopped firing)
  MISSING  — file absent
  ERROR    — file present but unparseable

Overall is RED if any non-optional job is EMPTY/MISSING/ERROR, YELLOW if any is
STALE, else GREEN.

RUN:
  python3 scripts/health_check.py                # check all, write report
  python3 scripts/health_check.py --job reddit   # single job
  python3 scripts/health_check.py --strict       # exit 1 if overall != GREEN
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "data" / "health" / "system_health.json"


# ── Substance predicates ──────────────────────────────────────────────────────
# Each returns (bool, detail). False == the payload is empty in the way that
# matters for that job, even though the file exists and is fresh.

def _reddit_substance(d: Any) -> tuple[bool, str]:
    posts = d.get("posts_scanned", 0)
    tickers = len(d.get("equity", {})) + len(d.get("forex", {}))
    return (posts > 0 and tickers > 0,
            f"posts_scanned={posts}, tickers+pairs={tickers}")


def _forex_prox_substance(d: Any) -> tuple[bool, str]:
    rows = d.get("pairs", [])
    # A scan that evaluated zero pairs is blind. NO_TRADE_TODAY across a full
    # pair list is a legitimate result and stays OK.
    scored = [r for r in rows if r.get("conviction") is not None]
    return len(scored) > 0, f"pairs_scored={len(scored)}, verdict={d.get('verdict')!r}"


def _forex_data_substance(d: Any) -> tuple[bool, str]:
    fake = d.get("fake_data_pairs", [])
    deg = d.get("degraded_pairs", [])
    # Substantive as long as it reported. Degradation surfaces via the detail.
    return True, f"overall={d.get('overall')}, degraded={deg}, fake_data={fake}"


def _components_substance(d: Any) -> tuple[bool, str]:
    comps = d.get("components", {})
    red = [k for k, v in comps.items() if v.get("status") == "RED"]
    return len(comps) > 0, f"components={len(comps)}, red={red}"


# ── Job registry ──────────────────────────────────────────────────────────────

JOBS: dict[str, dict] = {
    "reddit": {
        "path":        ROOT / "data" / "cache" / "reddit_sentiment.json",
        "max_age_min": 24 * 60,
        "substance":   _reddit_substance,
        "optional":    False,
    },
    "forex_scan": {
        "path":        ROOT / "data" / "agent" / "forex_proximity.json",
        "max_age_min": 24 * 60,
        "substance":   _forex_prox_substance,
        "optional":    False,
    },
    "forex_data": {
        "path":        ROOT / "data" / "health" / "forex_data_status.json",
        "max_age_min": 24 * 60,
        "substance":   _forex_data_substance,
        "optional":    False,
    },
    "api_health": {
        "path":        ROOT / "data" / "agent" / "health.json",
        "max_age_min": 24 * 60,
        "substance":   _components_substance,
        "optional":    True,
    },
}


def check_job(name: str, cfg: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    path: Path = cfg["path"]
    rel = str(path.relative_to(ROOT))

    if not path.exists():
        return {"status": "MISSING", "path": rel, "detail": "output file does not exist",
                "optional": cfg["optional"]}

    age_min = (now.timestamp() - path.stat().st_mtime) / 60.0
    base = {"path": rel, "age_minutes": round(age_min, 1),
            "max_age_minutes": cfg["max_age_min"], "optional": cfg["optional"]}

    try:
        payload = json.loads(path.read_text())
    except Exception as e:
        return {**base, "status": "ERROR", "detail": f"{type(e).__name__}: {e}"}

    substantive, detail = cfg["substance"](payload)

    if not substantive:
        # Fresh file, no content — the exact failure this check exists for.
        return {**base, "status": "EMPTY",
                "detail": f"file is fresh but payload is empty ({detail})"}
    if age_min > cfg["max_age_min"]:
        return {**base, "status": "STALE", "detail": f"{detail}; job may have stopped firing"}
    return {**base, "status": "OK", "detail": detail}


def run(only: str | None = None) -> dict:
    now = datetime.now(timezone.utc)
    jobs = {only: JOBS[only]} if only else JOBS
    results = {name: check_job(name, cfg, now) for name, cfg in jobs.items()}

    def _bad(r): return r["status"] in ("EMPTY", "MISSING", "ERROR") and not r["optional"]
    def _warn(r): return r["status"] == "STALE" or (r["status"] != "OK" and r["optional"])

    red = [n for n, r in results.items() if _bad(r)]
    yellow = [n for n, r in results.items() if not _bad(r) and _warn(r)]
    overall = "RED" if red else "YELLOW" if yellow else "GREEN"

    report = {
        "ts":       now.isoformat(),
        "overall":  overall,
        "red":      sorted(red),
        "yellow":   sorted(yellow),
        "jobs":     results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description='"Green but empty" job health check')
    ap.add_argument("--job", choices=sorted(JOBS), help="check a single job")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 unless overall is GREEN (for use in launchd/CI)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    report = run(args.job)

    if not args.quiet:
        print(f"system health: {report['overall']}")
        for name, r in sorted(report["jobs"].items()):
            mark = "  " if r["status"] == "OK" else "!!"
            opt = " (optional)" if r.get("optional") and r["status"] != "OK" else ""
            print(f"{mark} {r['status']:<8} {name}{opt}: {r['detail']}")

    if args.strict and report["overall"] != "GREEN":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
