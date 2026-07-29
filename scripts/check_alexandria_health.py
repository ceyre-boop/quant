#!/usr/bin/env python3
"""
scripts/check_alexandria_health.py
===================================
Daily sanity check confirming the Alexandrian Library's similarity-floor logic
is behaving correctly after the 2026-07-20 fix (SIMILARITY_FLOOR = 0.30).

What we verify
--------------
* When sim < 0.30  → library.regime == 'UNKNOWN'  (abstained correctly)
* When sim >= 0.30 → library.regime != 'UNKNOWN'  (fired correctly)

Data sources (in priority order)
---------------------------------
1. logs/scanner_state.json   — latest scan state, always present
2. data/oracle/pulses/       — pulse snapshots (last N files)

Output → data/health/alexandria_status.json

Schedule: daily at 18:00 ET via com.alta.alexandria_health plist.

Usage:
  python3 scripts/check_alexandria_health.py
  python3 scripts/check_alexandria_health.py --strict   # exit 1 on FAIL
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "health" / "alexandria_status.json"
SCANNER_STATE = ROOT / "logs" / "scanner_state.json"
PULSES_DIR = ROOT / "data" / "oracle" / "pulses"
SIMILARITY_FLOOR = 0.30


class Sample(NamedTuple):
    source: str          # which file
    sim: float           # primary similarity score
    regime: str          # library.regime at that scan
    abstained: bool      # True if regime == 'UNKNOWN'
    expected_abstain: bool  # True if sim < SIMILARITY_FLOOR
    verdict: str         # OK | WRONG


def _extract_library(d: dict) -> dict | None:
    """Pull the library sub-dict from a scanner_state or pulse blob."""
    if "library" in d:
        return d["library"]
    # Pulse files may nest it differently
    if "scanner_state" in d and "library" in d["scanner_state"]:
        return d["scanner_state"]["library"]
    return None


def _sim_from_library(lib: dict) -> float:
    """
    Best proxy for primary_similarity we can get from the written state.

    library_bridge writes threat_score which mirrors threat severity, but the
    advisory string contains 'sim=X.XXX' — parse that first for precision.
    Fall back to threat_score.
    """
    advisory = lib.get("advisory", "")
    import re
    m = re.search(r"sim=([0-9.]+)", advisory)
    if m:
        return float(m.group(1))
    return float(lib.get("threat_score", 0.0))


def collect_samples(max_pulse_files: int = 20) -> list[Sample]:
    samples: list[Sample] = []

    # --- scanner_state.json (always try this first) ---
    if SCANNER_STATE.exists():
        try:
            d = json.loads(SCANNER_STATE.read_text())
            lib = _extract_library(d)
            if lib is not None:
                sim = _sim_from_library(lib)
                regime = lib.get("regime", "UNKNOWN")
                abstained = regime == "UNKNOWN"
                expected = sim < SIMILARITY_FLOOR
                verdict = "OK" if abstained == expected else "WRONG"
                samples.append(Sample(
                    source="logs/scanner_state.json",
                    sim=sim, regime=regime,
                    abstained=abstained,
                    expected_abstain=expected,
                    verdict=verdict,
                ))
        except Exception as exc:
            samples.append(Sample(
                source="logs/scanner_state.json",
                sim=0.0, regime="ERROR",
                abstained=False, expected_abstain=False,
                verdict=f"PARSE_ERROR:{exc}",
            ))

    # --- pulse snapshots ---
    if PULSES_DIR.exists():
        pulse_files = sorted(PULSES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for pf in pulse_files[:max_pulse_files]:
            try:
                d = json.loads(pf.read_text())
                lib = _extract_library(d)
                if lib is None:
                    continue
                sim = _sim_from_library(lib)
                regime = lib.get("regime", "UNKNOWN")
                abstained = regime == "UNKNOWN"
                expected = sim < SIMILARITY_FLOOR
                verdict = "OK" if abstained == expected else "WRONG"
                samples.append(Sample(
                    source=str(pf.relative_to(ROOT)),
                    sim=sim, regime=regime,
                    abstained=abstained,
                    expected_abstain=expected,
                    verdict=verdict,
                ))
            except Exception:
                continue

    return samples


def run() -> dict:
    now = datetime.now(timezone.utc)
    samples = collect_samples()

    wrong = [s for s in samples if s.verdict.startswith("WRONG")]
    errors = [s for s in samples if s.verdict.startswith("PARSE")]

    if not samples:
        overall = "NO_DATA"
        detail = "No scanner_state.json or pulse files found"
    elif wrong:
        overall = "FAIL"
        detail = (f"{len(wrong)} sample(s) misbehaving: "
                  + "; ".join(f"{s.source}: sim={s.sim:.3f} regime={s.regime}" for s in wrong))
    else:
        overall = "PASS"
        low_sim  = [s for s in samples if s.expected_abstain]
        high_sim = [s for s in samples if not s.expected_abstain]
        detail = (f"{len(samples)} samples checked. "
                  f"Low-sim (<{SIMILARITY_FLOOR}) abstained correctly: {len(low_sim)}. "
                  f"High-sim fired correctly: {len(high_sim)}.")

    report = {
        "ts": now.isoformat(),
        "overall": overall,
        "detail": detail,
        "similarity_floor": SIMILARITY_FLOOR,
        "samples_checked": len(samples),
        "wrong_count": len(wrong),
        "error_count": len(errors),
        "samples": [
            {
                "source": s.source,
                "sim": round(s.sim, 4),
                "regime": s.regime,
                "abstained": s.abstained,
                "expected_abstain": s.expected_abstain,
                "verdict": s.verdict,
            }
            for s in samples
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Alexandria similarity-floor health check")
    ap.add_argument("--strict", action="store_true", help="exit 1 unless PASS")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    report = run()

    if not args.quiet:
        print(f"Alexandria health: {report['overall']}")
        print(f"  {report['detail']}")
        for s in report["samples"]:
            mark = "  " if s["verdict"] == "OK" else "!!"
            print(f"  {mark} {s['verdict']:<6} sim={s['sim']:.3f}  regime={s['regime']}  ({s['source']})")

    if args.strict and report["overall"] not in ("PASS",):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
