#!/usr/bin/env python3
"""
write_briefing_scorecard_status.py — emit the briefing scorecard's calibration/
maturity status as JSON for the dashboard.

WHY THIS EXISTS
---------------
sovereign/briefing/scorecard.py exposes report() (returns a dict with the
directional hit-rate, sample count, and a maturity string) but writes no
machine-readable snapshot file — it only computes the report in-process. The
dashboard is static HTML/JS and cannot invoke Python at runtime, so it needs
the report's fields on disk as JSON. This script imports scorecard.py's PUBLIC
report() (pure read of data/briefing/scorecard.jsonl — read-only), calls it,
and writes data/briefing/scorecard_status.json.

FREEZE-SAFE: this is a NEW, additive script. It does NOT modify scorecard.py or
any execution-path file. It only calls scorecard.py's public report() and writes
one new JSON file. No trading path, no parameter mutation.

Usage:  python3 scripts/write_briefing_scorecard_status.py
Output: data/briefing/scorecard_status.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sovereign.briefing.scorecard import report  # noqa: E402

OUT = REPO / "data" / "briefing" / "scorecard_status.json"


def main() -> None:
    r = report()  # pure read of scorecard.py's public API
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_scored_directional": r.get("n_scored_directional"),
        "directional_hit_rate": r.get("directional_hit_rate"),
        "maturity": r.get("maturity"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
