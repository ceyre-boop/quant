#!/usr/bin/env python3
"""Plist-hash watchdog — the live-organ integrity check.

Watches the launchd job surface (com.alta.* / com.sovereign.* plists in
~/Library/LaunchAgents plus the loaded-label set) against a recorded baseline.
Run after any file-touching batch: GREEN means no job file changed, appeared, or
vanished and the loaded set is unchanged — i.e. the session touched no live organ.

    python3 scripts/plist_watchdog.py                    # check vs baseline (exit 1 on RED)
    python3 scripts/plist_watchdog.py --rebaseline "why" # record a NEW baseline, reason logged

The baseline lives in data/system/plist_watchdog_baseline.json with an append-only
history of every rebaseline (ts + reason) — a job-surface change without a matching
rebaseline entry is by definition unauthorized.

Built 2026-07-03: the Day-2 mandate requires this check; the previously-assumed
"watchdog" (scripts/stray_tripwire.py) is a stray-FILE quarantine, a different organ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "data" / "system" / "plist_watchdog_baseline.json"
AGENTS = Path.home() / "Library" / "LaunchAgents"
PREFIXES = ("com.alta.", "com.sovereign.")


def snapshot() -> dict:
    plists = {}
    for p in sorted(AGENTS.glob("*.plist")):
        if p.name.startswith(PREFIXES):
            plists[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    r = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=10)
    loaded = sorted(line.split("\t")[-1].strip() for line in r.stdout.splitlines()
                    if line.split("\t")[-1].strip().startswith(PREFIXES))
    return {"plists": plists, "loaded": loaded}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebaseline", metavar="REASON", help="record a new baseline with a reason")
    a = ap.parse_args()
    now = datetime.now(timezone.utc).isoformat()
    snap = snapshot()

    if a.rebaseline:
        doc = {"ts": now, "reason": a.rebaseline, **snap, "history": []}
        if BASELINE.exists():
            old = json.loads(BASELINE.read_text())
            doc["history"] = old.get("history", []) + [{"ts": old.get("ts"), "reason": old.get("reason")}]
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"BASELINE RECORDED: {len(snap['plists'])} plists, {len(snap['loaded'])} loaded — {a.rebaseline}")
        return 0

    if not BASELINE.exists():
        print("RED: no baseline recorded — run with --rebaseline \"initial\" first")
        return 1
    base = json.loads(BASELINE.read_text())
    diffs = []
    for name, h in snap["plists"].items():
        if name not in base["plists"]:
            diffs.append(f"NEW plist: {name}")
        elif base["plists"][name] != h:
            diffs.append(f"MODIFIED plist: {name}")
    for name in base["plists"]:
        if name not in snap["plists"]:
            diffs.append(f"REMOVED plist: {name}")
    for lbl in set(snap["loaded"]) - set(base["loaded"]):
        diffs.append(f"NEWLY LOADED: {lbl}")
    for lbl in set(base["loaded"]) - set(snap["loaded"]):
        diffs.append(f"UNLOADED: {lbl}")
    if diffs:
        print(f"RED ({now}, baseline {base['ts']} — {base.get('reason')}):")
        for d in diffs:
            print(f"  ✗ {d}")
        return 1
    print(f"GREEN: {len(snap['plists'])} plists unchanged, {len(snap['loaded'])} loaded jobs match "
          f"(baseline {base['ts']} — {base.get('reason')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
