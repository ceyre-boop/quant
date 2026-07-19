#!/usr/bin/env python3
"""Reconcile AUTHORED plists against INSTALLED and LOADED launchd agents.

WHY THIS EXISTS
---------------
Authoring a plist and installing it are deliberately separate steps here — an
agent writes it into `scripts/`, the operator promotes it. That convention is
correct and is not what this tool changes.

What was missing is anything that checks the gap. On 2026-07-18 six plists had
been authored and never installed:

    com.alta.sentiment_update      (07:45 "so the board is fresh for the 08:00 scan")
    com.alta.gdelt_retry
    com.alta.esnq.brief
    com.alta.ib_shortable
    com.alta.hyp107_shadow
    com.alta.oracle.market_briefing

The first of those meant the entire sentiment pipeline ran dark for 11 days while
every dashboard read green, because nothing anywhere compared "jobs we wrote"
against "jobs that exist".

RELATIONSHIP TO plist_watchdog.py
---------------------------------
They guard opposite failure modes and are complementary, not redundant:

    plist_watchdog.py  -> catches UNEXPECTED plists   (intrusion / rogue writer)
    plist_manifest.py  -> catches EXPECTED-BUT-ABSENT (the silent-dark failure)

This tool REPORTS. It never installs, loads, or unloads anything — operator
promotion is preserved. Exits non-zero on drift so it can gate a health check.
"""
from __future__ import annotations

import argparse
import json
import plistlib
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORED_DIR = ROOT / "scripts"
INSTALLED_DIR = Path.home() / "Library" / "LaunchAgents"

#: Authored plists that are intentionally NOT installed. Each needs a reason, so
#: "deliberately parked" can never be confused with "forgotten".
INTENTIONALLY_UNINSTALLED = {
    "com.alta.esnq.brief":
        "ES/NQ 5-input bias engine KILLED on evidence 2026-06-10 (p=0.567, all "
        "inputs sub-base-rate). Code and tests remain but MUST NOT be revived.",
    "com.alta.hyp107_shadow":
        "Superseded by com.alta.execution_harness (TICK-038), which unifies both "
        "gapper legs and captures real bid/ask.",
    "com.alta.gapper_shadow_scan":
        "Superseded by com.alta.execution_harness (TICK-038). Currently exits 1.",
    "com.alta.gapper_shadow_close":
        "Superseded by com.alta.execution_harness (TICK-038).",
}


@dataclass
class Row:
    label: str
    authored: bool
    installed: bool
    loaded: bool
    last_exit: str | None
    identical: bool | None
    note: str = ""

    def state(self) -> str:
        if self.authored and not self.installed:
            return "PARKED" if self.label in INTENTIONALLY_UNINSTALLED else "NOT_INSTALLED"
        if self.installed and not self.loaded:
            return "INSTALLED_NOT_LOADED"
        if self.installed and not self.authored:
            return "UNTRACKED"
        if self.identical is False:
            return "DRIFTED"
        if self.last_exit not in (None, "0", "-"):
            return "FAILING"
        return "OK"


def _label_of(p: Path) -> str | None:
    try:
        with open(p, "rb") as fh:
            return plistlib.load(fh).get("Label")
    except Exception:                                        # noqa: BLE001
        m = re.match(r"(.+)\.plist$", p.name)
        return m.group(1) if m else None


def _launchctl_state() -> dict[str, str]:
    """label -> last exit status string, from `launchctl list`."""
    out: dict[str, str] = {}
    try:
        res = subprocess.run(["launchctl", "list"], capture_output=True,
                             text=True, timeout=20)
    except Exception:                                        # noqa: BLE001
        return out
    for line in res.stdout.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            out[parts[2].strip()] = parts[1].strip()
    return out


#: Only labels in these namespaces are ours. Everything else on the machine
#: (com.google.*, com.pai.*, ...) is third-party and not this repo's business.
OUR_NAMESPACES = ("com.alta.", "com.clawd.", "com.sovereign.")


def is_ours(label: str) -> bool:
    return label.startswith(OUR_NAMESPACES)


def scan() -> list[Row]:
    authored: dict[str, Path] = {}
    for p in sorted(AUTHORED_DIR.glob("*.plist")):
        lbl = _label_of(p)
        if lbl:
            authored[lbl] = p

    installed: dict[str, Path] = {}
    if INSTALLED_DIR.exists():
        for p in sorted(INSTALLED_DIR.glob("*.plist")):
            lbl = _label_of(p)
            if lbl:
                installed[lbl] = p

    loaded = _launchctl_state()

    rows: list[Row] = []
    for lbl in sorted(l for l in (set(authored) | set(installed)) if is_ours(l)):
        a, i = authored.get(lbl), installed.get(lbl)
        identical = None
        if a and i:
            identical = a.read_bytes() == i.read_bytes()
        rows.append(Row(
            label=lbl, authored=a is not None, installed=i is not None,
            loaded=lbl in loaded, last_exit=loaded.get(lbl),
            identical=identical,
            note=INTENTIONALLY_UNINSTALLED.get(lbl, ""),
        ))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reconcile authored vs installed launchd plists")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on any non-OK/PARKED state")
    args = ap.parse_args(argv)

    rows = scan()
    if args.json:
        print(json.dumps([{**asdict(r), "state": r.state()} for r in rows], indent=2))
    else:
        print(f"{'STATE':<21} {'LABEL':<38} A I L  EXIT")
        print("-" * 78)
        for r in rows:
            print(f"{r.state():<21} {r.label:<38} "
                  f"{'Y' if r.authored else '-'} {'Y' if r.installed else '-'} "
                  f"{'Y' if r.loaded else '-'}  {r.last_exit or '-'}")
            if r.note:
                print(f"{'':<21}   ↳ parked: {r.note}")

        bad = [r for r in rows if r.state() not in ("OK", "PARKED")]
        print()
        if not bad:
            print("GREEN: every authored plist is installed, loaded and exiting clean.")
        else:
            print(f"DRIFT: {len(bad)} job(s) need attention —")
            for r in bad:
                print(f"  {r.state():<21} {r.label}")
            untracked = [r for r in bad if r.state() == "UNTRACKED"]
            if untracked:
                print(f"\n  UNTRACKED ({len(untracked)}) is the serious class: these jobs are")
                print("  LOADED AND RUNNING with no plist committed in scripts/. They cannot")
                print("  be code-reviewed, reproduced on another machine, or recovered if this")
                print("  one dies. Export each with:")
                print("    cp ~/Library/LaunchAgents/<label>.plist scripts/ && git add scripts/<label>.plist")
            print("\nThis tool reports only. Promotion is the operator's:")
            print("  cp scripts/<label>.plist ~/Library/LaunchAgents/")
            print("  launchctl load ~/Library/LaunchAgents/<label>.plist")
            print("  python3 scripts/plist_watchdog.py --rebaseline \"loaded <label>\"")

    if args.strict:
        return 1 if any(r.state() not in ("OK", "PARKED") for r in rows) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
