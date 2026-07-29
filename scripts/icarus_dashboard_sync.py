#!/usr/bin/env python3
"""ICARUS dashboard sync — shadow results -> data/icarus_status.json -> master.

ICARUS is the operator-facing name of the sealed HYP-093 parabolic-gapper fade
(flew too high by 10:30; we sell the fall). This script aggregates the live
shadow record into one committed JSON the Render dashboard reads, and (with
--push) lands it on master via the standing worktree, data-only (814d1e2
pattern — never a merge).

Run: python3 scripts/icarus_dashboard_sync.py [--push]
Called automatically by live_shadow.py --close.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHADOW = REPO / "data/research/yield_frontier/shadow"
OUT = REPO / "data/icarus_status.json"
MASTER_WT = Path.home() / "quant-master-wt"

SEALED = {
    "hypothesis": "HYP-093", "verdict": "VALID_BUT_BELOW_FLOOR",
    "p_value": 0.031, "dsr_at_809_trials": 0.987, "holdout_events": 559,
    "event_median_net": 0.049, "event_mean_net": 0.016,
    "constitutional_floor_pct_day": 0.05,
    "status_line": "signal REAL (first to survive full-penalty significance on unseen data) · income gated below the constitutional floor · SHADOW ONLY — no live capital (Art. 6 source-tagged)",
}


def build():
    daily = []
    if (SHADOW / "shadow_daily.jsonl").exists():
        for line in (SHADOW / "shadow_daily.jsonl").read_text().splitlines():
            if line.strip():
                daily.append(json.loads(line))
    days = {}
    for d in daily:
        days[d["date"]] = {"date": d["date"], "ret": d["constitutional_day_ret"],
                           "n": d["n_signals"], "trades": []}
    for fp in sorted(SHADOW.glob("signals_*.json")):
        doc = json.loads(fp.read_text())
        dt = fp.stem.replace("signals_", "")
        for s in doc.get("signals", []):
            if dt in days and s.get("outcome") == "CLOSED":
                days[dt]["trades"].append({
                    "ticker": s["ticker"], "gain_1030": s["gain_1030"],
                    "entry": s["entry_open_1030"], "exit": s.get("exit_px"),
                    "ret": s.get("event_ret_net"), "stopped": s.get("stopped")})
    series = sorted(days.values(), key=lambda x: x["date"])
    rets = [d["ret"] for d in series]
    cum = 1.0
    for r in rets:
        cum *= (1 + r)
    doc = {
        "name": "ICARUS", "tagline": "The parabolic fade — flew too high by 10:30; we sell the fall.",
        "sealed": SEALED,
        "shadow": {
            "mode": "SHADOW (sim) — Art. 6 source-tagged, zero live capital",
            "days": len(series),
            "cum_return": round(cum - 1, 6),
            "mean_pct_day": round(sum(rets) / len(rets), 6) if rets else 0,
            "green_days": sum(1 for r in rets if r > 0),
            "red_days": sum(1 for r in rets if r < 0),
            "daily": series,
        },
        "next_gates": ["W6 RCK sizing simulator", "TICK-034 catalyst split",
                       ">=20 shadow days", "TICK-024 cascade", "clamps enforced (Jul 28)",
                       "broker account", "operator go"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"[icarus-sync] {OUT.name}: {len(series)} day(s), cum {cum - 1:+.4%}")
    return doc


def push_master():
    if not MASTER_WT.exists():
        print("[icarus-sync] master worktree missing — skipping push", file=sys.stderr)
        return
    dest = MASTER_WT / "data/icarus_status.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(OUT.read_text())
    digest = REPO / "data/oracle/daily_digest.json"
    if digest.exists():
        (MASTER_WT / "data/oracle").mkdir(parents=True, exist_ok=True)
        (MASTER_WT / "data/oracle/daily_digest.json").write_text(digest.read_text())
    def g(*args):
        return subprocess.run(["git", "-C", str(MASTER_WT), *args],
                              capture_output=True, text=True)

    # Realign onto origin/master BEFORE committing.
    #
    # Why: this worktree only ever regenerates the two snapshot files below. When
    # a dashboard commit lands on origin/master that this worktree never pulled,
    # every subsequent push is rejected non-fast-forward. That happened silently
    # from shadow day 6 and stranded 15 commits on one machine (TICK-097).
    #
    # Because the files are regenerated snapshots, divergent [AUTO] history carries
    # no information — only the newest content does. `reset --soft` moves the branch
    # onto origin/master while keeping the freshly-written files, which can never
    # conflict. Guarded: bail out rather than discard anything that is not our own
    # [AUTO] commit, so a human commit in this worktree is never silently dropped.
    g("fetch", "origin", "master")
    divergent = g("log", "--format=%s", "origin/master..HEAD").stdout.split("\n")
    divergent = [s for s in divergent if s.strip()]
    foreign = [s for s in divergent if not s.startswith("[AUTO] ICARUS")]
    if foreign:
        print("[icarus-sync] master worktree holds non-[AUTO] commits; refusing to "
              f"realign. Reconcile by hand: {foreign[:3]}", file=sys.stderr)
        sys.exit(1)
    if divergent:
        print(f"[icarus-sync] realigning onto origin/master, "
              f"folding {len(divergent)} stale [AUTO] commit(s)")
        g("reset", "--soft", "origin/master")

    g("add", "data/icarus_status.json", "data/oracle/daily_digest.json")
    r = g("commit", "-m", "[AUTO] ICARUS shadow daily sync", "--no-verify")
    if "nothing to commit" in (r.stdout + r.stderr):
        print("[icarus-sync] master: no change")
        return
    p = g("push", "origin", "master")
    if p.returncode != 0:
        # Must exit non-zero: printing the rejection as ordinary output is what let
        # this fail unnoticed for weeks while launchd recorded success every day.
        print(f"[icarus-sync] master push FAILED: {p.stderr.strip()[:300]}",
              file=sys.stderr)
        sys.exit(1)
    print("[icarus-sync] master push: ok")


if __name__ == "__main__":
    build()
    if "--push" in sys.argv:
        push_master()
