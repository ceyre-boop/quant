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
    def g(*args):
        return subprocess.run(["git", "-C", str(MASTER_WT), *args],
                              capture_output=True, text=True)
    g("add", "data/icarus_status.json")
    r = g("commit", "-m", "[AUTO] ICARUS shadow daily sync", "--no-verify")
    if "nothing to commit" in (r.stdout + r.stderr):
        print("[icarus-sync] master: no change")
        return
    p = g("push", "origin", "master")
    print(f"[icarus-sync] master push: {'ok' if p.returncode == 0 else p.stderr[:120]}")


if __name__ == "__main__":
    build()
    if "--push" in sys.argv:
        push_master()
