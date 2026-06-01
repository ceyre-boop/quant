"""
EdgeMonitor — sovereign/oracle/edge_monitor.py  (Loop 3: Monitor → Detect Decay → Retire)
=========================================================================================

Compares each codified lesson's LIVE performance to its backtest expectation and flags
decay. Runs monthly with the Oracle cycle.

GATED (consistent with Phase-3): EdgeMonitor DETECTS + flags + messages + marks a decayed
lesson SUSPENDED_DECAY in the research artifact (proven_research.json). It does NOT edit
live config — reverting a config-resident lesson goes through approve_edge.py with a logged
rationale (non-negotiable #4). Flagging/suspending the record is risk-reducing and safe;
mutating live config stays human-gated.

CURRENT STATE: lessons carry no expected_wr/costed_sharpe yet (fallback defaults used) and
closed trades carry no live volume (~1 real closed trade). So every lesson reports
INSUFFICIENT_DATA until trades accumulate (≥ MIN_TRADES) and are tagged with active_lessons.
This is correct, not a bug — the machinery activates automatically once live data flows.
"""
from __future__ import annotations

import glob
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROVEN = ROOT / "data" / "oracle" / "proven_research.json"
DECISION_LOG_DIR = ROOT / "data" / "decision_logs"
MESSAGES = ROOT / "data" / "agent" / "messages_to_colin.json"
REPORT = ROOT / "data" / "oracle" / "edge_monitor_report.json"

MIN_TRADES = 10            # below this → INSUFFICIENT_DATA
WR_DECAY_WARN = 0.15       # live WR this far below expected → DECAYING
WR_DECAY_SUSPEND = 0.25    # this far below → SUSPENDED_DECAY (severe)
DEFAULT_EXPECTED_WR = 0.41 # fallback when a lesson has no expected_wr yet

log = logging.getLogger("oracle.edge_monitor")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EdgeMonitor:
    def __init__(self, months: int = 3):
        self.months = months

    # ── data loaders ────────────────────────────────────────────────────────
    def _load_lessons(self) -> tuple[dict, list]:
        if not PROVEN.exists():
            return {}, []
        data = json.loads(PROVEN.read_text())
        return data, data.get("proven_lessons", [])

    def _load_closed_trades(self) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=30 * self.months)
        out = []
        for fp in glob.glob(str(DECISION_LOG_DIR / "decisions_*.jsonl")):
            for line in Path(fp).read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("outcome") in (None, "OPEN", "EXPIRED"):
                    continue   # real trades only
                try:
                    ts = datetime.fromisoformat(str(r["entry_timestamp"]).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < cutoff:
                        continue
                except Exception:
                    pass
                out.append(r)
        return out

    # ── messaging (reuse the messages_to_colin.json feed) ────────────────────
    def _message(self, priority: str, subject: str, body: str) -> None:
        try:
            data = json.loads(MESSAGES.read_text()) if MESSAGES.exists() else {"messages": []}
            data.setdefault("messages", []).insert(0, {
                "id": f"edgemon-{_now()[:19].replace(':', '').replace('-', '')}",
                "timestamp": _now(), "priority": priority, "source": "EDGE_MONITOR",
                "subject": subject, "message": body, "action_required": priority == "URGENT",
            })
            data["messages"] = data["messages"][:80]
            MESSAGES.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            log.warning("message write failed: %s", exc)

    # ── core ─────────────────────────────────────────────────────────────────
    def monthly_review(self) -> dict:
        data, lessons = self._load_lessons()
        trades = self._load_closed_trades()
        rows, suspended, decaying = [], [], []

        for lesson in lessons:
            lid = lesson.get("id")
            relevant = [t for t in trades if lid in (t.get("active_lessons") or [])]
            if len(relevant) < MIN_TRADES:
                lesson["monitor_status"] = "INSUFFICIENT_DATA"
                lesson["last_checked"] = _now()
                rows.append({"id": lid, "status": "INSUFFICIENT_DATA",
                             "n_trades": len(relevant), "need": MIN_TRADES})
                continue

            wins = sum(1 for t in relevant if (t.get("r_realized") or 0) > 0)
            live_wr = wins / len(relevant)
            live_avg_r = float(np.mean([t.get("r_realized") or 0 for t in relevant]))
            expected_wr = float(lesson.get("expected_wr", DEFAULT_EXPECTED_WR))
            wr_decay = expected_wr - live_wr

            row = {"id": lid, "n_trades": len(relevant), "live_wr": round(live_wr, 3),
                   "expected_wr": expected_wr, "wr_decay": round(wr_decay, 3),
                   "live_avg_r": round(live_avg_r, 3), "checked": _now()}

            if wr_decay > WR_DECAY_SUSPEND:
                lesson["status"] = "SUSPENDED_DECAY"      # research-artifact flag only
                lesson["monitor_status"] = "SUSPENDED_DECAY"
                lesson["live_wr"] = round(live_wr, 3)
                lesson["decay_detected"] = _now()
                row["status"] = "SUSPENDED_DECAY"
                suspended.append(lid)
                self._message("URGENT", f"AUTO-SUSPENDED (decay): {lid}",
                              f"{lid} live WR {live_wr:.0%} vs expected {expected_wr:.0%} "
                              f"(n={len(relevant)}). Marked SUSPENDED_DECAY in proven_research.json. "
                              f"Config NOT auto-changed — review + approve_edge.py to revert any "
                              f"config-resident effect (logged rationale, #4).")
            elif wr_decay > WR_DECAY_WARN:
                lesson["monitor_status"] = "DECAYING"
                row["status"] = "DECAYING"
                decaying.append(lid)
                self._message("IMPORTANT", f"DECAY DETECTED: {lid}",
                              f"{lid} live WR {live_wr:.0%} vs expected {expected_wr:.0%} "
                              f"(n={len(relevant)}). Watching; not yet suspended.")
            else:
                lesson["monitor_status"] = "HEALTHY"
                row["status"] = "HEALTHY"
            lesson["last_checked"] = _now()
            rows.append(row)

        # Persist lesson flags back to the research artifact (NOT config).
        if data:
            data["edge_monitor_last_run"] = _now()
            PROVEN.write_text(json.dumps(data, indent=2))

        summary = (f"{len(lessons)} lessons | "
                   f"{sum(1 for r in rows if r['status']=='INSUFFICIENT_DATA')} insufficient-data, "
                   f"{len(decaying)} decaying, {len(suspended)} suspended | "
                   f"{len(trades)} real closed trades in window")
        result = {"run_at": _now(), "months": self.months, "min_trades": MIN_TRADES,
                  "n_lessons": len(lessons), "n_closed_trades": len(trades),
                  "suspended": suspended, "decaying": decaying, "rows": rows,
                  "summary": summary,
                  "note": ("Inert until ≥%d real closed trades per lesson accumulate AND trades "
                           "carry active_lessons tags. Gated: suspends the record + messages; never "
                           "edits live config." % MIN_TRADES)}
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    r = EdgeMonitor().monthly_review()
    print(r["summary"])
    for row in r["rows"]:
        print(f"  {row['id']:8s} {row['status']:18s} n={row.get('n_trades', 0)}")
