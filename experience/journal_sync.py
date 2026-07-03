#!/usr/bin/env python3
"""Journal observer sync (W1 forward capture) — derives journal rows from logs.

READ-ONLY on every source (shadow log, decision logs, proof-of-life snapshot); never
imports or instruments the execution path. Idempotent (journal.upsert dedupes on
decision_id) — safe to run daily (com.alta.journal_sync, 09:15 ET Mon-Fri) or over
history (backfill.py drives the same functions with a window).
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience import journal  # noqa: E402

SHADOW_LOG = ROOT / "data" / "exec" / "exit_manager_shadow.jsonl"
DECISION_DIR = ROOT / "data" / "decision_logs"
PROOF_OF_LIFE = ROOT / "data" / "agent" / "proof_of_life.json"
FUNDED = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]


def rows_from_shadow(window: tuple[str, str] | None = None) -> list[dict]:
    if not SHADOW_LOG.exists():
        return []
    out = []
    for line in SHADOW_LOG.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d = r.get("bar_date") or str(r.get("run_ts", ""))[:10]
        if window and not (window[0] <= d <= window[1]):
            continue
        action = r.get("action")
        if action in ("SKIP", "SKIP_DUPLICATE"):
            continue
        out.append({
            "decision_ts": f"{d}T00:00:00+00:00",
            "decision_id": f"shadow:{r.get('trade_id')}:{d}",
            "engine": "exit_shadow",
            "pair": r.get("pair"),
            "board_ref": None,   # filled lazily below (board may lack weekend bar_dates)
            "thesis": {"kind": "structural_carry", "id": "v015",
                       "falsification_predicates": None},
            "action": action,
            "size": None,
            "detail": {"decision": r.get("decision"), "close": r.get("close"),
                       "hold_count": r.get("hold_count"),
                       "would_amend_stop_to": r.get("would_amend_stop_to")},
            "inferred": False,
            "source": "data/exec/exit_manager_shadow.jsonl",
        })
    return out


def rows_from_decision_logs(window: tuple[str, str] | None = None) -> list[dict]:
    out = []
    for f in sorted(DECISION_DIR.glob("decisions_*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("system") != "FOREX":
                continue
            d = str(r.get("entry_timestamp", ""))[:10]
            if window and not (window[0] <= d <= window[1]):
                continue
            out.append({
                "decision_ts": r.get("entry_timestamp"),
                "decision_id": f"carry:{r.get('pair')}:{r.get('entry_timestamp')}",
                "engine": "carry",
                "pair": str(r.get("pair", "")).replace("=X", "").replace("_", ""),
                "board_ref": None,
                "thesis": {"kind": "structural_carry", "id": "v015-carry",
                           "falsification_predicates": None,
                           "why": r.get("why_this_trade")},
                "action": "ENTER",
                "size": {"risk_pct": r.get("risk_pct")},
                "detail": {"direction": r.get("direction"), "outcome": r.get("outcome"),
                           "r_realized": r.get("r_realized"),
                           "rate_diff_z": r.get("rate_differential_zscore"),
                           "vix_at_entry": r.get("vix_at_entry"),
                           "exit_timestamp": r.get("exit_timestamp"),
                           "source_tag": (r.get("extra") or {}).get("source")},
                "inferred": False,
                "source": str(f.relative_to(ROOT)),
            })
    return out


def abstention_rows_today() -> list[dict]:
    """Today's NO_TRADE abstentions from the proof-of-life snapshot (Article 4 observable)."""
    if not PROOF_OF_LIFE.exists():
        return []
    try:
        pol = json.loads(PROOF_OF_LIFE.read_text())
    except json.JSONDecodeError:
        return []
    today = str(pol.get("as_of", datetime.now(timezone.utc).isoformat()))[:10]
    out = []
    for sig in pol.get("pairs", pol.get("signals", [])) or []:
        pair = (sig.get("pair") or "").replace("=X", "").replace("_", "")
        verdict = (sig.get("verdict") or sig.get("signal") or "").upper()
        if not pair or "NO_TRADE" not in verdict and verdict not in ("FLAT", "NEUTRAL"):
            continue
        out.append({
            "decision_ts": f"{today}T00:00:00+00:00",
            "decision_id": f"abstain:{pair}:{today}",
            "engine": "carry",
            "pair": pair,
            "board_ref": None,
            "thesis": {"kind": "abstention", "id": None, "falsification_predicates": None,
                       "reason": sig.get("reason") or "conviction below proven floor"},
            "action": "ABSTAIN",
            "size": None,
            "inferred": False,
            "source": "data/agent/proof_of_life.json",
        })
    return out


def inferred_abstentions(window: tuple[str, str], entered: set[tuple[str, str]]) -> list[dict]:
    """Backfill: weekdays in window with no carry ENTER for a funded pair -> inferred ABSTAIN.

    Marked inferred=True — reconstructed from absence of evidence, honestly labeled."""
    out = []
    d0, d1 = date.fromisoformat(window[0]), date.fromisoformat(window[1])
    cur = d0
    while cur <= d1:
        if cur.weekday() < 5:
            for pair in FUNDED:
                p = pair.replace("_", "")
                if (p, str(cur)) not in entered:
                    out.append({
                        "decision_ts": f"{cur}T00:00:00+00:00",
                        "decision_id": f"abstain:{p}:{cur}",
                        "engine": "carry", "pair": p, "board_ref": None,
                        "thesis": {"kind": "abstention", "id": None,
                                   "falsification_predicates": None,
                                   "reason": "no decision-log entry that day (inferred)"},
                        "action": "ABSTAIN", "size": None, "inferred": True,
                        "source": "backfill-inferred-from-absence",
                    })
        cur = date.fromordinal(cur.toordinal() + 1)
    return out


def attach_board_refs(rows: list[dict]) -> None:
    from sovereign.sentiment.store import connect
    con = connect(read_only=True)
    try:
        for r in rows:
            if r["board_ref"] is None and r.get("pair"):
                r["board_ref"] = journal.board_ref(r["decision_ts"][:10], r["pair"], con=con)
    finally:
        con.close()


def run_daily() -> int:
    rows = rows_from_shadow() + rows_from_decision_logs() + abstention_rows_today()
    attach_board_refs(rows)
    n = journal.upsert(rows)
    print(f"[journal_sync] {n} new rows (of {len(rows)} derived)")
    return n


if __name__ == "__main__":
    run_daily()
