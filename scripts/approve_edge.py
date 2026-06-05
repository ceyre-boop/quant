"""
approve_edge.py — the ONLY path that mutates live trading behavior.
===================================================================

EdgePipeline stages validated hypotheses in data/oracle/edge_review_queue.json as
VALIDATED_PENDING_APPROVAL. Nothing auto-applies. A human runs this to approve or
reject, and approval REQUIRES a rationale that is logged BEFORE any change —
honoring repo non-negotiable #4 ("No live parameter changes without logging").

Safety:
  • ICT scoring deltas (config/ict_params.yml) are applied in place (reversible yaml).
  • Forex VIX-gate deltas live in CODE (forex_backtester.py / signal_engine.py). This
    script does NOT auto-edit code — it logs the approval + prints the exact edit for
    a human to make, status APPROVED_MANUAL_APPLY. Programmatic code edits to live
    trading logic are deliberately out of scope.

Usage:
    python3 scripts/approve_edge.py --list
    python3 scripts/approve_edge.py <hyp_id> --rationale "why this is sound"
    python3 scripts/approve_edge.py --reject <hyp_id> --rationale "why not"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
QUEUE = ROOT / "data" / "oracle" / "edge_review_queue.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
CHANGE_LOG = ROOT / "data" / "agent" / "param_change_log.jsonl"
ICT_CONFIG = ROOT / "config" / "ict_params.yml"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_queue() -> dict:
    return json.loads(QUEUE.read_text()) if QUEUE.exists() else {"pending": []}


def _save_queue(q: dict) -> None:
    q["last_updated"] = _now()
    QUEUE.write_text(json.dumps(q, indent=2))


def _log_change(entry: dict) -> None:
    """Append the decision to the immutable param-change audit log (non-negotiable #4)."""
    CHANGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CHANGE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _set_ledger_status(hid: str, status: str, note: str) -> None:
    if not LEDGER.exists():
        return
    led = json.loads(LEDGER.read_text())
    for arr in ("ledger", "hypotheses"):
        for h in led.get(arr, []):
            if isinstance(h, dict) and h.get("id") == hid:
                h["status"] = status
                h["approval_note"] = note
    LEDGER.write_text(json.dumps(led, indent=2))


def _apply_ict_delta(param_delta: dict) -> str:
    """Apply an ICT scoring delta to config/ict_params.yml. Returns a description."""
    import yaml
    cfg = yaml.safe_load(ICT_CONFIG.read_text())
    scoring = cfg.setdefault("scoring", {})
    changes = []
    for k, v in param_delta.items():
        if k == "weights":
            w = scoring.setdefault("weights", {})
            for ck, cv in v.items():
                changes.append(f"weights.{ck}: {w.get(ck)} → {cv}")
                w[ck] = cv
        else:
            changes.append(f"{k}: {scoring.get(k)} → {v}")
            scoring[k] = v
    ICT_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return "; ".join(changes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hyp_id", nargs="?", help="hypothesis id to approve")
    ap.add_argument("--list", action="store_true", help="list pending approvals")
    ap.add_argument("--reject", metavar="ID", help="reject a pending hypothesis")
    ap.add_argument("--rationale", default="", help="REQUIRED for approve/reject (logged per #4)")
    args = ap.parse_args()

    q = _load_queue()
    pending = q.get("pending", [])

    if args.list or (not args.hyp_id and not args.reject):
        if not pending:
            print("Review queue empty.")
            return
        print(f"PENDING APPROVAL ({len(pending)}):")
        for v in pending:
            print(f"  {v['id']:12s} [{v.get('subsystem')}] p={v.get('p_value')}  {v.get('reason','')[:80]}")
            print(f"               delta: {json.dumps(v.get('param_delta'))}")
        return

    # ── Kill switch (HARD): block live-config mutation while frozen. --list above still works. ──
    from sovereign.utils.kill_switch import config_frozen
    frz = config_frozen()
    if frz:
        raise SystemExit(
            f"🧊 SYSTEM FROZEN (hard) — {frz.get('reason', '')}. approve_edge is blocked "
            f"(no live-config mutation while frozen). Thaw first: python3 scripts/alta.py thaw")

    target = args.reject or args.hyp_id
    item = next((v for v in pending if v["id"] == target), None)
    if item is None:
        raise SystemExit(f"'{target}' not in review queue. Run --list.")
    if not args.rationale.strip():
        raise SystemExit("--rationale is REQUIRED (logged before any change, per non-negotiable #4).")

    if args.reject:
        _log_change({"ts": _now(), "id": target, "action": "REJECT",
                     "rationale": args.rationale, "param_delta": item.get("param_delta")})
        _set_ledger_status(target, "REJECTED_BY_HUMAN", args.rationale)
        q["pending"] = [v for v in pending if v["id"] != target]
        _save_queue(q)
        print(f"REJECTED {target}. Logged. Removed from queue.")
        return

    # ── APPROVE ────────────────────────────────────────────────────────────
    delta = item.get("param_delta") or {}
    sub = item.get("subsystem")
    # Log the rationale FIRST (before any change) — non-negotiable #4.
    _log_change({"ts": _now(), "id": target, "action": "APPROVE", "subsystem": sub,
                 "rationale": args.rationale, "param_delta": delta,
                 "evidence": {"p_value": item.get("p_value"), "walkforward": item.get("walkforward")}})

    if sub == "ict":
        applied = _apply_ict_delta(delta)
        _set_ledger_status(target, "COMMITTED", f"{args.rationale} | applied: {applied}")
        print(f"APPROVED {target}. Applied to config/ict_params.yml: {applied}")
        print("Rationale logged to param_change_log.jsonl. Re-run backtests to confirm.")
    elif sub == "forex":
        # Forex VIX gates live in code — do NOT auto-edit code.
        _set_ledger_status(target, "APPROVED_MANUAL_APPLY", args.rationale)
        print(f"APPROVED {target} (rationale logged). Forex VIX gates are CODE-resident —")
        print("apply this edit by hand in BOTH files, then commit with the rationale:")
        print(f"   forex_backtester.py  PAIR_VIX_GATES  ← {json.dumps(delta.get('PAIR_VIX_GATES'))}")
        print(f"   signal_engine.py     _VIX_GATES      ← {json.dumps(delta.get('PAIR_VIX_GATES'))}")
    else:
        raise SystemExit(f"Unknown subsystem '{sub}' — cannot apply.")

    q["pending"] = [v for v in pending if v["id"] != target]
    _save_queue(q)


if __name__ == "__main__":
    main()
