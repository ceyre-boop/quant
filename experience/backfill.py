#!/usr/bin/env python3
"""W4 backfill: journal the window's decisions from existing logs (read-only) and run
attribution over the CLOSED ones — the first review's raw material.

Usage: python3 experience/backfill.py --window 2026-06-27..2026-07-02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experience import attribution as att  # noqa: E402
from experience import journal, journal_sync  # noqa: E402

RATES = {"EURUSD": ("EU", "US"), "GBPUSD": ("GB", "US"), "USDJPY": ("US", "JP"),
         "AUDUSD": ("AU", "US"), "AUDNZD": ("AU", "NZ")}
VIX_THR = {"USDJPY": 15.0, "EURUSD": 18.0, "GBPUSD": 18.0, "AUDUSD": 20.0, "AUDNZD": 15.0}
MACRO = ROOT / "data" / "cache" / "macro"


def _rate_series(cc: str) -> pd.Series | None:
    p = MACRO / f"{cc}_rates.parquet"
    if not p.exists():
        return None
    s = pd.read_parquet(p)["rate"]
    s.index = pd.to_datetime(s.index)
    return s


def rate_diff_sign(pair: str, d: str) -> int | None:
    legs = RATES.get(pair)
    if not legs:
        return None
    b, q = _rate_series(legs[0]), _rate_series(legs[1])
    if b is None or q is None:
        return None
    try:
        diff = float(b.asof(pd.Timestamp(d))) - float(q.asof(pd.Timestamp(d)))
    except Exception:
        return None
    if diff != diff:
        return None
    return 1 if diff > 0 else (-1 if diff < 0 else 0)


def vix_gate_state(pair: str, d: str) -> str | None:
    """Board-derivable gate discriminator: vix_level vs the pair threshold.

    (The full HYP-027-family gate also has a SPY-200SMA leg not present on the board;
    the rubric's proxy uses board values — this is the operative, evaluable component,
    recorded in evidence as vix_side.)"""
    from sovereign.sentiment.store import connect
    con = connect(read_only=True)
    try:
        row = con.execute("SELECT vix_level FROM sentiment_board_state WHERE date <= ? AND pair = ? "
                          "ORDER BY date DESC LIMIT 1", [d, pair]).fetchone()
    finally:
        con.close()
    if not row or row[0] is None:
        return None
    return "above_thr" if float(row[0]) > VIX_THR.get(pair, 18.0) else "below_thr"


_REASON = {"INITIAL_STOP": "STOP", "TRAILING_ATR": "TRAILING", "REVERSAL": "REVERSAL",
           "TIME": "TIME", "CB_REFRESH": "CB_REFRESH"}


def closed_decisions(rows: list[dict]) -> list[att.ClosedDecision]:
    out = []
    for r in rows:
        pair = (r.get("pair") or "").replace("_", "")
        if r["engine"] == "carry" and r["action"] == "ENTER":
            det = r.get("detail") or {}
            if det.get("outcome") not in ("WIN", "LOSS"):
                continue
            entry_d, exit_d = r["decision_ts"][:10], str(det.get("exit_timestamp") or "")[:10]
            out.append(att.ClosedDecision(
                decision_id=r["decision_id"], engine="carry", thesis_kind="structural_carry",
                rate_diff_sign_entry=rate_diff_sign(pair, entry_d),
                rate_diff_sign_exit=rate_diff_sign(pair, exit_d) if exit_d else None,
                vix_gate_entry=vix_gate_state(pair, entry_d),
                vix_gate_exit=vix_gate_state(pair, exit_d) if exit_d else None,
                # The outcome matcher records WIN/LOSS + R but NOT the exit mechanism —
                # honestly UNKNOWN → rubric routes these AMBIGUOUS. That gap is itself
                # first-review material (the machine should propose recording it).
                exit_reason="UNKNOWN",
                realized_r=det.get("r_realized"), fill_slippage_r=None))
        elif r["engine"] == "exit_shadow" and r["action"] == "CLOSE":
            det = r.get("detail") or {}
            d = r["decision_ts"][:10]
            out.append(att.ClosedDecision(
                decision_id=r["decision_id"], engine="exit_shadow", thesis_kind="structural_carry",
                rate_diff_sign_entry=None,   # entry date unknown from the shadow row alone
                rate_diff_sign_exit=rate_diff_sign(pair, d),
                vix_gate_entry=None, vix_gate_exit=vix_gate_state(pair, d),
                exit_reason=_REASON.get(det.get("decision"), "UNKNOWN"),
                realized_r=None,             # shadow never closed the broker trade
                fill_slippage_r=None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True, help="YYYY-MM-DD..YYYY-MM-DD")
    a = ap.parse_args()
    w = tuple(a.window.split(".."))

    rows = journal_sync.rows_from_shadow(w) + journal_sync.rows_from_decision_logs(w)
    entered = {(r["pair"], r["decision_ts"][:10]) for r in rows if r["action"] == "ENTER"}
    rows += journal_sync.inferred_abstentions(w, entered)
    journal_sync.attach_board_refs(rows)
    n = journal.upsert(rows)

    closed = closed_decisions(rows)
    atts = [att.classify(c) for c in closed]
    months = {r["decision_ts"][:7] for r in rows}
    n_att = sum(att.write_attributions(
        [x for x in atts for r in rows if r["decision_id"] == x.decision_id
         and r["decision_ts"][:7] == m], m) for m in months)
    from collections import Counter
    print(f"[backfill] window {a.window}: {n} journal rows written "
          f"({len(rows)} derived, {sum(1 for r in rows if r.get('inferred'))} inferred abstentions); "
          f"{n_att} attributions ({Counter(x.cls for x in atts)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
