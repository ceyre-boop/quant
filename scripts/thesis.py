#!/usr/bin/env python3
"""Thesis engine CLI.

  thesis new  spec.json            register a DRAFT thesis from a JSON spec
  thesis seal THS-001              hash-seal the basket (immutable afterwards)
  thesis live THS-001              mark as entered (first fill done)
  thesis score THS-001 [--asof D]  basket vs benchmark; appends a scorecard
  thesis list                      one line per thesis
  thesis verify THS-001            recompute the hash
  thesis reconcile --account A     measured TWR/MWR vs SPY from data/thesis/fills
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.thesis import ledger  # noqa: E402
from sovereign.thesis.prices import load_daily  # noqa: E402
from sovereign.thesis.score import format_scorecard, score  # noqa: E402
from sovereign.thesis.reconcile import format_reconciliation, reconcile  # noqa: E402

OUT = ROOT / "research" / "thesis"


def cmd_new(a):
    spec = json.loads(Path(a.spec).read_text())
    r = ledger.create(spec)
    print(f"DRAFT {r['id']} registered ({len(r['basket'])} legs). Next: thesis seal {r['id']}")


def cmd_seal(a):
    r = ledger.seal(a.id)
    print(f"SEALED {r['id']} hash {r['hash_lock']} at {r['sealed_at']}")
    print(f"  prereg: data/thesis/preregister/{r['id']}.json")


def cmd_live(a):
    r = ledger.get(a.id)
    if not r.get("hash_lock"):
        raise SystemExit(f"{a.id} is not sealed; seal before entering. No fills before the seal.")
    ledger.set_status(a.id, "LIVE", a.note)
    print(f"LIVE {a.id}")


def cmd_score(a):
    r = ledger.get(a.id)
    card = score(r, load_daily, a.asof)
    print(format_scorecard(r, card))
    if not a.dry_run:
        ledger.add_scorecard(a.id, card)
        d = OUT / a.id / "output"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"scorecard_{card['asof']}.json").write_text(json.dumps(card, indent=2) + "\n")
        print(f"  → {d.relative_to(ROOT)}/scorecard_{card['asof']}.json")
    if r.get("invalidation", {}).get("manual_check"):
        print(f"  MANUAL CHECK DUE: {r['invalidation']['manual_check']}")


def cmd_list(a):
    for r in ledger.read_all():
        last = r["scorecards"][-1] if r.get("scorecards") else None
        s = f"  {last['asof']} basket {last['basket_return']:+.1%} vs {last['benchmark']} {last['benchmark_return']:+.1%}" if last else ""
        print(f"{r['id']:9s} {r['status']:11s} {str(r.get('hash_lock') or '-')[:8]:8s} {r['entry_date']}  {r['title']}{s}")


def cmd_verify(a):
    ok = ledger.verify(a.id)
    print(f"{a.id}: {'HASH OK' if ok else 'HASH MISMATCH / not sealed'}")
    raise SystemExit(0 if ok else 1)


def cmd_reconcile(a):
    baskets = {}
    for b in a.basket or []:
        name, tk = b.split("=", 1)
        baskets[name] = set(t.strip().upper() for t in tk.split(","))
    r = reconcile(a.account, load_daily, a.asof, baskets)
    print(format_reconciliation(r))
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=2) + "\n")
        print(f"  → {p}")


def main():
    ap = argparse.ArgumentParser(prog="thesis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new"); p.add_argument("spec"); p.set_defaults(fn=cmd_new)
    p = sub.add_parser("seal"); p.add_argument("id"); p.set_defaults(fn=cmd_seal)
    p = sub.add_parser("live"); p.add_argument("id"); p.add_argument("--note"); p.set_defaults(fn=cmd_live)
    p = sub.add_parser("score"); p.add_argument("id"); p.add_argument("--asof"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_score)
    p = sub.add_parser("list"); p.set_defaults(fn=cmd_list)
    p = sub.add_parser("verify"); p.add_argument("id"); p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("reconcile"); p.add_argument("--account", required=True); p.add_argument("--asof", default=None)
    p.add_argument("--basket", action="append", help="NAME=T1,T2,...")
    p.add_argument("--out"); p.set_defaults(fn=cmd_reconcile)
    a = ap.parse_args()
    if getattr(a, "asof", None) is None and a.cmd in ("score", "reconcile"):
        import datetime as _dt
        a.asof = _dt.date.today().isoformat()
    try:
        a.fn(a)
    except ledger.LedgerError as e:
        raise SystemExit(f"REFUSED: {e}")


if __name__ == "__main__":
    main()
