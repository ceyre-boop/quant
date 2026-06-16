#!/usr/bin/env python3
"""Sign / verify the VRP options-backtest pre-registration.

The signature is a SHA-256 over the canonical JSON (sorted keys, tight separators) of the
`options_backtest.split` + `options_backtest.params` blocks. Once written, any change to a
frozen parameter changes the hash — `--check` then fails, which is the tripwire: a failing
check means someone retuned the pre-registration. Do not "fix" by re-signing; revert the
parameter or log a data/agent/param_change_log.jsonl entry.

Usage:
  python3 scripts/vrp_sign_prereg.py --write    # compute hash, write into the prereg file
  python3 scripts/vrp_sign_prereg.py --check     # recompute, compare, exit non-zero on mismatch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREREG = ROOT / "data" / "research" / "vrp_preregistration.json"


def _canonical_hash(block: dict) -> str:
    payload = {"split": block["split"], "params": block["params"]}
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()

    doc = json.loads(PREREG.read_text())
    ob = doc.get("options_backtest")
    if ob is None:
        print("FATAL: options_backtest block missing from pre-registration", file=sys.stderr)
        return 2
    digest = _canonical_hash(ob)

    if args.write:
        ob["content_sha256"] = digest
        PREREG.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"signed: content_sha256 = {digest}")
        return 0

    stored = ob.get("content_sha256")
    if stored == digest:
        print(f"OK: signature matches ({digest})")
        return 0
    print(f"MISMATCH — pre-registration was altered after signing.\n  stored:   {stored}\n  computed: {digest}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
