#!/usr/bin/env python3
"""ThetaData schema verifier — the ONLY file that touches the API. Deliberately tiny:
one probe, print the real schema, diff against the loader contract, exit. Fail-fast on
schema surprises.

DO NOT RUN until the ThetaData Options Value subscription is active and ThetaTerminal is
running locally. It makes exactly one HTTP GET. Its job is to tell you how to fill the
TODO bodies in ThetaDataLoader BEFORE you write any parsing code.

Usage (post-activation only):
  python3 scripts/vrp_schema_verify.py [--symbol SPY]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.research.vrp.data_loader import OPTION_CHAIN_COLUMNS  # noqa: E402

DEFAULT_BASE = "http://127.0.0.1:25510"


def _load_env() -> dict:
    env: dict = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _probe(base_url: str, symbol: str) -> dict:
    # VERIFY SCHEMA AGAINST LIVE RESPONSE — endpoint/params are ASSUMED until this runs.
    # List expirations is the smallest, cheapest probe (no strike loop).
    url = f"{base_url}/v2/list/expirations?root={symbol}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        status = resp.status
        body = resp.read().decode()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = {"_raw_text": body[:2000]}
    return {"http_status": status, "url": url, "response": parsed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    args = ap.parse_args()

    env = _load_env()
    base = env.get("THETADATA_BASE_URL", DEFAULT_BASE)
    print(f"== ThetaData schema probe ==\n  base_url: {base}\n  symbol:   {args.symbol}")
    print(f"  loader contract columns ({len(OPTION_CHAIN_COLUMNS)}): {OPTION_CHAIN_COLUMNS}\n")

    try:
        out = _probe(base, args.symbol)
    except Exception as e:                       # noqa: BLE001 — fail fast, print why
        print(f"PROBE FAILED: {e}", file=sys.stderr)
        print("  -> Is ThetaTerminal running and the Options Value tier active?", file=sys.stderr)
        return 2

    print(f"HTTP {out['http_status']}  {out['url']}")
    print("RAW RESPONSE (first level):")
    print(json.dumps(out["response"], indent=2)[:3000])
    print("\nNEXT: map the raw call/put bid/ask/iv/delta/volume/open_interest fields onto the")
    print("loader contract columns above, then fill the TODO bodies in ThetaDataLoader.")
    print("If a contract column has no raw source, STOP and reconcile the contract before coding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
