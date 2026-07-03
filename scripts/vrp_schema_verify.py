#!/usr/bin/env python3
"""ThetaData V3 schema verifier — the activation gate. Adaptive transport:

  * Local ThetaTerminal v3 : THETADATA_BASE_URL unset -> http://127.0.0.1:25503, NO auth
                             (the terminal authenticates itself; it must be RUNNING).
  * Hosted REST            : set THETADATA_BASE_URL=https://<host> -> Authorization: Bearer.

Only ever calls the host YOU configure in .env. Confirms the live v3 option-chain schema
against the loader contract and reports the earliest available expiration (IS-boundary).

Verified live 2026-06-16: v3 uses `symbol` (not `root`), CSV responses, dates YYYY-MM-DD,
strikes in decimal dollars; option EOD = one row per (strike,right) with bid/ask/close/
volume (no greeks/iv/oi -> optional NaN); stock history is FREE-tier-gated (403).

Usage:  python3 scripts/vrp_schema_verify.py [--symbol SPY]
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.research.vrp.data_loader import OPTION_CHAIN_COLUMNS  # noqa: E402

DEFAULT_BASE = "http://127.0.0.1:25503"
REQUIRED = ["strike", "call_bid", "call_ask", "call_mid", "put_bid", "put_ask", "put_mid"]


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


def _get(base: str, path: str, headers: dict, timeout: int = 20) -> str:
    req = urllib.request.Request(base + path, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    args = ap.parse_args()

    env = _load_env()
    base = env.get("THETADATA_BASE_URL", DEFAULT_BASE).rstrip("/")
    local = "127.0.0.1" in base or "localhost" in base
    headers = {} if local else {"Authorization": f"Bearer {env.get('THETADATA_API_KEY', '')}"}

    print("== ThetaData V3 schema probe ==")
    print(f"  base_url : {base}  ({'LOCAL terminal, no auth' if local else 'HOSTED, bearer auth'})")
    print(f"  required : {REQUIRED}\n")

    # 1) expirations (connectivity + IS-boundary)
    try:
        body = _get(base, f"/v3/option/list/expirations?symbol={args.symbol}", headers)
    except urllib.error.HTTPError as e:
        print(f"PROBE FAILED: HTTP {e.code} {e.reason}", file=sys.stderr)
        if e.code in (401, 403):
            print("  -> auth/subscription rejected. Check THETADATA_API_KEY / tier.", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"PROBE FAILED: {e}", file=sys.stderr)
        print(f"  -> nothing on {base}. Start ThetaTerminal v3 (java -jar ThetaTerminalv3.jar) or set "
              "THETADATA_BASE_URL to your hosted host.", file=sys.stderr)
        return 2

    exps = pd.read_csv(StringIO(body))["expiration"].astype(str).str[:10].sort_values()
    earliest = exps.iloc[0]
    print(f"/v3/option/list/expirations -> {len(exps)} expirations, earliest {earliest}")
    print(f"  IS-2022 boundary: {'OK (<= 2022-01-01)' if earliest <= '2022-01-01' else 'earliest > 2022-01-01 -> STOP, log param_change'}")

    # 2) one real chain -> schema diff against the contract
    # TICK-001: probe an expiration ON/AFTER the probe date (the old median-of-all-expirations
    # pick chose a ~2019 expired contract against the fixed 2022-03-07 probe -> HTTP 472 NO_DATA).
    probe_date = "2022-03-07"
    live_exps = exps[exps >= probe_date]
    if live_exps.empty:
        print(f"  no listed expiration on/after {probe_date} — cannot probe", file=sys.stderr)
        return 2
    exp = live_exps.iloc[0]
    raw = pd.read_csv(StringIO(_get(
        base, f"/v3/option/history/eod?symbol={args.symbol}&expiration={exp}"
              f"&start_date={probe_date}&end_date={probe_date}",
        headers)))
    print(f"  probe: expiration {exp} (nearest >= {probe_date})")
    print(f"\n/v3/option/history/eod raw columns: {list(raw.columns)}")
    have = {"strike", "right", "bid", "ask"}.issubset(raw.columns)
    print(f"  MATCH (strike,right,bid,ask present -> contract derivable): {have}")
    print(f"  optional (greeks/iv/oi): {'present' if {'delta','implied_vol'} & set(raw.columns) else 'absent -> NaN'}")
    print(f"\n  contract columns the loader emits: {OPTION_CHAIN_COLUMNS}")
    print("  NOTE: stock history is FREE-tier (403); the backtest sources spot from yfinance.")
    return 0 if have else 1


if __name__ == "__main__":
    raise SystemExit(main())
