#!/usr/bin/env python3
"""HYP-111 data gate — probe ThetaData for 1-minute STOCK bars. Read-only. Never prints credentials.

The terminal running here (~/ThetaTerminal/ThetaTerminalv3.jar, bundle STOCK.FREE / OPTION.VALUE)
answers the v2 REST API on :25510 (v3 paths on :25503 return 404 / no listener). Probes
`/v2/hist/stock/ohlc?ivl=60000` on five dates spanning 2020-2026 for SPY, then one date for
all ten HYP-111 instruments. Records HTTP status, ThetaData's own error text (471 = before
first-access date, 472 = no data, 403 = entitlement), row count, first/last bar, columns.

Probe dates are declared in the scope and EXCLUDED from the HYP-111 event sample.

  .venv313/bin/python scripts/research/probe_thetadata_stock_1m.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "research" / "hyp111" / "probe.json"
BASE = "http://127.0.0.1:25510"
SYMS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "XLF", "XLE"]
SPY_DATES = ["2020-03-16", "2021-06-15", "2022-06-13", "2024-01-03", "2026-08-03"]
ALL_DATE = "2024-01-03"     # one date for all ten; 2022-06-13 as scoped would 471 on this tier — try both
ALL_DATES = ["2022-06-13", "2024-01-03"]


def probe(sym: str, date: str) -> dict:
    d = date.replace("-", "")
    url = f"{BASE}/v2/hist/stock/ohlc"
    t0 = time.time()
    try:
        r = requests.get(url, params={"root": sym, "start_date": d, "end_date": d, "ivl": 60000,
                                      "rth": "true"}, timeout=60)
    except requests.RequestException as e:
        return {"symbol": sym, "date": date, "http": None, "error": type(e).__name__}
    rec = {"symbol": sym, "date": date, "http": r.status_code, "latency_s": round(time.time() - t0, 2)}
    if r.status_code != 200:
        rec["error_text"] = r.text.strip()[:200]
        return rec
    try:
        j = r.json()
    except ValueError:
        rec["error_text"] = "non-JSON body"; return rec
    fmt = j.get("header", {}).get("format", [])
    rows = j.get("response", [])
    rec.update({"columns": fmt, "rows": len(rows)})
    if rows:
        i = fmt.index("ms_of_day") if "ms_of_day" in fmt else None
        if i is not None:
            def hhmm(ms): return f"{ms // 3600000:02d}:{(ms % 3600000) // 60000:02d}"
            rec["first_bar"], rec["last_bar"] = hhmm(rows[0][i]), hhmm(rows[-1][i])
        rec["sample_row"] = rows[0]
    return rec


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = {"probed_at": datetime.now(timezone.utc).isoformat(), "base": BASE, "endpoint": "/v2/hist/stock/ohlc ivl=60000 rth=true",
           "spy_by_date": [], "all_symbols": {}}
    print(f"ThetaData 1-min STOCK probe  ({BASE})")
    for dt in SPY_DATES:
        rec = probe("SPY", dt); res["spy_by_date"].append(rec)
        print(f"  SPY {dt}: http {rec.get('http')}  rows {rec.get('rows', '-')}  "
              f"{rec.get('first_bar', '')}-{rec.get('last_bar', '')}  {rec.get('error_text', '')[:90]}")
        time.sleep(0.1)
    for dt in ALL_DATES:
        res["all_symbols"][dt] = []
        for s in SYMS:
            rec = probe(s, dt); res["all_symbols"][dt].append(rec)
            print(f"  {s:4s} {dt}: http {rec.get('http')}  rows {rec.get('rows', '-')}  {rec.get('error_text', '')[:70]}")
            time.sleep(0.1)
    # first-access date, parsed from ThetaData's 471 text if present
    fad = None
    for rec in res["spy_by_date"]:
        t = rec.get("error_text", "")
        if "first access date of" in t:
            fad = t.split("first access date of")[1].strip()[:8]
    res["first_access_date"] = fad
    ok_dates = [r["date"] for r in res["spy_by_date"] if r.get("http") == 200 and r.get("rows", 0) >= 370]
    res["verdict"] = ("SERVES_2020" if set(SPY_DATES) <= set(ok_dates)
                      else f"DEPTH_WALL first_access={fad} served={ok_dates}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nverdict: {res['verdict']}\nwritten {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
