#!/usr/bin/env python3
"""TICK-037 — Daily IB shortable-list snapshot vs the gapper signal universe.

Source: IB public shortable stock list, ftp.interactivebrokers.com
/usa.txt (host ftp2) (pipe-delimited, no auth). Fields include symbol,
available-to-borrow quantity and indicative fee rate. Availability tier:
  EASY        available >= 10,000 shares
  HARD        0 < available < 10,000 shares
  UNAVAILABLE listed with 0 available
  NOT_LISTED  not on the file at all
Cross-referenced against every distinct ticker in
data/research/gapper/per_candidate_enriched.csv (full candidate universe, not
just qualifying events — locate matters at scan time).
Output: data/research/gapper/ib_locate_YYYY-MM-DD.json
Read-only research data collection; touches no live system.
"""
import csv
import json
import sys
from datetime import date
from ftplib import FTP
from io import BytesIO
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data/research/gapper/per_candidate_enriched.csv"
OUTDIR = REPO / "data/research/gapper"
EASY_MIN = 10_000


def fetch_usa_txt():
    ftp = FTP("ftp2.interactivebrokers.com", timeout=60)
    ftp.login("shortstock", "")          # public list: user 'shortstock'
    buf = BytesIO()
    ftp.retrbinary("RETR usa.txt", buf.write)
    ftp.quit()
    return buf.getvalue().decode("latin-1")


def parse(txt):
    """usa.txt: '#SYM|CUR|NAME|CON|ISIN|REBATE|FEE|AVAILABLE|...' with #BOF/#EOF."""
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("|")
        if len(f) < 8:
            continue
        sym = f[0].strip()
        avail_raw = f[7].strip().replace(">", "").replace(",", "")
        try:
            avail = int(avail_raw)
        except ValueError:
            avail = 0
        try:
            fee = float(f[6])
        except ValueError:
            fee = None
        out[sym] = {"available": avail, "fee_pct": fee}
    return out


def main():
    universe = sorted({r["ticker"] for r in csv.DictReader(open(CSV))})
    txt = fetch_usa_txt()
    listed = parse(txt)
    if not listed:
        print("ERROR: usa.txt parsed to zero rows — aborting, no snapshot written")
        sys.exit(1)

    detail, counts = {}, {"EASY": 0, "HARD": 0, "UNAVAILABLE": 0, "NOT_LISTED": 0}
    for t in universe:
        rec = listed.get(t)
        if rec is None:
            tier = "NOT_LISTED"
        elif rec["available"] >= EASY_MIN:
            tier = "EASY"
        elif rec["available"] > 0:
            tier = "HARD"
        else:
            tier = "UNAVAILABLE"
        counts[tier] += 1
        detail[t] = {"tier": tier, **(rec or {})}

    today = date.today().isoformat()
    out = {"date": today, "source": "ftp.interactivebrokers.com/usa.txt",
           "n_universe": len(universe), "n_ib_listed_total": len(listed),
           "easy_min_shares": EASY_MIN, "summary": counts, "detail": detail}
    path = OUTDIR / f"ib_locate_{today}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
