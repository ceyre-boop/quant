#!/usr/bin/env python3
"""W5 forward-fill: daily borrow + halt snapshots. Every unsnapshotted day is
data lost forever (IBKR FTP is current-day-only; halt pages keep ~1yr).
Run daily 13:30 PT via scripts/com.alta.market_snapshots.plist (operator loads)."""
import gzip
import urllib.request
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data/research/yield_frontier"


def snap(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return "exists"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        with gzip.open(dest, "wb") as f:
            f.write(data)
        return f"{len(data)//1024}KB"
    except Exception as e:
        return f"FAILED {e}"


if __name__ == "__main__":
    d = str(date.today())
    # IBKR FTP unreachable from this network 2026-07-13 (ftp3 and ftp2+TLS both
    # timed out) — job keeps trying and FAILS LOUDLY in the log; if it never
    # succeeds, ticket a source switch (iBorrowDesk Patron CSV per W5 brief).
    print(d, "ibkr_borrow:", snap("ftp://shortstock:@ftp3.interactivebrokers.com/usa.txt",
                                  OUT / f"borrow_snapshots/{d}.txt.gz"), flush=True)
    print(d, "nasdaq_halts:", snap(
        "https://www.nasdaqtrader.com/dynamic/symdir/tradehalts.csv",
        OUT / f"halt_snapshots/nasdaq_{d}.csv.gz"), flush=True)
