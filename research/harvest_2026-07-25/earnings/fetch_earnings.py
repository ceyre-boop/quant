#!/usr/bin/env python3
"""Alpha Vantage EARNINGS harvest — research/harvest_2026-07-25/earnings/.

Extends the Petrules earnings-date spine. HARD BUDGET: 15 calls max.
Stops immediately on any rate-limit Note/Information response (no retries).
Re-runnable: skips tickers whose raw JSON already exists on disk.
"""
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_PATH = Path("/Users/taboost/quant/.env")

TICKERS = ["MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "JPM",
           "XOM", "UNH", "WMT", "COST", "NFLX", "ETSY", "WSM"]
SLEEP_S = 13
MAX_CALLS = 15

CSV_FIELDS = ["ticker", "fiscalDateEnding", "reportedDate", "reportedEPS",
              "estimatedEPS", "surprise", "surprisePercentage", "reportTime"]


def load_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith("ALPHA_VANTAGE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("ALPHA_VANTAGE_API_KEY not found in .env")


def fetch(ticker: str, key: str) -> dict:
    url = (f"https://www.alphavantage.co/query?function=EARNINGS"
           f"&symbol={ticker}&apikey={key}")
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def is_rate_limited(data: dict) -> bool:
    for k in ("Note", "Information"):
        v = data.get(k, "")
        if isinstance(v, str) and v:
            return True
    return False


def main() -> None:
    key = load_key()
    calls_made = 0
    succeeded, skipped, failed = [], [], []
    rate_limited_at = None

    for i, t in enumerate(TICKERS):
        out = HERE / f"{t}.json"
        if out.exists():
            skipped.append(t)
            continue
        if calls_made >= MAX_CALLS:
            print(f"BUDGET EXHAUSTED ({MAX_CALLS} calls) before {t}; stopping.")
            break
        if calls_made > 0:
            time.sleep(SLEEP_S)
        calls_made += 1
        try:
            data = fetch(t, key)
        except Exception as e:  # network failure — do NOT retry (budget)
            print(f"{t}: FETCH ERROR ({e}) — call counted, not retrying.")
            failed.append(t)
            continue
        if is_rate_limited(data):
            (HERE / f"{t}.RATELIMIT.json").write_text(json.dumps(data, indent=2))
            rate_limited_at = t
            print(f"{t}: RATE-LIMIT NOTE received — STOPPING. "
                  f"Calls made this run: {calls_made} "
                  f"(successful: {len(succeeded)}).")
            break
        if "quarterlyEarnings" not in data:
            (HERE / f"{t}.error.json").write_text(json.dumps(data, indent=2))
            print(f"{t}: unexpected payload (no quarterlyEarnings) — saved "
                  f"{t}.error.json, continuing.")
            failed.append(t)
            continue
        out.write_text(json.dumps(data, indent=2))
        succeeded.append(t)
        print(f"{t}: OK — {len(data['quarterlyEarnings'])} quarterly rows.")

    # Build spine from ALL {T}.json present (this run + any prior)
    rows = []
    per_ticker = {}
    for f in sorted(HERE.glob("*.json")):
        if f.name.endswith((".RATELIMIT.json", ".error.json")):
            continue
        d = json.loads(f.read_text())
        t = d.get("symbol", f.stem)
        q = d.get("quarterlyEarnings", [])
        per_ticker[t] = len(q)
        for rec in q:
            rows.append({
                "ticker": t,
                "fiscalDateEnding": rec.get("fiscalDateEnding", ""),
                "reportedDate": rec.get("reportedDate", ""),
                "reportedEPS": rec.get("reportedEPS", ""),
                "estimatedEPS": rec.get("estimatedEPS", ""),
                "surprise": rec.get("surprise", ""),
                "surprisePercentage": rec.get("surprisePercentage", ""),
                "reportTime": rec.get("reportTime", ""),
            })
    rows.sort(key=lambda r: (r["ticker"], r["fiscalDateEnding"]))
    spine = HERE / "earnings_spine.csv"
    with spine.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print("\n=== SUMMARY ===")
    print(f"calls_made={calls_made} succeeded={len(succeeded)} "
          f"skipped_existing={len(skipped)} failed={len(failed)} "
          f"rate_limited_at={rate_limited_at or 'none'}")
    print(f"spine_total_rows={len(rows)} -> {spine}")
    print("per_ticker_quarters:")
    for t in sorted(per_ticker):
        print(f"  {t}: {per_ticker[t]}")


if __name__ == "__main__":
    main()
