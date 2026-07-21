#!/usr/bin/env python3
"""Petrules Gate Phase 0 — data availability probes.

Read-only audit legwork. Hits each candidate data source with a small probe,
saves the raw response shape to research/petrules_audit/probes/, and computes
the two make-or-break measurements:

  Q1: do analyst-estimate sources expose point-in-time (as-of) values?
  Q2: disclosed-flow lead-time distributions vs. the events they should predict.

No model code. Imports nothing from the live execution path.
Availability timestamps use filing/publication dates ONLY — never
transaction dates or period-of-report dates.

Usage: python3 research/petrules_audit/probe_sources.py [probe_name ...]
       (no args = run all)
"""
import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
PROBES = HERE / "probes"
PROBES.mkdir(exist_ok=True)

# manual .env parse (same pattern as FRED loader elsewhere in repo)
ENV = {}
env_path = Path.home() / "quant" / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        ENV[k.strip()] = v.strip().strip('"').strip("'")

UA = {"User-Agent": "Alta Research colineyre222@gmail.com"}


def fetch(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def save(name, payload):
    p = PROBES / f"{name}.json"
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload, indent=1))
    print(f"  saved {p.name} ({p.stat().st_size} bytes)")


def truncate_shape(obj, depth=0):
    """Show structure of a JSON object without dumping everything."""
    if isinstance(obj, dict):
        return {k: truncate_shape(v, depth + 1) for k, v in list(obj.items())[:8]}
    if isinstance(obj, list):
        return [truncate_shape(obj[0], depth + 1), f"...({len(obj)} items)"] if obj else []
    return obj


# ---------------------------------------------------------------- Q1 probes

def probe_alphavantage():
    """EARNINGS (historical reported vs estimate) + EARNINGS_ESTIMATES (current)."""
    key = ENV["ALPHA_VANTAGE_API_KEY"]
    out = {}
    for fn, sym in [("EARNINGS", "AAPL"), ("EARNINGS", "DKS"), ("EARNINGS_ESTIMATES", "AAPL")]:
        url = f"https://www.alphavantage.co/query?function={fn}&symbol={sym}&apikey={key}"
        raw = fetch(url)
        out[f"{fn}_{sym}"] = json.loads(raw)
        time.sleep(1)
    save("alphavantage", out)
    q = out["EARNINGS_AAPL"].get("quarterlyEarnings", [])
    print(f"  AV EARNINGS AAPL: {len(q)} quarters; earliest {q[-1]['fiscalDateEnding'] if q else '?'}")
    if q:
        print(f"  fields: {list(q[0].keys())}")
    est = out["EARNINGS_ESTIMATES_AAPL"]
    print(f"  EARNINGS_ESTIMATES top-level keys: {list(est.keys())}")
    print(json.dumps(truncate_shape(est), indent=1)[:1200])


def probe_yahoo():
    """quoteSummary earningsTrend — expect current-only snapshot."""
    url = ("https://query2.finance.yahoo.com/v10/finance/quoteSummary/AAPL"
           "?modules=earningsTrend,earningsHistory")
    try:
        raw = fetch(url, headers={"User-Agent": "Mozilla/5.0"})
        save("yahoo_earningstrend", raw)
        obj = json.loads(raw)
        print(json.dumps(truncate_shape(obj), indent=1)[:1500])
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        save("yahoo_earningstrend_error", {"status": e.code, "body": body})
        print(f"  Yahoo HTTP {e.code}: {body[:200]}")


def probe_finnhub():
    """No key on file — probe the estimates endpoint unauthenticated to confirm access tier."""
    url = "https://finnhub.io/api/v1/stock/eps-estimate?symbol=AAPL&freq=quarterly"
    try:
        raw = fetch(url)
        save("finnhub", raw)
        print(raw[:300])
    except urllib.error.HTTPError as e:
        save("finnhub_error", {"status": e.code, "body": e.read().decode()[:300]})
        print(f"  Finnhub HTTP {e.code} (expected — no key)")


def probe_nasdaq():
    """Nasdaq.com public analyst endpoints — current-only expected."""
    for name, url in [
        ("nasdaq_eps", "https://api.nasdaq.com/api/analyst/AAPL/earnings-forecast"),
        ("nasdaq_ratings", "https://api.nasdaq.com/api/analyst/AAPL/ratings"),
    ]:
        try:
            raw = fetch(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            save(name, raw)
            print(f"  {name}: {raw[:200]}")
        except Exception as e:
            save(name + "_error", {"error": str(e)})
            print(f"  {name}: {e}")


def probe_thetadata():
    """Historical event-dated straddle: AAPL earnings 2023-08-03 (after close).
    Need chain quotes at T-1 close (2023-08-02) for expiry 2023-08-04."""
    base = "http://127.0.0.1:25503/v3"
    exps = fetch(f"{base}/option/list/expirations?symbol=AAPL")
    lines = exps.strip().splitlines()
    print(f"  expirations: {len(lines)-1} rows, first {lines[1]}, last {lines[-1]}")
    save("theta_expirations_head", {"first": lines[1], "last": lines[-1], "n": len(lines) - 1})
    # EOD chain quote on 2023-08-02 for the event-dated 2023-08-04 expiry
    url = (f"{base}/option/history/eod?symbol=AAPL&expiration=2023-08-04"
           f"&start_date=2023-08-02&end_date=2023-08-02")
    raw = fetch(url, timeout=90)
    save("theta_aapl_20230802_chain", raw[:20000])
    rows = raw.strip().splitlines()
    print(f"  event-dated chain rows for 2023-08-04 exp on 2023-08-02: {len(rows)-1}")
    print("  header:", rows[0][:200] if rows else "EMPTY")
    if len(rows) > 1:
        print("  sample:", rows[1][:200])


# ---------------------------------------------------------------- Q2 probes

TICKERS_CIK = {  # 12 large/mid caps across sectors
    "AAPL": 320193, "MSFT": 789019, "NVDA": 1045810, "JPM": 19617,
    "XOM": 34088, "UNH": 731766, "CAT": 18230, "DE": 315189,
    "DKS": 1089063, "ETSY": 1370637, "CROX": 1334036, "WSM": 719955,
}
INSTITUTION_CIK = {"Berkshire": 1067983, "Bridgewater": 1350694, "Renaissance": 1037389}


def _edgar_submissions(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    time.sleep(0.15)
    return json.loads(fetch(url))


def _recent_filings(sub):
    r = sub["filings"]["recent"]
    return list(zip(r["form"], r["filingDate"], r["reportDate"], r["accessionNumber"]))


def probe_edgar_form4():
    """Form 4 disclosure lag: filingDate - reportDate (transaction date), across tickers."""
    lags, per_ticker = [], {}
    for t, cik in TICKERS_CIK.items():
        sub = _edgar_submissions(cik)
        f4 = [(fd, rd) for form, fd, rd, _ in _recent_filings(sub) if form == "4" and rd]
        tl = []
        for fd, rd in f4:
            d = (date.fromisoformat(fd) - date.fromisoformat(rd)).days
            if 0 <= d <= 60:  # guard malformed
                tl.append(d)
        per_ticker[t] = {"n": len(tl), "median_lag_days": statistics.median(tl) if tl else None}
        lags += tl
        print(f"  {t}: n={len(tl)} median lag {per_ticker[t]['median_lag_days']}")
    result = {
        "n_total": len(lags),
        "median_lag_days": statistics.median(lags),
        "p90_lag_days": sorted(lags)[int(0.9 * len(lags))],
        "frac_within_2bd": sum(1 for x in lags if x <= 4) / len(lags),
        "per_ticker": per_ticker,
    }
    save("edgar_form4_lag", result)
    print(f"  FORM4 lag: n={result['n_total']} median={result['median_lag_days']}d "
          f"p90={result['p90_lag_days']}d frac<=4d={result['frac_within_2bd']:.2f}")


def probe_edgar_13f():
    """13F staleness: filingDate - reportDate (period end)."""
    out = {}
    lags = []
    for name, cik in INSTITUTION_CIK.items():
        sub = _edgar_submissions(cik)
        f13 = [(fd, rd) for form, fd, rd, _ in _recent_filings(sub)
               if form.startswith("13F-HR") and rd]
        tl = [(date.fromisoformat(fd) - date.fromisoformat(rd)).days for fd, rd in f13]
        out[name] = {"n": len(tl), "median_staleness_days": statistics.median(tl) if tl else None,
                     "sample": f13[:3]}
        lags += tl
        print(f"  {name}: n={len(tl)} median staleness {out[name]['median_staleness_days']}d")
    out["all_median"] = statistics.median(lags)
    save("edgar_13f_staleness", out)


def probe_edgar_13d():
    """13D/G event volume via EDGAR full-text search API — enough events for training?"""
    out = {}
    for yr in [2021, 2023, 2025]:
        url = (f"https://efts.sec.gov/LATEST/search-index?q=%22%22&forms=SC%2013D"
               f"&dateRange=custom&startdt={yr}-01-01&enddt={yr}-12-31")
        # full-text search JSON endpoint:
        url = (f"https://efts.sec.gov/LATEST/search-index?forms=SC+13D")
        try:
            raw = fetch(f"https://efts.sec.gov/LATEST/search-index?q=&forms=SC%2013D&startdt={yr}-01-01&enddt={yr}-12-31")
        except Exception:
            raw = None
        if raw is None:
            try:
                raw = fetch(f"https://efts.sec.gov/LATEST/search-index?q=%22schedule+13d%22&startdt={yr}-01-01&enddt={yr}-12-31")
            except Exception as e:
                out[str(yr)] = {"error": str(e)}
                continue
        obj = json.loads(raw)
        out[str(yr)] = {"total": obj.get("hits", {}).get("total", {})}
        print(f"  13D {yr}: {out[str(yr)]}")
        time.sleep(0.3)
    save("edgar_13d_volume", out)


def probe_congress():
    """House/Senate Stock Watcher S3 dumps: disclosure_date - transaction_date lag."""
    sources = {
        "house": "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
        "senate": "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json",
    }
    out = {}
    for name, url in sources.items():
        try:
            raw = fetch(url, timeout=120)
            data = json.loads(raw)
            lags = []
            dates = []
            for row in data:
                dd = row.get("disclosure_date")
                td = row.get("transaction_date")
                try:
                    ddd = datetime.strptime(dd, "%m/%d/%Y").date()
                    tdd = date.fromisoformat(td)
                except (ValueError, TypeError):
                    continue
                lag = (ddd - tdd).days
                if 0 <= lag <= 400:
                    lags.append(lag)
                    dates.append(str(ddd))
            out[name] = {
                "n_rows": len(data), "n_clean": len(lags),
                "median_lag_days": statistics.median(lags) if lags else None,
                "p90_lag_days": sorted(lags)[int(0.9 * len(lags))] if lags else None,
                "frac_within_45d": sum(1 for x in lags if x <= 45) / len(lags) if lags else None,
                "earliest_disclosure": min(dates) if dates else None,
                "latest_disclosure": max(dates) if dates else None,
                "sample_row": data[0] if data else None,
            }
            print(f"  {name}: n={out[name]['n_clean']} median lag {out[name]['median_lag_days']}d "
                  f"range {out[name]['earliest_disclosure']}..{out[name]['latest_disclosure']}")
        except Exception as e:
            out[name] = {"error": str(e)[:300]}
            print(f"  {name}: ERROR {e}")
        save("congress_lag", out)


def probe_form4_lead_vs_earnings():
    """Lead time from Form-4 filing clusters (>=3 filings in 30d) to the NEXT earnings
    report date (AV spine). Cluster dated by LAST filing date in the cluster.
    Buy/sell parsing is Phase 1; this measures the timing geometry only."""
    key = ENV["ALPHA_VANTAGE_API_KEY"]
    leads = []
    per = {}
    for t in ["AAPL", "DKS", "CROX", "CAT"]:  # 4 AV calls (25/day budget)
        av = json.loads(fetch(
            f"https://www.alphavantage.co/query?function=EARNINGS&symbol={t}&apikey={key}"))
        edates = sorted(date.fromisoformat(q["reportedDate"])
                        for q in av.get("quarterlyEarnings", []) if q.get("reportedDate"))
        time.sleep(1)
        sub = _edgar_submissions(TICKERS_CIK[t])
        f4dates = sorted(date.fromisoformat(fd) for form, fd, rd, _ in _recent_filings(sub)
                         if form == "4")
        # clusters: >=3 filings within any 30d window; cluster date = last filing in window
        clusters = []
        for i in range(len(f4dates)):
            w = [d for d in f4dates if 0 <= (f4dates[i] - d).days <= 30]
            if len(w) >= 3:
                clusters.append(f4dates[i])
        clusters = sorted(set(clusters))
        tl = []
        for c in clusters:
            nxt = next((e for e in edates if e > c), None)
            if nxt:
                tl.append((nxt - c).days)
        per[t] = {"n_clusters": len(tl),
                  "median_lead_days": statistics.median(tl) if tl else None,
                  "frac_lead_ge_1d": sum(1 for x in tl if x >= 1) / len(tl) if tl else None,
                  "earnings_spine_n": len(edates),
                  "earnings_spine_range": [str(edates[0]), str(edates[-1])] if edates else None}
        leads += tl
        print(f"  {t}: clusters={len(tl)} median lead {per[t]['median_lead_days']}d")
    result = {"n_total": len(leads),
              "median_lead_days": statistics.median(leads) if leads else None,
              "p10_lead_days": sorted(leads)[int(0.1 * len(leads))] if leads else None,
              "frac_lead_ge_1d": sum(1 for x in leads if x >= 1) / len(leads) if leads else None,
              "per_ticker": per}
    save("form4_lead_vs_earnings", result)
    print(f"  LEAD: n={result['n_total']} median={result['median_lead_days']}d "
          f"frac>=1d pre-event={result['frac_lead_ge_1d']}")


def probe_finra_short_interest():
    """FINRA equity short interest files — public, twice monthly."""
    url = ("https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
           "?limit=5")
    try:
        raw = fetch(url, headers={"Accept": "application/json", **UA})
        save("finra_short_interest", raw[:5000])
        print(raw[:400])
    except Exception as e:
        save("finra_short_interest_error", {"error": str(e)[:300]})
        print(f"  FINRA API: {e} — trying legacy file listing")
        try:
            raw = fetch("https://cdn.finra.org/equity/otcmarket/biweekly/shrt20250613.csv",
                        headers={"User-Agent": "Mozilla/5.0"})
            save("finra_short_interest_csv_head", raw[:3000])
            print("  legacy CSV head:", raw[:200])
        except Exception as e2:
            save("finra_short_interest_csv_error", {"error": str(e2)[:300]})
            print(f"  legacy CSV: {e2}")


ALL = {
    "alphavantage": probe_alphavantage,
    "yahoo": probe_yahoo,
    "finnhub": probe_finnhub,
    "nasdaq": probe_nasdaq,
    "thetadata": probe_thetadata,
    "form4": probe_edgar_form4,
    "13f": probe_edgar_13f,
    "13d": probe_edgar_13d,
    "congress": probe_congress,
    "lead": probe_form4_lead_vs_earnings,
    "shortint": probe_finra_short_interest,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        print(f"== {n} ==")
        try:
            ALL[n]()
        except Exception as e:
            print(f"  PROBE FAILED: {type(e).__name__}: {e}")
    print("done")
