#!/usr/bin/env python3
"""Probe every credential in .env against its live endpoint. Read-only.

A key being present in .env proves nothing — tier, entitlement and quota are all
invisible until you call. This reports, per source: reachable, what tier it
behaves like, and the ONE thing that matters for research use, which is whether
the payload carries a usable publication timestamp.

No key material is ever printed.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# NewsAPI rejects a bare contact string; SEC requires one. This form satisfies both.
UA = {"User-Agent": "AltaResearch/1.0 (colineyre222@gmail.com)"}


def get(url: str, headers: dict | None = None, timeout: int = 20):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", errors="replace")


def probe(name: str, fn) -> tuple[str, str]:
    try:
        return fn()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:120]
        except Exception:
            pass
        return "FAIL", f"HTTP {e.code} {body}"
    except Exception as e:  # noqa: BLE001
        return "FAIL", f"{type(e).__name__}: {str(e)[:120]}"


# ── individual probes ────────────────────────────────────────────────────────

def p_fmp():
    k = os.environ["FMP_API_KEY"]
    # The v3 API is retired ("Legacy Endpoint ... no longer supported"). The
    # /stable/ surface is the live one, and analyst-estimates requires `period`.
    s, t = get(f"https://financialmodelingprep.com/stable/analyst-estimates"
               f"?symbol=AAPL&period=annual&limit=2&apikey={k}")
    d = json.loads(t)
    if isinstance(d, dict) and d.get("Error Message"):
        return "FAIL", str(d["Error Message"])[:110]
    if not d:
        return "LIMITED", "reachable but analyst-estimates empty (tier may exclude it)"
    r = d[0]
    return "OK", (f"forward estimates to {r.get('date')}: "
                  f"epsAvg={r.get('epsAvg')} revAvg={r.get('revenueAvg')}")


def p_fmp_grades():
    """Forward guidance / rating changes — the gap free sources cannot fill."""
    k = os.environ["FMP_API_KEY"]
    s, t = get(f"https://financialmodelingprep.com/stable/grades?symbol=AAPL&apikey={k}")
    d = json.loads(t)
    if isinstance(d, dict) and d.get("Error Message"):
        return "FAIL", str(d["Error Message"])[:110]
    if not d:
        return "LIMITED", "no grade history returned"
    return "OK", f"rating changes: {d[0].get('date')} {d[0].get('gradingCompany')} -> {d[0].get('newGrade')}"


def p_fmp_earnings():
    """est vs actual WITH forward rows — the forward-consensus gap free sources
    cannot fill. A future date with epsActual=null IS the current consensus."""
    k = os.environ["FMP_API_KEY"]
    s, t = get(f"https://financialmodelingprep.com/stable/earnings?symbol=AAPL&apikey={k}")
    d = json.loads(t)
    if not isinstance(d, list) or not d:
        return "LIMITED", str(d)[:110]
    fwd = [r for r in d if r.get("epsActual") is None and r.get("epsEstimated") is not None]
    return "OK", (f"{len(d)} prints, {len(fwd)} forward; next {fwd[0].get('date') if fwd else '-'} "
                  f"est={fwd[0].get('epsEstimated') if fwd else '-'}")


def p_finnhub():
    k = os.environ["FINNHUB_API_KEY"]
    s, t = get(f"https://finnhub.io/api/v1/stock/earnings?symbol=AAPL&token={k}")
    d = json.loads(t)
    if isinstance(d, dict) and d.get("error"):
        return "FAIL", str(d["error"])[:110]
    if not d:
        return "LIMITED", "empty earnings surprise history"
    r = d[0]
    return "OK", f"earnings surprise: {r.get('period')} est={r.get('estimate')} act={r.get('actual')}"


def p_finnhub_cal():
    """Earnings calendar with the BMO/AMC flag — the catalyst timing field."""
    k = os.environ["FINNHUB_API_KEY"]
    s, t = get(f"https://finnhub.io/api/v1/calendar/earnings?from=2026-07-25&to=2026-08-05&token={k}")
    d = json.loads(t)
    rows = d.get("earningsCalendar", []) if isinstance(d, dict) else []
    if not rows:
        return "LIMITED", "calendar empty (often a paid endpoint)"
    r = rows[0]
    return "OK", f"{len(rows)} events; sample {r.get('symbol')} {r.get('date')} hour={r.get('hour')}"


def p_bls():
    """CPI. BLS release times are precise, which makes them point-in-time usable."""
    k = os.environ["BLS_API_KEY"]
    body = json.dumps({"seriesid": ["CUUR0000SA0"], "startyear": "2025",
                       "endyear": "2026", "registrationkey": k}).encode()
    req = urllib.request.Request("https://api.bls.gov/publicAPI/v2/timeseries/data/",
                                 data=body, headers={**UA, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as r:
        d = json.loads(r.read().decode())
    if d.get("status") != "REQUEST_SUCCEEDED":
        return "FAIL", str(d.get("message"))[:110]
    ser = d["Results"]["series"][0]["data"]
    return "OK", f"CPI-U: {len(ser)} points, latest {ser[0]['year']}-{ser[0]['period']} = {ser[0]['value']}"


def p_bea():
    k = os.environ["BEA_API_KEY"]
    q = urllib.parse.urlencode({"UserID": k, "method": "GetParameterList",
                                "datasetname": "NIPA", "ResultFormat": "JSON"})
    s, t = get(f"https://apps.bea.gov/api/data/?{q}")
    d = json.loads(t)
    res = d.get("BEAAPI", {}).get("Results", {})
    err = res.get("Error") if isinstance(res, dict) else None
    if err:
        desc = str(err.get("APIErrorDescription", err))
        if "not active" in desc:
            return "FAIL", "key valid but NOT ACTIVATED — click the activation link BEA emailed"
        return "FAIL", desc[:110]
    if not res:
        return "FAIL", "empty result"
    return "OK", f"NIPA params: {len(res.get('Parameter', []))}"


def p_census():
    k = os.environ["CENSUS_API_KEY"]
    # ACS is the stable, always-available dataset; the EITS timeseries surface
    # rejects a bare year and returns HTML rather than JSON.
    s, t = get(f"https://api.census.gov/data/2023/acs/acs1?get=NAME&for=state:*&key={k}")
    # Census answers HTTP 200 with an HTML error page rather than a JSON error,
    # so a naive json.loads reports a parse failure and hides the real cause.
    if t.lstrip().startswith("<"):
        if "Invalid Key" in t:
            return "FAIL", "key rejected as invalid — check it, or activate via the Census email"
        if "Missing Key" in t:
            return "FAIL", "no key sent"
        return "FAIL", "HTML error page returned"
    d = json.loads(t)
    return "OK", f"ACS reachable: {len(d)-1} states"


def p_eia():
    k = os.environ["EIA_API_KEY"]
    s, t = get("https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
               f"?api_key={k}&frequency=weekly&data[0]=value&sort[0][column]=period"
               "&sort[0][direction]=desc&length=3")
    d = json.loads(t)
    rows = d.get("response", {}).get("data", [])
    if not rows:
        return "FAIL", str(d)[:110]
    return "OK", f"weekly petroleum stocks: latest period {rows[0].get('period')}"


def p_senate_lda():
    """Lobbying disclosures — a novel, timestamped event class."""
    k = os.environ["SENATE_LDA_API_KEY"]
    s, t = get("https://lda.senate.gov/api/v1/filings/?filing_year=2026&page_size=2",
               headers={"Authorization": f"Token {k}"})
    d = json.loads(t)
    n = d.get("count")
    if n is None:
        return "FAIL", str(d)[:110]
    r = (d.get("results") or [{}])[0]
    return "OK", (f"{n:,} 2026 filings; sample posted={r.get('dt_posted')} "
                  f"client={(r.get('client') or {}).get('name','?')[:28]}")


def p_openfda():
    """FDA actions — a discrete biotech catalyst class."""
    k = os.environ["OPENFDA_API_KEY"]
    s, t = get("https://api.fda.gov/drug/event.json"
               f"?api_key={k}&limit=1")
    d = json.loads(t)
    if "results" not in d:
        return "FAIL", str(d)[:110]
    meta = d.get("meta", {}).get("results", {})
    return "OK", f"adverse events reachable; total={meta.get('total'):,}"


def p_data_gov():
    k = os.environ["DATA_GOV_API_KEY"]
    # api.data.gov keys are shared across .gov APIs. Use a fast endpoint —
    # NASA's APOD regularly times out and says nothing about the key.
    s, t = get(f"https://api.govinfo.gov/collections?api_key={k}", timeout=25)
    d = json.loads(t)
    n = len(d.get("collections", []))
    return ("OK", f"govinfo reachable: {n} collections") if n else ("LIMITED", str(d)[:110])


def p_tiingo():
    k = os.environ["TIINGO_API_KEY"]
    s, t = get("https://api.tiingo.com/tiingo/daily/AAPL/prices?startDate=2026-08-25",
               headers={"Authorization": f"Token {k}"})
    d = json.loads(t)
    if not isinstance(d, list) or not d:
        return "LIMITED", str(d)[:110]
    return "OK", f"daily bars: {len(d)} rows, latest {d[-1].get('date','')[:10]}"


def p_polygon():
    k = os.environ["POLYGON_API_KEY"]
    s, t = get(f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2026-08-25/2026-08-29?apiKey={k}")
    d = json.loads(t)
    if d.get("status") == "ERROR":
        return "FAIL", str(d.get("error"))[:110]
    return "OK", f"{d.get('resultsCount', 0)} daily bars ({d.get('status')})"


def p_nasdaq_dl():
    k = os.environ["NASDAQ_DATA_LINK_API_KEY"]
    try:
        s, t = get(f"https://data.nasdaq.com/api/v3/datasets/LBMA/GOLD.json?rows=2&api_key={k}")
    except urllib.error.HTTPError as e:
        if e.code == 403:
            # Edge bot-protection, not a key problem. This source is health-ping
            # only with no consumer anyway, so it is not worth working around.
            return "BLOCKED", "403 at the CDN edge (bot protection), not a key failure"
        raise
    d = json.loads(t)
    if "dataset" not in d:
        return "FAIL", str(d)[:110]
    return "OK", f"LBMA/GOLD reachable, latest {d['dataset']['newest_available_date']}"


def p_alpha_vantage():
    k = os.environ["ALPHA_VANTAGE_API_KEY"]
    s, t = get(f"https://www.alphavantage.co/query?function=EARNINGS&symbol=AAPL&apikey={k}")
    d = json.loads(t)
    if "Note" in d or "Information" in d:
        return "LIMITED", "rate-limited / free tier note returned"
    n = len(d.get("quarterlyEarnings", []))
    return ("OK", f"{n} quarterly earnings") if n else ("FAIL", str(d)[:110])


def p_news():
    k = os.environ["NEWS_API_KEY"]
    s, t = get(f"https://newsapi.org/v2/top-headlines?category=business&pageSize=1&apiKey={k}")
    d = json.loads(t)
    if d.get("status") != "ok":
        return "FAIL", str(d.get("message"))[:110]
    return "OK", f"{d.get('totalResults')} business headlines"


PROBES = [
    ("FMP estimates",       "FMP_API_KEY",              p_fmp),
    ("FMP rating changes",  "FMP_API_KEY",              p_fmp_grades),
    ("FMP est vs actual",   "FMP_API_KEY",              p_fmp_earnings),
    ("Finnhub surprises",   "FINNHUB_API_KEY",          p_finnhub),
    ("Finnhub earn cal",    "FINNHUB_API_KEY",          p_finnhub_cal),
    ("Senate LDA",          "SENATE_LDA_API_KEY",       p_senate_lda),
    ("openFDA",             "OPENFDA_API_KEY",          p_openfda),
    ("BLS",                 "BLS_API_KEY",              p_bls),
    ("BEA",                 "BEA_API_KEY",              p_bea),
    ("Census",              "CENSUS_API_KEY",           p_census),
    ("EIA",                 "EIA_API_KEY",              p_eia),
    ("api.data.gov",        "DATA_GOV_API_KEY",         p_data_gov),
    ("Tiingo",              "TIINGO_API_KEY",           p_tiingo),
    ("Polygon",             "POLYGON_API_KEY",          p_polygon),
    ("Nasdaq Data Link",    "NASDAQ_DATA_LINK_API_KEY", p_nasdaq_dl),
    ("Alpha Vantage",       "ALPHA_VANTAGE_API_KEY",    p_alpha_vantage),
    ("NewsAPI",             "NEWS_API_KEY",             p_news),
]


def main() -> int:
    print(f"{'source':<20} {'status':<9} detail")
    print("-" * 96)
    results = {}
    for label, env, fn in PROBES:
        if not os.environ.get(env):
            print(f"{label:<20} {'NO KEY':<9} {env} not set")
            results[label] = "NO KEY"
            continue
        status, detail = probe(label, fn)
        results[label] = status
        print(f"{label:<20} {status:<9} {detail}")

    ok = sum(1 for v in results.values() if v == "OK")
    print(f"\n{ok}/{len(PROBES)} usable")
    if "--json" in sys.argv:
        out = ROOT / "data" / "agent" / "key_probe.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
