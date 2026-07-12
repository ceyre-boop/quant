#!/usr/bin/env python3
"""HYP-092 POST-HOC descriptive scan — "method in the madness" pass.

STAMP: DESCRIPTIVE / HYPOTHESIS-GENERATING ONLY. This runs AFTER outcomes were
observed, crosses ~30 feature buckets against continuation, and adjusts for
nothing. Any bucket that looks alive is a candidate for a NEW prereg, never
evidence. (The sealed HYP-092 verdict is untouched by anything here.)

Catalyst labels use ONLY news published before 10:30 ET on the day (window
opens prior day 16:00 ET) — the scan stays look-ahead-clean even though its
statistics are post-hoc.
"""
import gzip
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/research/gapper"
ACACHE = OUT / "cache/alpaca"
NCACHE = OUT / "cache/news"
BASE_TARGET = 0.03  # card's CONTINUED label


def load_keys():
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env["ALPACA_API_KEY"], env["ALPACA_SECRET_KEY"]


def api_get(url, kid, sec):
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
    for _ in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10)
                continue
            raise
        except Exception:
            time.sleep(5)
    raise RuntimeError(url[:120])


def et_start(bar):
    return datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(ET).time()


# ---------- catalyst classification (priority order) ----------
BUCKETS = [
    ("FDA_CLINICAL", ["fda", "phase 1", "phase 2", "phase 3", "clinical",
                      "trial", "approval", "clearance", "breakthrough", "nda",
                      "510(k)", "orphan drug"]),
    ("MERGER_ACQ", ["merger", "acquisition", "acquire", "buyout", "takeover",
                    "definitive agreement", "letter of intent", "strategic alternatives",
                    "going private"]),
    ("OFFERING", ["offering", "private placement", "registered direct",
                  "dilution", "warrant", "at-the-market"]),
    ("EARNINGS", ["earnings", "quarterly results", "q1 ", "q2 ", "q3 ", "q4 ",
                  "revenue", "eps", "guidance", "fiscal"]),
    ("CRYPTO", ["bitcoin", "crypto", "ethereum", "solana", "token",
                "digital asset", "treasury strategy"]),
    ("AI", [" ai ", "artificial intelligence", "ai-powered", "genai"]),
    ("CONTRACT_PARTNER", ["contract", "partnership", "collaboration",
                          "purchase order", "agreement", "award", "deal with"]),
    ("ANALYST", ["upgrade", "price target", "initiates"]),
]


def classify_headlines(heads):
    if not heads:
        return "NO_NEWS_PRE1030"
    blob = " " + " ".join(h.lower() for h in heads) + " "
    for name, kws in BUCKETS:
        if any(k in blob for k in kws):
            return name
    return "OTHER_NEWS"


def fetch_news_day(day, symbols, kid, sec):
    fp = NCACHE / f"{day}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    d = date.fromisoformat(day)
    start = datetime.combine(d - timedelta(days=1), dtime(16, 0), tzinfo=ET).astimezone(UTC)
    end = datetime.combine(d, dtime(10, 30), tzinfo=ET).astimezone(UTC)
    per_sym = defaultdict(list)
    for i in range(0, len(symbols), 40):
        chunk = symbols[i:i + 40]
        token = None
        while True:
            params = {"symbols": ",".join(chunk),
                      "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "limit": "50"}
            if token:
                params["page_token"] = token
            r = api_get("https://data.alpaca.markets/v1beta1/news?" +
                        urllib.parse.urlencode(params), kid, sec)
            for a in r.get("news", []):
                for s in a.get("symbols", []):
                    per_sym[s].append(a.get("headline", ""))
            token = r.get("next_page_token")
            if not token:
                break
            time.sleep(0.2)
        time.sleep(0.2)
    NCACHE.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(per_sym))
    return per_sym


def fetch_spy(kid, sec):
    fp = OUT / "cache/spy_morning.json"
    if fp.exists():
        return json.loads(fp.read_text())
    out = {}
    # one query for the whole window, paginated
    start = datetime(2025, 7, 1, tzinfo=UTC)
    end = datetime(2026, 7, 1, tzinfo=UTC)
    token, bars = None, []
    while True:
        params = {"symbols": "SPY", "timeframe": "5Min",
                  "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "feed": "sip", "adjustment": "split", "limit": "10000"}
        if token:
            params["page_token"] = token
        r = api_get("https://data.alpaca.markets/v2/stocks/bars?" +
                    urllib.parse.urlencode(params), kid, sec)
        bars.extend((r.get("bars") or {}).get("SPY", []))
        token = r.get("next_page_token")
        if not token:
            break
        time.sleep(0.2)
    by_day = defaultdict(list)
    for b in bars:
        ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
        if dtime(9, 30) <= ts.time() <= dtime(10, 25):
            by_day[ts.date().isoformat()].append(b)
    for d, bs in by_day.items():
        bs.sort(key=lambda b: b["t"])
        out[d] = round(bs[-1]["c"] / bs[0]["o"] - 1, 5)
    fp.write_text(json.dumps(out))
    return out


def enrich(df, kid, sec):
    feats = []
    spy = fetch_spy(kid, sec)
    for day, sub in df.groupby("date"):
        with gzip.open(ACACHE / f"{day}.json.gz", "rt") as f:
            payload = json.load(f)
        symbols = sorted(sub["ticker"])
        news = fetch_news_day(day, symbols, kid, sec)
        for _, r in sub.iterrows():
            bars = payload["intraday"][r["ticker"]]
            slice_bars = [b for b in bars
                          if dtime(9, 30) <= et_start(b) <= dtime(10, 25)]
            daily = payload["daily"].get(r["ticker"], [])
            mean_dvol = (sum(b["v"] for b in daily) / len(daily)) if daily else None
            first_open = slice_bars[0]["o"]
            H = max(b["h"] for b in slice_bars)
            hbar = next(b for b in slice_bars if b["h"] == H)
            t_high = et_start(hbar)
            mins_high = (t_high.hour - 9) * 60 + t_high.minute - 30
            feats.append({
                "date": day, "ticker": r["ticker"],
                "catalyst": classify_headlines(news.get(r["ticker"], [])),
                "overnight_gap": round(first_open / r["prev_close"] - 1, 4),
                "intraday_push": round(r["price_1030"] / first_open - 1, 4),
                "rvol_1030": round(r["cum_vol_1030"] / mean_dvol, 1) if mean_dvol else None,
                "dollar_vol_m": round(r["price_1030"] * r["cum_vol_1030"] / 1e6, 1),
                "mins_to_high": mins_high,
                "vwap_dist": round(r["price_1030"] / r["vwap_1030"] - 1, 4),
                "dow": date.fromisoformat(day).weekday(),
                "spy_am": spy.get(day),
            })
    return df.merge(pd.DataFrame(feats), on=["date", "ticker"])


def cross(df, col, buckets):
    """buckets: list of (label, mask_fn)"""
    base = (df["outcome_pct"] > BASE_TARGET).mean()
    rows = []
    for label, fn in buckets:
        sub = df[fn(df)]
        if len(sub) < 40:
            continue
        o = sub["outcome_pct"]
        rows.append({
            "feature": col, "bucket": label, "n": len(sub),
            "cont_rate": round(float((o > BASE_TARGET).mean()), 3),
            "rev_rate": round(float((o < -BASE_TARGET).mean()), 3),
            "median_pct": round(float(o.median()), 4),
            "edge_vs_base": round(float((o > BASE_TARGET).mean() - base), 3),
        })
    return rows


def main():
    kid, sec = load_keys()
    df = pd.read_csv(OUT / "per_candidate.csv")
    df = enrich(df, kid, sec)
    df.to_csv(OUT / "per_candidate_enriched.csv", index=False)

    R = []
    cats = sorted(df["catalyst"].unique())
    R += cross(df, "catalyst", [(c, lambda d, c=c: d["catalyst"] == c) for c in cats])
    R += cross(df, "gain_1030", [
        ("30-50%", lambda d: d["gain_1030"] < 0.5),
        ("50-100%", lambda d: (d["gain_1030"] >= 0.5) & (d["gain_1030"] < 1.0)),
        (">=100%", lambda d: d["gain_1030"] >= 1.0)])
    R += cross(df, "price_1030", [
        ("$2-5", lambda d: d["price_1030"] < 5),
        ("$5-10", lambda d: (d["price_1030"] >= 5) & (d["price_1030"] < 10)),
        (">=$10", lambda d: d["price_1030"] >= 10)])
    R += cross(df, "rvol_1030", [
        ("<25x", lambda d: d["rvol_1030"] < 25),
        ("25-100x", lambda d: (d["rvol_1030"] >= 25) & (d["rvol_1030"] < 100)),
        (">=100x", lambda d: d["rvol_1030"] >= 100)])
    R += cross(df, "dollar_vol", [
        ("<$5M", lambda d: d["dollar_vol_m"] < 5),
        ("$5-25M", lambda d: (d["dollar_vol_m"] >= 5) & (d["dollar_vol_m"] < 25)),
        (">=$25M", lambda d: d["dollar_vol_m"] >= 25)])
    R += cross(df, "gap_style", [
        ("overnight-gap led", lambda d: (d["overnight_gap"] >= 0.20) & (d["intraday_push"] < 0.10)),
        ("intraday-grind led", lambda d: (d["overnight_gap"] < 0.10) & (d["intraday_push"] >= 0.20)),
        ("both/mixed", lambda d: ~(((d["overnight_gap"] >= 0.20) & (d["intraday_push"] < 0.10)) |
                                   ((d["overnight_gap"] < 0.10) & (d["intraday_push"] >= 0.20))))])
    R += cross(df, "mins_to_high", [
        ("high in first 15m", lambda d: d["mins_to_high"] <= 15),
        ("high 20-40m", lambda d: (d["mins_to_high"] > 15) & (d["mins_to_high"] <= 40)),
        ("still making highs", lambda d: d["mins_to_high"] > 40)])
    R += cross(df, "vwap_dist", [
        ("below VWAP", lambda d: d["vwap_dist"] < 0),
        ("0-5% above", lambda d: (d["vwap_dist"] >= 0) & (d["vwap_dist"] < 0.05)),
        (">5% above", lambda d: d["vwap_dist"] >= 0.05)])
    for v in ["C1", "C2", "C3", "C4", "E1", "E2", "E3", "E4"]:
        R += cross(df, f"vote_{v}", [
            (f"{v}=1", lambda d, v=v: d[v] == 1), (f"{v}=0", lambda d, v=v: d[v] == 0)])
    R += cross(df, "spy_am", [
        ("SPY am down", lambda d: d["spy_am"] < 0),
        ("SPY am up", lambda d: d["spy_am"] >= 0)])
    R += cross(df, "dow", [(n, lambda d, i=i: d["dow"] == i)
                           for i, n in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])])

    base = float((df["outcome_pct"] > BASE_TARGET).mean())
    result = {
        "stamp": "POST-HOC DESCRIPTIVE — hypothesis-generating only, ~30 uncorrected comparisons; nothing here is evidence; sealed HYP-092 verdict unaffected",
        "base_cont_rate": round(base, 3),
        "n": int(len(df)),
        "crosses": R,
        "catalyst_counts": df["catalyst"].value_counts().to_dict(),
    }
    (OUT / "posthoc_scan.json").write_text(json.dumps(result, indent=2))
    R2 = sorted(R, key=lambda r: -abs(r["edge_vs_base"]))
    print(f"base continuation rate {base:.1%}  (n={len(df)})\n")
    print(f"{'feature':<14} {'bucket':<20} {'n':>5} {'cont':>6} {'rev':>6} {'med%':>7} {'edge':>6}")
    for r in R2[:24]:
        print(f"{r['feature']:<14} {r['bucket']:<20} {r['n']:>5} {r['cont_rate']:>6.1%} "
              f"{r['rev_rate']:>6.1%} {r['median_pct']:>7.2%} {r['edge_vs_base']:>+6.1%}")


if __name__ == "__main__":
    main()
