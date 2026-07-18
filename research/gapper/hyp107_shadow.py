#!/usr/bin/env python3
"""HYP-107 LIVE SHADOW — observational forward test of the de-biased runner
filter. SHADOW ONLY: no capital, no orders, source-tagged "shadow_hyp107".

HYP-105/106 were retracted (look-ahead: universe selected on the 10:30 outcome
but entered 09:31). HYP-107 is the honest, de-biased version — filter selected
using ONLY 09:31-available info. Backtest holdout: gross median +5.4%, win 70%,
tail 4.4, p=0.0005. At ~+5% gross the question is whether 09:31 microcap spread
+ halts leave anything. This tracker logs hypothetical 09:31→10:30 trades and
compares realized spread/return to the backtest. Target: 40 events.

FROZEN filter (commit 48303cd — do not re-fit):
  overnight_gap <= 0.577  AND  log10(first-minute volume) <= 5.854
first-minute range recorded descriptively (weak feature, not a gate).

Passes (launchd scripts/com.alta.hyp107_shadow.plist — operator loads):
  --scan  (~10:50 ET): movers -> 1-min bars -> filter -> log 09:31 entry /
          10:30 exit / realized first-bar spread. outcome=OPEN.
  --close (~16:20 ET): finalize gross return, append daily, update tracking.
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/research/gapper/hyp107_shadow"

# FROZEN HYP-107 filter (commit 48303cd)
OG_MAX = 0.577            # overnight_gap = 09:30 open / prev_close - 1
LOGVOL_MAX = 5.854        # log10(09:30-minute volume)
GAP_FLOOR = 0.30         # honest pre-selection: gapped up >=30% at the open
STOP_PCT = 0.25          # descriptive only (would-a-stop-hit flag)
SOURCE_TAG = "shadow_hyp107"
BASELINE = {"median": 0.054, "win": 0.70, "tail": 4.4}   # backtest holdout


def keys():
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env["ALPACA_API_KEY"], env["ALPACA_SECRET_KEY"]


def get(url, kid, sec):
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10); continue
            if e.code == 403:                 # free-tier recency window
                time.sleep(65); continue
            raise
        except Exception:
            time.sleep(5)
    raise RuntimeError(url[:110])


def min_bars(sym, day, kid, sec):
    s = datetime.combine(day, dtime(9, 30), tzinfo=ET).astimezone(UTC)
    e = datetime.combine(day, dtime(16, 10), tzinfo=ET).astimezone(UTC)
    e = min(e, datetime.now(UTC) - timedelta(minutes=17))   # SIP free-tier lag
    q = urllib.parse.urlencode({"symbols": sym, "timeframe": "1Min",
                                "start": s.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "end": e.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "feed": "sip", "adjustment": "split",
                                "limit": "10000"})
    return (get(f"https://data.alpaca.markets/v2/stocks/bars?{q}", kid, sec)
            .get("bars") or {}).get(sym, [])


def daily_prev_close(sym, day, kid, sec):
    s = datetime.combine(day - timedelta(days=10), dtime(0, 1),
                         tzinfo=ET).astimezone(UTC)
    e = datetime.combine(day - timedelta(days=1), dtime(23, 0),
                         tzinfo=ET).astimezone(UTC)
    q = urllib.parse.urlencode({"symbols": sym, "timeframe": "1Day",
                                "start": s.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "end": e.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "feed": "sip", "adjustment": "split", "limit": "20"})
    d = (get(f"https://data.alpaca.markets/v2/stocks/bars?{q}", kid, sec)
         .get("bars") or {}).get(sym, [])
    return d[-1]["c"] if d else None


def et_t(bar):
    return datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(ET).time()


def bar_at(bars, hh, mm):
    tgt = dtime(hh, mm)
    for b in bars:
        if et_t(b) >= tgt:
            return b
    return None


import math


def scan():
    kid, sec = keys()
    today = datetime.now(ET).date()
    OUT.mkdir(parents=True, exist_ok=True)
    movers = get("https://data.alpaca.markets/v1beta1/screener/stocks/movers?top=50",
                 kid, sec).get("gainers", [])
    cands = [m["symbol"] for m in movers if m.get("percent_change", 0) >= 30]
    signals, checked = [], 0
    for sym in cands:
        if not (sym.isalpha() and len(sym) <= 5 and
                not (len(sym) == 5 and sym[-1] in "WRU")):
            continue
        bars = min_bars(sym, today, kid, sec)
        pc = daily_prev_close(sym, today, kid, sec)
        if not bars or not pc or pc <= 0:
            continue
        b0930 = bar_at(bars, 9, 30)
        b0931 = bar_at(bars, 9, 31)
        b1030 = bar_at(bars, 10, 30)
        if not (b0930 and b0931 and b1030):
            continue
        checked += 1
        overnight_gap = b0930["o"] / pc - 1
        first_vol = b0930["v"]
        log_vol = math.log10(first_vol + 1)
        first_range = (b0930["h"] - b0930["l"]) / b0930["o"] if b0930["o"] else 0.0
        # honest pre-selection + frozen filter (all 09:31-available)
        if overnight_gap < GAP_FLOOR:
            continue
        passes = overnight_gap <= OG_MAX and log_vol <= LOGVOL_MAX
        if not passes:
            continue
        entry = b0931["o"]
        realized_spread = (b0931["h"] - b0931["l"]) / entry if entry else 0.0
        signals.append({
            "source": SOURCE_TAG, "date": str(today), "ticker": sym,
            "prev_close": round(pc, 4),
            "overnight_gap": round(overnight_gap, 4),
            "first_min_vol": int(first_vol), "log_vol": round(log_vol, 4),
            "first_min_range": round(first_range, 4),
            "entry_0931": round(entry, 4),
            "realized_spread_pct": round(realized_spread, 4),
            "exit_1030_ref": round(b1030["c"], 4),   # provisional (finalized at close)
            "logged_at": datetime.now(UTC).isoformat(), "outcome": "OPEN",
        })
    fp = OUT / f"signals_{today}.json"
    fp.write_text(json.dumps({"source": SOURCE_TAG, "signals": signals,
                              "movers_checked": checked}, indent=2))
    print(f"[hyp107] {today}: {len(signals)} filter-pass signal(s) from {checked} "
          f"verified gappers -> {fp.name}", flush=True)


def close():
    kid, sec = keys()
    today = datetime.now(ET).date()
    fp = OUT / f"signals_{today}.json"
    if not fp.exists():
        fp.write_text(json.dumps({"source": SOURCE_TAG, "signals": [],
                                  "movers_checked": 0,
                                  "note": "no scan output — 0-signal day"}))
    doc = json.loads(fp.read_text())
    for s in doc["signals"]:
        if s.get("outcome") == "CLOSED":
            continue
        bars = min_bars(s["ticker"], today, kid, sec)
        b1030 = bar_at(bars, 10, 30)
        if not b1030:
            s["outcome"] = "NO_DATA"; continue
        exit_px = b1030["c"]
        entry = s["entry_0931"]
        gross = exit_px / entry - 1 if entry else 0.0
        # would a 25% stop have hit between 09:31 and 10:30? (descriptive)
        stop_hit = any(b["l"] <= entry * (1 - STOP_PCT) for b in bars
                       if dtime(9, 31) <= et_t(b) <= dtime(10, 30))
        s.update({"exit_1030": round(exit_px, 4), "gross_ret": round(gross, 5),
                  "stop_would_hit": bool(stop_hit), "outcome": "CLOSED",
                  "closed_at": datetime.now(UTC).isoformat()})
    fp.write_text(json.dumps(doc, indent=2))
    # append closed events to the daily ledger + refresh tracking
    with open(OUT / "hyp107_daily.jsonl", "a") as f:
        for s in doc["signals"]:
            if s.get("outcome") == "CLOSED":
                f.write(json.dumps(s) + "\n")
    _refresh_tracking()


def _refresh_tracking():
    rows = []
    p = OUT / "hyp107_daily.jsonl"
    if p.exists():
        seen = set()
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = (r["date"], r["ticker"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    rets = [r["gross_ret"] for r in rows if "gross_ret" in r]
    spreads = [r["realized_spread_pct"] for r in rows if "realized_spread_pct" in r]
    wins = [x for x in rets if x > 0]; losses = [x for x in rets if x < 0]
    tail = (sum(wins) / len(wins)) / (-sum(losses) / len(losses)) \
        if wins and losses else None
    track = {
        "hypothesis": "HYP-107", "source": SOURCE_TAG,
        "n_events": len(rets), "target": 40,
        "median_return": round(median(rets), 5) if rets else None,
        "win_rate": round(len(wins) / len(rets), 3) if rets else None,
        "tail_ratio": round(tail, 2) if tail else None,
        "median_realized_spread": round(median(spreads), 4) if spreads else None,
        "backtest_baseline": BASELINE,
        "note": ("Observational, no capital. At ~+5% gross vs realized spread, "
                 "this tells whether the de-biased edge survives real 09:31 fills."),
        "updated": datetime.now(UTC).isoformat(),
    }
    (OUT / "hyp107_tracking.json").write_text(json.dumps(track, indent=2))
    print(f"[hyp107] tracking: {track['n_events']}/40 events, "
          f"median {track['median_return']} vs backtest {BASELINE['median']}, "
          f"median spread {track['median_realized_spread']}", flush=True)


if __name__ == "__main__":
    if "--scan" in sys.argv:
        scan()
    elif "--close" in sys.argv:
        close()
    else:
        raise SystemExit("use --scan (~10:50 ET) or --close (~16:20 ET)")
