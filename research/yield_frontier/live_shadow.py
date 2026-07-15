#!/usr/bin/env python3
"""W7 — LIVE SHADOW of the frozen HYP-093 signal. SHADOW ONLY — places no orders.

Art. 6 carve-out compliant: every record is source-tagged "shadow_gapper" so a
paper outcome can never masquerade as live evidence. Signal constants are the
sealed HYP-093 spec VERBATIM — this file may never diverge from the prereg.

Two passes (launchd: scripts/com.alta.gapper_shadow.plist — operator loads):
  --scan  (10:50 ET): Alpaca movers screener -> verify each candidate on
          15-min-delayed SIP bars through 10:30 -> frozen filters + M&A news
          exclusion -> log signals with entry = 10:30 bar open + constitutional
          sizing stamp.
  --close (16:20 ET): complete outcomes for today's signals; append the daily
          constitutional net %/day to the shadow equity series; close the loop
          same-day (no outcome ever left dangling).
"""
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/research/yield_frontier/shadow"

# HYP-093 frozen spec (prereg c5b10616) — DO NOT EDIT without a new prereg
GAIN_MIN, PRICE_MIN, VOL_MIN = 0.50, 2.00, 500_000
QUAL_GAIN = 1.30
SLIP, LOCATE_W, NOTIONAL = 0.005, 0.50, 0.0125
APR = {0.5: 2.00, 1.0: 4.00, 1.5: 6.00}
SOURCE_TAG = "shadow_gapper"   # Art. 6: never mistakable for live

BUCKETS_MNA = ["merger", "acquisition", "acquire", "buyout", "takeover",
               "definitive agreement", "letter of intent",
               "strategic alternatives", "going private"]


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
                time.sleep(10)
                continue
            if e.code == 403:          # free-tier recency window — wait it out
                time.sleep(65)
                continue
            raise
        except Exception:
            time.sleep(5)
    raise RuntimeError(url[:110])


def bars_for(sym, day, kid, sec, tf="5Min", days_back=0):
    s = datetime.combine(day - timedelta(days=days_back), dtime(0, 1) if days_back
                         else dtime(9, 30), tzinfo=ET).astimezone(UTC)
    e = datetime.combine(day, dtime(16, 10), tzinfo=ET).astimezone(UTC)
    # free-tier SIP cannot serve the most recent 15 min — cap the window
    e = min(e, datetime.now(UTC) - timedelta(minutes=17))
    q = urllib.parse.urlencode({"symbols": sym, "timeframe": tf,
                                "start": s.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "end": e.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "feed": "sip", "adjustment": "split", "limit": "10000"})
    return (get(f"https://data.alpaca.markets/v2/stocks/bars?{q}", kid, sec)
            .get("bars") or {}).get(sym, [])


def et_t(bar):
    return datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(ET).time()


def notify(msg):
    try:
        urllib.request.urlopen(urllib.request.Request(
            "http://localhost:31337/notify", method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"message": msg, "voice_enabled": True}).encode()),
            timeout=3)
    except Exception:
        pass


def scan():
    kid, sec = keys()
    today = datetime.now(ET).date()
    OUT.mkdir(parents=True, exist_ok=True)
    movers = get("https://data.alpaca.markets/v1beta1/screener/stocks/movers?top=50",
                 kid, sec).get("gainers", [])
    cands = [m["symbol"] for m in movers if m.get("percent_change", 0) >= 40]
    signals, checked = [], 0
    for sym in cands:
        if not (sym.isalpha() and len(sym) <= 5 and
                not (len(sym) == 5 and sym[-1] in "WRU")):
            continue
        daily = bars_for(sym, today - timedelta(days=1), kid, sec, "1Day", 14)
        bars = bars_for(sym, today, kid, sec)
        if not daily or not bars:
            continue
        checked += 1
        pc = daily[-1]["c"]
        sl = [b for b in bars if dtime(9, 30) <= et_t(b) <= dtime(10, 25)]
        if len(sl) < 8 or et_t(sl[-1]) < dtime(10, 15) or pc <= 0:
            continue
        P = sl[-1]["c"]
        vol = sum(b["v"] for b in sl)
        gain = P / pc - 1
        if not (P >= QUAL_GAIN * pc and P >= PRICE_MIN and vol >= VOL_MIN
                and gain >= GAIN_MIN):
            continue
        # M&A exclusion (pre-10:30 headlines only)
        s = datetime.combine(today - timedelta(days=1), dtime(16, 0),
                             tzinfo=ET).astimezone(UTC)
        e = datetime.combine(today, dtime(10, 30), tzinfo=ET).astimezone(UTC)
        news = get("https://data.alpaca.markets/v1beta1/news?" + urllib.parse.urlencode(
            {"symbols": sym, "start": s.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "end": e.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": "50"}), kid, sec)
        blob = " ".join(a.get("headline", "").lower()
                        for a in news.get("news", []))
        if any(k in blob for k in BUCKETS_MNA):
            continue
        entry_bars = [b for b in bars if dtime(10, 30) <= et_t(b) < dtime(11, 0)]
        if not entry_bars:
            continue
        apr = APR[1.5] if gain >= 1.5 else APR[1.0] if gain >= 1.0 else APR[0.5]
        signals.append({
            "source": SOURCE_TAG, "date": str(today), "ticker": sym,
            "prev_close": round(pc, 4), "price_1030": round(P, 4),
            "gain_1030": round(gain, 4), "cum_vol_1030": int(vol),
            "entry_open_1030": round(entry_bars[0]["o"], 4),
            "stop_px": round(entry_bars[0]["o"] * 1.30, 4),
            "borrow_apr": apr, "notional_w": NOTIONAL, "locate_w": LOCATE_W,
            "logged_at": datetime.now(UTC).isoformat(), "outcome": "OPEN",
        })
    fp = OUT / f"signals_{today}.json"
    fp.write_text(json.dumps({"source": SOURCE_TAG, "signals": signals,
                              "movers_checked": checked}, indent=2))
    print(f"[shadow] {today}: {len(signals)} signal(s) from {checked} verified "
          f"movers -> {fp.name}", flush=True)
    if signals:
        notify(f"Gapper shadow: {len(signals)} fade signal"
               f"{'s' if len(signals) != 1 else ''} logged this morning.")


def close():
    kid, sec = keys()
    today = datetime.now(ET).date()
    fp = OUT / f"signals_{today}.json"
    if not fp.exists():
        fp.write_text(json.dumps({"source": SOURCE_TAG, "signals": [],
                                  "movers_checked": 0,
                                  "note": "no scan output — recorded as 0-signal day"}))
        print(f"[shadow] no signal file for {today} — recording 0-signal day")
    doc = json.loads(fp.read_text())
    day_ret = 0.0
    for s in doc["signals"]:
        bars = bars_for(s["ticker"], today, kid, sec)
        post = [b for b in bars if dtime(10, 30) <= et_t(b) < dtime(16, 0)]
        if not post:
            s["outcome"] = "NO_DATA"
            continue
        entry, stop_px = s["entry_open_1030"], s["stop_px"]
        exit_px = None
        for b in post[1:]:
            if b["o"] >= stop_px:
                exit_px = b["o"]
                break
            if b["h"] >= stop_px:
                exit_px = stop_px
                break
        stopped = exit_px is not None
        if exit_px is None:
            exit_px = post[-1]["c"]
        ret = (entry - exit_px) / entry - 2 * SLIP - s["borrow_apr"] / 365
        s.update({"exit_px": round(exit_px, 4), "stopped": stopped,
                  "event_ret_net": round(ret, 5), "outcome": "CLOSED",
                  "closed_at": datetime.now(UTC).isoformat()})
        day_ret += ret * NOTIONAL * LOCATE_W
    fp.write_text(json.dumps(doc, indent=2))
    with open(OUT / "shadow_daily.jsonl", "a") as f:
        f.write(json.dumps({"source": SOURCE_TAG, "date": str(today),
                            "n_signals": len(doc["signals"]),
                            "constitutional_day_ret": round(day_ret, 6)}) + "\n")
    print(f"[shadow] {today}: closed {len(doc['signals'])} -> "
          f"constitutional day ret {day_ret:+.5f}", flush=True)
    notify(f"Gapper shadow closed the day at "
           f"{day_ret * 100:+.3f} basis-adjusted percent.")
    # ICARUS dashboard sync (data-only worktree push to master, 814d1e2 pattern)
    import subprocess
    subprocess.run([sys.executable, str(REPO / "scripts/icarus_dashboard_sync.py"),
                    "--push"], timeout=120)


if __name__ == "__main__":
    if "--scan" in sys.argv:
        scan()
    elif "--close" in sys.argv:
        close()
    else:
        raise SystemExit("use --scan (10:50 ET) or --close (16:20 ET)")
