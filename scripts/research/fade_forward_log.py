#!/usr/bin/env python3
"""Forward log of the post-shock next-session fade — NOT an edge (HYP-114 FAIL, 2026-09-03: absent on
2016-2019 and on 20 other ETFs). Kept running as a live NULL CHECK of the rule that looked best in
2020-2026, so a future session can see whether the forward tape agrees with the out-of-sample null.

Frozen rule (HYP-111a incumbent, negated; HYP-113 says do not condition on size):
  shock at t  : |close-to-close log return| >= p90 of the trailing 252 sessions' |r| (t excluded)
  trade       : session t+1, direction −sign(r_t), 09:30 open -> 15:55 close, 3.0 bp round trip
  instruments : SPY QQQ IWM DIA TLT GLD EFA EEM XLF XLE, one unit each, no sizing

Pure observation. No orders. Daily bars and 1-min bars from the local ThetaTerminal (STOCK.FREE
serves both from 2023-06-01). Appends one JSON line per (instrument, event) to
data/research/hyp111/fade_forward_log.jsonl, idempotent, and prints the running expectancy in
the operator's form: E = W*avgWin − (1−W)*avgLoss. This is the 'log & tag 50–100 trades' step,
run on the rule that was sealed BEFORE any of these trades happened.

  .venv313/bin/python scripts/research/fade_forward_log.py            # append today's completed events
  .venv313/bin/python scripts/research/fade_forward_log.py --backfill  # from 2026-07-17 (day after HYP-111a's window)
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import theta_v2 as th  # noqa: E402
from research.hyp111.engine import naive, INSTRUMENTS  # noqa: E402

LOG = ROOT / "data" / "research" / "hyp111" / "fade_forward_log.jsonl"
FIRST_FORWARD = "2026-07-17"        # HYP-111a window ended 2026-07-16; nothing before this is forward
COST = 3.0 / 1e4


def eod(sym: str, start: str, end: str) -> pd.DataFrame:
    # the terminal 475s on long EOD ranges (Java index bug); request in ~120-day chunks
    # history from the tracked daily cache (through 2026-07-16); only the tail comes from the terminal,
    # which 475s (Java index bug) on several older ranges
    hist = pd.read_parquet(ROOT / "data" / "cache" / "daily_universe" / f"{sym}.parquet")
    hist["date"] = pd.to_datetime(hist["date"])
    rows = [(d, float(c)) for d, c in zip(hist["date"], hist["close"]) if d >= pd.Timestamp(start)]
    start = (hist["date"].max() + pd.Timedelta(days=1)).strftime("%Y%m%d")
    for a, b in _chunks(start, end, 45):
        code, body = th._get("/v2/hist/stock/eod", {"root": sym, "start_date": a, "end_date": b})
        if code == 472:
            continue
        if code != 200:
            raise SystemExit(f"{sym} EOD {a}-{b}: HTTP {code} {body}")
        fmt = body["header"]["format"]; ic, idt = fmt.index("close"), fmt.index("date")
        rows += [(pd.Timestamp(str(r[idt])), float(r[ic])) for r in body["response"]]
    df = pd.DataFrame(rows, columns=["date", "close"]).drop_duplicates("date").set_index("date").sort_index()
    df["r"] = np.log(df["close"]).diff()
    a = df["r"].abs(); thr = a.shift(1).rolling(252).quantile(0.90)
    df["shock"] = (a >= thr) & thr.notna()
    return df


def _chunks(start: str, end: str, days: int):
    a = pd.Timestamp(start); e = pd.Timestamp(end)
    while a <= e:
        b = min(a + pd.Timedelta(days=days - 1), e)
        yield a.strftime("%Y%m%d"), b.strftime("%Y%m%d")
        a = b + pd.Timedelta(days=1)


def main(argv) -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    seen = {(json.loads(l)["sym"], json.loads(l)["t"]) for l in LOG.read_text().splitlines()} if LOG.exists() else set()
    today = date.today().strftime("%Y%m%d")
    new = []
    for sym in INSTRUMENTS:
        df = eod(sym, "20230601", today)
        idx = df.index
        for i in np.flatnonzero(df["shock"].values):
            if i + 1 >= len(idx):
                continue                                           # t+1 not complete yet
            t, t1 = idx[i].strftime("%Y-%m-%d"), idx[i + 1].strftime("%Y-%m-%d")
            if t1 < FIRST_FORWARD or (sym, t) in seen:
                continue
            bars = th.stock_1m(sym, t1)
            if len(bars) < 370:
                continue                                           # session incomplete / not served yet
            s = 1.0 if df["r"].iloc[i] > 0 else -1.0
            fade = -naive(bars, s) - COST
            new.append({"sym": sym, "t": t, "t1": t1, "shock_r": float(df["r"].iloc[i]), "dir": "short" if s > 0 else "long",
                        "fade_net": float(fade), "logged": date.today().isoformat(), "rule": "HYP-111a incumbent negated, 3bp"})
    with LOG.open("a") as fh:
        for r in new:
            fh.write(json.dumps(r) + "\n")
    rows = [json.loads(l) for l in LOG.read_text().splitlines()] if LOG.exists() else []
    print(f"appended {len(new)}; forward log now {len(rows)} events since {FIRST_FORWARD}")
    if rows:
        f = np.array([r["fade_net"] for r in rows]); w = (f > 0).mean()
        aw = f[f > 0].mean() if (f > 0).any() else 0.0; al = -f[f <= 0].mean() if (f <= 0).any() else 0.0
        print(f"forward expectancy: n={len(f)} W={w:.3f} avgWin={aw*100:.3f}% avgLoss={al*100:.3f}% "
              f"E={(w*aw-(1-w)*al)*100:+.4f}%/trade   (backtest 2023-06→2026-07: n=798 W=0.569 E=+0.150%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
