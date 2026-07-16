#!/usr/bin/env python3
"""1-year forward simulation of the SEALED HYP-093 gapper-fade rule on unseen
data (2025-07-02..2026-06-30 — entirely after HYP-093's validation window,
which ended 2025-06-30). Descriptive forward-tracking of an already-confirmed
rule; no parameter was chosen using this data.

Spec (from the session mandate + HYP-099 prereg sim section):
- Entry: HYP-093 frozen signal, short at entry_open_1030, zero lookahead.
- Sizing: 2% notional per trade (2x constitutional stress test).
- Stop: 25% adverse, evaluated on the post-10:30 intraday high (Alpaca SIP minute bars);
  post-entry high >= entry*1.25 => stopped, fill entry*1.25 plus slip.
  Missing data => treated as stopped (conservative).
- Exit otherwise: EOD close (close_eod).
- Frictions: slip 0.005 per side, locate 0.50 * APR(gain)/252,
  APR tiers {>=0.5x: 2.00, >=1.0x: 4.00, >=1.5x: 6.00} annualized.
Outputs: research/gapper/sim_annual_HYP093_forward.csv (daily P&L),
stats printed + JSON summary.
"""
import csv
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data/research/gapper/per_candidate_enriched.csv"
HIGHS = REPO / "data/research/gapper/event_post1030_highs.json"
OUT = Path(__file__).with_name("sim_annual_HYP093_forward.csv")

GAIN_MIN, PRICE_MIN, VOL_MIN = 1.00, 2.00, 500_000
SLIP, LOCATE_W, NOTIONAL, STOP = 0.005, 0.50, 0.02, 0.25
MNA = ["merger", "acquisition", "acquire", "buyout", "takeover",
       "definitive agreement", "letter of intent", "strategic alternatives",
       "going private"]


def apr(g):
    return 6.00 if g >= 1.5 else 4.00 if g >= 1.0 else 2.00


def main():
    highs = json.loads(HIGHS.read_text())
    daily = {}
    events = []
    for r in csv.DictReader(open(CSV)):
        try:
            g, p, v = (float(r["gain_1030"]), float(r["price_1030"]),
                       float(r["cum_vol_1030"]))
            entry, close = float(r["entry_open_1030"]), float(r["close_eod"])
        except (ValueError, KeyError):
            continue
        if g < GAIN_MIN or p < PRICE_MIN or v < VOL_MIN:
            continue
        if any(m in (r.get("catalyst") or "").lower() for m in MNA):
            continue
        hi = highs.get(f"{r['date']}|{r['ticker']}")
        stopped = hi is None or hi >= entry * (1 + STOP)
        if stopped:
            gross = -(STOP + SLIP)              # short loses 25% + slip on fill
        else:
            gross = (entry - close) / entry
        net = gross - 2 * SLIP - LOCATE_W * apr(g) / 252
        events.append((r["date"], r["ticker"], stopped, round(net, 5)))
        daily[r["date"]] = daily.get(r["date"], 0.0) + net * NOTIONAL

    dates = sorted(daily)
    equity, rows, cum = 1.0, [], []
    peak, maxdd = 1.0, 0.0
    for d in dates:
        equity *= (1 + daily[d])
        peak = max(peak, equity)
        maxdd = max(maxdd, 1 - equity / peak)
        rows.append((d, round(daily[d], 6), round(equity, 6)))
        cum.append(daily[d])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "day_return", "equity"])
        w.writerows(rows)

    mu, sd = statistics.mean(cum), statistics.stdev(cum)
    stats = {
        "rule": "HYP-093 frozen (forward year 2025-07-02..2026-06-30, unseen)",
        "n_events": len(events),
        "n_stopped": sum(1 for e in events if e[2]),
        "n_event_days": len(dates),
        "sizing_notional": NOTIONAL,
        "total_return": round(equity - 1, 5),
        "max_drawdown": round(maxdd, 5),
        "sharpe_annual_event_days": round(mu / sd * math.sqrt(252), 3),
        "mean_day_ret": round(mu, 6),
        "win_rate_events": round(sum(1 for e in events if e[3] > 0) / len(events), 3),
    }
    Path(__file__).with_name("sim_annual_HYP093_forward_stats.json").write_text(
        json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
