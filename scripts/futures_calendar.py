#!/usr/bin/env python3
"""Track-2 futures P&L calendar — the Warrior-style monthly view.

Reads data/futures/trade_log.jsonl, groups by day, renders a static monthly grid HTML
(green/red cells = daily $ P&L, with trade count + win%) → data/futures/calendar.html.
No server — re-run to refresh, open the file in a browser. Standalone (Track 2).

$ P&L is computed from CLOSED trades (entry+exit present); trades still open are counted but
contribute $0 until their exit is logged. Honest by construction.

Usage:  python3 scripts/futures_calendar.py
"""
from __future__ import annotations

import calendar as _cal
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRADE_LOG = ROOT / "data" / "futures" / "trade_log.jsonl"
OUT = ROOT / "data" / "futures" / "calendar.html"
POINT_VALUE = {"MES": 5.0, "MNQ": 2.0}
GOAL = 150


def _load():
    days = defaultdict(lambda: {"pnl": 0.0, "n": 0, "wins": 0, "closed": 0})
    total = 0
    if TRADE_LOG.exists():
        for line in TRADE_LOG.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("size_contracts", 0) == 0:   # macro-hold notes etc. — not a trade
                continue
            day = str(r.get("ts", ""))[:10]
            if not day:
                continue
            d = days[day]
            d["n"] += 1
            total += 1
            entry, exit_, size = r.get("entry"), r.get("exit"), r.get("size_contracts") or 0
            if entry is not None and exit_ is not None and size:
                pv = POINT_VALUE.get(r.get("instrument"), 5.0)
                mult = 1 if r.get("direction") == "LONG" else -1
                pnl = (float(exit_) - float(entry)) * mult * pv * float(size)
                d["pnl"] += pnl
                d["closed"] += 1
                if pnl > 0:
                    d["wins"] += 1
    return days, total


def _cell(day_str, info):
    if info is None:
        return '<td class="empty"></td>'
    pnl, n, wins, closed = info["pnl"], info["n"], info["wins"], info["closed"]
    cls = "flat" if closed == 0 else ("win" if pnl > 0 else "loss" if pnl < 0 else "flat")
    wr = f"{(wins/closed*100):.0f}%" if closed else "—"
    dnum = int(day_str[-2:])
    pnl_str = f"${pnl:+,.0f}" if closed else "·"
    return (f'<td class="{cls}"><div class="dnum">{dnum}</div>'
            f'<div class="pnl">{pnl_str}</div><div class="meta">{n} tr · {wr}</div></td>')


def _month_grid(year, month, days):
    rows = []
    for week in _cal.Calendar(firstweekday=6).monthdatescalendar(year, month):  # Sun-first
        cells = []
        for d in week:
            if d.month != month:
                cells.append('<td class="empty"></td>')
            else:
                cells.append(_cell(d.isoformat(), days.get(d.isoformat())))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join(f"<th>{x}</th>" for x in ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"))
    title = f"{_cal.month_name[month]} {year}"
    mtot = sum(v["pnl"] for k, v in days.items() if k.startswith(f"{year}-{month:02d}"))
    mn = sum(v["n"] for k, v in days.items() if k.startswith(f"{year}-{month:02d}"))
    return (f'<h2>{title} <span class="mtot {"win" if mtot>0 else "loss" if mtot<0 else ""}">'
            f'${mtot:+,.0f}</span> <span class="msub">{mn} trades</span></h2>'
            f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def main():
    days, total = _load()
    # Render the 3 most recent months that have data (or current month if empty).
    months = sorted({k[:7] for k in days}) or [datetime.now(timezone.utc).strftime("%Y-%m")]
    grids = "".join(_month_grid(int(m[:4]), int(m[5:7]), days) for m in months[-3:][::-1])
    gtot = sum(v["pnl"] for v in days.values())
    pct = min(100, total / GOAL * 100)
    html = f"""<!doctype html><meta charset=utf-8><title>Futures P&L Calendar</title>
<style>
 body{{background:#0d1117;color:#e6edf3;font:14px -apple-system,Segoe UI,sans-serif;padding:24px;max-width:760px;margin:auto}}
 h1{{font-size:20px}} h2{{font-size:16px;margin:24px 0 8px;border-bottom:1px solid #30363d;padding-bottom:6px}}
 table{{border-collapse:collapse;width:100%}} th{{color:#7d8590;font-weight:500;padding:4px;font-size:11px}}
 td{{border:1px solid #21262d;height:62px;width:14%;vertical-align:top;padding:4px;border-radius:4px}}
 .empty{{background:transparent;border:none}} .win{{background:#0f2e1a}} .loss{{background:#3a1518}} .flat{{background:#161b22}}
 .dnum{{color:#7d8590;font-size:11px}} .pnl{{font-weight:700;font-size:15px;margin-top:2px}}
 .win .pnl{{color:#3fb950}} .loss .pnl{{color:#f85149}} .meta{{color:#7d8590;font-size:10px;margin-top:2px}}
 .mtot.win{{color:#3fb950}} .mtot.loss{{color:#f85149}} .msub{{color:#7d8590;font-size:12px;font-weight:400}}
 .bar{{background:#161b22;border-radius:6px;height:10px;overflow:hidden;margin:6px 0}}
 .bar>div{{background:#1f6feb;height:100%;width:{pct:.1f}%}}
 .tot{{font-size:18px;font-weight:700}} .tot.win{{color:#3fb950}} .tot.loss{{color:#f85149}}
</style>
<h1>Futures Sandbox — P&L Calendar</h1>
<p>Total realized: <span class="tot {'win' if gtot>0 else 'loss' if gtot<0 else ''}">${gtot:+,.0f}</span>
 &nbsp;·&nbsp; {total} / {GOAL} trades toward validation</p>
<div class="bar"><div></div></div>
{grids}
<p style="color:#7d8590;font-size:11px;margin-top:24px">Paper sandbox · $ from closed trades (open trades counted, $0 until exit logged) · regenerate: python3 scripts/futures_calendar.py</p>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"Calendar: {total} trades, total realized ${gtot:+,.0f} → {OUT.relative_to(ROOT)}")
    print(f"  Open in browser: file://{OUT}")


if __name__ == "__main__":
    main()
