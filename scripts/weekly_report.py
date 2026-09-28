#!/usr/bin/env python3
"""
scripts/weekly_report.py — weekly v015 performance report from committed live data.

Run by .github/workflows/weekly-report.yml (stdlib only, no requirements install).
Replaces the archived ML report (clawd_trading/meta_evaluator/performance_monitor.py),
whose imports no longer exist.

Sources (committed on sovereign-v2 by the evening data sync):
  data/agent/equity_curve_live.jsonl      OANDA NAV snapshots  -> NAV change, drawdown, Sharpe
  data/decision_logs/decisions_*.jsonl    decision rows        -> FOREX trades closed in window
  RISK_CONSTITUTION.md  Article 3         first breaker        -> drawdown alert threshold

Every number carries its age. Exit 1 when the newest NAV point is older than
--stale-days (or there is none): a dead writer must turn CI red, not sit quiet.

Usage:
  python scripts/weekly_report.py [--as-of ISO] [--days 7] [--stale-days 7] [--out weekly_report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.reporting.equity_curve import build_from_nav  # noqa: E402

GAP_DAYS = 3.0
_FRAC = re.compile(r"(\.\d{6})\d+")


def parse_ts(s: str | None) -> datetime | None:
    """ISO timestamp -> aware UTC datetime. Tolerates 'Z' and nanosecond fractions."""
    if not s:
        return None
    s = _FRAC.sub(r"\1", str(s).strip().replace("Z", "+00:00"))
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_dd_cap(text: str) -> float:
    """First bold percentage in RISK_CONSTITUTION.md Article 3 (the size-halving breaker)."""
    m = re.search(r"^## Article 3\b(.*?)(?=^## |\Z)", text, re.S | re.M)
    if not m:
        raise ValueError("RISK_CONSTITUTION.md: Article 3 not found")
    pct = re.search(r"\*\*(\d+(?:\.\d+)?)%\*\*", m.group(1))
    if not pct:
        raise ValueError("RISK_CONSTITUTION.md: no bold drawdown % in Article 3")
    return float(pct.group(1))


def _norm_pair(p: str | None) -> str:
    return str(p or "").upper().replace("=X", "").replace("_", "").replace("/", "")


def nav_section(rows: list[dict], as_of: datetime, days: int) -> dict:
    pts = []
    for r in rows:
        t = parse_ts(r.get("t"))
        if t is not None and r.get("nav") is not None and t <= as_of:
            pts.append((t, float(r["nav"])))
    pts.sort()
    if not pts:
        return {"n_points": 0}

    start = as_of - timedelta(days=days)
    latest_t, latest_nav = pts[-1]
    anchor = [p for p in pts if p[0] <= start]
    start_t, start_nav = anchor[-1] if anchor else pts[0]

    peak = max(n for _, n in pts)
    gaps = [{"from": a[0].isoformat(), "to": b[0].isoformat(),
             "days": round((b[0] - a[0]).total_seconds() / 86400, 2)}
            for a, b in zip(pts, pts[1:])
            if (b[0] - a[0]).total_seconds() / 86400 > GAP_DAYS]

    inception = build_from_nav([{"t": t.isoformat(), "nav": n} for t, n in pts])["stats"]
    return {
        "n_points": len(pts),
        "first_t": pts[0][0].isoformat(),
        "latest_t": latest_t.isoformat(),
        "age_days": round((as_of - latest_t).total_seconds() / 86400, 2),
        "latest_nav": latest_nav,
        "window_start_t": start_t.isoformat(),
        "window_start_nav": start_nav,
        "week_change_pct": round((latest_nav / start_nav - 1) * 100, 3) if start_nav else None,
        # > days when the anchor sits before a writer gap: the "week" change is really longer
        "window_span_days": round((latest_t - start_t).total_seconds() / 86400, 2),
        "peak_nav": peak,
        "current_drawdown_pct": round((latest_nav / peak - 1) * 100, 3) if peak else None,
        "inception": inception,
        "gaps": gaps,
    }


def trades_section(rows: list[dict], as_of: datetime, days: int) -> dict:
    """FOREX outcomes closed in (as_of - days, as_of], windowed by exit_timestamp.

    Resolved rows are de-duped on (pair, direction, exit time): the fills backfill can
    match several decision rows to one OANDA close, which would otherwise count one
    trade N times. Rows whose exit precedes their entry are cross-matched backfill
    artefacts, excluded and counted in `invalid_rows` so they stay visible.
    """
    start = as_of - timedelta(days=days)
    seen: set[tuple] = set()
    wins = losses = expired = open_ = invalid = 0
    r_vals: list[float] = []
    by_pair: dict[str, dict[str, int]] = {}
    for r in rows:
        if r.get("system") != "FOREX":
            continue
        pair, direction = _norm_pair(r.get("pair")), r.get("direction")
        outcome = r.get("outcome")
        entry_t = parse_ts(r.get("entry_timestamp"))
        if outcome is None:
            key = ("open", pair, direction, entry_t)
            if key not in seen and (entry_t is None or entry_t <= as_of):
                seen.add(key)
                open_ += 1
            continue
        exit_t = parse_ts(r.get("exit_timestamp"))
        if exit_t is None or not (start < exit_t <= as_of):
            continue
        key = ("closed", pair, direction, exit_t)
        if key in seen:
            continue
        seen.add(key)
        if entry_t is not None and exit_t < entry_t:
            invalid += 1
            continue
        if outcome == "EXPIRED":
            expired += 1
        elif outcome in ("WIN", "LOSS"):
            wins += outcome == "WIN"
            losses += outcome == "LOSS"
            slot = by_pair.setdefault(pair, {"W": 0, "L": 0})
            slot["W" if outcome == "WIN" else "L"] += 1
            if r.get("r_realized") is not None:
                r_vals.append(float(r["r_realized"]))
    closed = wins + losses
    return {
        "closed": closed, "wins": wins, "losses": losses,
        "win_rate_pct": round(wins / closed * 100, 1) if closed else None,
        "avg_r": round(sum(r_vals) / len(r_vals), 3) if r_vals else None,
        "n_r": len(r_vals),
        "expired_in_window": expired,
        "open": open_,
        "invalid_rows": invalid,
        "by_pair": dict(sorted(by_pair.items())),
    }


def build_report(repo: Path, as_of: datetime, *, days: int, stale_days: float) -> dict:
    nav = nav_section(_read_jsonl(repo / "data/agent/equity_curve_live.jsonl"), as_of, days)
    decisions: list[dict] = []
    for f in sorted((repo / "data/decision_logs").glob("decisions_*.jsonl")):
        decisions.extend(_read_jsonl(f))
    trades = trades_section(decisions, as_of, days)
    dd_cap = parse_dd_cap((repo / "RISK_CONSTITUTION.md").read_text())

    stale = nav.get("n_points", 0) == 0 or nav["age_days"] > stale_days
    cur_dd = nav.get("current_drawdown_pct")
    return {
        "schema": "weekly_report.v2",
        "as_of": as_of.isoformat(),
        "window_days": days,
        "week_ending": as_of.date().isoformat(),
        "stale": stale,
        "stale_days": stale_days,
        "dd_cap_pct": dd_cap,
        "dd_breach": cur_dd is not None and cur_dd <= -dd_cap,
        "nav": nav,
        "trades": trades,
    }


def _fmt(v, suffix: str = "") -> str:
    return "n/a" if v is None else f"{v}{suffix}"


def render_markdown(rep: dict) -> str:
    nav, t = rep["nav"], rep["trades"]
    lines = [f"## Weekly Performance Report — week ending {rep['week_ending']}", ""]
    if rep["stale"]:
        age = nav.get("age_days")
        lines += [f"> **STALE DATA** — newest NAV point is "
                  f"{'missing' if age is None else f'{age} days old'} "
                  f"(limit {rep['stale_days']}d). The equity writer or the data push is dead.", ""]
    if rep["dd_breach"]:
        lines += [f"> **DRAWDOWN BREACH** — current {nav.get('current_drawdown_pct')}% "
                  f"vs Article 3 cap -{rep['dd_cap_pct']}%.", ""]
    if nav.get("n_points"):
        inc = nav["inception"]
        lines += [
            "| Metric | Value | As of |", "|---|---|---|",
            f"| NAV | {nav['latest_nav']:,.2f} | {nav['latest_t'][:16]} ({nav['age_days']}d old) |",
            f"| {rep['window_days']}d NAV change | {_fmt(nav['week_change_pct'], '%')} "
            f"| from {nav['window_start_t'][:16]}"
            + (f" — **spans {nav['window_span_days']}d** (no snapshot at window start)"
               if nav["window_span_days"] > rep["window_days"] + 1 else "") + " |",
            f"| Current drawdown from peak | {_fmt(nav['current_drawdown_pct'], '%')} "
            f"| cap -{rep['dd_cap_pct']}% |",
            f"| Since-inception return | {_fmt(inc.get('total_return_pct'), '%')} "
            f"| {nav['first_t'][:10]} → {nav['latest_t'][:10]} |",
            f"| Since-inception max drawdown | {_fmt(inc.get('max_drawdown_pct'), '%')} | {nav['n_points']} snapshots |",
            f"| Since-inception Sharpe (snapshot steps) | {_fmt(inc.get('sharpe'))} | {nav['n_points']} snapshots |",
        ]
    lines += [
        f"| FOREX trades closed ({rep['window_days']}d) | {t['closed']} ({t['wins']}W / {t['losses']}L) "
        f"| by exit_timestamp |",
        f"| Win rate | {_fmt(t['win_rate_pct'], '%')} | n={t['closed']} |",
        f"| Avg R | {_fmt(t['avg_r'])} | n={t['n_r']} with r_realized |",
        f"| Expired in window / unresolved (outcome null) | {t['expired_in_window']} / {t['open']} "
        f"| unresolved ≠ open positions; see NAV open_trade_count |",
    ]
    if t["by_pair"]:
        lines.append("| By pair | " + ", ".join(f"{p} {c['W']}W/{c['L']}L" for p, c in t["by_pair"].items())
                     + " | closed in window |")
    if t["invalid_rows"]:
        lines.append(f"| Excluded rows (exit before entry) | {t['invalid_rows']} | backfill artefacts |")
    lines.append("")
    if nav.get("gaps"):
        lines += [f"**NAV snapshot gaps > {GAP_DAYS:g} days** (writer outages):", ""]
        lines += [f"- {g['from'][:10]} → {g['to'][:10]}: {g['days']}d" for g in nav["gaps"]]
        lines.append("")
    return "\n".join(lines)


def _github_outputs(rep: dict) -> dict:
    nav = rep["nav"]
    b = lambda x: "true" if x else "false"  # noqa: E731
    return {
        "stale": b(rep["stale"]),
        "dd_breach": b(rep["dd_breach"]),
        "nav_age_days": _fmt(nav.get("age_days")),
        "week_nav_change_pct": _fmt(nav.get("week_change_pct")),
        "current_drawdown_pct": _fmt(nav.get("current_drawdown_pct")),
        "max_drawdown_pct": _fmt((nav.get("inception") or {}).get("max_drawdown_pct")),
        "dd_cap_pct": _fmt(rep["dd_cap_pct"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Weekly v015 performance report from committed live data")
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--as-of", help="ISO timestamp (default: now, UTC)")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--stale-days", type=float, default=7.0)
    ap.add_argument("--out", default="weekly_report.json")
    args = ap.parse_args(argv)

    as_of = parse_ts(args.as_of) if args.as_of else datetime.now(timezone.utc)
    if as_of is None:
        ap.error(f"unparseable --as-of: {args.as_of}")

    rep = build_report(Path(args.repo), as_of, days=args.days, stale_days=args.stale_days)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    md = render_markdown(rep)
    print(md)

    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md + "\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            for k, v in _github_outputs(rep).items():
                f.write(f"{k}={v}\n")

    if rep["stale"]:
        print(f"STALE: newest NAV point age {rep['nav'].get('age_days')}d > {args.stale_days}d",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
