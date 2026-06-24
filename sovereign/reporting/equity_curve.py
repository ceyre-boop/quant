"""
sovereign/reporting/equity_curve.py
====================================
The shared money-over-time artifact for the Alta proof system.

ONE schema ("equity_curve.v1") renders BOTH the backtest proof curve and the
live OANDA-practice NAV curve, so a single dashboard panel and `alta money` can
answer "has this actually made money?" from either source.

Pure transforms only — no backtest, no broker, no network. Callers feed closed
trades (backtest) or NAV snapshots (live); this builds the curve + stats.

Money-curve convention: each trade contributes its own conviction-sized return
`risk_adjusted_pnl_pct` (= pnl_pct * risk_pct, the field the backtester already
records). Sharpe is annualised by EMPIRICAL trades/year to match
forex_backtester, and the headline portfolio Sharpe is the √n-weighted mean of
per-pair Sharpes — the same aggregation holdout_validation uses, so the proof
number equals the ~1.25 OOS figure recorded in CLAUDE.md.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Iterable

SCHEMA = "equity_curve.v1"


# ─── small helpers ────────────────────────────────────────────────────────────

def _iso_date(x: Any) -> str:
    """Best-effort YYYY-MM-DD from a pandas Timestamp / datetime / str."""
    if x is None:
        return ""
    if hasattr(x, "date"):
        try:
            return x.date().isoformat()
        except Exception:
            pass
    return str(x)[:10]


def _isnan(x: float) -> bool:
    try:
        return math.isnan(x)
    except TypeError:
        return False


def _years_span(iso_dates: list[str]) -> float:
    ds = [d for d in iso_dates if d]
    if len(ds) < 2:
        return max(len(ds) / 252.0, 1e-9)
    try:
        a = date.fromisoformat(ds[0][:10])
        b = date.fromisoformat(ds[-1][:10])
        yrs = (b - a).days / 365.25
        return yrs if yrs > 0 else max(len(ds) / 252.0, 1e-9)
    except Exception:
        return max(len(ds) / 252.0, 1e-9)


def _cagr_pct(init: float, final: float, n_years: float) -> float:
    if init <= 0 or final <= 0 or n_years <= 0:
        return 0.0
    return round(((final / init) ** (1.0 / n_years) - 1.0) * 100, 3)


def _max_drawdown_pct(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (e - peak) / peak
            if dd < mdd:
                mdd = dd
    return round(mdd * 100, 3)


def _sharpe(returns: list[float], n_years: float) -> float:
    """Annualised Sharpe from per-trade (or per-step) returns.

    Annualised by empirical events/year (n/years) — matches forex_backtester so
    the figure is comparable to the trusted v015 number.
    """
    n = len(returns)
    if n < 2 or n_years <= 0:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    sd = math.sqrt(var)
    if sd <= 1e-12:
        return 0.0
    return round((mean / sd) * math.sqrt(n / n_years), 3)


def weighted_portfolio_sharpe(pairs: Iterable[tuple]) -> float:
    """√n-weighted mean of per-pair Sharpes (Lo 2002: SE ∝ 1/√n).

    Reproduces holdout_validation_v014._sharpe_from_results so the proof headline
    equals the OOS Sharpe recorded in CLAUDE.md (~1.25).
    """
    rows = [(s, n) for (s, n) in pairs if s is not None and not _isnan(s) and n and n > 0]
    wsum = sum(math.sqrt(n) for _, n in rows)
    if not wsum:
        return 0.0
    return round(sum(s * math.sqrt(n) for s, n in rows) / wsum, 4)


# ─── builders ─────────────────────────────────────────────────────────────────

def build_from_trades(
    trades: list[dict],
    *,
    initial_equity: float = 100_000.0,
    label: str = "backtest",
    source: str = "backtest",
    return_field: str = "risk_adjusted_pnl_pct",
) -> dict:
    """Compound a list of closed trades into an equity_curve.v1 artifact.

    Trades are ordered by exit date so the curve is chronological across pairs.
    Each trade contributes `return_field` (falls back to raw `pnl_pct`).
    """
    rows = []
    for t in trades:
        r = t.get(return_field)
        if r is None:
            r = t.get("pnl_pct", 0.0)
        rows.append({
            "exit": _iso_date(t.get("exit_date") or t.get("exit") or t.get("date")),
            "pair": t.get("pair", ""),
            "ret": float(r),
        })
    rows.sort(key=lambda x: x["exit"])

    equity = float(initial_equity)
    eq_series = [equity]
    points: list[dict] = []
    rets: list[float] = []
    for row in rows:
        prev = equity
        equity *= (1.0 + row["ret"])
        eq_series.append(equity)
        rets.append(row["ret"])
        points.append({
            "t": row["exit"],
            "pair": row["pair"],
            "equity": round(equity, 2),
            "pnl": round(equity - prev, 2),
            "ret": round(row["ret"], 6),
        })

    n_years = _years_span([p["t"] for p in points])
    stats = _trade_stats(eq_series, rets, initial_equity, n_years)

    return {
        "schema": SCHEMA,
        "source": source,
        "label": label,
        "initial_equity": round(float(initial_equity), 2),
        "final_equity": round(equity, 2),
        "n_points": len(points),
        "points": points,
        "stats": stats,
    }


def build_from_nav(nav_points: list[dict], *, label: str = "live", source: str = "live") -> dict:
    """Build equity_curve.v1 from a NAV time series (live broker snapshots).

    nav_points: list of {"t": iso, "nav": float}. Stats computed on the NAV
    series directly (total return vs first point, drawdown, step-return Sharpe).
    """
    pts = [p for p in nav_points if p.get("nav") is not None]
    pts.sort(key=lambda p: str(p.get("t", "")))
    if not pts:
        return {"schema": SCHEMA, "source": source, "label": label,
                "initial_equity": 0.0, "final_equity": 0.0, "n_points": 0,
                "points": [], "stats": {}}

    init = float(pts[0]["nav"])
    eq_series = [float(p["nav"]) for p in pts]
    points: list[dict] = []
    rets: list[float] = []
    prev = init
    for p in pts:
        nav = float(p["nav"])
        ret = (nav - prev) / prev if prev else 0.0
        rets.append(ret)
        points.append({"t": _iso_date(p.get("t")), "equity": round(nav, 2),
                       "pnl": round(nav - prev, 2), "ret": round(ret, 6)})
        prev = nav

    n_years = _years_span([p["t"] for p in points])
    final = eq_series[-1]
    stats = {
        "total_return_pct": round((final / init - 1) * 100, 3) if init else 0.0,
        "cagr_pct": _cagr_pct(init, final, n_years),
        "sharpe": _sharpe(rets[1:], n_years) if len(rets) > 2 else 0.0,
        "max_drawdown_pct": _max_drawdown_pct(eq_series),
        "win_rate": None,
        "profit_factor": None,
        "n_trades": None,
        "n_snapshots": len(points),
        "years": round(n_years, 2),
    }
    return {"schema": SCHEMA, "source": source, "label": label,
            "initial_equity": round(init, 2), "final_equity": round(final, 2),
            "n_points": len(points), "points": points, "stats": stats}


def _trade_stats(eq_series: list[float], rets: list[float], init: float, n_years: float) -> dict:
    final = eq_series[-1]
    n = len(rets)
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "total_return_pct": round((final / init - 1) * 100, 3) if init else 0.0,
        "cagr_pct": _cagr_pct(init, final, n_years),
        "sharpe": _sharpe(rets, n_years),
        "max_drawdown_pct": _max_drawdown_pct(eq_series),
        "win_rate": round(len(wins) / n, 4) if n else 0.0,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 1e-12 else None,
        "n_trades": n,
        "years": round(n_years, 2),
    }


def points_to_csv(curve: dict) -> str:
    """Serialise the curve's points to CSV text (t,equity,pnl,ret[,pair])."""
    pts = curve.get("points", [])
    has_pair = any("pair" in p for p in pts)
    header = "t,equity,pnl,ret" + (",pair" if has_pair else "")
    lines = [header]
    for p in pts:
        row = f'{p.get("t","")},{p.get("equity","")},{p.get("pnl","")},{p.get("ret","")}'
        if has_pair:
            row += f',{p.get("pair","")}'
        lines.append(row)
    return "\n".join(lines) + "\n"
