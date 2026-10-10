"""Score a sealed basket vs its benchmark. Pure functions; the price loader is injected."""
from __future__ import annotations

from datetime import date
from typing import Any, Callable

import pandas as pd

Loader = Callable[[str], pd.Series]


def basket_curve(legs: list[dict[str, Any]], entry_date: str, loader: Loader,
                 asof: str | None = None) -> pd.Series:
    """Buy-and-hold value of $1 invested at entry_date with the given weights.
    Weights are set once at entry (no rebalance) — that is the thesis contract."""
    start = pd.Timestamp(entry_date).normalize()
    end = pd.Timestamp(asof).normalize() if asof else None
    parts = []
    for leg in legs:
        s = loader(leg["ticker"])
        s = s[s.index >= start]
        if end is not None:
            s = s[s.index <= end]
        if s.empty:
            raise ValueError(f"no prices for {leg['ticker']} from {entry_date}")
        parts.append((s / s.iloc[0]) * float(leg["weight"]))
    df = pd.concat(parts, axis=1).ffill()
    return df.sum(axis=1)


def max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def _mark(curve: pd.Series, months: int) -> float | None:
    t = curve.index[0] + pd.DateOffset(months=months)
    s = curve[curve.index <= t]
    if curve.index[-1] < t:
        return None  # horizon not reached yet
    return float(s.iloc[-1] / curve.iloc[0] - 1.0)


def score(record: dict[str, Any], loader: Loader, asof: str | None = None) -> dict[str, Any]:
    legs = record["basket"]
    entry = record["entry_date"]
    bench = record.get("benchmark", "SPY")
    b = basket_curve(legs, entry, loader, asof)
    m = basket_curve([{"ticker": bench, "weight": 1.0}], entry, loader, asof)
    idx = b.index.intersection(m.index)
    b, m = b.loc[idx], m.loc[idx]
    inv = record.get("invalidation", {})
    dd = max_drawdown(b)
    invalidated = False
    if inv.get("metric") == "basket_drawdown" and inv.get("threshold") is not None:
        invalidated = dd <= -abs(float(inv["threshold"]))
    days = int((b.index[-1] - b.index[0]).days)
    return {
        "asof": str(b.index[-1].date()),
        "entry_date": str(b.index[0].date()),
        "days_held": days,
        "basket_return": float(b.iloc[-1] - 1.0),
        "benchmark": bench,
        "benchmark_return": float(m.iloc[-1] - 1.0),
        "excess_return": float(b.iloc[-1] - m.iloc[-1]),
        "basket_max_drawdown": dd,
        "benchmark_max_drawdown": max_drawdown(m),
        "marks": {f"{k}m": {"basket": _mark(b, k), "benchmark": _mark(m, k)} for k in (1, 3, 6, 12)},
        "invalidation": inv,
        "invalidated": bool(invalidated),
        "legs": [
            {"ticker": l["ticker"], "weight": l["weight"],
             "return": float(loader(l["ticker"]).loc[idx].iloc[-1] / loader(l["ticker"]).loc[idx].iloc[0] - 1.0)}
            for l in legs
        ],
    }


def format_scorecard(rec: dict[str, Any], card: dict[str, Any]) -> str:
    out = [f"{rec['id']}  {rec['title']}  [{rec.get('status')}]  hash {str(rec.get('hash_lock') or '')[:8]}",
           f"  {card['entry_date']} → {card['asof']} ({card['days_held']}d)",
           f"  basket {card['basket_return']:+.1%}   {card['benchmark']} {card['benchmark_return']:+.1%}   "
           f"excess {card['excess_return']:+.1%}   maxDD {card['basket_max_drawdown']:+.1%} (bench {card['benchmark_max_drawdown']:+.1%})"]
    for k, v in card["marks"].items():
        if v["basket"] is not None:
            out.append(f"  @{k}: basket {v['basket']:+.1%}  bench {v['benchmark']:+.1%}")
    for l in card["legs"]:
        out.append(f"    {l['ticker']:6s} w={l['weight']:.2f}  {l['return']:+.1%}")
    inv = card["invalidation"]
    out.append(f"  invalidation: {inv.get('rule')}  → {'INVALIDATED' if card['invalidated'] else 'holding'}")
    return "\n".join(out)
