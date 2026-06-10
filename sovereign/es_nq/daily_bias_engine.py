"""ES/NQ daily bias engine — 5 pre-registered inputs → UP/DOWN/NEUTRAL + confidence.

Pure: no I/O, no network — everything arrives as frames/dicts. Every input is
computable strictly BEFORE 09:30 ET on the session date (lookahead cutoffs are
pre-registered in data/research/es_nq_preregistration.json):

  overnight     0.30  px@09:25 ET vs prior 16:00 RTH close; 0 on roll days
  calendar      0.25  backtest: Amendment A1 (0 direction, 0.75 confidence mult
                      on FOMC/CPI/NFP days); live: real tone from the scraper
  vix           0.20  prior close: −sign(5d change)/2 − sign(vs SMA20)/2
  hurst         0.15  variance-ratio H on trailing 20 daily log returns (t−1);
                      ±sign(5d return) for H>0.55 / H<0.45, else 0
  international 0.10  Nikkei same-day close sign + DAX prior-session sign, ½ each

NEUTRAL below confidence 0.40 — the skip is a trade.
Weights are NEVER fit (tripwire: tests/unit/test_es_nq_isolation.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from sovereign.es_nq.config import es_nq_params


@dataclass(frozen=True)
class BiasResult:
    date: str
    direction: str                 # UP | DOWN | NEUTRAL
    confidence: float              # 0..1
    raw_score: float               # weighted sum in [-1, 1]
    components: dict = field(default_factory=dict)
    event_day: bool = False
    roll_day: bool = False
    reasoning: str = ""


def overnight_score(overnight_ret: float, roll_day: bool, params: Optional[dict] = None) -> float:
    p = (params or es_nq_params())["bias"]
    if roll_day or overnight_ret is None or not math.isfinite(overnight_ret):
        return 0.0
    return float(np.clip(overnight_ret / p["overnight_strong_pct"], -1.0, 1.0))


def calendar_score(date: str, calendar: dict, live_tone: Optional[float] = None
                   ) -> tuple[float, bool]:
    """(directional score, event_day). Backtest: tone unavailable → 0 (Amendment A1).
    Live passes a real tone in [-1, 1]."""
    events = (calendar.get(date) or {}).get("events", [])
    event_day = len(events) > 0
    if live_tone is not None:
        return float(np.clip(live_tone, -1.0, 1.0)), event_day
    return 0.0, event_day


def vix_score(vix: pd.Series, date: str, params: Optional[dict] = None) -> float:
    """vix: daily closes indexed by YYYY-MM-DD strings. Uses data through t−1 only."""
    p = (params or es_nq_params())["bias"]["vix"]
    prior = vix[vix.index < date].dropna()
    need = max(p["direction_days"] + 1, p["ma_days"])
    if len(prior) < need:
        return 0.0
    last = float(prior.iloc[-1])
    dir5 = float(np.sign(last - float(prior.iloc[-1 - p["direction_days"]])))
    rel = float(np.sign(last - float(prior.tail(p["ma_days"]).mean())))
    return 0.5 * (-dir5) + 0.5 * (-rel)


def variance_ratio_hurst(log_returns: np.ndarray) -> float:
    """Variance-ratio Hurst on ONE window of 1-period log returns — the same math as
    universe_sweep._rolling_hurst, single-window form. Clipped to [0.1, 0.9]."""
    r = np.asarray(log_returns, dtype=np.float64)
    if len(r) < 4:
        return 0.5
    var1 = float(np.mean((r - r.mean()) ** 2))
    r2 = r[:-1] + r[1:]
    var2 = float(np.mean((r2 - r2.mean()) ** 2))
    ratio = var2 / (2.0 * var1 + 1e-12)
    ratio = float(np.clip(ratio, 1e-6, 100.0))
    h = 0.5 + 0.5 * math.log(ratio) / math.log(2.0)
    return float(np.clip(h, 0.1, 0.9))


def hurst_score(daily_closes: pd.Series, date: str, params: Optional[dict] = None) -> float:
    """daily_closes: NQ RTH closes indexed by YYYY-MM-DD. Data through t−1 only."""
    p = (params or es_nq_params())["bias"]["hurst"]
    prior = daily_closes[daily_closes.index < date].dropna()
    if len(prior) < p["window_days"] + 1:
        return 0.0
    logret = np.log(prior.values[1:] / prior.values[:-1])
    h = variance_ratio_hurst(logret[-p["window_days"]:])
    ctx = float(np.sign(logret[-p["context_return_days"]:].sum()))
    if ctx == 0.0:
        return 0.0
    if h > p["momentum_above"]:
        return ctx
    if h < p["reversion_below"]:
        return -ctx
    return 0.0


def international_score(nikkei: pd.Series, dax: pd.Series, date: str) -> float:
    """Nikkei: same-calendar-date close (Tokyo completes pre-02:00 ET — usable same day);
    0 if Tokyo holiday. DAX: prior COMPLETED session close-to-close (stale by design)."""
    nik = 0.0
    n = nikkei.dropna()
    if date in n.index:
        pos = n.index.get_loc(date)
        if pos >= 1:
            nik = float(np.sign(float(n.iloc[pos]) - float(n.iloc[pos - 1])))
    dax_s = 0.0
    d = dax.dropna()
    prior = d[d.index < date]
    if len(prior) >= 2:
        dax_s = float(np.sign(float(prior.iloc[-1]) - float(prior.iloc[-2])))
    return 0.5 * nik + 0.5 * dax_s


def compute_bias(date: str, components: dict[str, float], event_day: bool,
                 roll_day: bool, *, calendar_active: bool = False,
                 params: Optional[dict] = None) -> BiasResult:
    """Weighted vote → BiasResult.

    `calendar_active=False` (backtest, Amendment A1): the calendar weight is excluded
    from the normalizer and event days apply the 0.75 confidence multiplier.
    `calendar_active=True` (live): full pre-registered weights, no multiplier.
    """
    p = (params or es_nq_params())["bias"]
    w = p["weights"]
    active = dict(w)
    if not calendar_active:
        active.pop("calendar")
    raw = sum(active[k] * components.get(k, 0.0) for k in active)
    denom = sum(active.values())
    confidence = abs(raw) / denom if denom > 0 else 0.0
    if not calendar_active and event_day:
        confidence *= p["calendar_event_confidence_mult"]
    direction = "NEUTRAL"
    if confidence >= p["neutral_below_confidence"] and raw != 0.0:
        direction = "UP" if raw > 0 else "DOWN"
    parts = ", ".join(f"{k}={components.get(k, 0.0):+.2f}(w{active.get(k, w.get(k)):.2f})"
                      for k in w)
    reasoning = (f"raw={raw:+.3f} conf={confidence:.3f} → {direction}; {parts}"
                 f"{'; EVENT_DAY×0.75' if (event_day and not calendar_active) else ''}"
                 f"{'; ROLL_DAY overnight zeroed' if roll_day else ''}")
    return BiasResult(date=date, direction=direction, confidence=round(confidence, 4),
                      raw_score=round(raw, 4), components=dict(components),
                      event_day=event_day, roll_day=roll_day, reasoning=reasoning)


def build_feature_table(daily: pd.DataFrame, aux: pd.DataFrame, calendar: dict,
                        start: str, end: str, params: Optional[dict] = None) -> pd.DataFrame:
    """One row per session in [start, end]: component scores + bias + realized outcome.

    `daily` = data_store.build_daily_table output; `aux` = vix/nikkei/dax daily closes.
    Outcome (pre-registered): direction_real = sign(rth_close − rth_open); zeros NaN.
    """
    p = params or es_nq_params()
    closes = daily["rth_close"]
    rows = []
    for date, row in daily.iterrows():
        if not (start <= date <= end):
            continue
        comp = {
            "overnight": overnight_score(row["overnight_ret"], bool(row["roll_day"]), p),
            "vix": vix_score(aux["vix"], date, p),
            "hurst": hurst_score(closes, date, p),
            "international": international_score(aux["nikkei"], aux["dax"], date),
        }
        cal_s, event_day = calendar_score(date, calendar)
        comp["calendar"] = cal_s
        bias = compute_bias(date, comp, event_day, bool(row["roll_day"]),
                            calendar_active=False, params=p)
        move = row["rth_close"] - row["rth_open"]
        move_pct = move / row["rth_open"]
        rows.append({
            "date": date, **{f"s_{k}": v for k, v in comp.items()},
            "direction": bias.direction, "confidence": bias.confidence,
            "raw_score": bias.raw_score, "event_day": event_day,
            "roll_day": bool(row["roll_day"]),
            "direction_real": ("UP" if move > 0 else "DOWN") if move != 0 else None,
            "move_pct": float(move_pct),
            "flat_secondary": bool(abs(move_pct) < p["outcome"]["secondary_noise_threshold_pct"]),
        })
    out = pd.DataFrame(rows).set_index("date")
    if len(out) == 0:
        raise ValueError(f"build_feature_table: no sessions in [{start}, {end}]")
    return out
