"""Options-leg event logic for the positioning family (HYP-074/075/076/078/079/080 + 077-full).

Pure functions — no I/O, no network, no clock — so every semantic is offline-testable,
mirroring sovereign/research/positioning/event_study.py (which this module builds on).

Declared interpretations under the LOCKED protocol (stamped verbatim on every seal;
declared 2026-07-03 from data-availability arithmetic BEFORE any statistic was computed):
- "trailing 252 obs" (rr25_z / bf25_z / tone_z) = the DAILY board series
  (sentiment_board_state, ASOF-carried weekly surface), pandas rolling(window=252,
  min_periods=252), std ddof=1. 252 is the trading-year idiom; the weekly-series
  reading would leave <90 usable obs on the 2020+ Value-tier depth and was rejected
  on that arithmetic alone. Trailing-only by construction; truncation-invariance is
  asserted at run time.
- Event t0 for EOD-derived signals (surface z, tone z, spot extremes): first trading
  day STRICTLY AFTER the signal date — EOD option marks and the EOD spot close are
  simultaneous, so same-close execution would not be "post-information" (the COT
  15:30-vs-17:00 argument does not transfer). Implemented by shifting publish_date
  one calendar day before effective_t0. Surprise releases (08:30 ET, pre-market)
  keep same-day t0 per the SENTIMENT-ECON-SURPRISE standard.
- Primary horizons (one pooled primary per HYP): 074 h=20 · 075 h=20 · 076 h=5 ·
  078 h=10 (sole) · 079 h=10 (sole) · 080 h=20; other listed horizons sealed secondary.
- Base-currency space: rr25/bf25 (FX-ETF underlyings: FXE/FXB/FXY/FXA), cot_net_pct_1y
  (6E/6B/6J/6A) and gdelt tone are BASE-currency-signed; pair-space direction applies
  event_study.PAIR_FLIP (USDJPY −1) exactly once.
- HYP-077 full composite rr term: "rr25_z aligned" mapped into the composite's [0,1]
  mean-with-0.90-threshold scale via the standard normal CDF Φ(z) (deterministic, no
  new fitted window): aligned_rr = Φ(z) if the open v015 position is long base else
  1−Φ(z); per-pair component = mean(aligned_cot, aligned_rr) when both exist, else
  aligned_cot; composite = mean over included funded pairs.
- HYP-076 cells: events are de-overlapped releases (max-|surprise_z| release per
  publish_date) with |z|>=1.5; CROWDED-AGAINST cell = crowd extreme (0.90/0.10)
  opposing the pair-space surprise direction; CONTROL cell = same-|z| events with no
  crowd extreme; extreme-ALIGNED events belong to neither cell. Null = per-pair
  permutation of the crowded label over the pooled cell union, preserving per-pair
  crowded counts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sovereign.research.positioning import event_study as es  # noqa: E402

OPT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
# pair-space sign of a USD-POSITIVE surprise (USDJPY is the only USD-base pair here)
USD_SURPRISE_PAIR_SIGN = {"EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "USDJPY": 1}

PRIMARY_H = {"HYP-074": 20, "HYP-075": 20, "HYP-076": 5, "HYP-078": 10,
             "HYP-079": 10, "HYP-080": 20}
SECONDARY_H = {"HYP-074": 10, "HYP-075": 10, "HYP-076": 10, "HYP-080": 10}


# ── z construction ────────────────────────────────────────────────────────────

def z_trailing(s: pd.Series, window: int = 252) -> pd.Series:
    """Trailing z vs the last `window` obs INCLUDING today; strict min_periods."""
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=1)
    return (s - mu) / sd.replace(0.0, np.nan)


def assert_truncation_invariant(s: pd.Series, window: int = 252, cut: int = 100) -> None:
    """z computed on a truncated series must equal the full-series z on the overlap."""
    if len(s.dropna()) < window + cut + 5:
        return  # not enough data to test; caller stamps sample status anyway
    full = z_trailing(s, window)
    part = z_trailing(s.iloc[:-cut], window)
    a, b = full.iloc[:-cut].dropna(), part.dropna()
    j = a.index.intersection(b.index)
    if len(j) and not np.allclose(a.loc[j].values, b.loc[j].values, atol=1e-10, equal_nan=True):
        raise SystemExit("TRUNCATION-INVARIANCE FAILED: z is not trailing-only. HALT.")


def shift_pub(d: date) -> date:
    """EOD-signal publish shift: +1 calendar day so effective_t0 is strictly after."""
    return d + timedelta(days=1)


# ── HYP-074: rr extreme reversion ────────────────────────────────────────────

def events_074(rr_z: dict[str, pd.Series]) -> dict[str, list[es.Event]]:
    out = {}
    for pair in OPT_PAIRS:
        z = rr_z[pair].dropna()
        cross = es.detect_crossings(
            [i.date() for i in z.index], list(z.values),
            enter=lambda v: 1 if v > 2.0 else (-1 if v < -2.0 else 0),
            rearm=lambda v: abs(v) <= 1.0)
        cross = [(shift_pub(d), s, v) for d, s, v in cross]
        out[pair] = es.make_events(pair, cross, lambda s: -s, lambda v: abs(v) - 2.0)
    return out


# ── HYP-075: spot extreme without rr confirmation ────────────────────────────

def events_075(closes: dict[str, pd.Series], rr_z: dict[str, pd.Series],
               lookback: int = 60, suppress: int = 20) -> dict[str, list[es.Event]]:
    out = {}
    for pair in OPT_PAIRS:
        flip = es.PAIR_FLIP.get(pair, 1)
        c = closes[pair]
        rrp = (rr_z[pair] * flip).reindex(c.index)
        evs, last_pos = [], {1: -10**9, -1: -10**9}
        for i in range(lookback, len(c)):
            z = rrp.iloc[i]
            if not np.isfinite(z):
                continue
            win_c = c.iloc[i - lookback:i]
            win_r = rrp.iloc[i - lookback:i]
            if not np.isfinite(win_r).any():
                continue
            hi, lo = float(win_c.max()), float(win_c.min())
            px = float(c.iloc[i])
            side = 0
            if px > hi and not (z >= float(np.nanmax(win_r.values))):
                side = -1          # new high unconfirmed -> reversal short
                strength = math.log(px / hi)
            elif px < lo and not (z <= float(np.nanmin(win_r.values))):
                side = +1          # new low unconfirmed -> reversal long
                strength = math.log(lo / px)
            if side and i - last_pos[side] >= suppress:
                evs.append(es.Event(pair, shift_pub(c.index[i].date()), side, strength))
                last_pos[side] = i
        out[pair] = evs
    return out


# ── HYP-078 / HYP-079 joint-condition detectors ──────────────────────────────

@dataclass(frozen=True)
class StateEvent:
    pair: str
    signal_date: date       # board date of onset (t0 = first trading day AFTER)
    crowd_pair_side: int    # ±1 pair-space crowded direction (0 impossible at onset)


def _crowd_side(pct: float, hi: float = 0.90, lo: float = 0.10) -> int:
    return 1 if pct >= hi else (-1 if pct <= lo else 0)


def events_078(slope: dict[str, pd.Series], cot_pct: dict[str, pd.Series]) -> dict[str, list[StateEvent]]:
    out = {}
    for pair in OPT_PAIRS:
        flip = es.PAIR_FLIP.get(pair, 1)
        sl, pc = slope[pair], cot_pct[pair].reindex(slope[pair].index)
        evs, armed = [], True
        for ts, v in sl.items():
            p = pc.get(ts)
            if not (np.isfinite(v) and p is not None and np.isfinite(p)):
                continue
            joint = (v > 0.0) and (_crowd_side(float(p)) != 0)
            if armed and joint:
                evs.append(StateEvent(pair, ts.date(), _crowd_side(float(p)) * flip))
                armed = False
            elif not armed and v < 0.0:      # prereg: re-arm when slope < 0
                armed = True
        out[pair] = evs
    return out


def events_079(bf_z: dict[str, pd.Series], cot_pct: dict[str, pd.Series]) -> dict[str, list[StateEvent]]:
    out = {}
    for pair in OPT_PAIRS:
        flip = es.PAIR_FLIP.get(pair, 1)
        z, pc = bf_z[pair], cot_pct[pair].reindex(bf_z[pair].index)
        evs, armed = [], True
        for ts, v in z.items():
            p = pc.get(ts)
            if not (np.isfinite(v) and p is not None and np.isfinite(p)):
                continue
            if armed and v >= 2.0 and _crowd_side(float(p)) != 0:
                evs.append(StateEvent(pair, ts.date(), _crowd_side(float(p)) * flip))
                armed = False
            elif not armed and v < 1.0:
                armed = True
        out[pair] = evs
    return out


# ── magnitude/range statistics + per-pair date-shuffle permutation ───────────

def _t0_pos(idx: pd.DatetimeIndex, signal_date: date) -> int | None:
    t0 = es.effective_t0(shift_pub(signal_date), idx)
    if t0 is None:
        return None
    return int(idx.get_indexer([t0])[0])


def range_ratio(highs: pd.Series, lows: pd.Series, signal_date: date,
                h: int = 10, base_lb: int = 60) -> float | None:
    """(fwd h-day sum of daily H−L) / (h × trailing base_lb-day median daily H−L)."""
    idx = highs.index
    pos = _t0_pos(idx, signal_date)
    if pos is None or pos < base_lb or pos + h > len(idx):
        return None
    fwd = (highs.iloc[pos:pos + h] - lows.iloc[pos:pos + h]).sum()
    base = float((highs.iloc[pos - base_lb:pos] - lows.iloc[pos - base_lb:pos]).median())
    if not (np.isfinite(fwd) and np.isfinite(base)) or base <= 0:
        return None
    return float(fwd) / (h * base)


def abs_move(closes: pd.Series, signal_date: date, h: int = 10) -> float | None:
    idx = closes.index
    pos = _t0_pos(idx, signal_date)
    if pos is None or pos + h >= len(idx):
        return None
    c0, ch = float(closes.iloc[pos]), float(closes.iloc[pos + h])
    if not (c0 > 0 and ch > 0):
        return None
    return abs(math.log(ch / c0))


def perm_p_stat(stat_by_date: dict[str, pd.Series], event_dates: dict[str, list[date]],
                rng: np.random.Generator, n_perm: int = 10000) -> es.PermResult:
    """Pooled-median permutation for per-date scalar statistics (range ratio, |move|).

    Null: within each pair, k event labels land on k random eligible dates (the dates
    where the statistic is computable) — same shuffle family as pooled_primary_p.
    """
    obs_vals, counts, raw = [], {}, {}
    for pair, dates_ in event_dates.items():
        s = stat_by_date[pair].dropna()
        raw[pair] = s.values
        vals = [s.get(pd.Timestamp(d)) for d in dates_]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        obs_vals.extend(vals)
        counts[pair] = len(vals)
    if not obs_vals:
        return es.PermResult(float("nan"), float("nan"), n_perm, 0)
    obs = float(np.median(obs_vals))
    n_ge = 0
    for _ in range(n_perm):
        draw = []
        for pair, k in counts.items():
            pool = raw[pair]
            if k == 0 or len(pool) < k:
                continue
            draw.extend(rng.choice(pool, size=k, replace=False))
        if draw and float(np.median(draw)) >= obs:
            n_ge += 1
    return es.PermResult(obs, (n_ge + 1) / (n_perm + 1), n_perm, len(obs_vals))


# ── HYP-076: surprise vs crowding (two-cell) ─────────────────────────────────

@dataclass(frozen=True)
class ReleaseEvent:
    pair: str
    publish_date: date      # 08:30 ET release — same-day t0 (no shift)
    side: int               # pair-space surprise direction
    crowded: bool           # True = crowded-AGAINST cell, False = uncrowded control


def events_076(releases: pd.DataFrame, cot_pct: dict[str, pd.Series],
               z_min: float = 1.5) -> list[ReleaseEvent]:
    """releases: columns publish_date, surprise_z, usd_sign (one row per release)."""
    df = releases.dropna(subset=["surprise_z", "usd_sign"]).copy()
    df["absz"] = df["surprise_z"].abs()
    df = df[df["absz"] >= z_min]
    # de-overlap: ONE event per publish_date = the max-|z| release that day
    df = df.sort_values("absz").groupby("publish_date", as_index=False).last()
    out = []
    for _, r in df.iterrows():
        d = r["publish_date"] if isinstance(r["publish_date"], date) else pd.Timestamp(r["publish_date"]).date()
        usd = float(np.sign(r["surprise_z"] * r["usd_sign"]))
        if usd == 0:
            continue
        for pair in OPT_PAIRS:
            side = int(usd * USD_SURPRISE_PAIR_SIGN[pair])
            pc = cot_pct[pair]
            prior = pc[pc.index <= pd.Timestamp(d)]
            if prior.empty or not np.isfinite(prior.iloc[-1]):
                continue
            crowd = _crowd_side(float(prior.iloc[-1])) * es.PAIR_FLIP.get(pair, 1)
            if crowd == -side:
                out.append(ReleaseEvent(pair, d, side, True))
            elif crowd == 0:
                out.append(ReleaseEvent(pair, d, side, False))
            # crowd == side (extreme-aligned): neither cell, per declared interpretation
    return out


def perm_p_two_cell(events: list[ReleaseEvent], closes: dict[str, pd.Series], h: int,
                    rng: np.random.Generator, n_perm: int = 10000) -> dict:
    """obs = median(crowded signed fwd ret) − median(control); label-shuffle null
    preserving per-pair crowded counts. One-sided (crowded > control)."""
    scored = []   # (pair, signed_ret, crowded)
    for e in events:
        c = closes[e.pair]
        t0 = es.effective_t0(e.publish_date, c.index)
        if t0 is None:
            continue
        r = es.forward_signed_return(c, t0, h, e.side)
        if r is not None:
            scored.append((e.pair, r, e.crowded))
    cr = [r for _, r, k in scored if k]
    co = [r for _, r, k in scored if not k]
    if len(cr) == 0 or len(co) == 0:
        return {"obs": None, "p": None, "n_crowded": len(cr), "n_control": len(co),
                "n_perm": n_perm, "note": "empty cell — statistic undefined"}
    obs = float(np.median(cr) - np.median(co))
    by_pair: dict[str, list[tuple[float, bool]]] = {}
    for pair, r, k in scored:
        by_pair.setdefault(pair, []).append((r, k))
    n_ge = 0
    for _ in range(n_perm):
        a, b = [], []
        for pair, rows in by_pair.items():
            rs = np.array([r for r, _ in rows])
            k = sum(1 for _, kk in rows if kk)
            pick = rng.choice(len(rs), size=k, replace=False)
            mask = np.zeros(len(rs), bool)
            mask[pick] = True
            a.extend(rs[mask]); b.extend(rs[~mask])
        if a and b and float(np.median(a) - np.median(b)) >= obs:
            n_ge += 1
    return {"obs": round(obs, 6), "p": (n_ge + 1) / (n_perm + 1),
            "n_crowded": len(cr), "n_control": len(co), "n_perm": n_perm,
            "median_crowded": round(float(np.median(cr)), 6),
            "median_control": round(float(np.median(co)), 6)}


# ── HYP-080: gdelt tone × positioning (two predicted cells, one pooled median) ─

def events_080(tone_z: dict[str, pd.Series], cot_pct: dict[str, pd.Series]) -> dict[str, list[es.Event]]:
    out = {}
    for pair in OPT_PAIRS:
        flip = es.PAIR_FLIP.get(pair, 1)
        z = tone_z[pair].dropna()
        pc = cot_pct[pair].reindex(z.index)
        cross = es.detect_crossings(
            [i.date() for i in z.index], list(z.values),
            enter=lambda v: 1 if v >= 1.5 else (-1 if v <= -1.5 else 0),
            rearm=lambda v: abs(v) < 0.75)
        evs = []
        for d, tone_side, v in cross:
            p = pc.get(pd.Timestamp(d))
            if p is None or not np.isfinite(p):
                continue
            crowd = _crowd_side(float(p), hi=0.80, lo=0.20)   # 080's own thresholds
            if crowd == 0:
                continue
            base_side = -tone_side if tone_side == crowd else tone_side  # aligned→fade, opposed→follow
            evs.append(es.Event(pair, shift_pub(d), base_side * flip, abs(v) - 1.5))
        out[pair] = evs
    return out


# ── HYP-077 full composite ───────────────────────────────────────────────────

def crowding_composite_full(cot: pd.DataFrame, rr_z: dict[str, pd.Series],
                            trades: pd.DataFrame, funded_pairs: list[str],
                            open_positions_on) -> tuple[list[date], list[float], list[dict]]:
    """Locked composite with BOTH terms: aligned cot pct + Φ(rr25_z) aligned."""
    from scipy.stats import norm
    dates_, values, mapping = [], [], []
    weekly = cot.pivot_table(index="publish_date", columns="pair", values="net_pct_1y")
    for d, row in weekly.sort_index().iterrows():
        open_pos = open_positions_on(trades, d)
        comps, used = [], {}
        for pair in funded_pairs:
            v = row.get(pair)
            if pair not in open_pos or v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            long_base = open_pos[pair] == 1
            aligned_cot = float(v) if long_base else 1.0 - float(v)
            terms = [aligned_cot]
            info = {"direction": open_pos[pair], "net_pct_1y": float(v), "aligned_cot": aligned_cot}
            zs = rr_z.get(pair)
            if zs is not None:
                prior = zs[zs.index <= pd.Timestamp(d)].dropna()
                if len(prior):
                    phi = float(norm.cdf(float(prior.iloc[-1])))
                    aligned_rr = phi if long_base else 1.0 - phi
                    terms.append(aligned_rr)
                    info.update({"rr25_z": round(float(prior.iloc[-1]), 4), "aligned_rr": aligned_rr})
            comps.append(float(np.mean(terms)))
            used[pair] = info
        if comps:
            dates_.append(d if isinstance(d, date) else pd.Timestamp(d).date())
            values.append(float(np.mean(comps)))
            mapping.append({"date": str(d)[:10], "pairs": used})
    return dates_, values, mapping
