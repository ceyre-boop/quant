"""sovereign/sentiment/geometry_feed.py — trailing price-geometry features (G2).

Substrate for the LOCKED HYP-082/083/084 hypotheses (specs hash-locked under
data/research/preregister/ + the GEOMETRY-2026-07 manifest, commit 01cacbd — this module may
not alter any spec'd constant). Three independent, trailing-only reads of the daily OHLC cache:

  - corridor_stats     — trailing log-price linregress: how "locked" price is in a linear
                          corridor (R²) and how many residual-sigmas the current close sits
                          off that corridor's fitted line (signed deviation).
  - detect_fvgs_daily  — REPLICATED 3-bar Fair Value Gap kernel (see ISOLATION below): how many
                          gaps opened in the trailing window and how many remain unfilled.
  - tri_state_stats    — is the daily H-L range actively contracting (a consolidation/wedge
                          read): a shrinking trailing-window range AND a current range small
                          relative to its own trailing 252-day history.

ISOLATION (bidirectional, AST-tested by tests/test_sentiment_board.py::test_sentiment_does_not_
import_ict): this module MUST NOT import ict/ or ict-engine/. The FVG kernel is therefore
REPLICATED here rather than imported from ict.fvg_detector.FVGDetector — which, independent of
the wall, has a real bug for historical-as-of use anyway: it computes ATR over the *whole* frame
passed to it, so unless the caller pre-truncates to df.iloc[:t+1] its ATR silently leaks bars
after t. tests/test_sentiment_geometry.py pins this replica to the ict canon on truncated frames
(legal there — a test file, not this module, importing both sides).

TRAILING-ONLY / WARMUP: every feature is computed from df.iloc[:t+1] alone — never a value at or
after t. Each helper independently reports "not yet computable" (None / NaN) until ITS OWN
window is fully populated (120 bars for corridor, a full fvg_max_age-bar lookback for the FVG
count, 252 bars for tri-state's percentile leg) — warmup rows are NULL, never dropped and never
backfilled with a partial-window value, so the board's ASOF join can't carry a stale/fabricated
early read forward across a gap.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config.loader import params
from sovereign.sentiment.store import connect, upsert

ROOT = Path(__file__).resolve().parents[2]
SPOT_CACHE = ROOT / "data" / "research" / "positioning_family" / "spot_cache"

# Trailing lookback for the tri-state percentile leg ("trailing 252d ranges" — the plan's fixed
# prose constant, not exposed as a config knob; only tri_window/tri_pctile are tunable).
_TRI_PCTILE_WINDOW = 252
# Below this many bars the FVG kernel cannot even attempt its first 3-bar triplet.
_FVG_MIN_BARS = 5
# Near-zero guard for the corridor residual std (avoids a NaN/inf dev_sigma on a synthetic,
# effectively-noiseless fit — real market residuals never sit this close to zero).
_RESID_STD_EPS = 1e-12


def _none_if_nan(x: float) -> float | None:
    return None if x is None or x != x else x  # `x != x` is NaN-safe for float/np.float64 alike


def corridor_stats(close: pd.Series, window: int) -> tuple[float, float]:
    """Trailing log-price linregress over the *window* bars ENDING at the last element of *close*.

    r2 = R² of the log-price-vs-time fit ("corridor integrity" — math adapted from
    research/validate_corridor_feature.py:16-29; the threshold logic there is NOT reused, only
    the r_value**2 math). dev_sigma = (log p_t - fit_t) / std(residuals) — how many residual-
    sigmas the last point sits off the fitted line, signed (positive = above the corridor).

    Returns (nan, nan) when *close* has fewer than *window* observations (warmup) or contains a
    non-positive price (log undefined) — never a value computed on a short/invalid window.
    """
    close = pd.Series(close)
    if len(close) < window or window < 2:
        return (float("nan"), float("nan"))
    seg = close.iloc[-window:]
    vals = seg.to_numpy(dtype=float)
    if np.any(vals <= 0) or np.any(np.isnan(vals)):
        return (float("nan"), float("nan"))

    t_idx = np.arange(window, dtype=float)
    log_p = np.log(vals)
    slope, intercept, r_value, _p_value, _std_err = stats.linregress(t_idx, log_p)
    fit = intercept + slope * t_idx
    resid = log_p - fit
    resid_std = float(np.std(resid, ddof=1)) if window > 2 else 0.0

    r2 = float(r_value ** 2)
    if resid_std < _RESID_STD_EPS:
        dev_sigma = 0.0
    else:
        dev_sigma = float((log_p[-1] - fit[-1]) / resid_std)
    return (r2, dev_sigma)


def _true_range(d: pd.DataFrame) -> pd.Series:
    h, l, c = d["High"], d["Low"], d["Close"]
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


def _compute_atr(d: pd.DataFrame, period: int = 14) -> float:
    """REPLICATED from ict/_atr_utils.py::compute_atr (isolation wall forbids importing ict/).

    Kept byte-for-byte equivalent (same formula, same NaN/zero fallback) so the FVG parity test
    can pin this module's output to the ict canon exactly. Any drift here must be re-verified
    against that test.
    """
    tr = _true_range(d)
    val = float(tr.rolling(period).mean().iloc[-1])
    if val > 0 and val == val:  # val == val is False only for NaN
        return val
    last_close = float(d["Close"].iloc[-1])
    return max(0.0001 * last_close, 1e-9)


def _require_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ("Open", "High", "Low", "Close") if c not in df.columns]
    if missing:
        raise ValueError(f"geometry_feed requires OHLC columns, missing {missing}")
    return df


def detect_fvgs_daily(df: pd.DataFrame, max_age: int, min_atr_frac: float) -> tuple[int, int]:
    """REPLICATED 3-bar Fair Value Gap kernel (ict.fvg_detector.FVGDetector._find_fvgs, restated
    here — see module docstring ISOLATION). Operates on *df* as given: the caller passes
    df.iloc[:t+1] so "now" (as_of) is the last row and the ATR trails to exactly t, never leaking
    a bar after t (the bug in calling FVGDetector.detect on an un-truncated frame).

    Bullish gap: c1.High < c3.Low over consecutive bars (c1, c2, c3). Bearish: c1.Low > c3.High.
    A gap counts only when its size >= min_atr_frac * ATR(14) (trailing, ending at the last row).
    "Unfilled" mirrors FVGDetector's own fill test: bullish filled when price_now < bottom,
    bearish filled when price_now > top.

    Returns (0, 0) below _FVG_MIN_BARS (nothing CAN have formed yet — a true zero, not a warmup
    NULL) and (None, None) until a FULL, uncapped max_age-bar lookback exists (n >= max_age + 2:
    the same floor-until-fully-populated discipline as every other geometry feature here — a
    partial lookback would silently mean something different from a later full-window count).
    """
    d = _require_ohlc(df)
    n = len(d)
    if n < _FVG_MIN_BARS:
        return (0, 0)
    if n < max_age + 2:
        return (None, None)

    as_of = n - 1
    price_now = float(d["Close"].iloc[as_of])
    atr = _compute_atr(d, period=14)
    lookback = min(max_age, as_of - 1)
    start = as_of - lookback

    count = 0
    unfilled = 0
    for i in range(start, as_of - 1):
        c1 = d.iloc[i]
        c3 = d.iloc[i + 2]
        if c1["High"] < c3["Low"]:
            top, bottom = float(c3["Low"]), float(c1["High"])
            if (top - bottom) >= min_atr_frac * atr:
                count += 1
                filled = price_now < bottom          # canon's own bullish fill test
                if not filled:
                    unfilled += 1
        if c1["Low"] > c3["High"]:
            top, bottom = float(c1["Low"]), float(c3["High"])
            if (top - bottom) >= min_atr_frac * atr:
                count += 1
                filled = price_now > top             # canon's own bearish fill test
                if not filled:
                    unfilled += 1
    return (count, unfilled)


def tri_state_stats(df: pd.DataFrame, window: int, pctile: float) -> tuple[bool | None, int | None, float]:
    """Contraction/consolidation read on the daily H-L range, over df.iloc[:t+1] (last row = t).

    state = True iff BOTH hold at t: (a) the trailing *window*-bar linregress slope of the H-L
    range series is < 0 (the range has been shrinking), AND (b) today's range sits below the
    *pctile* quantile of the trailing 252-bar range history (today's range is unusually small).
    days_in_consolidation = the current consecutive run of state==True ending at t (walked
    backward only as far as both windows stay fully populated — never fabricated past that
    point). range_slope = the trailing *window*-bar slope at t (defined whenever state is).

    Returns (None, None, nan) until n >= max(window, 252) — the tri-state read needs a full
    252-bar percentile history before it means anything; a partial-window percentile would be
    silently non-comparable to a later full-window one.
    """
    d = _require_ohlc(df)
    rng = (d["High"] - d["Low"]).to_numpy(dtype=float)
    n = len(rng)
    floor = max(window, _TRI_PCTILE_WINDOW)
    if n < floor or window < 2:
        return (None, None, float("nan"))

    t_idx = np.arange(window, dtype=float)
    last = n - 1
    state_now: bool | None = None
    slope_now = float("nan")
    days = 0
    i = last
    while i >= floor - 1:
        w = rng[i - window + 1: i + 1]
        slope, *_rest = stats.linregress(t_idx, w)
        thresh = float(np.quantile(rng[i - _TRI_PCTILE_WINDOW + 1: i + 1], pctile))
        state_i = bool(slope < 0 and rng[i] < thresh)
        if i == last:
            state_now = state_i
            slope_now = float(slope)
        if not state_i:
            break
        days += 1
        i -= 1
    return (state_now, days, slope_now)


def compute_pair(df: pd.DataFrame, pair: str, cfg: dict) -> pd.DataFrame:
    """Compute the 7 geometry features for every row of *df* (one row per trading day).

    *df* must be sorted ascending with a date-like index and Open/High/Low/Close columns (the
    spot_cache parquet contract). Every feature at row t is computed from df.iloc[:t+1] ONLY —
    a literal slice per row, so truncation-invariance and no-look-ahead hold by construction.
    Warmup rows carry NULL (pd.NA / NaN, per column dtype below) for every column that isn't yet
    computable (never dropped, never a fabricated/partial-window stand-in) — and render
    identically regardless of what other rows share the frame (explicit dtypes below; a plain
    DataFrame(list-of-dicts) would otherwise infer per-column dtype from ALL rows at once and let
    the SAME warmup value print as Python None in one call and float NaN in another). Output
    columns match sentiment_geometry_daily minus
    fetched_at (stamped by update()): date, pair, corridor_r2, corridor_dev, corridor_window,
    fvg_count_20d, fvg_unfilled, tri_state, days_in_consolidation, range_slope, src_last_bar_date.
    """
    d = _require_ohlc(df).sort_index()
    corridor_window = int(cfg["corridor_window"])
    fvg_max_age = int(cfg["fvg_max_age"])
    fvg_min_atr_frac = float(cfg["fvg_min_atr_frac"])
    tri_window = int(cfg["tri_window"])
    tri_pctile = float(cfg["tri_pctile"])

    close_all = d["Close"]
    rows = []
    for t in range(len(d)):
        sub = d.iloc[: t + 1]
        close_sub = close_all.iloc[: t + 1]

        r2, dev = corridor_stats(close_sub, corridor_window)
        count_20d, unfilled = detect_fvgs_daily(sub, fvg_max_age, fvg_min_atr_frac)
        state, days, slope = tri_state_stats(sub, tri_window, tri_pctile)

        row_date = pd.Timestamp(sub.index[-1]).date()
        rows.append({
            "date": row_date,
            "pair": pair,
            "corridor_r2": _none_if_nan(r2),
            "corridor_dev": _none_if_nan(dev),
            "corridor_window": corridor_window,
            "fvg_count_20d": count_20d,
            "fvg_unfilled": unfilled,
            "tri_state": None if state is None else int(state),
            "days_in_consolidation": days,
            "range_slope": _none_if_nan(slope),
            "src_last_bar_date": row_date,
        })

    out = pd.DataFrame(rows, columns=[
        "date", "pair", "corridor_r2", "corridor_dev", "corridor_window", "fvg_count_20d",
        "fvg_unfilled", "tri_state", "days_in_consolidation", "range_slope", "src_last_bar_date",
    ])
    # Explicit dtypes so a warmup row renders identically (pd.NA / float NaN) whether it's the
    # only row in the frame or sits alongside later fully-computed rows — pandas' DataFrame(list
    # of dicts) otherwise infers per-column dtype from ALL rows at once, so the SAME warmup value
    # would silently render as Python None here and float NaN there depending on what else is in
    # the frame, which would break truncation-invariance at the byte-representation level even
    # though the underlying NULL-ness is identical. Nullable "Int64" also keeps integer columns
    # as clean integers instead of accidentally upcasting to float (e.g. "4.0") whenever the
    # column contains any NA.
    for col in ("corridor_r2", "corridor_dev", "range_slope"):
        out[col] = out[col].astype("float64")
    for col in ("corridor_window", "fvg_count_20d", "fvg_unfilled", "tri_state", "days_in_consolidation"):
        out[col] = out[col].astype("Int64")
    start = cfg.get("start")
    if start:
        out = out[out["date"] >= pd.Timestamp(start).date()].reset_index(drop=True)
    return out


def update(con=None, pairs: list[str] | None = None, start: str | None = None) -> dict:
    """Compute geometry features for each pair's cached daily OHLC parquet and upsert.

    Reads ONLY the local spot_cache parquet (SPOT_CACHE/{PAIR}_ohlc.parquet) — never fetches
    over the network (feeder idiom: sovereign/sentiment/vix_feed.py). A missing parquet is
    skipped LOUDLY (printed + counted in the returned coverage dict), matching the plan's
    data-source contract exactly. pairs defaults to the board's pair universe (config
    sentiment.news.pairs, the same source board_state.PAIRS reads — not imported directly here
    to avoid a needless cross-module coupling for a one-line list).
    """
    cfg = dict(params["sentiment"]["geometry"])
    pairs = pairs or list(params["sentiment"]["news"]["pairs"].keys())
    cfg["start"] = start or cfg.get("start", "2015-01-01")
    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    coverage: dict = {}
    try:
        for pair in pairs:
            path = SPOT_CACHE / f"{pair}_ohlc.parquet"
            if not path.exists():
                print(f"  [geometry] {pair}: MISSING parquet at {path} — SKIPPED (never fetched inline)")
                coverage[pair] = {"rows": 0, "note": "missing parquet"}
                continue
            raw = pd.read_parquet(path)
            out = compute_pair(raw, pair, cfg)
            if out.empty:
                coverage[pair] = {"rows": 0, "note": "no rows on/after start"}
                continue
            out = out.assign(fetched_at=now)
            upsert(con, "sentiment_geometry_daily", out, [
                "date", "pair", "corridor_r2", "corridor_dev", "corridor_window", "fvg_count_20d",
                "fvg_unfilled", "tri_state", "days_in_consolidation", "range_slope",
                "src_last_bar_date", "fetched_at",
            ])
            coverage[pair] = {"rows": int(len(out)), "start": str(out["date"].min()), "end": str(out["date"].max())}
    finally:
        if own:
            con.close()
    return coverage
