#!/usr/bin/env python3
"""
intelligence_run.py — host-side harness for the nightly Oracle claim tests.

WHY THIS EXISTS
---------------
The `oracle-market-intelligence` scheduled task is supposed to test one piece of
trading folklore per night against real data. It has resolved ZERO claims since it
was created, because its sandbox mounts only the Obsidian vault: no repo, no cached
bars. It wrote a rigorous pre-registration for Claim 1 and then had nothing to
measure.

This script is the missing half. It runs ON THE HOST, where `data/cache/` is
reachable, computes the statistic, and stages the result into the vault at
`Trading/System/Mirror/intelligence/`. The scheduled task then reads that staged
result and writes it up. Compute here, narrate there.

DISCIPLINE
----------
- The pre-registration in `Trading/Research/Oracle-Intelligence-Log.md` BINDS this
  code. Definitions are transcribed from it, not invented here, and any deviation
  is recorded in the output under `deviations` rather than silently applied.
- No result is written without an honest `n`. Below the pre-registered minimum the
  verdict is INCONCLUSIVE regardless of how good the point estimate looks.
- If data is missing the script says so and exits non-zero. It never substitutes a
  looser source to produce a number.

Usage:
    python3 scripts/intelligence_run.py --claim 1
    python3 scripts/intelligence_run.py --claim 1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
VAULT = Path(os.environ.get("ALTA_VAULT", "/Users/taboost/Obsidian/Obsidian"))
OUT_DIR = VAULT / "Trading" / "System" / "Mirror" / "intelligence"
UNIVERSE = REPO / "data" / "cache" / "daily_universe"

NOW = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Claim 1 pre-registration — transcribed from Oracle-Intelligence-Log.md.
# Do not edit these to fit a result. Changing them invalidates the test.
# ---------------------------------------------------------------------------
PREREG_CLAIM_1 = {
    "claim": "Fair value gaps always get filled",
    "fvg_definition": (
        "three-candle pattern where candle 1's high < candle 3's low (bullish) or "
        "candle 1's low > candle 3's high (bearish); gap zone is the interval between "
        "those two levels"
    ),
    "fill_definition": (
        "price trades to the MIDPOINT of the gap zone (not first touch of the near "
        "edge) — matches the live entry model at ict/pipeline.py:378"
    ),
    "horizons_sessions": [1, 5, 20],
    "H0": "fill rate <= 50% at each horizon",
    "H1": "fill rate > 50%",
    "test": "one-sided binomial per horizon",
    "correction": "Bonferroni across 3 horizons",
    "alpha": 0.05 / 3,
    "min_n": 100,
    "precommitted_caveat": (
        "the 20-session horizon is mechanically easiest to confirm — over a long "
        "enough window almost any level is revisited by drift alone. The 1-session "
        "number is the one that matters for the trading system."
    ),
    "precommitted_trap": (
        "any FVG formed within 20 sessions of the data end date must be EXCLUDED, "
        "not counted as unfilled"
    ),
}

# Matches ict/fvg_detector.py's live configuration so results are comparable to the
# live detector rather than to a fresh reimplementation. READ FROM THE DETECTOR, not
# guessed: the first version of this file assumed 0.10 and was wrong — the live
# value is 0.30, which is a materially stricter size filter. Verify with:
#   python3 -c "from ict.fvg_detector import FVGDetector; print(FVGDetector()._fvg_min_atr)"
FVG_MIN_ATR = 0.30
ATR_PERIOD = 14

# Order-block parameters, likewise read from the live detector.
OB_IMPULSE_ATR = 1.5
OB_LOOKBACK = 20


@dataclass
class Formation:
    ticker: str
    kind: str
    formed_idx: int
    formed_date: str
    top: float
    bottom: float
    midpoint: float
    size_atr_ratio: float


def atr_series(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Point-in-time ATR. The live detector computes one ATR for the frame at call
    time; a historical sweep must use a trailing value at each bar or it leaks
    future information into the size filter. Recorded as a deviation."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def find_formations(df: pd.DataFrame, ticker: str) -> list[Formation]:
    """Enumerate every FVG formation, using the detector's own three-candle rule
    and its FVG_MIN_ATR size filter."""
    out: list[Formation] = []
    atr = atr_series(df)
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    a = atr.to_numpy()
    dates = df["date"].astype(str).to_numpy()

    for i in range(len(df) - 2):
        cur_atr = a[i + 2]
        if not np.isfinite(cur_atr) or cur_atr <= 0:
            continue
        floor = FVG_MIN_ATR * cur_atr

        if h[i] < lo[i + 2]:
            top, bottom = float(lo[i + 2]), float(h[i])
            size = top - bottom
            if size >= floor:
                out.append(Formation(ticker, "BULLISH", i + 2, dates[i + 2], top,
                                     bottom, (top + bottom) / 2, size / cur_atr))
        if lo[i] > h[i + 2]:
            top, bottom = float(lo[i]), float(h[i + 2])
            size = top - bottom
            if size >= floor:
                out.append(Formation(ticker, "BEARISH", i + 2, dates[i + 2], top,
                                     bottom, (top + bottom) / 2, size / cur_atr))
    return out


def filled_within(df: pd.DataFrame, f: Formation, horizon: int) -> bool | None:
    """Did price trade to the gap midpoint within `horizon` sessions after
    formation? Returns None if the window extends past the data end — those are
    EXCLUDED per the pre-registered trap, never counted as unfilled."""
    start = f.formed_idx + 1
    end = f.formed_idx + horizon
    if end >= len(df):
        return None
    window = df.iloc[start:end + 1]
    if f.kind == "BULLISH":
        return bool((window["low"] <= f.midpoint).any())
    return bool((window["high"] >= f.midpoint).any())


EXACT_BINOM_MAX_N = 5000


def binom_sf(k: int, n: int, p: float = 0.5) -> tuple[float, str]:
    """One-sided P(X >= k) under Binomial(n, p).

    Exact summation for n <= EXACT_BINOM_MAX_N. Above that the exact terms
    overflow a float, so a normal approximation with continuity correction is
    used — valid here because np and n(1-p) are both in the tens of thousands,
    far beyond the usual >5 rule of thumb. The method used is returned alongside
    the value so the output never hides which one produced it.
    """
    if n == 0:
        return float("nan"), "none"
    if n <= EXACT_BINOM_MAX_N:
        # Log-space. math.comb(n, i) for n in the thousands overflows a float long
        # before the product with the tiny probability terms brings it back down,
        # so the naive form raises OverflowError. Each exponentiated term here is
        # a probability in [0, 1] and sums safely.
        lgam = math.lgamma
        log_p, log_q = math.log(p), math.log1p(-p)
        total = 0.0
        for i in range(k, n + 1):
            log_c = lgam(n + 1) - lgam(i + 1) - lgam(n - i + 1)
            total += math.exp(log_c + i * log_p + (n - i) * log_q)
        return min(1.0, total), "exact"
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p))
    z = (k - 0.5 - mu) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2)), "normal_approx_cc"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_claim_1(verbose: bool = True) -> dict:
    if not UNIVERSE.exists():
        raise SystemExit(f"FATAL: universe not found at {UNIVERSE}. Nothing to measure.")
    files = sorted(UNIVERSE.glob("*.parquet"))
    if not files:
        raise SystemExit(f"FATAL: no parquet files in {UNIVERSE}. Nothing to measure.")

    horizons = PREREG_CLAIM_1["horizons_sessions"]
    counts = {h: {"filled": 0, "n": 0} for h in horizons}
    by_kind = {"BULLISH": 0, "BEARISH": 0}
    excluded_truncated = 0
    tickers_used = 0
    bars_scanned = 0
    skipped: list[str] = []

    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{fp.name}: unreadable ({exc})")
            continue
        need = {"date", "open", "high", "low", "close"}
        if not need.issubset(df.columns):
            skipped.append(f"{fp.name}: missing columns {sorted(need - set(df.columns))}")
            continue
        df = df.reset_index(drop=True)
        if len(df) < ATR_PERIOD + 25:
            skipped.append(f"{fp.name}: only {len(df)} bars")
            continue

        tickers_used += 1
        bars_scanned += len(df)
        ticker = fp.stem
        for f in find_formations(df, ticker):
            by_kind[f.kind] += 1
            for h in horizons:
                res = filled_within(df, f, h)
                if res is None:
                    if h == max(horizons):
                        excluded_truncated += 1
                    continue
                counts[h]["n"] += 1
                counts[h]["filled"] += int(res)

    results = {}
    alpha = PREREG_CLAIM_1["alpha"]
    for h in horizons:
        k, n = counts[h]["filled"], counts[h]["n"]
        rate = k / n if n else float("nan")
        p, p_method = binom_sf(k, n) if n else (float("nan"), "none")
        ci = wilson_ci(k, n)
        if n < PREREG_CLAIM_1["min_n"]:
            verdict = "INCONCLUSIVE"
            why = f"n={n} below pre-registered minimum of {PREREG_CLAIM_1['min_n']}"
        elif p < alpha:
            verdict = "CONFIRMED"
            why = f"p={p:.2e} < Bonferroni alpha {alpha:.4f}"
        else:
            verdict = "NOT_SIGNIFICANT"
            why = f"p={p:.4f} >= Bonferroni alpha {alpha:.4f}"
        results[f"{h}_session"] = {
            "horizon_sessions": h, "n": n, "filled": k,
            "fill_rate": round(rate, 4) if n else None,
            "wilson_95ci": [round(ci[0], 4), round(ci[1], 4)] if n else None,
            "p_value_one_sided": p, "p_method": p_method,
            "verdict": verdict, "reason": why,
        }
        if verbose:
            print(f"  h={h:>2}  n={n:>7}  filled={k:>7}  rate={rate:.4f}  "
                  f"p={p:.3e}  {verdict}")

    headline = results["1_session"]
    return {
        "claim_number": 1,
        "claim": PREREG_CLAIM_1["claim"],
        "generated_at": NOW.isoformat(),
        "status": "COMPLETE",
        "prereg": PREREG_CLAIM_1,
        "data": {
            "source": str(UNIVERSE.relative_to(REPO)),
            "tickers_used": tickers_used,
            "tickers_skipped": len(skipped),
            "skipped_detail": skipped[:20],
            "daily_bars_scanned": bars_scanned,
            "formations_bullish": by_kind["BULLISH"],
            "formations_bearish": by_kind["BEARISH"],
            "excluded_truncated_at_20s": excluded_truncated,
        },
        "results": results,
        "headline_verdict": headline["verdict"],
        "headline_reason": (
            f"1-session fill rate {headline['fill_rate']} on n={headline['n']} — "
            "this is the horizon that matters for the trading system, per the "
            "pre-committed caveat."
        ),
        "deviations": [
            "ATR for the size filter is computed point-in-time (trailing 14-bar) "
            "rather than once per frame as ict/fvg_detector.py does at call time. "
            "A frame-level ATR would leak future information into a historical "
            "sweep. Strictly more conservative; recorded rather than applied "
            "silently.",
            "Detection reimplements the detector's three-candle rule and its "
            "FVG_MIN_ATR=0.10 size filter rather than calling FVGDetector.detect(), "
            "because detect() returns only gaps unfilled as-of a single point and "
            "cannot enumerate historical formations. The formation rule is "
            "transcribed line-for-line from _find_fvgs.",
            "Universe is the cached daily equity universe, not the gapper event "
            "universe. Stated so the population under test is not misread.",
        ],
        "intelligence_log_line": {
            "date": NOW.strftime("%Y-%m-%d"),
            "claim": PREREG_CLAIM_1["claim"],
            "verdict": headline["verdict"],
            "p": headline["p_value_one_sided"],
            "n": headline["n"],
            "implication": "",
        },
    }


# ---------------------------------------------------------------------------
# Claims 7 and 10.
#
# INTEGRITY NOTE — these differ from Claim 1 in one important way. Claim 1's
# pre-registration was written on 2026-07-20, before any data was reachable, and
# this harness only transcribed it. Claims 7 and 10 have their pre-registrations
# written HERE, in the same pass that first ran them. That is weaker. It is
# disclosed in every result rather than glossed, and the registrations below are
# now frozen for all future runs — a re-run cannot redefine them after seeing
# these numbers.
# ---------------------------------------------------------------------------
PREREG_CLAIM_7 = {
    "claim": "The gap fill is the trade — gaps >5% fill on day 1, 2, 5",
    "population": "daily bars where |open / prior_close - 1| > 0.05",
    "fill_definition": "price trades back to the prior session's close",
    "horizons_sessions": [1, 2, 5],
    "H0": "fill rate <= 50% at each horizon",
    "H1": "fill rate > 50%",
    "test": "one-sided binomial per horizon",
    "correction": "Bonferroni across 3 horizons",
    "alpha": 0.05 / 3,
    "min_n": 100,
    "precommitted_caveat": (
        "gap-fill studies are survivorship-prone in the other direction from FVGs: "
        "a gap that never fills stays open forever, so any window truncated by the "
        "data end must be excluded, not counted unfilled."
    ),
    "prereg_written": "2026-07-21, same pass as first execution — see integrity note",
}

PREREG_CLAIM_10 = {
    "claim": "Session highs and lows are magnets — price returns to test the prior "
             "session high/low within 3 sessions more than 50% of the time",
    "population": "every daily bar with a valid prior session",
    "fill_definition": "a subsequent session's high >= prior high (for the high "
                       "test) or low <= prior low (for the low test)",
    "horizons_sessions": [1, 3, 5],
    "H0": "retest rate <= 50% at each horizon",
    "H1": "retest rate > 50%",
    "test": "one-sided binomial per horizon",
    "correction": "Bonferroni across 3 horizons",
    "alpha": 0.05 / 3,
    "min_n": 100,
    "precommitted_caveat": (
        "this claim is close to trivially true for a random walk — over any window "
        "price frequently exceeds a recent extreme. A CONFIRMED verdict here is "
        "weak evidence of a tradable magnet effect and should be read as a base "
        "rate, not an edge. Stated before the numbers were seen."
    ),
    "prereg_written": "2026-07-21, same pass as first execution — see integrity note",
}


def _iter_universe():
    files = sorted(UNIVERSE.glob("*.parquet"))
    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception:  # noqa: BLE001
            continue
        need = {"date", "open", "high", "low", "close"}
        if not need.issubset(df.columns) or len(df) < 40:
            continue
        yield fp.stem, df.reset_index(drop=True)


def _score(counts: dict, prereg: dict) -> dict:
    out = {}
    alpha = prereg["alpha"]
    for h, c in counts.items():
        k, n = c["hit"], c["n"]
        rate = k / n if n else float("nan")
        p, method = binom_sf(k, n) if n else (float("nan"), "none")
        ci = wilson_ci(k, n)
        if n < prereg["min_n"]:
            verdict, why = "INCONCLUSIVE", f"n={n} below minimum {prereg['min_n']}"
        elif p < alpha:
            verdict, why = "CONFIRMED", f"p={p:.2e} < alpha {alpha:.4f}"
        else:
            verdict, why = "NOT_SIGNIFICANT", f"p={p:.4f} >= alpha {alpha:.4f}"
        out[f"{h}_session"] = {
            "horizon_sessions": h, "n": n, "hit": k,
            "rate": round(rate, 4) if n else None,
            "wilson_95ci": [round(ci[0], 4), round(ci[1], 4)] if n else None,
            "p_value_one_sided": p, "p_method": method,
            "verdict": verdict, "reason": why,
        }
        print(f"  h={h:>2}  n={n:>7}  hit={k:>7}  rate={rate:.4f}  {verdict}")
    return out


def run_claim_7() -> dict:
    horizons = PREREG_CLAIM_7["horizons_sessions"]
    counts = {h: {"hit": 0, "n": 0} for h in horizons}
    gaps_up = gaps_down = excluded = 0
    tickers = 0

    for ticker, df in _iter_universe():
        tickers += 1
        o = df["open"].to_numpy(); h = df["high"].to_numpy()
        lo = df["low"].to_numpy(); c = df["close"].to_numpy()
        for i in range(1, len(df)):
            prev_close = c[i - 1]
            if prev_close <= 0:
                continue
            gap = o[i] / prev_close - 1.0
            if abs(gap) <= 0.05:
                continue
            up = gap > 0
            gaps_up += int(up); gaps_down += int(not up)
            for hz in horizons:
                end = i + hz - 1
                if end >= len(df):
                    if hz == max(horizons):
                        excluded += 1
                    continue
                win_lo = lo[i:end + 1].min(); win_hi = h[i:end + 1].max()
                hit = win_lo <= prev_close if up else win_hi >= prev_close
                counts[hz]["n"] += 1
                counts[hz]["hit"] += int(hit)

    print("Claim 7 — gaps >5% fill to prior close:")
    results = _score(counts, PREREG_CLAIM_7)
    head = results["1_session"]
    return {
        "claim_number": 7, "claim": PREREG_CLAIM_7["claim"],
        "generated_at": NOW.isoformat(), "status": "COMPLETE",
        "prereg": PREREG_CLAIM_7,
        "data": {"source": str(UNIVERSE.relative_to(REPO)), "tickers_used": tickers,
                 "gaps_up": gaps_up, "gaps_down": gaps_down,
                 "excluded_truncated": excluded},
        "results": results,
        "headline_verdict": head["verdict"],
        "headline_reason": f"day-1 fill rate {head['rate']} on n={head['n']}",
        "deviations": [
            "Gap measured open-vs-prior-close on daily bars from the cached equity "
            "universe; this is not the gapper event universe and the population "
            "will differ from the live gapper program.",
            "Pre-registration written in the same pass as first execution — weaker "
            "than Claim 1. Frozen from now on.",
        ],
    }


def run_claim_10() -> dict:
    horizons = PREREG_CLAIM_10["horizons_sessions"]
    counts_h = {h: {"hit": 0, "n": 0} for h in horizons}
    counts_l = {h: {"hit": 0, "n": 0} for h in horizons}
    tickers = 0; excluded = 0

    for ticker, df in _iter_universe():
        tickers += 1
        hi = df["high"].to_numpy(); lo = df["low"].to_numpy()
        for i in range(1, len(df)):
            ph, pl = hi[i - 1], lo[i - 1]
            for hz in horizons:
                end = i + hz - 1
                if end >= len(df):
                    if hz == max(horizons):
                        excluded += 1
                    continue
                counts_h[hz]["n"] += 1
                counts_h[hz]["hit"] += int(hi[i:end + 1].max() >= ph)
                counts_l[hz]["n"] += 1
                counts_l[hz]["hit"] += int(lo[i:end + 1].min() <= pl)

    print("Claim 10 — prior session HIGH retested:")
    res_h = _score(counts_h, PREREG_CLAIM_10)
    print("Claim 10 — prior session LOW retested:")
    res_l = _score(counts_l, PREREG_CLAIM_10)
    head = res_h["3_session"]
    return {
        "claim_number": 10, "claim": PREREG_CLAIM_10["claim"],
        "generated_at": NOW.isoformat(), "status": "COMPLETE",
        "prereg": PREREG_CLAIM_10,
        "data": {"source": str(UNIVERSE.relative_to(REPO)), "tickers_used": tickers,
                 "excluded_truncated": excluded},
        "results": {"prior_high": res_h, "prior_low": res_l},
        "headline_verdict": head["verdict"],
        "headline_reason": (
            f"prior-high retest within 3 sessions: {head['rate']} on n={head['n']}. "
            "Read as a base rate, not an edge — see the pre-committed caveat."
        ),
        "deviations": [
            "Bonferroni applied across 3 horizons as pre-registered, NOT across the "
            "6 tests actually run (high and low separately). Stated rather than "
            "quietly widened; the correct correction for 6 tests would be stricter.",
            "Pre-registration written in the same pass as first execution.",
        ],
    }


PREREG_CLAIM_3 = {
    "claim": "Monday opens are a trap — Monday gap direction is more likely to "
             "reverse intraday than Tuesday–Friday",
    "population": "daily bars with a non-zero gap vs prior close, |gap| >= 0.5% "
                  "to exclude noise-level opens",
    "reversal_definition": "gap up and close < open, or gap down and close > open "
                           "— the open's direction is given back by the close",
    "H0": "Monday reversal rate <= Tue-Fri reversal rate",
    "H1": "Monday reversal rate > Tue-Fri reversal rate",
    "test": "two-proportion z-test, one-sided",
    "alpha": 0.05,
    "min_n": 100,
    "precommitted_caveat": (
        "a daily-bar proxy cannot see the intraday path. 'Reverses intraday' is "
        "approximated as close-vs-open, which misses gaps that fade and recover. "
        "A NOT_SIGNIFICANT result here does not fully clear the claim; a CONFIRMED "
        "one would still need intraday confirmation before being traded."
    ),
    "precommitted_trap": (
        "Monday has fewer observations than the other four days combined, so the "
        "comparison is inherently unbalanced. Report both n values, never just the "
        "rates."
    ),
    "prereg_written": "2026-07-21, same pass as first execution — see integrity note",
}


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """One-sided z-test for p1 > p2. Returns (z, p)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    return z, 0.5 * math.erfc(z / math.sqrt(2))


def run_claim_3() -> dict:
    mon = {"rev": 0, "n": 0}
    rest = {"rev": 0, "n": 0}
    tickers = 0

    for ticker, df in _iter_universe():
        tickers += 1
        d = pd.to_datetime(df["date"])
        dow = d.dt.dayofweek.to_numpy()  # Monday == 0
        o = df["open"].to_numpy(); c = df["close"].to_numpy()
        for i in range(1, len(df)):
            prev_close = c[i - 1]
            if prev_close <= 0:
                continue
            gap = o[i] / prev_close - 1.0
            if abs(gap) < 0.005:
                continue
            reversed_ = (c[i] < o[i]) if gap > 0 else (c[i] > o[i])
            bucket = mon if dow[i] == 0 else rest
            bucket["n"] += 1
            bucket["rev"] += int(reversed_)

    rate_m = mon["rev"] / mon["n"] if mon["n"] else float("nan")
    rate_r = rest["rev"] / rest["n"] if rest["n"] else float("nan")
    z, p = two_prop_z(mon["rev"], mon["n"], rest["rev"], rest["n"])
    ci_m, ci_r = wilson_ci(mon["rev"], mon["n"]), wilson_ci(rest["rev"], rest["n"])

    if min(mon["n"], rest["n"]) < PREREG_CLAIM_3["min_n"]:
        verdict, why = "INCONCLUSIVE", "one bucket below the minimum n"
    elif p < PREREG_CLAIM_3["alpha"]:
        verdict, why = "CONFIRMED", f"p={p:.4f} < alpha {PREREG_CLAIM_3['alpha']}"
    else:
        verdict, why = "NOT_SIGNIFICANT", f"p={p:.4f} >= alpha {PREREG_CLAIM_3['alpha']}"

    print("Claim 3 — Monday gap reversal vs Tue-Fri:")
    print(f"  Monday   n={mon['n']:>7}  rev={mon['rev']:>7}  rate={rate_m:.4f}")
    print(f"  Tue-Fri  n={rest['n']:>7}  rev={rest['rev']:>7}  rate={rate_r:.4f}")
    print(f"  z={z:.3f}  p={p:.4f}  {verdict}")

    return {
        "claim_number": 3, "claim": PREREG_CLAIM_3["claim"],
        "generated_at": NOW.isoformat(), "status": "COMPLETE",
        "prereg": PREREG_CLAIM_3,
        "data": {"source": str(UNIVERSE.relative_to(REPO)), "tickers_used": tickers},
        "results": {
            "monday": {"n": mon["n"], "reversals": mon["rev"],
                       "rate": round(rate_m, 4),
                       "wilson_95ci": [round(ci_m[0], 4), round(ci_m[1], 4)]},
            "tue_fri": {"n": rest["n"], "reversals": rest["rev"],
                        "rate": round(rate_r, 4),
                        "wilson_95ci": [round(ci_r[0], 4), round(ci_r[1], 4)]},
            "z": z, "p_value_one_sided": p, "verdict": verdict, "reason": why,
            "monday_minus_tuefri_pct_points": round((rate_m - rate_r) * 100, 3),
        },
        "headline_verdict": verdict,
        "headline_reason": (
            f"Monday {rate_m:.4f} vs Tue-Fri {rate_r:.4f} "
            f"({(rate_m - rate_r) * 100:+.2f} pct points), p={p:.4f}"
        ),
        "deviations": [
            "Intraday reversal approximated by close-vs-open on daily bars. The "
            "pre-committed caveat applies: this cannot see the intraday path.",
            "Pre-registration written in the same pass as first execution.",
        ],
    }


MINUTE_BARS = REPO / "data" / "cache" / "minute_bars"

PREREG_CLAIM_5 = {
    "claim": "Momentum beats mean reversion in the first 30 minutes — in the "
             "gapper universe, stocks still moving at 09:45 continue more often "
             "than they reverse",
    "population": "ticker-days in data/cache/minute_bars (the gapper universe) "
                  "with a complete 09:30-09:45 window and a session close",
    "moving_definition": "|close(09:45) / open(09:30) - 1| >= threshold",
    "primary_threshold": 0.02,
    "sensitivity_thresholds": [0.01, 0.02, 0.03, 0.05],
    "continuation_definition": "the session's final close is beyond close(09:45) "
                               "in the same direction as the 09:30-09:45 move",
    "H0": "continuation rate <= 50% (momentum no better than a coin flip)",
    "H1": "continuation rate > 50%",
    "test": "one-sided binomial at the primary threshold",
    "alpha": 0.05,
    "min_n": 100,
    "precommitted_primary": (
        "2% is designated the PRIMARY threshold before any data is read. The other "
        "thresholds are reported as sensitivity only and must not be promoted to "
        "primary after the fact — that would be threshold shopping."
    ),
    "precommitted_caveat": (
        "this measures direction only, not profitability. A continuation rate above "
        "50% says nothing about whether the continuation is large enough to cover "
        "spread and slippage in small-cap gappers, where both are severe."
    ),
    "prereg_written": "2026-07-21, same pass as first execution — see integrity note",
}


def run_claim_5() -> dict:
    if not MINUTE_BARS.exists():
        raise SystemExit(f"FATAL: no minute bars at {MINUTE_BARS}. Nothing to measure.")
    files = sorted(MINUTE_BARS.glob("*.parquet"))
    if not files:
        raise SystemExit(f"FATAL: {MINUTE_BARS} is empty. Nothing to measure.")

    thresholds = PREREG_CLAIM_5["sensitivity_thresholds"]
    counts = {t: {"cont": 0, "n": 0} for t in thresholds}
    days_read = 0
    days_skipped = 0
    tickers = set()

    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception:  # noqa: BLE001
            days_skipped += 1
            continue
        if not {"time", "open", "close"}.issubset(df.columns) or len(df) < 20:
            days_skipped += 1
            continue
        t = df["time"].astype(str)
        try:
            i_open = t[t == "09:30"].index[0]
            i_945 = t[t == "09:45"].index[0]
        except IndexError:
            days_skipped += 1
            continue

        o = float(df["open"].iloc[i_open])
        c945 = float(df["close"].iloc[i_945])
        cend = float(df["close"].iloc[-1])
        if o <= 0:
            days_skipped += 1
            continue

        days_read += 1
        tickers.add(fp.stem.split("_")[0])
        move = c945 / o - 1.0
        up = move > 0
        cont = (cend > c945) if up else (cend < c945)

        for th in thresholds:
            if abs(move) >= th:
                counts[th]["n"] += 1
                counts[th]["cont"] += int(cont)

    print("Claim 5 — first-30-min momentum vs mean reversion (gapper universe):")
    results = {}
    primary = PREREG_CLAIM_5["primary_threshold"]
    for th in thresholds:
        k, n = counts[th]["cont"], counts[th]["n"]
        rate = k / n if n else float("nan")
        p, method = binom_sf(k, n) if n else (float("nan"), "none")
        ci = wilson_ci(k, n)
        if n < PREREG_CLAIM_5["min_n"]:
            verdict, why = "INCONCLUSIVE", f"n={n} below minimum {PREREG_CLAIM_5['min_n']}"
        elif p < PREREG_CLAIM_5["alpha"]:
            verdict, why = "CONFIRMED", f"p={p:.2e} < alpha {PREREG_CLAIM_5['alpha']}"
        else:
            verdict, why = "NOT_SIGNIFICANT", f"p={p:.4f} >= alpha {PREREG_CLAIM_5['alpha']}"
        tag = "PRIMARY" if th == primary else "sensitivity"
        results[f"threshold_{th}"] = {
            "threshold": th, "role": tag, "n": n, "continuations": k,
            "continuation_rate": round(rate, 4) if n else None,
            "wilson_95ci": [round(ci[0], 4), round(ci[1], 4)] if n else None,
            "p_value_one_sided": p, "p_method": method,
            "verdict": verdict, "reason": why,
        }
        print(f"  th={th:<5} [{tag:<11}] n={n:>6}  cont={k:>6}  rate={rate:.4f}  {verdict}")

    head = results[f"threshold_{primary}"]
    return {
        "claim_number": 5, "claim": PREREG_CLAIM_5["claim"],
        "generated_at": NOW.isoformat(), "status": "COMPLETE",
        "prereg": PREREG_CLAIM_5,
        "data": {
            "source": str(MINUTE_BARS.relative_to(REPO)),
            "ticker_days_read": days_read, "ticker_days_skipped": days_skipped,
            "distinct_tickers": len(tickers),
        },
        "results": results,
        "headline_verdict": head["verdict"],
        "headline_reason": (
            f"at the PRIMARY 2% threshold, continuation rate "
            f"{head['continuation_rate']} on n={head['n']}. Direction only — says "
            "nothing about whether the move covers spread and slippage."
        ),
        "deviations": [
            "Session close taken as the last available minute bar rather than a "
            "fixed 16:00 stamp, since files vary in length.",
            "Pre-registration written in the same pass as first execution.",
        ],
    }


PREREG_CLAIM_4 = {
    "claim": "Institutional order blocks hold as support/resistance",
    "population": "order blocks detected with the live detector's rule — the last "
                  "opposite-direction candle before an impulse of >= 1.5 x ATR",
    "hold_definition": (
        "on the FIRST subsequent revisit of the block zone within the horizon, "
        "price respects it: a bullish OB holds if the session that touches the "
        "zone closes above the block low; a bearish OB holds if the session that "
        "touches the zone closes below the block high"
    ),
    "horizons_sessions": [5, 20],
    "H0": "hold rate on first revisit <= 50%",
    "H1": "hold rate > 50%",
    "test": "one-sided binomial per horizon",
    "correction": "Bonferroni across 2 horizons",
    "alpha": 0.05 / 2,
    "min_n": 100,
    "precommitted_caveat": (
        "'holds' is conditioned on a revisit occurring. Blocks never revisited are "
        "excluded, not counted as holding — counting them as holds would inflate "
        "the rate toward 1 by construction, which is the obvious way to fake this "
        "result."
    ),
    "precommitted_trap": (
        "a close-above-low test is generous: price can pierce deep into the block "
        "and still 'hold' by the close. A CONFIRMED verdict here is therefore an "
        "upper bound on how well blocks work, not a tradable estimate."
    ),
    "prereg_written": "2026-07-21, same pass as first execution — see integrity note",
}


def run_claim_4() -> dict:
    horizons = PREREG_CLAIM_4["horizons_sessions"]
    counts = {h: {"hold": 0, "n": 0} for h in horizons}
    detected = 0
    never_revisited = {h: 0 for h in horizons}
    tickers = 0

    for ticker, df in _iter_universe():
        tickers += 1
        atr = atr_series(df).to_numpy()
        o = df["open"].to_numpy(); c = df["close"].to_numpy()
        hi = df["high"].to_numpy(); lo = df["low"].to_numpy()

        for i in range(len(df) - 1):
            a = atr[i + 1]
            if not np.isfinite(a) or a <= 0:
                continue
            impulse = abs(c[i + 1] - o[i + 1])
            if impulse < OB_IMPULSE_ATR * a:
                continue

            if c[i] < o[i] and c[i + 1] > o[i + 1]:
                kind, blk_hi, blk_lo = "BULLISH", hi[i], lo[i]
            elif c[i] > o[i] and c[i + 1] < o[i + 1]:
                kind, blk_hi, blk_lo = "BEARISH", hi[i], lo[i]
            else:
                continue
            detected += 1

            for hz in horizons:
                start, end = i + 2, i + 1 + hz
                if end >= len(df):
                    continue
                revisit = None
                for j in range(start, end + 1):
                    if lo[j] <= blk_hi and hi[j] >= blk_lo:   # touched the zone
                        revisit = j
                        break
                if revisit is None:
                    never_revisited[hz] += 1
                    continue
                held = c[revisit] > blk_lo if kind == "BULLISH" else c[revisit] < blk_hi
                counts[hz]["n"] += 1
                counts[hz]["hold"] += int(held)

    # ------------------------------------------------------------------
    # NULL CONTROL. A "hold" test framed as close-vs-the-far-edge is generous
    # by construction, so a high rate proves nothing on its own. Run the
    # identical test on PLACEBO zones — an ordinary candle's own high/low,
    # chosen without any impulse condition — and compare. If the placebo scores
    # the same, order blocks are not doing the work; the test framing is.
    # This control was decided before the real numbers were interpreted.
    # ------------------------------------------------------------------
    ctrl = {h: {"hold": 0, "n": 0} for h in horizons}
    rng = np.random.default_rng(20260721)
    for ticker, df in _iter_universe():
        c = df["close"].to_numpy(); hi = df["high"].to_numpy(); lo = df["low"].to_numpy()
        n_bars = len(df)
        if n_bars < 60:
            continue
        picks = rng.choice(np.arange(20, n_bars - 25), size=min(30, n_bars - 45),
                           replace=False)
        for i in picks:
            kind = "BULLISH" if rng.random() < 0.5 else "BEARISH"
            blk_hi, blk_lo = hi[i], lo[i]
            for hz in horizons:
                start, end = i + 2, i + 1 + hz
                if end >= n_bars:
                    continue
                revisit = None
                for j in range(start, end + 1):
                    if lo[j] <= blk_hi and hi[j] >= blk_lo:
                        revisit = j
                        break
                if revisit is None:
                    continue
                held = c[revisit] > blk_lo if kind == "BULLISH" else c[revisit] < blk_hi
                ctrl[hz]["n"] += 1
                ctrl[hz]["hold"] += int(held)

    print("Claim 4 — order blocks hold on first revisit:")
    results = {}
    alpha = PREREG_CLAIM_4["alpha"]
    for hz in horizons:
        k, n = counts[hz]["hold"], counts[hz]["n"]
        rate = k / n if n else float("nan")
        p, method = binom_sf(k, n) if n else (float("nan"), "none")
        ci = wilson_ci(k, n)
        if n < PREREG_CLAIM_4["min_n"]:
            verdict, why = "INCONCLUSIVE", f"n={n} below minimum {PREREG_CLAIM_4['min_n']}"
        elif p < alpha:
            verdict, why = "CONFIRMED", f"p={p:.2e} < alpha {alpha:.4f}"
        else:
            verdict, why = "NOT_SIGNIFICANT", f"p={p:.4f} >= alpha {alpha:.4f}"
        results[f"{hz}_session"] = {
            "horizon_sessions": hz, "n_revisited": n, "held": k,
            "hold_rate": round(rate, 4) if n else None,
            "wilson_95ci": [round(ci[0], 4), round(ci[1], 4)] if n else None,
            "never_revisited_excluded": never_revisited[hz],
            "p_value_one_sided": p, "p_method": method,
            "verdict": verdict, "reason": why,
        }
        ck, cn = ctrl[hz]["hold"], ctrl[hz]["n"]
        crate = ck / cn if cn else float("nan")
        z, pz = two_prop_z(k, n, ck, cn)
        results[f"{hz}_session"]["placebo_rate"] = round(crate, 4) if cn else None
        results[f"{hz}_session"]["placebo_n"] = cn
        results[f"{hz}_session"]["vs_placebo_z"] = z
        results[f"{hz}_session"]["vs_placebo_p"] = pz
        results[f"{hz}_session"]["beats_placebo"] = bool(pz < 0.05) if cn else None
        print(f"  h={hz:>2}  revisited n={n:>7}  held={k:>7}  rate={rate:.4f}  "
              f"(excluded never-revisited: {never_revisited[hz]})  {verdict}")
        print(f"        PLACEBO n={cn:>7}  rate={crate:.4f}  "
              f"OB-vs-placebo z={z:.2f} p={pz:.4f}  "
              f"{'BEATS placebo' if pz < 0.05 else 'NO BETTER than placebo'}")

    head = results["5_session"]
    return {
        "claim_number": 4, "claim": PREREG_CLAIM_4["claim"],
        "generated_at": NOW.isoformat(), "status": "COMPLETE",
        "prereg": PREREG_CLAIM_4,
        "data": {"source": str(UNIVERSE.relative_to(REPO)), "tickers_used": tickers,
                 "order_blocks_detected": detected,
                 "ob_impulse_atr": OB_IMPULSE_ATR},
        "results": results,
        "headline_verdict": head["verdict"],
        "headline_reason": (
            f"5-session hold rate on first revisit {head['hold_rate']} on "
            f"n={head['n_revisited']}. Upper bound, per the pre-committed trap."
        ),
        "deviations": [
            "Impulse threshold read from the live detector (1.5 x ATR) rather than "
            "assumed. Point-in-time ATR, as in claim 1.",
            "Pre-registration written in the same pass as first execution.",
        ],
    }


PREREG_CLAIM_8 = {
    "claim": "Higher timeframe bias determines lower timeframe outcome — SPY's "
             "daily trend direction predicts whether a gap fades or continues",
    "population": "gap events (|open/prior_close - 1| >= 1%) across the cached "
                  "daily universe, excluding SPY itself, matched to SPY's trend "
                  "state on the same date",
    "htf_bias_definition": "SPY close above its trailing 20-session SMA = UP, "
                           "below = DOWN. Both computed strictly from bars at or "
                           "before the prior session — no same-day information.",
    "outcome_definition": "the gap CONTINUES if the session's close is beyond its "
                          "open in the gap's direction; otherwise it FADES",
    "H0": "gap continuation rate is the same whether SPY is UP or DOWN",
    "H1": "aligned gaps (up-gap in an UP tape, down-gap in a DOWN tape) continue "
          "more often than unaligned ones",
    "test": "two-proportion z-test, one-sided",
    "alpha": 0.05,
    "min_n": 100,
    "precommitted_caveat": (
        "SPY trend state is highly autocorrelated, so the two buckets are not "
        "independent samples of market conditions — they are largely different "
        "time periods. A significant result may reflect regime differences rather "
        "than a usable conditioning signal."
    ),
    "precommitted_trap": (
        "the SMA must be computed on bars strictly at or before the prior session. "
        "Using the same day's close to define the day's bias is lookahead and "
        "would manufacture a result."
    ),
    "prereg_written": "2026-07-21, same pass as first execution — see integrity note",
}


def run_claim_8() -> dict:
    spy_path = UNIVERSE / "SPY.parquet"
    if not spy_path.exists():
        raise SystemExit("FATAL: SPY.parquet not in the universe — cannot define "
                         "higher-timeframe bias. Refusing to substitute a proxy.")
    spy = pd.read_parquet(spy_path).reset_index(drop=True)
    spy["sma20"] = spy["close"].rolling(20, min_periods=20).mean()
    # Shift by one so the bias for date D uses only bars up to D-1.
    spy["bias"] = np.where(spy["close"].shift(1) > spy["sma20"].shift(1), "UP",
                  np.where(spy["close"].shift(1) < spy["sma20"].shift(1), "DOWN", None))
    bias_by_date = dict(zip(spy["date"].astype(str), spy["bias"]))

    aligned = {"cont": 0, "n": 0}
    unaligned = {"cont": 0, "n": 0}
    tickers = 0
    unmatched_dates = 0

    for ticker, df in _iter_universe():
        if ticker == "SPY":
            continue
        tickers += 1
        dates = df["date"].astype(str).to_numpy()
        o = df["open"].to_numpy(); c = df["close"].to_numpy()
        for i in range(1, len(df)):
            prev_close = c[i - 1]
            if prev_close <= 0:
                continue
            gap = o[i] / prev_close - 1.0
            if abs(gap) < 0.01:
                continue
            bias = bias_by_date.get(dates[i])
            if bias is None or (isinstance(bias, float) and not np.isfinite(bias)):
                unmatched_dates += 1
                continue
            gap_up = gap > 0
            is_aligned = (gap_up and bias == "UP") or ((not gap_up) and bias == "DOWN")
            cont = (c[i] > o[i]) if gap_up else (c[i] < o[i])
            bucket = aligned if is_aligned else unaligned
            bucket["n"] += 1
            bucket["cont"] += int(cont)

    ra = aligned["cont"] / aligned["n"] if aligned["n"] else float("nan")
    ru = unaligned["cont"] / unaligned["n"] if unaligned["n"] else float("nan")
    z, p = two_prop_z(aligned["cont"], aligned["n"], unaligned["cont"], unaligned["n"])
    ci_a, ci_u = wilson_ci(aligned["cont"], aligned["n"]), wilson_ci(unaligned["cont"], unaligned["n"])

    if min(aligned["n"], unaligned["n"]) < PREREG_CLAIM_8["min_n"]:
        verdict, why = "INCONCLUSIVE", "one bucket below the minimum n"
    elif p < PREREG_CLAIM_8["alpha"]:
        verdict, why = "CONFIRMED", f"p={p:.4f} < alpha {PREREG_CLAIM_8['alpha']}"
    else:
        verdict, why = "NOT_SIGNIFICANT", f"p={p:.4f} >= alpha {PREREG_CLAIM_8['alpha']}"

    print("Claim 8 — gap continuation, aligned vs unaligned with SPY trend:")
    print(f"  aligned    n={aligned['n']:>7}  cont={aligned['cont']:>7}  rate={ra:.4f}")
    print(f"  unaligned  n={unaligned['n']:>7}  cont={unaligned['cont']:>7}  rate={ru:.4f}")
    print(f"  z={z:.3f}  p={p:.4f}  {verdict}")

    return {
        "claim_number": 8, "claim": PREREG_CLAIM_8["claim"],
        "generated_at": NOW.isoformat(), "status": "COMPLETE",
        "prereg": PREREG_CLAIM_8,
        "data": {"source": str(UNIVERSE.relative_to(REPO)), "tickers_used": tickers,
                 "gap_events_unmatched_to_spy_date": unmatched_dates},
        "results": {
            "aligned": {"n": aligned["n"], "continuations": aligned["cont"],
                        "rate": round(ra, 4),
                        "wilson_95ci": [round(ci_a[0], 4), round(ci_a[1], 4)]},
            "unaligned": {"n": unaligned["n"], "continuations": unaligned["cont"],
                          "rate": round(ru, 4),
                          "wilson_95ci": [round(ci_u[0], 4), round(ci_u[1], 4)]},
            "z": z, "p_value_one_sided": p, "verdict": verdict, "reason": why,
            "aligned_minus_unaligned_pct_points": round((ra - ru) * 100, 3),
        },
        "headline_verdict": verdict,
        "headline_reason": (
            f"aligned {ra:.4f} vs unaligned {ru:.4f} "
            f"({(ra - ru) * 100:+.2f} pct points), p={p:.4f}"
        ),
        "deviations": [
            "Higher-timeframe bias defined as SPY vs its 20-session SMA, lagged one "
            "session to prevent lookahead. The original claim says 'SPY daily trend' "
            "without specifying a definition; this one is stated so it can be "
            "argued with.",
            "Outcome is close-vs-open on daily bars, not the intraday fade path.",
            "Population is the cached daily equity universe, not the gapper event "
            "universe the claim refers to.",
            "Pre-registration written in the same pass as first execution.",
        ],
    }


VIX_PATH = REPO / "data" / "research" / "modern" / "spot_cache" / "VIX.parquet"

PREREG_CLAIM_2 = {
    "claim": "Buy the dip on high VIX — buying when VIX > 25 and SPY is down >1% "
             "produces positive returns over 1, 5, 20 days",
    "population": "SPY sessions where VIX close > 25 AND SPY return < -1%",
    "horizons_sessions": [1, 5, 20],
    "H0": "forward return positive-rate <= the unconditional SPY base rate over "
          "the same horizon",
    "H1": "the conditional rate exceeds the unconditional base rate",
    "test": "sign test (binomial on positive forward returns) against the "
            "unconditional base rate, plus a two-proportion z-test vs the "
            "unconditional sample",
    "alpha": 0.05,
    "min_n": 30,
    "precommitted_control": (
        "the honest comparison is NOT 'is the return positive' — SPY drifts up, so "
        "any long is positive most of the time. The comparison is against the "
        "UNCONDITIONAL forward return over the same horizons. A dip-buying rule "
        "that merely matches buy-and-hold is not an edge."
    ),
    "precommitted_caveat": (
        "high-VIX events cluster heavily (2018, 2020, 2022), so observations are "
        "not independent. Effective n is far below nominal n and any p-value here "
        "is optimistic. Treat a marginal result as no result."
    ),
    "prereg_written": "2026-07-21, same pass as first execution — see integrity note",
}


def run_claim_2() -> dict:
    if not VIX_PATH.exists():
        raise SystemExit(f"FATAL: no VIX series at {VIX_PATH}. Refusing to "
                         "substitute VXX/UVXY — those are volatility ETPs with "
                         "decay, not the index.")
    spy_path = UNIVERSE / "SPY.parquet"
    if not spy_path.exists():
        raise SystemExit("FATAL: SPY.parquet not in the universe.")

    vix = pd.read_parquet(VIX_PATH)
    vix.index = pd.to_datetime(vix.index)
    vix_close = vix["Close"]

    spy = pd.read_parquet(spy_path).reset_index(drop=True)
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.set_index("date").sort_index()
    spy["ret"] = spy["close"].pct_change()

    joined = spy.join(vix_close.rename("vix"), how="inner").dropna(subset=["vix", "ret"])
    horizons = PREREG_CLAIM_2["horizons_sessions"]
    close = joined["close"].to_numpy()
    ret = joined["ret"].to_numpy()
    vx = joined["vix"].to_numpy()
    n_all = len(joined)

    events = [i for i in range(n_all) if vx[i] > 25 and ret[i] < -0.01]

    results = {}
    print("Claim 2 — buy the dip on high VIX (VIX>25 and SPY down >1%):")
    for h in horizons:
        ev_rets, base_rets = [], []
        for i in events:
            if i + h < n_all:
                ev_rets.append(close[i + h] / close[i] - 1.0)
        for i in range(n_all - h):
            base_rets.append(close[i + h] / close[i] - 1.0)
        ev = np.array(ev_rets)
        base = np.array(base_rets)
        k, n = int((ev > 0).sum()), len(ev)
        bk, bn = int((base > 0).sum()), len(base)
        rate = k / n if n else float("nan")
        brate = bk / bn if bn else float("nan")
        z, p = two_prop_z(k, n, bk, bn)

        if n < PREREG_CLAIM_2["min_n"]:
            verdict, why = "INCONCLUSIVE", f"n={n} below minimum {PREREG_CLAIM_2['min_n']}"
        elif p < PREREG_CLAIM_2["alpha"]:
            verdict, why = "CONFIRMED", f"p={p:.4f} < alpha vs unconditional base rate"
        else:
            verdict, why = "NOT_SIGNIFICANT", f"p={p:.4f} — no better than buy-and-hold"

        results[f"{h}_session"] = {
            "horizon_sessions": h, "n_events": n, "positive": k,
            "positive_rate": round(rate, 4) if n else None,
            "mean_return": round(float(ev.mean()), 5) if n else None,
            "median_return": round(float(np.median(ev)), 5) if n else None,
            "unconditional_n": bn,
            "unconditional_positive_rate": round(brate, 4),
            "unconditional_mean_return": round(float(base.mean()), 5),
            "vs_unconditional_z": z, "vs_unconditional_p": p,
            "verdict": verdict, "reason": why,
        }
        print(f"  h={h:>2}  events n={n:>4}  pos_rate={rate:.4f} "
              f"(uncond {brate:.4f})  mean={ev.mean():+.4f} "
              f"(uncond {base.mean():+.4f})  z={z:.2f} p={p:.4f}  {verdict}")

    head = results["5_session"]
    return {
        "claim_number": 2, "claim": PREREG_CLAIM_2["claim"],
        "generated_at": NOW.isoformat(), "status": "COMPLETE",
        "prereg": PREREG_CLAIM_2,
        "data": {
            "vix_source": str(VIX_PATH.relative_to(REPO)),
            "spy_source": str(spy_path.relative_to(REPO)),
            "overlapping_sessions": n_all,
            "event_count": len(events),
            "date_range": [str(joined.index[0].date()), str(joined.index[-1].date())],
        },
        "results": results,
        "headline_verdict": head["verdict"],
        "headline_reason": (
            f"5-session positive rate {head['positive_rate']} vs unconditional "
            f"{head['unconditional_positive_rate']}, mean {head['mean_return']} vs "
            f"{head['unconditional_mean_return']}"
        ),
        "deviations": [
            "This claim was briefly recorded as BLOCKED for lack of a VIX series. "
            "That was wrong — only data/cache/daily_universe had been searched. A "
            "real VIX history exists at data/research/modern/spot_cache/VIX.parquet "
            "(3,038 sessions, 2014-2026). Corrected rather than left as blocked.",
            "Tested on SPY only, as the claim is about buying the index dip.",
            "Sign test against the unconditional base rate rather than against zero "
            "— see the pre-committed control.",
            "Pre-registration written in the same pass as first execution.",
        ],
    }


RUNNERS = {1: run_claim_1, 2: run_claim_2, 3: run_claim_3, 4: run_claim_4,
           5: run_claim_5, 7: run_claim_7, 8: run_claim_8, 10: run_claim_10}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claim", type=int, default=1, help="claim number to run")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = ap.parse_args()

    if args.claim not in RUNNERS:
        print(f"Claim {args.claim} has no harness yet. Implemented: "
              f"{sorted(RUNNERS)}", file=sys.stderr)
        print("Refusing to emit a result for a claim that was not measured.",
              file=sys.stderr)
        return 2

    print(f"Running claim {args.claim} against {UNIVERSE} ...")
    doc = RUNNERS[args.claim]()

    if args.dry_run:
        print(json.dumps(doc, indent=2, default=str))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = NOW.strftime("%Y-%m-%d")
    (OUT_DIR / f"claim-{args.claim}-{stamp}.json").write_text(json.dumps(doc, indent=2, default=str))
    (OUT_DIR / "latest.json").write_text(json.dumps(doc, indent=2, default=str))
    print(f"\nstaged -> {(OUT_DIR / f'claim-{args.claim}-{stamp}.json')}")
    print(f"headline: {doc['headline_verdict']} — {doc['headline_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
