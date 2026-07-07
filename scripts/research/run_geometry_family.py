#!/usr/bin/env python3
"""HYP-082 (Gα, corridor deviation) + HYP-083 (Gβ, FVG diversifier) — GEOMETRY-2026-07 runner.

TICK-019 Phase 2. Mirrors scripts/research/run_positioning_family_options.py architecture
exactly: gate-zero hash verify over ALL FOUR geometry prereg files (082/083/084 + the family
manifest) BEFORE any data read; seals are dated ledger ANNOTATIONS while prereg status stays
PREREGISTERED (this script never writes to the hash-locked prereg files themselves); interim
seals carry NO verdict until --adjudicate; --dry-run previews without any ledger write.

Both members are computed and sealed together in the same run (the manifest's 2-member BH
family is complete in one pass — unlike the staged 10-member positioning-options family this
mirrors). --adjudicate runs the family Benjamini-Hochberg (m=2, alpha=.05) over the two primary
p-values and stamps per-member verdict annotations + the ledger's verdict field, per the plan's
Phase 2 Adjudication mapping:
    BH fail                                            -> NOT_SIGNIFICANT
    N < 50 pooled (either member)                     -> UNDERPOWERED (checked before BH)
    Gα: BH-pass, fold-sign UNSTABLE                    -> NOT_ROBUST
    Gα: BH-pass, fold-sign OK, |IC|<0.05 or cost-floor  -> NOT_SIGNIFICANT ("anything else", prereg)
    Gα: BH-pass + |IC|>=0.05 + fold-sign + cost-floor   -> CONFIRMED
    Gβ: BH-pass, diversifier gate fails                -> NOT_DIVERSIFIER
    Gβ: BH-pass + gate passes + N>=50                   -> CONFIRMED
A CONFIRMED member is reported with the E4 recommendation (`python -m factory.train --hyp
HYP-08x`) but this script NEVER invokes training itself — ledger + report only, per the plan.

Usage: python3 scripts/research/run_geometry_family.py [--dry-run] [--adjudicate]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sovereign.discovery import cpcv                                    # noqa: E402
from sovereign.research.positioning import event_study as es            # noqa: E402
from sovereign.research.positioning import v015_replay as vr            # noqa: E402
from sovereign.research.vrp import data_loader as vrp_data              # noqa: E402
from sovereign.sentiment import geometry_feed                            # noqa: E402
from sovereign.sentiment.store import connect                            # noqa: E402
from scripts.research import positioning_options_legs as ol             # noqa: E402

PREREG = ROOT / "data" / "research" / "preregister"
OUT = ROOT / "data" / "research" / "geometry_family"
SPOT_CACHE = ROOT / "data" / "research" / "positioning_family" / "spot_cache"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
FAMILY = "GEOMETRY-2026-07"
SEAL_BY = f"{FAMILY}/interim-geometry"
N_PERM, SEED = 10000, 42

# Gate-zero covers all FOUR geometry docs (HYP-084 is outside the BH family by design, but its
# hash is still verified here — plan Phase 0/2: "ALL FOUR geometry files (082, 083, 084, manifest)").
ALL_PREREG = ["HYP-082_fractal_beyond_carry.json", "HYP-083_fvg_diversifier.json",
              "HYP-084_triangle_precedent_quality.json", "GEOMETRY-2026-07_family.json"]

PAIRS = ol.OPT_PAIRS  # ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"] — same universe as HYP-082/083
PIP_SIZE = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "USDJPY": 0.01}
COST_FLOOR_PIPS = 3.0
GAMMA_START = "2015-07-01"
CRISIS_WINDOWS = [("2020-02-20", "2020-04-30"), ("2022-01-01", "2022-12-31")]

INTERP_TEXT = (
    "interpretations (A1 banner restated: research/whole_chart_harvester.py, "
    "research/validate_corridor_feature.py, the pre-gate FVG explorations, and any in-sample "
    "permutation artifacts are NON-EVIDENTIARY — thesis text only; thresholds set from mechanism "
    "+ realistic costs, never reverse-fit). Gα: obs = every 5th trading day per pair (position "
    "0,5,10,... over the FULL geometry-board calendar, so the sampling anchor is stable "
    "regardless of the start-date/non-NaN filters applied after), non-NaN corridor_dev, from "
    f"{GAMMA_START}; carry_residual = pair fwd h20 log return - beta_t * v015-equity fwd h20 log "
    "return, beta_t = trailing-252d rolling Cov/Var slope (equivalent to the OLS slope for a "
    "no-intercept-needed univariate regression), min_periods=252, trailing-only ending at the "
    "obs date, undefined-beta obs dropped AND counted; primary = pooled Spearman IC(corridor_dev, "
    "carry_residual), TWO-SIDED on |IC| (prereg-declared deviation from the family's one-sided "
    "convention); null = within-pair permutation of the FEATURE vector only (carry_residual held "
    "fixed) across that pair's own beta-defined eligible obs. Cost floor: per-pair top-decile "
    "(90th pct) |corridor_dev| threshold, raw |close[t0+h20]-close[t0]| price move per obs "
    "converted to PIPS via that obs's own pair pip size, then pooled across all 4 pairs into one "
    "combined median vs the 3.0-pip floor. Gβ: events from geometry_feed.fvg_formation_events "
    "called per trading day on df.iloc[:t+1] (REPLICATED kernel, no ict import — the board carries "
    "counts, not events); side = gap direction, PAIR SPACE, NO PAIR_FLIP applied (prereg "
    "event_definition: 'spot-derived, no flip' — unlike the COT-derived legs, this is already "
    "pair-space by construction); de-overlap one event/pair-day (structural: a 3-bar triplet can "
    "register at most one direction) + same-direction suppression 5 TRADING days (measured on the "
    "pair's own calendar positions, not event-list index); t0 = shift_pub (+1 calendar day) then "
    "es.effective_t0 (first trading day strictly after formation). Diversifier gate evaluated "
    "independently vs v015 daily curve AND vs DBV carry proxy — BOTH benchmarks must "
    "independently satisfy |rho_full|<0.25 AND max crisis |rho|<0.35 (prereg lists both "
    "benchmarks under one joint_success_condition; a crisis window with n<20 is excluded from "
    "the max rather than counted as failing, and is reported per-window)."
)
COVERAGE_TEXT = (
    "coverage: sentiment_geometry_daily substrate from the daily OHLC spot cache (2014-06+, "
    "output floor 2015-01-01 per config sentiment.geometry.start) — DAILY bars only, FX intraday "
    "does not exist in this system (prereg data_substrate: small-N honesty over resolution "
    "fantasy). DBV carry-proxy overlap ends 2023-03 (vrp.data_loader.load_carry_proxy) — "
    "crisis-window/full-period n stated per benchmark in the diversifier_gate block."
)


# ── gate zero ──────────────────────────────────────────────────────────────────────────────

def _canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gate_zero() -> dict:
    checks = {}
    for name in ALL_PREREG:
        doc = json.loads((PREREG / name).read_text())
        ok = _canonical_hash(doc) == doc.get("hash_lock")
        checks[name] = {"hash": doc.get("hash_lock", "")[:16], "ok": ok}
        if not ok:
            raise SystemExit(f"GATE ZERO: PREREGISTER HASH MISMATCH in {name}. HALT.")
    return checks


# ── data loads ─────────────────────────────────────────────────────────────────────────────

def load_ohlc(pair: str) -> pd.DataFrame:
    """Daily OHLC from the SAME local parquet cache geometry_feed.py reads — read-only, no
    network fetch (Phase 1 of the plan already populated these; this runner never hits yfinance).
    """
    path = SPOT_CACHE / f"{pair}_ohlc.parquet"
    if not path.exists():
        raise SystemExit(f"no cached OHLC for {pair} at {path} (run Phase 1 geometry_feed.update() first)")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[["Open", "High", "Low", "Close"]].astype(float).sort_index()


def load_geometry(pairs: list[str]) -> dict[str, pd.DataFrame]:
    """sentiment_geometry_daily — the prereg-named data_substrate (NOT the joined board:
    HYP-082/083's validation_protocol.data_substrate names this table explicitly)."""
    con = connect(read_only=True)
    try:
        df = con.execute(
            "SELECT date, pair, corridor_dev, corridor_r2, fvg_count_20d, fvg_unfilled, "
            "range_slope, days_in_consolidation FROM sentiment_geometry_daily "
            "WHERE pair IN (?,?,?,?) ORDER BY pair, date", pairs).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    out = {p: g.set_index("date").sort_index() for p, g in df.groupby("pair")}
    return {p: out.get(p, pd.DataFrame(columns=["corridor_dev", "corridor_r2", "fvg_count_20d",
                                                 "fvg_unfilled", "range_slope",
                                                 "days_in_consolidation"])) for p in pairs}


# ── Gα (HYP-082): corridor deviation beyond carry ────────────────────────────────────────────

def gamma_observations(geo_pair: pd.DataFrame, start: str = GAMMA_START) -> pd.DataFrame:
    """Every 5th trading day on the pair's OWN geometry-board calendar (position 0,5,10,... over
    ALL rows the board carries — the grid anchor is fixed BEFORE the start/non-NaN filters, per
    the prereg's observation_definition word order), then restricted to date >= start and
    non-NaN corridor_dev."""
    every5 = geo_pair.iloc[::5]
    return every5[(every5.index >= pd.Timestamp(start)) & every5["corridor_dev"].notna()]


def carry_residual_frame(pair_close: pd.Series, v015_eq: pd.Series, h: int = 20,
                         beta_window: int = 252) -> pd.DataFrame:
    """Per-date beta_t / carry_residual table, aligned on dates common to both series.

    beta_t = trailing beta_window-day rolling Cov(pair_ret, v015_ret) / Var(v015_ret) —
    algebraically the OLS slope of pair_ret on v015_ret (no intercept term needed for a slope),
    min_periods=beta_window so beta is undefined (NaN) until a full trailing window exists;
    the rolling window ending at t never reads a return after t (trailing-only, no look-ahead).
    carry_residual = pair_fwd_h - beta_t * v015_fwd_h, forward log returns over the SAME h-day
    window starting at t (undefined near the end of history where h forward bars don't exist).
    """
    merged = pd.DataFrame({"pair_close": pair_close, "v015_eq": v015_eq}).dropna()
    idx = merged.index
    n = len(idx)
    pair_ret = np.log(merged["pair_close"] / merged["pair_close"].shift(1))
    v015_ret = np.log(merged["v015_eq"] / merged["v015_eq"].shift(1))
    cov = pair_ret.rolling(beta_window, min_periods=beta_window).cov(v015_ret)
    var = v015_ret.rolling(beta_window, min_periods=beta_window).var()
    beta = (cov / var.replace(0.0, np.nan)).to_numpy()

    pc = merged["pair_close"].to_numpy()
    ve = merged["v015_eq"].to_numpy()
    pair_fwd = np.full(n, np.nan)
    v015_fwd = np.full(n, np.nan)
    exit_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    if n > h:
        valid = np.arange(n - h)
        pair_fwd[valid] = np.log(pc[valid + h] / pc[valid])
        v015_fwd[valid] = np.log(ve[valid + h] / ve[valid])
        exit_date[valid] = idx.values[valid + h]

    residual = pair_fwd - beta * v015_fwd
    return pd.DataFrame({"beta": beta, "pair_fwd": pair_fwd, "v015_fwd": v015_fwd,
                        "carry_residual": residual, "exit_date": exit_date}, index=idx)


def pooled_ic_permutation(scored_by_pair: dict[str, pd.DataFrame], rng: np.random.Generator,
                          n_perm: int = 10000) -> dict:
    """Pooled Spearman IC(corridor_dev, carry_residual) + within-pair FEATURE-permutation null
    (carry_residual held fixed; corridor_dev shuffled within each pair's own obs). TWO-SIDED p on
    |IC| — the prereg-declared deviation from the family's one-sided convention."""
    per_pair_feat: dict[str, np.ndarray] = {}
    feat_all: list[float] = []
    resid_all: list[float] = []
    for pair, df in scored_by_pair.items():
        feat = df["corridor_dev"].to_numpy(dtype=float)
        resid = df["carry_residual"].to_numpy(dtype=float)
        per_pair_feat[pair] = feat
        feat_all.extend(feat.tolist())
        resid_all.extend(resid.tolist())
    n = len(feat_all)
    if n < 3:
        return {"ic": None, "p": None, "n": n, "n_perm": n_perm}
    obs_ic = spearmanr(feat_all, resid_all).statistic
    if not np.isfinite(obs_ic):
        return {"ic": None, "p": None, "n": n, "n_perm": n_perm}
    obs_ic = float(obs_ic)
    n_ge = 0
    pairs_order = list(per_pair_feat.keys())
    for _ in range(n_perm):
        draw: list[float] = []
        for pair in pairs_order:
            draw.extend(rng.permutation(per_pair_feat[pair]))
        ic = spearmanr(draw, resid_all).statistic
        if np.isfinite(ic) and abs(float(ic)) >= abs(obs_ic):
            n_ge += 1
    p = (n_ge + 1) / (n_perm + 1)
    return {"ic": round(obs_ic, 6), "p": round(p, 5), "n": n, "n_perm": n_perm}


def cpcv_fold_sign_consistency(scored_by_pair: dict[str, pd.DataFrame], n_groups: int = 6,
                               test_groups: int = 1, embargo_frac: float = 0.02) -> dict:
    """IC recomputed per CPCV TEST fold (pooled across pairs; label interval = [obs_date,
    exit_date]); ALL folds with a defined sign must agree for 'ROBUST' fold-sign consistency."""
    rows = []
    for _pair, df in scored_by_pair.items():
        for dt_, r in df.iterrows():
            if pd.isna(r.get("exit_date")):
                continue
            rows.append((dt_, r["exit_date"], float(r["corridor_dev"]), float(r["carry_residual"])))
    n = len(rows)
    if n < n_groups:
        return {"folds": [], "n_folds": 0, "n_defined": 0, "all_same_sign": None,
                "note": f"fewer than n_groups={n_groups} pooled obs ({n}) — cannot form CPCV groups"}
    entry_dt = np.array([r[0] for r in rows], dtype="datetime64[ns]")
    exit_dt = np.array([r[1] for r in rows], dtype="datetime64[ns]")
    feats = np.array([r[2] for r in rows], dtype=float)
    resid = np.array([r[3] for r in rows], dtype=float)
    fold_ics: list[float | None] = []
    for _train_idx, test_idx in cpcv.combinatorial_purged_splits(
            entry_dt, exit_dt, n_groups=n_groups, test_groups=test_groups, embargo_frac=embargo_frac):
        if len(test_idx) < 3:
            fold_ics.append(None)
            continue
        ic = spearmanr(feats[test_idx], resid[test_idx]).statistic
        fold_ics.append(None if not np.isfinite(ic) else round(float(ic), 6))
    defined = [ic for ic in fold_ics if ic is not None]
    all_same_sign = None
    if len(defined) >= 2:
        signs = {(1 if ic > 0 else (-1 if ic < 0 else 0)) for ic in defined}
        signs.discard(0)
        all_same_sign = bool(len(signs) <= 1)
    return {"folds": fold_ics, "n_folds": len(fold_ics), "n_defined": len(defined),
            "all_same_sign": all_same_sign}


def cost_floor(obs_by_pair: dict[str, pd.DataFrame], closes: dict[str, pd.Series],
              h: int = 20) -> dict:
    """Median |fwd h20 PRICE move| among each pair's own top-decile-|corridor_dev| obs, each
    move normalized to PIPS via its own pair's pip size, then pooled across all 4 pairs into one
    combined median — compared to the 3.0-pip floor (HYP-082 success_criteria)."""
    pooled_pips: list[float] = []
    per_pair: dict[str, dict] = {}
    for pair, df in obs_by_pair.items():
        d = df.dropna(subset=["corridor_dev"])
        if d.empty:
            per_pair[pair] = {"n_top_decile": 0, "n_scored": 0, "median_pips": None}
            continue
        thresh = float(d["corridor_dev"].abs().quantile(0.90))
        top = d[d["corridor_dev"].abs() >= thresh]
        c = closes[pair]
        pip = PIP_SIZE[pair]
        moves_pips = []
        for dt_ in top.index:
            pos = int(c.index.get_indexer([dt_])[0])
            if pos < 0 or pos + h >= len(c):
                continue
            move = abs(float(c.iloc[pos + h] - c.iloc[pos]))
            moves_pips.append(move / pip)
        pooled_pips.extend(moves_pips)
        per_pair[pair] = {"n_top_decile": int(len(top)), "n_scored": len(moves_pips),
                          "median_pips": round(float(np.median(moves_pips)), 3) if moves_pips else None,
                          "threshold_abs_corridor_dev": round(thresh, 4)}
    median_pips = float(np.median(pooled_pips)) if pooled_pips else None
    # Round BEFORE the floor comparison (not just for display): raw price arithmetic on decimals
    # like 1.1000 + 0.00030 introduces float noise at the 1e-13 relative level (e.g. 2.9999999999996696
    # instead of 3.0), which would wrongly fail an exact-boundary obs on a >= comparison. 3 decimal
    # places of a PIP is far finer than anything economically meaningful here, so rounding first
    # both matches what's reported and removes the float-noise false negative.
    median_pips_rounded = round(median_pips, 3) if median_pips is not None else None
    return {
        "method": ("per-pair top-decile |corridor_dev| threshold (90th pct of that pair's own "
                   "eligible-obs |corridor_dev|); raw |close[t0+h20]-close[t0]| price move per "
                   "obs converted to PIPS via that obs's own pair pip size (EURUSD/GBPUSD/AUDUSD "
                   "pip=0.0001, USDJPY pip=0.01 — equivalently the prereg's 0.00030/0.030 price "
                   "floors), then ALL pairs' top-decile pip-moves pooled into one combined median"),
        "per_pair": per_pair,
        "pooled_median_pips": median_pips_rounded,
        "floor_pips": COST_FLOOR_PIPS,
        "pass": bool(median_pips_rounded is not None and median_pips_rounded >= COST_FLOOR_PIPS),
    }


def run_gamma(geo: dict[str, pd.DataFrame], closes: dict[str, pd.Series], v015_eq: pd.Series,
             rng: np.random.Generator) -> dict:
    base_obs = {p: gamma_observations(geo[p]) for p in PAIRS}
    scored: dict[str, pd.DataFrame] = {}
    beta_stats: dict[str, dict] = {}
    for p in PAIRS:
        crf = carry_residual_frame(closes[p], v015_eq, h=20, beta_window=252)
        joined = crf.reindex(base_obs[p].index)
        undefined_beta = joined["beta"].isna()
        # carry_residual can also be NaN with a DEFINED beta (obs sits within the final h bars
        # of history, where pair_fwd/v015_fwd have no forward window) — spec text only names
        # "undefined-beta obs dropped AND counted", so the two reasons are reported separately
        # here rather than folded into one (possibly misleading) "undefined_beta" count.
        undefined_fwd_given_beta = joined["carry_residual"].isna() & ~undefined_beta
        keep = joined["beta"].notna() & joined["carry_residual"].notna()
        beta_stats[p] = {"total_obs": int(len(joined)),
                         "undefined_beta_dropped": int(undefined_beta.sum()),
                         "undefined_fwd_return_dropped": int(undefined_fwd_given_beta.sum()),
                         "scored": int(keep.sum())}
        extra_cols = ["corridor_dev", "fvg_count_20d", "fvg_unfilled", "range_slope",
                     "days_in_consolidation"]
        scored[p] = joined[keep].join(base_obs[p][extra_cols])

    primary = pooled_ic_permutation(scored, rng, N_PERM)
    fold = cpcv_fold_sign_consistency(scored)
    cf = cost_floor(base_obs, closes, h=20)

    secondaries: dict[str, float | None] = {}
    for col in ("fvg_count_20d", "fvg_unfilled", "range_slope", "days_in_consolidation"):
        feat_all: list[float] = []
        resid_all: list[float] = []
        for p in PAIRS:
            d = scored[p].dropna(subset=[col])
            feat_all.extend(d[col].astype(float).tolist())
            resid_all.extend(d["carry_residual"].astype(float).tolist())
        secondaries[col] = es.spearman_ic(feat_all, resid_all)

    n_total = primary.get("n") or 0
    ic_val = primary.get("ic")
    ic_pass = bool(ic_val is not None and abs(ic_val) >= 0.05)
    cost_pass = bool(cf.get("pass"))
    sample_status = "OK" if n_total >= 50 else "UNDERPOWERED"

    return {
        "hyp": "HYP-082",
        "per_pair_N": {p: int(len(scored[p])) for p in PAIRS},
        "beta": beta_stats,
        "primary": {"statistic_ic": ic_val, "raw_p": primary.get("p"), "n_perm": primary.get("n_perm"),
                    "N": n_total, "sidedness": "two-sided on |IC| (prereg-declared deviation)"},
        "gates": {"ic_threshold_pass": ic_pass,
                  "ic_abs": None if ic_val is None else round(abs(ic_val), 6),
                  "fold_sign_consistent": fold.get("all_same_sign"), "cpcv": fold,
                  "cost_floor_pass": cost_pass, "cost_floor": cf},
        "secondaries_exploratory": secondaries,
        "sample_status": sample_status,
        "deviations": [],
    }


# ── Gβ (HYP-083): FVG event book diversifier ────────────────────────────────────────────────

def formation_log(df: pd.DataFrame, cfg: dict) -> list[tuple[date, int]]:
    """Full-history (date, direction) event log: call geometry_feed.fvg_formation_events per
    trading day on df.iloc[:t+1] (REPLICATED kernel — plan's stated seam: 'the board carries
    counts, not events'), keeping only the freshest hit each day (the triplet whose c3 == today;
    at most one triplet can complete on any given day, by construction of the 3-bar kernel)."""
    out: list[tuple[date, int]] = []
    n = len(df)
    for t in range(n):
        sub = df.iloc[: t + 1]
        today = pd.Timestamp(sub.index[-1]).date()
        for d, side in geometry_feed.fvg_formation_events(sub, cfg):
            if d == today:
                out.append((today, side))
    return out


def deoverlap_and_suppress(events: list[tuple[date, int]], trading_index: pd.DatetimeIndex,
                           suppress: int = 5) -> list[tuple[date, int]]:
    """One event per pair-day (safety-net dedupe — formation_log already guarantees this
    structurally) + same-direction suppression: drop a same-side event landing within
    `suppress` TRADING DAYS (calendar positions on the pair's own index, not event-list index)
    of the last accepted same-side event."""
    seen: set = set()
    deduped = []
    for d, side in events:
        if d in seen:
            continue
        seen.add(d)
        deduped.append((d, side))
    pos_of = {pd.Timestamp(d).normalize(): i for i, d in enumerate(trading_index)}
    out: list[tuple[date, int]] = []
    last_pos = {1: -10 ** 9, -1: -10 ** 9}
    for d, side in deduped:
        p = pos_of.get(pd.Timestamp(d).normalize())
        if p is None:
            continue
        if p - last_pos[side] >= suppress:
            out.append((d, side))
            last_pos[side] = p
    return out


def _corr_to_benchmark(events: list, closes_by_pair: dict[str, pd.Series], h: int,
                       bench_ret: pd.Series, index: pd.DatetimeIndex,
                       window: tuple[str, str] | None, min_n: int) -> dict:
    """Event-book daily returns vs one benchmark's daily returns — generalises
    run_positioning_family_options._corr_carry to an arbitrary benchmark series and an optional
    date sub-window (crisis-window rho), and reports n honestly rather than assuming coverage."""
    book = pd.Series(0.0, index=index)
    n_active = pd.Series(0, index=index)
    for e in events:
        closes = closes_by_pair.get(e.pair)
        if closes is None:
            continue
        t0 = es.effective_t0(e.publish_date, closes.index)
        if t0 is None:
            continue
        pos = closes.index.get_indexer([t0])[0]
        seg = closes.iloc[pos:pos + h + 1]
        rets = e.side * np.log(seg / seg.shift(1)).dropna()
        rets = rets.reindex(index).fillna(0.0)
        book = book + rets
        n_active = n_active + (rets != 0).astype(int)
    has_event = n_active > 0
    bench_aligned = bench_ret.reindex(index)
    has_bench = bench_aligned.notna()
    mask = has_event & has_bench
    if window:
        in_window = (index >= pd.Timestamp(window[0])) & (index <= pd.Timestamp(window[1]))
        mask = mask & in_window
    n = int(mask.sum())
    if n < min_n:
        return {"rho": None, "n": n, "note": f"n={n} < min_n={min_n}"}
    book_ret = (book[mask] / n_active[mask])
    rho = float(np.corrcoef(book_ret.to_numpy(), bench_aligned[mask].to_numpy())[0, 1])
    return {"rho": round(rho, 4) if np.isfinite(rho) else None, "n": n}


def diversifier_gate(events: list, closes_by_pair: dict[str, pd.Series], index: pd.DatetimeIndex,
                     h: int, v015_ret: pd.Series, dbv_ret: pd.Series) -> dict:
    """true_diversifier_iff, evaluated independently vs BOTH benchmarks (prereg lists both under
    one joint_success_condition — pass requires the condition to hold vs both, each recorded)."""
    benchmarks = {}
    for name, ret in (("v015", v015_ret), ("dbv", dbv_ret)):
        full = _corr_to_benchmark(events, closes_by_pair, h, ret, index, window=None, min_n=30)
        crises = {}
        for w0, w1 in CRISIS_WINDOWS:
            crises[f"{w0}_{w1}"] = _corr_to_benchmark(events, closes_by_pair, h, ret, index,
                                                       window=(w0, w1), min_n=20)
        crisis_rhos = [c["rho"] for c in crises.values() if c and c.get("rho") is not None]
        max_crisis = max((abs(r) for r in crisis_rhos), default=None)
        full_rho = full.get("rho") if full else None
        passed = bool(full_rho is not None and abs(full_rho) < 0.25
                     and (max_crisis is None or max_crisis < 0.35))
        benchmarks[name] = {"full": full, "crisis_windows": crises,
                            "max_crisis_abs_rho": None if max_crisis is None else round(max_crisis, 4),
                            "pass": passed}
    overall = bool(all(b["pass"] for b in benchmarks.values()))
    return {"benchmarks": benchmarks, "pass": overall,
            "rule": ("abs(full_rho) < 0.25 AND max crisis abs(rho) < 0.35, evaluated "
                    "independently vs v015 daily curve AND vs DBV carry proxy — BOTH must pass")}


def run_beta(closes: dict[str, pd.Series], ohlc: dict[str, pd.DataFrame], v015_eq: pd.Series,
            cfg: dict, rng: np.random.Generator) -> dict:
    events_by_pair: dict[str, list] = {}
    raw_counts: dict[str, int] = {}
    for pair in PAIRS:
        log = formation_log(ohlc[pair], cfg)
        raw_counts[pair] = len(log)
        deo = deoverlap_and_suppress(log, ohlc[pair].index, suppress=5)
        events_by_pair[pair] = [es.Event(pair, ol.shift_pub(d), side, 1.0) for d, side in deo]

    eligible = {}
    for pair in PAIRS:
        all_shifted = [ol.shift_pub(d.date()) for d in closes[pair].index]
        eligible[pair] = es.eligible_dates(closes[pair], all_shifted, 10)

    prim = es.pooled_primary_p(events_by_pair, closes, eligible, 10, rng, N_PERM)
    all_events = [e for evs in events_by_pair.values() for e in evs]
    rets, _kept, dropped = es.signed_returns(all_events, closes, 10)

    v015_ret = v015_eq.pct_change()
    dbv_ret = vrp_data.load_carry_proxy()
    gate = diversifier_gate(all_events, closes, v015_eq.index, 10, v015_ret, dbv_ret)

    n = prim.n_events
    sample_status = "OK" if n >= 50 else "UNDERPOWERED"
    return {
        "hyp": "HYP-083",
        "per_pair_N": {p: len(v) for p, v in events_by_pair.items()},
        "raw_formation_counts_pre_deoverlap": raw_counts,
        "primary": {"h": 10, "statistic_median": None if np.isnan(prim.obs) else round(prim.obs, 6),
                    "raw_p": None if np.isnan(prim.p) else round(prim.p, 5),
                    "n_perm": prim.n_perm, "N": n, "sidedness": "one-sided (continuation)"},
        "medians": {"full_h10": round(float(np.median(rets)), 6) if rets else None},
        "dropped_missing_data": dropped,
        "diversifier_gate": gate,
        "sample_status": sample_status,
        "deviations": [],
    }


# ── ledger + sealing (mirrors run_positioning_family_options.py verbatim) ──────────────────

def _annotate_ledger(hyp_id: str, annotation: dict, verdict: str | None = None) -> str:
    ledger = json.loads(LEDGER.read_text())
    assert isinstance(ledger, list)
    backup = LEDGER.with_suffix(f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    shutil.copy2(LEDGER, backup)
    entry = next(e for e in ledger if e.get("id") == hyp_id)
    assert entry["status"] == "PREREGISTERED", f"{hyp_id} status is {entry['status']} — refusing"
    before = len(entry.get("annotations", []))
    entry.setdefault("annotations", []).append(annotation)
    if verdict is not None:
        entry["verdict"] = verdict
    tmp = tempfile.NamedTemporaryFile("w", dir=LEDGER.parent, delete=False, suffix=".tmp")
    tmp.write(json.dumps(ledger, indent=2) + "\n")
    tmp.close()
    Path(tmp.name).replace(LEDGER)
    check = json.loads(LEDGER.read_text())
    e2 = next(e for e in check if e.get("id") == hyp_id)
    assert len(e2["annotations"]) == before + 1 and e2["status"] == "PREREGISTERED"
    if verdict is not None:
        assert e2["verdict"] == verdict
    return str(backup.name)


def seal(hyp_id: str, res: dict, dry_run: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{hyp_id}.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    prim = res.get("primary", {})
    note = (f"INTERIM SEAL — NO VERDICT (family BH across the 2 GEOMETRY-2026-07 primaries). "
            f"raw_p={prim.get('raw_p')}, N={prim.get('N')}, statistic={prim.get('statistic_ic', prim.get('statistic_median'))}, "
            f"sample_status={res.get('sample_status')}, deviations={res.get('deviations', [])}; "
            f"{COVERAGE_TEXT}; {INTERP_TEXT}; artifacts=data/research/geometry_family/{hyp_id}.json")
    if dry_run:
        print(f"[dry-run] would annotate {hyp_id}: {note[:150]}…")
        return
    backup = _annotate_ledger(hyp_id, {"date": datetime.now(timezone.utc).isoformat(),
                                       "by": SEAL_BY, "note": note})
    print(f"sealed {hyp_id} (ledger backup {backup})")


# ── adjudication ─────────────────────────────────────────────────────────────────────────────

def _member_verdict(hyp: str, bh_pass: bool, underpowered: bool, doc: dict) -> str:
    if underpowered:
        return "UNDERPOWERED"
    if not bh_pass:
        return "NOT_SIGNIFICANT"
    if hyp == "HYP-082":
        gates = doc.get("gates", {})
        if gates.get("fold_sign_consistent") is False:
            return "NOT_ROBUST"
        if not (gates.get("ic_threshold_pass") and gates.get("cost_floor_pass")):
            return "NOT_SIGNIFICANT"
        return "CONFIRMED"
    if hyp == "HYP-083":
        gate = doc.get("diversifier_gate", {})
        if not gate.get("pass"):
            return "NOT_DIVERSIFIER"
        return "CONFIRMED"
    return "NOT_SIGNIFICANT"


def adjudicate(dry_run: bool) -> int:
    sources = {"HYP-082": "HYP-082.json", "HYP-083": "HYP-083.json"}
    ps: dict[str, float] = {}
    docs: dict[str, dict] = {}
    missing = []
    for hyp, fname in sources.items():
        f = OUT / fname
        if not f.exists():
            missing.append(f"{hyp} ({fname} absent)")
            continue
        doc = json.loads(f.read_text())
        p = doc.get("primary", {}).get("raw_p")
        if p is None:
            missing.append(f"{hyp} (primary p null — {doc.get('sample_status')})")
            continue
        ps[hyp] = float(p)
        docs[hyp] = doc
    if missing:
        print("FAMILY BH REFUSED — not all 2 primaries exist:", "; ".join(missing))
        return 1

    m, alpha = len(ps), 0.05
    ranked = sorted(ps.items(), key=lambda kv: kv[1])
    kmax = 0
    for i, (_hyp, p) in enumerate(ranked, 1):
        if p <= i / m * alpha:
            kmax = i

    table = []
    for i, (hyp, p) in enumerate(ranked, 1):
        bh_pass = i <= kmax
        doc = docs[hyp]
        n = doc.get("primary", {}).get("N")
        underpowered = (n is not None and n < 50) or doc.get("sample_status") == "UNDERPOWERED"
        verdict = _member_verdict(hyp, bh_pass, underpowered, doc)
        table.append({"hyp": hyp, "raw_p": p, "rank": i, "bh_threshold": round(i / m * alpha, 5),
                      "bh_pass": bh_pass, "verdict": verdict})
    print(json.dumps(table, indent=2))
    if dry_run:
        print("[dry-run] no verdict annotations written")
        return 0

    ts = datetime.now(timezone.utc).isoformat()
    for row in table:
        note = (f"FAMILY VERDICT ({FAMILY}, BH alpha=0.05, m=2): raw_p={row['raw_p']}, "
                f"rank={row['rank']}, threshold={row['bh_threshold']}, bh_pass={row['bh_pass']} "
                f"-> {row['verdict']}. {COVERAGE_TEXT} kill-criterion (manifest): nulls here close "
                f"the geometry thread at daily resolution — prior explorations never re-enter as "
                f"evidence (A1).")
        _annotate_ledger(row["hyp"], {"date": ts, "by": f"{FAMILY}/adjudication", "note": note},
                         verdict=row["verdict"])
        print(f"verdict annotated: {row['hyp']} -> {row['verdict']}")
        if row["verdict"] == "CONFIRMED":
            print(f"E4 PROTOCOL: {row['hyp']} is a CONFIRMED-class survivor — ledger + report "
                  f"only. Recommended next command: python -m factory.train --hyp {row['hyp']} "
                  f"(NOT run by this script — halting per plan Phase 2 Adjudication).")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--adjudicate", action="store_true", help="family BH (both primaries required)")
    a = ap.parse_args()

    checks = gate_zero()
    print("gate zero: all 4 geometry prereg hashes verified", all(v["ok"] for v in checks.values()))
    if a.adjudicate:
        return adjudicate(a.dry_run)

    from config.loader import params
    cfg = dict(params["sentiment"]["geometry"])

    ohlc = {p: load_ohlc(p) for p in PAIRS}
    closes = {p: ohlc[p]["Close"] for p in PAIRS}
    trades = vr.load_trades()
    got = vr.reconcile_guard(trades, closes)
    print(f"reconcile guard: 0.6886 -> {got}")
    v015_eq = vr.daily_portfolio_equity(trades, closes["EURUSD"].index)

    geo = load_geometry(PAIRS)
    rng = np.random.default_rng(SEED)

    res_082 = run_gamma(geo, closes, v015_eq, rng)
    res_083 = run_beta(closes, ohlc, v015_eq, cfg, np.random.default_rng(SEED))

    manifest = {"run_ts": datetime.now(timezone.utc).isoformat(), "seed": SEED, "n_perm": N_PERM,
                "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip(),
                "gate_zero": checks, "interpretations": INTERP_TEXT, "coverage": COVERAGE_TEXT,
                "reconcile_guard": {"target": 0.6886, "recomputed": got}, "hyp_order": ["HYP-082", "HYP-083"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    seal("HYP-082", res_082, a.dry_run)
    seal("HYP-083", res_083, a.dry_run)

    print("\n== geometry family interim summary (NO verdicts until --adjudicate) ==")
    for hyp_id, res in (("HYP-082", res_082), ("HYP-083", res_083)):
        prim = res.get("primary") or {}
        print(f"{hyp_id}: raw_p={prim.get('raw_p')} N={prim.get('N')} [{res.get('sample_status')}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
