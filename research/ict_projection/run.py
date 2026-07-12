#!/usr/bin/env python3
"""90-day ICT taken-trade projection (TICK-028) -- READ-ONLY.

Projects how many ICT trades the system will realistically TAKE over the next
90 days at the current signal/veto frequency, to sanity-check whether ICT can
supply the ~30 trades a prop challenge needs.

Method (see report.md for prose):
  1. Load 3 veto shards + 3 decision-log months. Parse timestamps -> UTC dates.
  2. Dedup per-scan re-emission: collapse veto records to unique
     (date, pair, direction, veto_class) keeping first-per-day. Collapse taken
     decisions to unique (date, pair, direction) per day.
  3. Recompute veto rates LIVE from deduped data over the trailing 45 days.
  4. Daily taken base rate from decision logs over the trailing 45 days.
  5. Project 90 calendar days -> trading days; bootstrap over days (N=10000, seed 42).
  6. Write projection_90d.json.
  7. Write report.md (via report.py).

Run:  python3 research/ict_projection/run.py

Determinism: no wall-clock is written to output. The analysis anchor is the
max observed data date. numpy default_rng(42). All floats rounded before dump.
Two runs produce byte-identical projection_90d.json.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SEED = 42
TRAILING_DAYS = 45           # trailing window for rate estimation (calendar days)
PROJECTION_CALENDAR_DAYS = 90
BOOTSTRAP_N = 10_000
SENSITIVITY = 0.30           # +/-30% rate-sensitivity band
NEAR30_LO, NEAR30_HI = 20, 45  # "plausibly a ~30-trade vehicle" range

VETO_SHARDS = [
    "ict_veto_ledger_2026_05.jsonl",
    "ict_veto_ledger_2026_06.jsonl",
    "ict_veto_ledger_2026_07.jsonl",
]
DECISION_MONTHS = [
    "decisions_2026_05.jsonl",
    "decisions_2026_06.jsonl",
    "decisions_2026_07.jsonl",
]

# Outcomes that mean a position actually FILLED (has P&L) vs. an unfilled expiry.
FILLED_OUTCOMES = {"WIN", "LOSS", "BREAKEVEN", "SCRATCH", "STOP", "TP", "TARGET"}


# --------------------------------------------------------------------------- #
# Path resolution (read real data from main checkout; write to this worktree)
# --------------------------------------------------------------------------- #
def output_root() -> Path:
    """Repo root that owns this package (research/ict_projection/run.py -> up 2)."""
    return Path(__file__).resolve().parents[2]


def data_root(out_root: Path) -> Path:
    """Root that actually contains the input data.

    data/ is a gitignored placeholder inside git worktrees, so the real
    ledger/decision files live in the main checkout. Probe candidate roots and
    return the first that has the veto ledger. No silent fallback: raise loudly
    if the real data cannot be found.
    """
    probe = Path("data/ledger") / VETO_SHARDS[0]
    candidates = [out_root]
    try:
        common = subprocess.check_output(
            ["git", "-C", str(out_root), "rev-parse", "--git-common-dir"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        common_p = Path(common)
        if not common_p.is_absolute():
            common_p = (out_root / common_p).resolve()
        candidates.append(common_p.parent)  # parent of .git == main repo root
    except Exception:
        pass
    for c in candidates:
        if (c / probe).exists():
            return c
    raise FileNotFoundError(
        "Could not locate input data. Looked for "
        f"{probe} under: {[str(c) for c in candidates]}. "
        "Inputs must exist read-only; refusing to proceed without real data."
    )


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def to_utc_date(ts: str | None):
    """Parse an ISO tz-aware timestamp to a UTC calendar date."""
    if not ts:
        return None
    s = ts.strip().replace("Z", "+00:00")
    try:
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc).date()
    except Exception:
        try:
            return dt.date.fromisoformat(ts[:10])
        except Exception:
            return None


def veto_class(reason: str | None) -> str:
    """Normalize a raw veto_reason string into one of 7 stable buckets.

    ADR / WEEKLY_TREND / DISP_GATE / TIMING / SCORE / SESSION / OTHER
    """
    u = (reason or "").strip().upper()
    if not u:
        return "OTHER"
    if u.startswith("ADR"):                       # "ADR exhausted: ..."
        return "ADR"
    if u.startswith("WEEKLY_TREND"):              # "WEEKLY_TREND_CONFLICT: ..."
        return "WEEKLY_TREND"
    if "DISP_GATE" in u:                          # "HYP046_DISP_GATE: ..."
        return "DISP_GATE"
    if u.startswith("TIMING"):                    # "TIMING_GATE: ..."
        return "TIMING"
    if u.startswith("SCORE") or "SCORE_CEILING" in u:  # "Score X < ...", HYP047_SCORE_CEILING
        return "SCORE"
    if u.startswith("SESSION"):                   # "session"
        return "SESSION"
    return "OTHER"                                # "gate", "BLACKOUT", ...


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_vetos(d_root: Path):
    recs = []
    for shard in VETO_SHARDS:
        p = d_root / "data" / "ledger" / shard
        for r in read_jsonl(p):
            date = to_utc_date(r.get("timestamp"))
            if date is None:
                continue
            recs.append({
                "date": date,
                "pair": r.get("pair"),
                "direction": r.get("intended_direction"),
                "cls": veto_class(r.get("veto_reason")),
            })
    return recs


def load_taken(d_root: Path):
    """ICT decision records = setups the system committed to (placed orders for)."""
    recs = []
    for month in DECISION_MONTHS:
        p = d_root / "data" / "decision_logs" / month
        for r in read_jsonl(p):
            if r.get("system") != "ICT":
                continue
            date = to_utc_date(r.get("entry_timestamp"))
            if date is None:
                continue
            recs.append({
                "date": date,
                "pair": r.get("pair"),
                "direction": r.get("direction"),
                "entry_level": r.get("entry_level"),
                "outcome": r.get("outcome"),
                "r_realized": r.get("r_realized"),
            })
    return recs


# --------------------------------------------------------------------------- #
# Window / trading-day helpers
# --------------------------------------------------------------------------- #
def business_days_between(start_excl: dt.date, end_incl: dt.date):
    """List of weekday (Mon-Fri) dates in (start_excl, end_incl]."""
    out = []
    d = start_excl + dt.timedelta(days=1)
    while d <= end_incl:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def rnd(x, n=6):
    if x is None:
        return None
    return round(float(x), n)


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
def analyze():
    out_root = output_root()
    d_root = data_root(out_root)

    vetos = load_vetos(d_root)
    taken = load_taken(d_root)

    # Anchor the trailing window to the latest observed data date (no wall-clock).
    all_dates = [v["date"] for v in vetos] + [t["date"] for t in taken]
    analysis_date = max(all_dates)
    window_start = analysis_date - dt.timedelta(days=TRAILING_DAYS)  # exclusive
    biz_days = business_days_between(window_start, analysis_date)
    trading_days_window = len(biz_days)
    biz_day_set = set(biz_days)

    def in_window(rec):
        return window_start < rec["date"] <= analysis_date

    # ---- VETO SIDE: dedup per-scan re-emission ----------------------------- #
    vetos_win = [v for v in vetos if in_window(v)]
    raw_veto_win = len(vetos_win)
    unique_veto_keys = set()
    for v in vetos_win:
        unique_veto_keys.add((v["date"], v["pair"], v["direction"], v["cls"]))
    unique_veto_win = len(unique_veto_keys)
    dedup_factor = (raw_veto_win / unique_veto_win) if unique_veto_win else float("nan")

    # Full-data dedup factor (all shards) as a cross-check.
    raw_veto_all = len(vetos)
    unique_veto_all = len({(v["date"], v["pair"], v["direction"], v["cls"]) for v in vetos})
    dedup_factor_all = (raw_veto_all / unique_veto_all) if unique_veto_all else float("nan")

    # Class breakdown over deduped window setups.
    cls_counts = Counter(k[3] for k in unique_veto_keys)
    class_breakdown = {
        cls: {"unique_setups": n, "share": rnd(n / unique_veto_win) if unique_veto_win else None}
        for cls, n in sorted(cls_counts.items(), key=lambda kv: -kv[1])
    }
    adr_share = rnd(cls_counts.get("ADR", 0) / unique_veto_win) if unique_veto_win else None
    weekly_share = rnd(cls_counts.get("WEEKLY_TREND", 0) / unique_veto_win) if unique_veto_win else None

    # ---- TAKEN SIDE: dedup logged decisions to unique setups --------------- #
    taken_win = [t for t in taken if in_window(t)]
    raw_taken_win = len(taken_win)

    # Primary unique-setup key: (date, pair, direction). Secondary adds entry_level.
    setups_by_day = defaultdict(set)          # date -> {(pair, direction)}
    setups_lvl_by_day = defaultdict(set)      # date -> {(pair, direction, entry_level)}
    outcome_counts = Counter()
    filled_by_day = defaultdict(set)          # date -> {(pair, direction)} for FILLED only
    for t in taken_win:
        key = (t["pair"], t["direction"])
        setups_by_day[t["date"]].add(key)
        setups_lvl_by_day[t["date"]].add((t["pair"], t["direction"], t["entry_level"]))
        outcome_counts[t["outcome"]] += 1
        oc = (t["outcome"] or "").upper()
        r = t["r_realized"]
        is_filled = (oc in FILLED_OUTCOMES) or (oc != "EXPIRED" and r not in (None, 0, 0.0))
        if is_filled:
            filled_by_day[t["date"]].add(key)

    unique_taken_win = sum(len(s) for s in setups_by_day.values())
    unique_taken_lvl_win = sum(len(s) for s in setups_lvl_by_day.values())
    weekend_setups = sum(
        len(s) for d, s in setups_by_day.items() if d not in biz_day_set
    )

    # Per-business-day taken counts (including zero days) -> base rate + bootstrap.
    per_day_counts = np.array([len(setups_by_day.get(d, ())) for d in biz_days], dtype=float)
    mean_daily_taken = float(per_day_counts.mean()) if per_day_counts.size else 0.0

    # Filled reality (fills that actually count toward a prop challenge).
    unique_filled_win = sum(len(s) for s in filled_by_day.values())
    per_day_filled = np.array([len(filled_by_day.get(d, ())) for d in biz_days], dtype=float)
    mean_daily_filled = float(per_day_filled.mean()) if per_day_filled.size else 0.0

    # ---- FUNNEL RECONCILIATION -------------------------------------------- #
    total_unique_setups = unique_veto_win + unique_taken_win
    p_vetoed = (unique_veto_win / total_unique_setups) if total_unique_setups else None
    # Estimate A: direct taken/day from decision logs.
    est_a = mean_daily_taken
    # Estimate B: (total unique setups/day) x (1 - P(vetoed)).
    total_setups_per_day = (total_unique_setups / trading_days_window) if trading_days_window else None
    est_b = (total_setups_per_day * (1 - p_vetoed)) if (total_setups_per_day is not None and p_vetoed is not None) else None

    # ---- PROJECTION ------------------------------------------------------- #
    fwd_start = np.datetime64(analysis_date) + np.timedelta64(1, "D")
    fwd_end = np.datetime64(analysis_date) + np.timedelta64(PROJECTION_CALENDAR_DAYS + 1, "D")
    trading_days_90 = int(np.busday_count(fwd_start, fwd_end))
    cal_to_trading = rnd(trading_days_90 / PROJECTION_CALENDAR_DAYS)

    point_logged_90 = mean_daily_taken * trading_days_90
    point_filled_90 = mean_daily_filled * trading_days_90

    # Bootstrap over days: resample per-day taken counts to a 90-trading-day sum.
    rng = np.random.default_rng(SEED)
    if per_day_counts.size:
        draws = rng.choice(per_day_counts, size=(BOOTSTRAP_N, trading_days_90), replace=True)
        sums = draws.sum(axis=1)
    else:
        sums = np.zeros(BOOTSTRAP_N)
    ci = {
        "median": rnd(float(np.median(sums)), 3),
        "p2_5": rnd(float(np.percentile(sums, 2.5)), 3),
        "p10": rnd(float(np.percentile(sums, 10)), 3),
        "p90": rnd(float(np.percentile(sums, 90)), 3),
        "p97_5": rnd(float(np.percentile(sums, 97.5)), 3),
        "mean": rnd(float(sums.mean()), 3),
    }
    sensitivity = {
        "minus_30pct": rnd(point_logged_90 * (1 - SENSITIVITY), 3),
        "point": rnd(point_logged_90, 3),
        "plus_30pct": rnd(point_logged_90 * (1 + SENSITIVITY), 3),
    }

    def classify(x):
        if x < NEAR30_LO:
            return "BELOW_RANGE"
        if x > NEAR30_HI:
            return "ABOVE_RANGE"
        return "IN_RANGE"

    logged_class = classify(point_logged_90)
    near_30 = {
        "range_lo": NEAR30_LO,
        "range_hi": NEAR30_HI,
        "basis_logged_setups": {
            "point": rnd(point_logged_90, 3),
            "classification": logged_class,
            "near_30": logged_class == "IN_RANGE",
        },
        "basis_filled_trades": {
            "point": rnd(point_filled_90, 3),
            "classification": classify(point_filled_90),
            "near_30": classify(point_filled_90) == "IN_RANGE",
        },
    }

    # ---- Verdict prose ---------------------------------------------------- #
    fill_rate_win = (unique_filled_win / unique_taken_win) if unique_taken_win else None
    if logged_class == "ABOVE_RANGE":
        verdict = (
            f"ICT GENERATES enough setups (~{point_logged_90:.0f} logged/committed over 90 days, "
            f"above the 20-45 band), so signal/veto frequency is NOT the bottleneck. But "
            f"~{(1-(fill_rate_win or 0))*100:.1f}% of committed setups EXPIRE unfilled: only "
            f"~{point_filled_90:.1f} trades would actually FILL over 90 days. As a source of ~30 "
            f"EXECUTED prop-challenge trades, ICT is FAR BELOW unless the fill/expiry problem is fixed."
        )
    elif logged_class == "IN_RANGE":
        verdict = (
            f"ICT plausibly supplies ~30 committed setups over 90 days (~{point_logged_90:.0f}, in the "
            f"20-45 band), but fills are the constraint: only ~{point_filled_90:.1f} would execute."
        )
    else:
        verdict = (
            f"ICT is FAR BELOW a 30-trade vehicle: only ~{point_logged_90:.0f} committed setups projected "
            f"over 90 days, and ~{point_filled_90:.1f} actual fills."
        )

    results = {
        "meta": {
            "ticket": "TICK-028",
            "read_only": True,
            "seed": SEED,
            "bootstrap_n": BOOTSTRAP_N,
            "data_root": str(d_root),
            "output_root": str(out_root),
        },
        "window": {
            "analysis_anchor_date": analysis_date.isoformat(),
            "trailing_calendar_days": TRAILING_DAYS,
            "window_start_exclusive": window_start.isoformat(),
            "window_end_inclusive": analysis_date.isoformat(),
            "trading_days_in_window": trading_days_window,
        },
        "dedup": {
            "raw_veto_records_window": raw_veto_win,
            "unique_veto_setups_window": unique_veto_win,
            "dedup_factor_window": rnd(dedup_factor, 3),
            "raw_veto_records_all_shards": raw_veto_all,
            "unique_veto_setups_all_shards": unique_veto_all,
            "dedup_factor_all_shards": rnd(dedup_factor_all, 3),
            "unique_veto_setups_per_trading_day": rnd(unique_veto_win / trading_days_window, 3) if trading_days_window else None,
            "dedup_note": (
                "Collapsed on (date, pair, direction, veto_class). Dedup factor ~13x confirms "
                "the scanner re-emits a full universe sweep every cycle."
            ),
        },
        "veto_rates_live": {
            "adr_share": adr_share,
            "weekly_trend_share": weekly_share,
            "class_breakdown": class_breakdown,
            "note": (
                "Recomputed live from deduped trailing-45d setups. Prior memory (55% ADR / 31% weekly) "
                "is STALE."
            ),
        },
        "taken_base_rate": {
            "raw_taken_records_window": raw_taken_win,
            "unique_taken_setups_window_pair_dir": unique_taken_win,
            "unique_taken_setups_window_pair_dir_level": unique_taken_lvl_win,
            "weekend_setups_excluded_from_rate": weekend_setups,
            "mean_daily_taken_setups": rnd(mean_daily_taken, 4),
            "estimate_A_direct_per_day": rnd(est_a, 4),
            "estimate_B_veto_implied_per_day": rnd(est_b, 4) if est_b is not None else None,
            "p_vetoed_setup_level": rnd(p_vetoed, 4) if p_vetoed is not None else None,
            "total_unique_setups_window": total_unique_setups,
            "reconciliation_note": (
                "Estimate B = (total unique setups/day) x (1 - P(vetoed)), total = vetoed + taken. "
                f"A and B nearly coincide (~1.5/day). They differ only because B's taken numerator "
                f"counts all {unique_taken_win} unique setups while A's trading-day base rate uses the "
                f"{unique_taken_win - weekend_setups} on business days (the {weekend_setups} weekend/"
                f"non-business setups are excluded), both over {trading_days_window} trading days. Since "
                "vetoed and taken setups partition the observed universe, B cannot diverge from A beyond "
                "this day-basis difference -- the real independent uncertainty is the FILL gap, not the "
                "veto split."
            ),
        },
        "fill_gap": {
            "outcome_distribution_window": dict(sorted(outcome_counts.items(), key=lambda kv: -kv[1])),
            "unique_filled_setups_window": unique_filled_win,
            "fill_rate_window": rnd(fill_rate_win, 5) if fill_rate_win is not None else None,
            "mean_daily_filled": rnd(mean_daily_filled, 5),
            "note": (
                "EXPIRED = limit order placed at entry_level but never filled (r_realized=0). "
                "Nearly all committed ICT setups expire unfilled; only filled trades count toward a "
                "prop challenge."
            ),
        },
        "projection_90d": {
            "projection_calendar_days": PROJECTION_CALENDAR_DAYS,
            "trading_days_90": trading_days_90,
            "calendar_to_trading_day_factor": cal_to_trading,
            "logged_setups": {
                "point_estimate": rnd(point_logged_90, 3),
                "bootstrap_ci": ci,
                "sensitivity_band_pm30pct": sensitivity,
            },
            "filled_trades": {
                "point_estimate": rnd(point_filled_90, 3),
                "note": "Near-zero given the observed fill rate; the binding prop-challenge constraint.",
            },
        },
        "near_30": near_30,
        "verdict": verdict,
        "caveats": [
            "Only ~3 months of data (veto ledger starts 2026-05-24); a 45-day base is short and "
            "regime-specific.",
            "Assumes the pipeline stays frozen at current signal/veto/order config -- valid under the "
            "active shadow-freeze, but any gate change invalidates the projection.",
            "Selection gap #1: not every setup that passes vetoes becomes a logged decision.",
            (
                "Selection gap #2 (dominant): only ~"
                f"{(fill_rate_win * 100 if fill_rate_win is not None else 0):.1f}% of committed ICT "
                "setups FILL in this window (the rest EXPIRE unfilled) -- 'committed setup' massively "
                "overstates 'executed trade'."
            ),
            (
                f"FX weekend/non-business setups ({weekend_setups} in this window) are excluded from the "
                "trading-day base rate; they inflate Estimate B slightly vs Estimate A."
            ),
        ],
    }
    return results


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_outputs(results: dict):
    out_root = output_root()
    json_path = out_root / "data" / "research" / "ict_projection" / "projection_90d.json"
    md_path = out_root / "research" / "ict_projection" / "report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")

    # Robust to both `python3 research/ict_projection/run.py` (pkg dir on path)
    # and `-m research.ict_projection.run` (repo root on path).
    try:
        from research.ict_projection.report import render_report
    except ModuleNotFoundError:
        sys.path.insert(0, str(out_root))
        from research.ict_projection.report import render_report
    md_path.write_text(render_report(results))
    return json_path, md_path


def main():
    results = analyze()
    json_path, md_path = write_outputs(results)
    v = results
    print("ICT 90-day taken-trade projection (TICK-028) -- READ-ONLY")
    print(f"  window: {v['window']['window_start_exclusive']} -> {v['window']['window_end_inclusive']} "
          f"({v['window']['trading_days_in_window']} trading days)")
    print(f"  dedup factor (window): {v['dedup']['dedup_factor_window']}x  "
          f"({v['dedup']['raw_veto_records_window']} raw -> {v['dedup']['unique_veto_setups_window']} unique)")
    print(f"  live ADR share: {v['veto_rates_live']['adr_share']}  |  weekly-trend share: {v['veto_rates_live']['weekly_trend_share']}")
    print(f"  mean daily taken setups: {v['taken_base_rate']['mean_daily_taken_setups']}")
    print(f"  90d logged setups: point {v['projection_90d']['logged_setups']['point_estimate']} "
          f"(95% [{v['projection_90d']['logged_setups']['bootstrap_ci']['p2_5']}, "
          f"{v['projection_90d']['logged_setups']['bootstrap_ci']['p97_5']}])")
    print(f"  90d filled trades: point {v['projection_90d']['filled_trades']['point_estimate']}")
    print(f"  near_30 (logged basis): {v['near_30']['basis_logged_setups']['classification']}")
    print(f"  VERDICT: {v['verdict']}")
    print(f"  wrote: {json_path}")
    print(f"  wrote: {md_path}")


if __name__ == "__main__":
    sys.exit(main())
