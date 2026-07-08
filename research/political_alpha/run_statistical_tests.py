"""Phase 4 — pre-registered statistical tests + charts + summary (HYP-085, §7-P4).

EXACTLY three tests, no others (spec §10 forbids adding BH/permutation/CAR here):
  1. Normality: QQ-plot of pooled standardized pre-announcement returns vs N(0,1)
     + Shapiro-Wilk; direction-aligned skew (down-events sign-flipped — positive
     skew = pre-drift toward the eventual announcement direction).
  2. SD exceedance: observed big-move rate across evaluable event rows vs what the
     baseline predicts (normal-theory two-day reference + the placebo mean).
  3. Bootstrap null (THE decision test): 10,000 statement-level placebo sets —
     one random eligible weekday (pinned 12:00 ET) per real evaluable statement,
     applied to all its instrument rows via the identical T0 mapping; eligibility
     excludes ±5 trading days around any real event T0 on those instruments and
     the pre-listed FOMC/CPI/NFP dates. p = (n_ge + 1) / (N + 1), one-sided
     (H1 = more exceedances). Seed 42. All locked in the HYP-085 prereg.

Outputs: output/normality_plot.png (QQ + null-distribution panel),
         output/sd_test_results.json, output/summary_report.md.

Run:  python3 research/political_alpha/run_statistical_tests.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402
from classification_rules import SCHEDULED_MACRO_DATES, STUDY_START  # noqa: E402

N_BOOT = 10_000
SEED = 42
MIN_POS = 61          # >=60 prior returns so sigma60 exists (returns[0] is NaN)


# ── per-instrument market state ──────────────────────────────────────────────────────

class Market:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.px = _lib.fetch_daily(ticker)
        self.ok = not self.px.empty
        if self.ok:
            self.index = self.px.index
            r = _lib.log_returns(self.px["Close"])
            sig = _lib.trailing_sigma60(r)
            with np.errstate(invalid="ignore"):
                self.big_at = (r.abs() > 2.0 * sig).to_numpy()
            self.valid_at = (r.notna() & sig.notna() & (sig > 0)).to_numpy()
            self.dates = [str(d.date()) for d in self.index]

    def t0_pos(self, ts_utc: str) -> int | None:
        t0 = _lib.map_t0(ts_utc, self.index, _lib.ASSET_CLASS[self.ticker])
        return None if t0 is None else int(self.index.get_loc(t0))

    def row_eval(self, pos: int) -> tuple[bool, bool]:
        """(evaluable, big_move) for a T0 position — identical rule to Phase 2."""
        if pos < MIN_POS or pos + 1 >= len(self.index):
            return False, False
        if not (self.valid_at[pos] and self.valid_at[pos + 1]):
            return False, False
        return True, bool(self.big_at[pos] or self.big_at[pos + 1])


def main() -> int:
    events = _lib.read_jsonl(_lib.DATA_DIR / "trump_events.jsonl")
    study = _lib.read_jsonl(_lib.DATA_DIR / "event_study_results.jsonl")
    flags = _lib.read_jsonl(_lib.DATA_DIR / "manipulation_flags.jsonl")
    if not events or not study:
        print("STOP: run Phases 1-2 first (Phase 3 flags are reported if present).")
        return 2

    mkts = {t: Market(t) for t in _lib.UNIVERSE}
    study_by_key = {(r["event_id"], r["instrument_tagged"]): r for r in study}

    # ── observed statistic (from Phase-2 rows; identical rule re-derived here) ──
    eval_rows = [r for r in study if r["big_move"] is not None]
    n_big = sum(1 for r in eval_rows if r["big_move"])
    observed_rate = n_big / len(eval_rows) if eval_rows else float("nan")

    # statements with >=1 evaluable row, with their instrument lists
    stmts: dict[str, list[str]] = {}
    for r in eval_rows:
        stmts.setdefault(r["event_id"], []).append(r["instrument_tagged"])
    print(f"Phase 4 — observed: {n_big}/{len(eval_rows)} rows big_move "
          f"({observed_rate:.3%}) across {len(stmts)} statements")

    # real-event T0 positions per instrument (exclusion zones for placebos)
    real_pos: dict[str, list[int]] = {t: [] for t in _lib.UNIVERSE}
    for r in study:
        m = mkts[r["instrument_tagged"]]
        if m.ok and r.get("t0"):
            d = pd.Timestamp(r["t0"])
            if d in m.index:
                real_pos[r["instrument_tagged"]].append(int(m.index.get_loc(d)))

    macro = set(SCHEDULED_MACRO_DATES)

    # ── eligible placebo dates per unique instrument-combo ──
    last_dates = [m.index[-2] for m in mkts.values() if m.ok]
    horizon_end = min(last_dates) if last_dates else None
    weekdays = pd.bdate_range(STUDY_START, horizon_end) if horizon_end is not None else []

    def eligible_dates(combo: tuple[str, ...]) -> list[str]:
        out = []
        for d in weekdays:
            ts = f"{d.date()}T16:00:00Z"          # 12:00 ET == 16:00/17:00 UTC; UTC date == d
            ok = True
            for inst in combo:
                m = mkts[inst]
                if not m.ok:
                    ok = False
                    break
                pos = m.t0_pos(ts)
                if pos is None:
                    ok = False
                    break
                evaluable, _ = m.row_eval(pos)
                if not evaluable:
                    ok = False
                    break
                if any(abs(pos - rp) < 5 for rp in real_pos[inst]):
                    ok = False
                    break
                if m.dates[pos] in macro or m.dates[pos + 1] in macro:
                    ok = False
                    break
            if ok:
                out.append(str(d.date()))
        return out

    combos: dict[tuple[str, ...], list[str]] = {}
    stmt_combo: dict[str, tuple[str, ...]] = {}
    for eid, insts in stmts.items():
        combo = tuple(sorted(set(insts)))
        stmt_combo[eid] = combo
        if combo not in combos:
            combos[combo] = eligible_dates(combo)
    placeable = {eid for eid, c in stmt_combo.items() if combos[c]}
    skipped_stmts = sorted(set(stmts) - placeable)
    if skipped_stmts:
        print(f"  ⚠️ {len(skipped_stmts)} statements have zero eligible placebo dates "
              f"and are excluded from the null: {skipped_stmts}")
    print(f"  eligible-date pools: " +
          ", ".join(f"{'+'.join(c)}:{len(v)}" for c, v in combos.items()))

    # ── bootstrap ──
    rng = np.random.default_rng(SEED)
    placebo_rates = np.empty(N_BOOT)
    stmt_list = [(eid, stmt_combo[eid], stmts[eid]) for eid in sorted(placeable)]
    for b in range(N_BOOT):
        hits = total = 0
        for eid, combo, insts in stmt_list:
            pool = combos[combo]
            day = pool[rng.integers(len(pool))]
            ts = f"{day}T16:00:00Z"
            for inst in insts:                      # duplicates preserved: row-level stat
                m = mkts[inst]
                pos = m.t0_pos(ts)
                evaluable, big = m.row_eval(pos) if pos is not None else (False, False)
                if evaluable:
                    total += 1
                    hits += int(big)
        placebo_rates[b] = hits / total if total else np.nan
    placebo_rates = placebo_rates[np.isfinite(placebo_rates)]
    n_ge = int((placebo_rates >= observed_rate).sum())
    p_value = (n_ge + 1) / (len(placebo_rates) + 1)

    # ── Test 1: normality of pooled standardized pre-announcement returns ──
    pre_std, pre_aligned = [], []
    for r in eval_rows:
        m = mkts[r["instrument_tagged"]]
        if not m.ok or not r.get("t0"):
            continue
        d = pd.Timestamp(r["t0"])
        if d not in m.index:
            continue
        pos = int(m.index.get_loc(d))
        rr = _lib.log_returns(m.px["Close"])
        mu, sg = r["est_mu"], r["est_sigma"]
        for back in (2, 1):                         # the 2 pre-window trading days
            if pos - back >= 1 and np.isfinite(rr.iloc[pos - back]):
                z = (float(rr.iloc[pos - back]) - mu) / sg
                pre_std.append(z)
                if r["direction"] in ("up", "down"):
                    pre_aligned.append(z if r["direction"] == "up" else -z)
    pre_std = np.array(pre_std)
    from scipy import stats as sps
    shapiro_stat, shapiro_p = (sps.shapiro(pre_std) if len(pre_std) >= 3 else (np.nan, np.nan))
    aligned_skew = float(sps.skew(np.array(pre_aligned))) if len(pre_aligned) >= 3 else float("nan")

    # ── charts (one figure, two panels: QQ + null distribution) ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if len(pre_std) >= 3:
        sps.probplot(pre_std, dist="norm", plot=ax1)
    ax1.set_title(f"QQ: standardized pre-announcement returns (n={len(pre_std)})\n"
                  f"Shapiro-Wilk p={shapiro_p:.4f}  aligned skew={aligned_skew:+.3f}")
    ax2.hist(placebo_rates, bins=50, color="#8899aa", edgecolor="white")
    ax2.axvline(observed_rate, color="crimson", lw=2,
                label=f"observed {observed_rate:.3%}")
    ax2.set_title(f"Bootstrap null: big-move exceedance rate "
                  f"({len(placebo_rates):,} placebo sets)\np = {p_value:.4f} (one-sided)")
    ax2.set_xlabel("exceedance rate")
    ax2.legend()
    fig.tight_layout()
    _lib.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_lib.OUTPUT_DIR / "normality_plot.png", dpi=150)
    plt.close(fig)

    # ── results json ──
    n_events_catalog = len({r["event_id"] for r in events})
    n_flags = sum(1 for f in flags if f.get("manipulation_signal"))
    n_pos_avail = sum(1 for f in flags if f.get("positioning_available"))
    results = {
        "hypothesis": "HYP-085",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observed_rate": round(float(observed_rate), 6),
        "n_rows_evaluable": len(eval_rows),
        "n_big_move": n_big,
        "n_statements_in_null": len(placeable),
        "n_statements_skipped_no_placebo": len(skipped_stmts),
        "placebo_mean": round(float(placebo_rates.mean()), 6),
        "placebo_std": round(float(placebo_rates.std(ddof=1)), 6),
        "placebo_p95": round(float(np.percentile(placebo_rates, 95)), 6),
        "normal_theory_two_day_reference": round(1 - (1 - 0.0455) ** 2, 6),
        "p_value": round(float(p_value), 6),
        "n_boot": int(len(placebo_rates)),
        "seed": SEED,
        "shapiro_stat": round(float(shapiro_stat), 6) if np.isfinite(shapiro_stat) else None,
        "shapiro_p": round(float(shapiro_p), 6) if np.isfinite(shapiro_p) else None,
        "aligned_skew": round(aligned_skew, 6) if np.isfinite(aligned_skew) else None,
        "n_pre_returns": int(len(pre_std)),
        "null_rejected": bool(p_value < 0.05),
        "catalog_events": n_events_catalog,
        "catalog_shortfall_vs_30": max(0, 30 - n_events_catalog),
        "manipulation_signals": n_flags,
        "positioning_available_rows": n_pos_avail,
        "notes": "hourly not used — daily mapping per spec §6 default; "
                 "exactly the 3 pre-registered tests were run",
    }
    (_lib.OUTPUT_DIR / "sd_test_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # ── summary report ──
    verdict = ("**H0 REJECTED** (p = {p:.4f} < 0.05) — a CANDIDATE RESULT only: promotion to "
               "an edge requires the standard discovery gauntlet as a separate ledgered step "
               "(spec §10)." if results["null_rejected"] else
               "**H0 NOT REJECTED** (p = {p:.4f} ≥ 0.05) — the pre-registered prior "
               "(NOT_SIGNIFICANT). A null result is a real, publishable result.").format(p=p_value)
    gaps2 = pd.Series([r["gap_reason"] for r in study if r["gap_reason"]]).value_counts().to_dict()
    gaps3 = pd.Series([f["gap_reason"] for f in flags if f.get("gap_reason")]).value_counts().to_dict()
    per_source = pd.Series([e["source"] for e in events]).value_counts().to_dict()
    shortfall = ""
    if n_events_catalog < 30:
        shortfall = (f"\n> ⚠️ **SAMPLE-SIZE LIMITATION:** only {n_events_catalog} qualifying "
                     "events (< 30). The definition was NOT loosened to reach the count "
                     "(honesty gate, spec §7/§9).\n")
    report = f"""# Political-Alpha Summary Report — HYP-085

Generated {results['generated_utc']} · pre-registration
`data/research/preregister/HYP-085_political_alpha_trump_events.json` (hash-locked
BEFORE data collection) · governing spec: vault `Political-Alpha-Claude-Code-Spec.md`.

## Verdict

{verdict}
{shortfall}
## The three pre-registered tests (and no others)

**1. Normality / directional skew (pre-announcement returns).**
Pooled standardized pre-window returns (2 trading days before each event, n={len(pre_std)}):
Shapiro–Wilk W={results['shapiro_stat']}, p={results['shapiro_p']} — {"non-normal" if (results['shapiro_p'] or 1) < 0.05 else "normality not rejected"}.
Direction-aligned skew = {results['aligned_skew']} (positive = pre-drift toward the eventual
announcement direction). See `normality_plot.png` (left panel).

**2. SD exceedance (primary decision statistic).**
Observed: {n_big}/{len(eval_rows)} evaluable event rows ({observed_rate:.2%}) exceeded ±2σ of the
trailing 60-day SD on the event day or the day after. References: normal-theory two-day
baseline ≈ {results['normal_theory_two_day_reference']:.2%}; placebo-null mean = {results['placebo_mean']:.2%} (σ = {results['placebo_std']:.2%}, p95 = {results['placebo_p95']:.2%}).

**3. Bootstrap null (decides the hypothesis).**
{results['n_boot']:,} statement-level placebo sets (seed {SEED}), identical mapping and big-move
rule, excluding ±5 trading days around real events and scheduled FOMC/CPI/NFP dates:
**p = {p_value:.4f}** (one-sided, `(n_ge+1)/(N+1)`). See `normality_plot.png` (right panel).

## Catalog

{n_events_catalog} qualifying statements → {len(study)} event×instrument rows
({len(eval_rows)} evaluable). By source: {per_source}.
Statements in the null: {len(placeable)} (skipped, no eligible placebo dates: {len(skipped_stmts)}).

## Positioning overlay (descriptive — carries no p-value)

manipulation_signal = post big-move AND pre-announcement rr25/put-call-volume moved
directionally (T-48h→T-0, FXE proxy for forex rows): **{n_flags}** of {n_pos_avail} rows with
positioning data available.

## Data gaps (recorded, never fabricated)

Phase 2: {gaps2 if gaps2 else 'none'}
Phase 3: {gaps3 if gaps3 else 'none'}

## Method notes

- Hourly bars NOT used — daily T0/T+1 mapping per spec §6 (the stated default).
- Estimation window T-252→T-10, mean-adjusted model; big-move yardstick = trailing 60d SD
  (shifted; never includes the tested day). These are distinct by design (spec §6).
- Power: with ~{len(eval_rows)} rows against a ~{results['placebo_mean']:.0%} null rate, only large effects are
  detectable; a null here does not prove absence — it bounds the effect size at this N.
- No BH/permutation/CAR were run (spec §10 — deliberately excluded from this build).
"""
    (_lib.OUTPUT_DIR / "summary_report.md").write_text(report)

    print(f"\n  ── PHASE 4 RESULT ──")
    print(f"  observed exceedance rate: {observed_rate:.3%}  placebo mean: {placebo_rates.mean():.3%}")
    print(f"  p = {p_value:.4f}  ({'H0 REJECTED — candidate result only' if p_value < 0.05 else 'H0 NOT rejected'})")
    print(f"  outputs: {_lib.OUTPUT_DIR}/normality_plot.png, sd_test_results.json, summary_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
