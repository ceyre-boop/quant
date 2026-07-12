#!/usr/bin/env python3
"""Render the human-readable report.md from the projection results dict.

Pure formatting -- no data access, no imports from the live ICT path.
"""
from __future__ import annotations


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render_report(r: dict) -> str:
    w = r["window"]
    d = r["dedup"]
    vr = r["veto_rates_live"]
    tb = r["taken_base_rate"]
    fg = r["fill_gap"]
    pj = r["projection_90d"]
    ls = pj["logged_setups"]
    ci = ls["bootstrap_ci"]
    n30 = r["near_30"]

    lines = []
    A = lines.append

    A("# ICT 90-Day Taken-Trade Projection (TICK-028)")
    A("")
    A("> READ-ONLY research. Reads only `data/ledger/` veto shards and "
      "`data/decision_logs/` decisions. Nothing on the execution/exit path is imported "
      "or modified (shadow-freeze respected).")
    A("")
    A("## Verdict")
    A("")
    A(f"**{r['verdict']}**")
    A("")
    A(f"- 90-day **logged/committed setups**: point **{ls['point_estimate']:.0f}** "
      f"(95% bootstrap [{ci['p2_5']:.0f}, {ci['p97_5']:.0f}], 80% [{ci['p10']:.0f}, {ci['p90']:.0f}])")
    A(f"- 90-day **actually-filled trades**: point **{pj['filled_trades']['point_estimate']:.1f}** "
      f"(the number that would count toward a prop challenge)")
    A(f"- `near_30` (logged-setup basis): **{n30['basis_logged_setups']['classification']}** "
      f"vs the {n30['range_lo']}-{n30['range_hi']} band  |  "
      f"(filled basis: **{n30['basis_filled_trades']['classification']}**)")
    A("")
    A("## Window")
    A("")
    A(f"- Analysis anchored to latest observed data date: **{w['analysis_anchor_date']}** (no wall-clock; deterministic).")
    A(f"- Trailing window: `{w['window_start_exclusive']}` (exclusive) -> `{w['window_end_inclusive']}` "
      f"= {w['trailing_calendar_days']} calendar days = **{w['trading_days_in_window']} trading days**.")
    A("")
    A("## 1. Dedup (the load-bearing step)")
    A("")
    A("The scanner re-emits a full universe sweep every cycle, so a standing condition "
      "(e.g. \"ADR exhausted\") re-vetoes the same pair all day. Records are collapsed to unique "
      "`(date, pair, direction, veto_class)`, first-per-day.")
    A("")
    A(f"- Trailing-window vetoes: **{d['raw_veto_records_window']} raw -> {d['unique_veto_setups_window']} unique** "
      f"= **{d['dedup_factor_window']:.1f}x** dedup.")
    A(f"- All-shards cross-check: {d['raw_veto_records_all_shards']} raw -> {d['unique_veto_setups_all_shards']} unique "
      f"= {d['dedup_factor_all_shards']:.1f}x.")
    A(f"- Unique vetoed setups per trading day: **{d['unique_veto_setups_per_trading_day']}**.")
    A(f"- {d['dedup_note']}")
    A("")
    A("## 2. Live veto-rate breakdown (trailing 45d, deduped)")
    A("")
    A(f"- **ADR share: {_pct(vr['adr_share'])}**  |  **weekly-trend share: {_pct(vr['weekly_trend_share'])}**")
    A(f"- {vr['note']}")
    A("")
    A("| veto class | unique setups | share |")
    A("|---|---:|---:|")
    for cls, info in vr["class_breakdown"].items():
        A(f"| {cls} | {info['unique_setups']} | {_pct(info['share'])} |")
    A("")
    A("## 3. Daily taken base rate")
    A("")
    A(f"- Raw ICT decision records in window: {tb['raw_taken_records_window']} -> "
      f"unique `(date,pair,direction)` setups: **{tb['unique_taken_setups_window_pair_dir']}** "
      f"(with entry_level granularity: {tb['unique_taken_setups_window_pair_dir_level']}).")
    A(f"- **Mean daily taken (committed) setups: {tb['mean_daily_taken_setups']}** per trading day.")
    A(f"- Estimate A (direct): {tb['estimate_A_direct_per_day']}/day.  "
      f"Estimate B (veto-implied): {tb['estimate_B_veto_implied_per_day']}/day.  "
      f"P(vetoed) at setup level: {_pct(tb['p_vetoed_setup_level'])}.")
    A(f"- {tb['reconciliation_note']}")
    A("")
    A("## 4. Fill gap (why committed != executed)")
    A("")
    A(f"- Outcome distribution (window): {fg['outcome_distribution_window']}")
    A(f"- Unique filled setups in window: **{fg['unique_filled_setups_window']}**; "
      f"fill rate: **{_pct(fg['fill_rate_window'])}**.")
    A(f"- {fg['note']}")
    A("")
    A("## 5. 90-day projection")
    A("")
    A(f"- Calendar->trading-day factor: {pj['calendar_to_trading_day_factor']} "
      f"({pj['projection_calendar_days']} cal days -> **{pj['trading_days_90']} trading days**).")
    A(f"- Logged setups: point **{ls['point_estimate']:.1f}**; bootstrap over days (N=10000, seed 42): "
      f"median {ci['median']:.1f}, 80% [{ci['p10']:.1f}, {ci['p90']:.1f}], 95% [{ci['p2_5']:.1f}, {ci['p97_5']:.1f}].")
    A(f"- Sensitivity (+/-30% rate): [{ls['sensitivity_band_pm30pct']['minus_30pct']:.1f}, "
      f"{ls['sensitivity_band_pm30pct']['plus_30pct']:.1f}].")
    A(f"- Filled trades: point **{pj['filled_trades']['point_estimate']:.1f}** -- {pj['filled_trades']['note']}")
    A("")
    A("## Caveats")
    A("")
    for c in r["caveats"]:
        A(f"- {c}")
    A("")
    return "\n".join(lines)
