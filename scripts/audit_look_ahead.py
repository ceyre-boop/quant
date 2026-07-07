#!/usr/bin/env python3
"""Automated look-ahead auditor for the sentiment/positioning tables (ALFRED standard).

Every provenance-bearing table is checked with the DB-level COUNT(*)==0 shape the
repo's no-look-ahead tests use (test_sentiment_board.py::TestCOT/TestVRP), plus an
EMPIRICAL board check: a fused board value must be reproducible from a row that was
PUBLIC on the board date — a value that only matches a future-published row is a leak.

Exit code 0 = zero violations; 1 = any violation (wire-able into runners and CI).
Telemetry (not violations): COT publish-Fridays falling on US federal holidays — the
weeks where the +3d publish approximation can run ≤1 business day early (documented
bias, config sentiment.cot.release_time_et note).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sovereign.sentiment.store import connect  # noqa: E402


def _exists(con, table: str) -> bool:
    return bool(con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchone()[0])


def _count(con, sql: str) -> int:
    return int(con.execute(sql).fetchone()[0])


def audit(con) -> list[dict]:
    """Run every check; returns [{table, check, violations, total, note}]."""
    out = []

    def add(table, check, violations, total, note=""):
        out.append({"table": table, "check": check, "violations": int(violations),
                    "total": int(total), "note": note})

    for table, prov in (("sentiment_cot_weekly", "publish_date <= measurement_date"),
                        ("sentiment_cot_tff_weekly", "publish_date <= measurement_date")):
        if not _exists(con, table):
            continue
        total = _count(con, f"SELECT COUNT(*) FROM {table}")
        add(table, "publish_after_measurement",
            _count(con, f"SELECT COUNT(*) FROM {table} WHERE {prov}"), total)
        add(table, "release_ts_on_publish_date",
            _count(con, f"SELECT COUNT(*) FROM {table} WHERE release_ts IS NOT NULL "
                        f"AND CAST(release_ts AS DATE) != publish_date"), total)

    if _exists(con, "sentiment_vrp_daily"):
        total = _count(con, "SELECT COUNT(*) FROM sentiment_vrp_daily")
        add("sentiment_vrp_daily", "provenance_not_after_date",
            _count(con, "SELECT COUNT(*) FROM sentiment_vrp_daily "
                        "WHERE iv_obs_date > date OR rv_last_date > date"), total)

    if _exists(con, "sentiment_options_surface"):
        total = _count(con, "SELECT COUNT(*) FROM sentiment_options_surface")
        add("sentiment_options_surface", "provenance_not_after_date",
            _count(con, "SELECT COUNT(*) FROM sentiment_options_surface WHERE iv_obs_date > date"), total)

    # Geometry (corridor/FVG/tri-state) is a same-day daily feed — unlike VRP/options-surface's
    # independently-sampled weekly source, src_last_bar_date must EQUAL date always, not merely
    # not-be-after it (any mismatch, either direction, signals a feeder bug, not legitimate staleness).
    if _exists(con, "sentiment_geometry_daily"):
        total = _count(con, "SELECT COUNT(*) FROM sentiment_geometry_daily")
        add("sentiment_geometry_daily", "provenance_last_bar_matches_date",
            _count(con, "SELECT COUNT(*) FROM sentiment_geometry_daily WHERE src_last_bar_date != date"),
            total)

    if _exists(con, "sentiment_surprise_release"):
        total = _count(con, "SELECT COUNT(*) FROM sentiment_surprise_release")
        add("sentiment_surprise_release", "publish_after_ref",
            _count(con, "SELECT COUNT(*) FROM sentiment_surprise_release "
                        "WHERE publish_date <= ref_date"), total)

    # ── empirical board leak checks: a fused value must match SOME row public on the board date ──
    if _exists(con, "sentiment_board_state"):
        for col, src, src_col, pub in (
            ("cot_net_pct", "sentiment_cot_weekly", "net_pct", "publish_date"),
            ("cot_net_pct_1y", "sentiment_cot_weekly", "net_pct_1y", "publish_date"),
            ("tff_lev_net_pct", "sentiment_cot_tff_weekly", "lev_net_pct", "publish_date"),
            ("vrp_signal", "sentiment_vrp_daily", "vrp_signal", "date"),
            ("corridor_r2", "sentiment_geometry_daily", "corridor_r2", "date"),
            ("corridor_dev", "sentiment_geometry_daily", "corridor_dev", "date"),
            ("fvg_count_20d", "sentiment_geometry_daily", "fvg_count_20d", "date"),
            ("fvg_unfilled", "sentiment_geometry_daily", "fvg_unfilled", "date"),
            ("tri_state", "sentiment_geometry_daily", "tri_state", "date"),
            ("days_in_consolidation", "sentiment_geometry_daily", "days_in_consolidation", "date"),
            ("range_slope", "sentiment_geometry_daily", "range_slope", "date"),
        ):
            if not _exists(con, src):
                continue
            has_col = bool(con.execute(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_name='sentiment_board_state' AND column_name=?", [col]).fetchone()[0])
            if not has_col:
                continue
            total = _count(con, f"SELECT COUNT(*) FROM sentiment_board_state WHERE {col} IS NOT NULL")
            if total == 0:
                add("sentiment_board_state", f"asof_{col}", 0, 0, "no fused values yet")
                continue
            viol = _count(con, f"""
                SELECT COUNT(*) FROM sentiment_board_state b
                WHERE b.{col} IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM {src} s
                    WHERE s.pair = b.pair AND s.{pub} <= b.date
                      AND abs(s.{src_col} - b.{col}) < 1e-9)""")
            add("sentiment_board_state", f"asof_{col}", viol, total,
                "value only reproducible from a future-published row = leak" if viol else "")

    # ── telemetry: holiday-week publish approximation (bias count, NOT a violation) ──
    if _exists(con, "sentiment_cot_weekly"):
        try:
            import pandas as pd
            from pandas.tseries.holiday import USFederalHolidayCalendar
            pubs = con.execute(
                "SELECT DISTINCT publish_date FROM sentiment_cot_weekly ORDER BY 1").df()["publish_date"]
            pubs = pd.to_datetime(pubs)
            if len(pubs):
                hol = USFederalHolidayCalendar().holidays(start=pubs.min(), end=pubs.max())
                n = int(pubs.isin(hol).sum())
                out.append({"table": "sentiment_cot_weekly", "check": "holiday_week_publish_bias",
                            "violations": 0, "total": int(len(pubs)),
                            "note": f"{n} publish-Fridays on US federal holidays — weeks where the +3d "
                                    f"approximation may run early (documented bias, not a violation)"})
        except Exception:
            pass
    return out


def main() -> int:
    con = connect()
    try:
        results = audit(con)
    finally:
        con.close()
    bad = 0
    print("LOOK-AHEAD AUDIT (violations/total):")
    for r in results:
        flag = "  ✗" if r["violations"] else "  ✓"
        print(f"{flag} {r['table']:28s} {r['check']:28s} {r['violations']}/{r['total']}"
              + (f"  — {r['note']}" if r["note"] else ""))
        bad += r["violations"]
    print(f"TOTAL VIOLATIONS: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
