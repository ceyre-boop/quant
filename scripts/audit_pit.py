#!/usr/bin/env python3
"""Standing point-in-time audit of the fundamentals store.

Shaped after scripts/audit_look_ahead.py: every check is a COUNT(*) that must be
zero, and any violation exits 1 so a runner or CI wrapper can see teeth.

Three checks per point-in-time fact:

  NULL PUBLICATION (warning, not a violation)
      Rows whose publication instant is unknown. These are stored but the as-of
      reader correctly refuses them, so they are invisible to research rather
      than dangerous. Reported because a large or growing count means a source
      is silently contributing nothing — fund_short_interest holds 250 such rows
      today because Nasdaq supplies no publication date.

  PUBLISHED BEFORE THE EVENT (violation only above a threshold)
      What this really detects is a SWAPPED event/published mapping, which would
      invert the temporal contract for a whole fact.

      A few such rows are legitimate and must not fail the run: an 8-K can
      announce something scheduled for the NEXT day, so its acceptance instant
      precedes its period-of-report. Six AMZN annual-meeting 8-Ks do exactly
      this (accepted ~18:00 ET on day D-1, reportDate day D). That is knowing
      about an event before it happens — an announcement, the opposite of a leak.

      So: a swap would make ~100% of rows trip this. Legitimate forward
      announcements are a rounding error. Violation above SWAP_RATIO, reported
      as a warning below it.

  DUPLICATE VINTAGE (violation)
      The same identity at the same publication instant appearing twice. Writes
      are append-only, so re-observing an identical vintage must be a no-op; a
      duplicate means the dedup in sovereign/pit/store.append() is not holding
      and the store is inflating on every harvest.

--self-test injects each violation into a THROWAWAY database and asserts the
audit catches it. A detector nobody has seen fail is not evidence of anything.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from sovereign.pit.spec import FACTS, FactSpec  # noqa: E402
from sovereign.pit.store import DB_PATH  # noqa: E402

#: Fraction of rows with published < event above which we call it a SWAP rather
#: than legitimate forward announcements. A genuine column swap trips ~100% of
#: rows; real announcements are a fraction of a percent.
SWAP_RATIO = 0.05


def _table_exists(con, table: str) -> bool:
    try:
        return bool(con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone())
    except duckdb.Error:
        return False


def audit_fact(con, spec: FactSpec) -> dict:
    """Returns counts for one fact. Never raises on a missing table."""
    out = {
        "fact": spec.name, "table": spec.table, "present": False,
        "rows": 0, "null_published": 0,
        "published_before_valid": 0, "duplicate_vintage": 0,
    }
    if not _table_exists(con, spec.table):
        return out
    out["present"] = True

    pub, valid = spec.published_col, spec.valid_col
    out["rows"] = con.execute(f"SELECT count(*) FROM {spec.table}").fetchone()[0]
    out["null_published"] = con.execute(
        f"SELECT count(*) FROM {spec.table} WHERE {pub} IS NULL"
    ).fetchone()[0]

    # CAST both sides: a DATE valid column against a TIMESTAMP publication
    # column would otherwise compare inconsistently.
    out["published_before_valid"] = con.execute(
        f"SELECT count(*) FROM {spec.table} "
        f"WHERE {pub} IS NOT NULL AND {valid} IS NOT NULL "
        f"AND CAST({pub} AS DATE) < CAST({valid} AS DATE)"
    ).fetchone()[0]

    ident = ", ".join(spec.identity)
    out["duplicate_vintage"] = con.execute(
        f"SELECT coalesce(sum(n - 1), 0) FROM ("
        f"  SELECT count(*) AS n FROM {spec.table} "
        f"  WHERE {pub} IS NOT NULL GROUP BY {ident}, {pub} HAVING count(*) > 1)"
    ).fetchone()[0]
    return out


def run(db: Path) -> tuple[list[dict], int, int]:
    if not db.exists():
        return [], 0, 0
    con = duckdb.connect(str(db), read_only=True)
    try:
        con.execute("SET TimeZone='UTC'")
        results = [
            audit_fact(con, spec)
            for _, spec in sorted(FACTS.items()) if spec.is_point_in_time
        ]
    finally:
        con.close()
    violations = 0
    for r in results:
        r["forward_ratio"] = (r["published_before_valid"] / r["rows"]) if r["rows"] else 0.0
        r["swap_suspected"] = r["forward_ratio"] > SWAP_RATIO
        violations += r["duplicate_vintage"]
        if r["swap_suspected"]:
            violations += r["published_before_valid"]
    warnings = sum(r["null_published"] for r in results)
    return results, violations, warnings


def render(results: list[dict], violations: int, warnings: int) -> None:
    print(f"point-in-time audit — {DB_PATH}\n")
    print(f"{'fact':<16} {'rows':>7} {'null pub':>9} {'pub<valid':>10} {'dup vintage':>12}  status")
    print("-" * 74)
    for r in results:
        if not r["present"]:
            print(f"{r['fact']:<16} {'-':>7} {'-':>9} {'-':>10} {'-':>12}  (no table yet)")
            continue
        bad = r["duplicate_vintage"] + (r["published_before_valid"] if r.get("swap_suspected") else 0)
        status = "VIOLATION" if bad else ("warn" if r["null_published"] else "ok")
        print(f"{r['fact']:<16} {r['rows']:>7} {r['null_published']:>9} "
              f"{r['published_before_valid']:>10} {r['duplicate_vintage']:>12}  {status}")

    blocked = [n for n, s in sorted(FACTS.items()) if not s.is_point_in_time]
    if blocked:
        print(f"\nblocked facts (cannot be read as-of): {', '.join(blocked)}")
        print("  each names its fix in sovereign/pit/spec.py")

    fwd = [r for r in results if r["published_before_valid"] and not r.get("swap_suspected")]
    if fwd:
        print()
        for r in fwd:
            print(f"{r['fact']}: {r['published_before_valid']} row(s) published BEFORE their event "
                  f"({r['forward_ratio']*100:.2f}%) — forward announcements, not a leak.")
        print("  (a swapped event/published mapping would trip ~100%, not a fraction of a percent)")

    if warnings:
        print(f"\n{warnings} row(s) have no publication instant. These are stored but the")
        print("as-of reader refuses them, so they are invisible to research, not unsafe.")
    print(f"\n{'FAIL' if violations else 'PASS'} — {violations} violation(s)")


def self_test() -> int:
    """Inject each violation into a throwaway DB and assert we catch it."""
    print("self-test: injecting known-bad rows into a temporary database\n")
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "probe.db"
        con = duckdb.connect(str(db))
        con.execute("""
            CREATE TABLE fund_earnings_event (
                ticker VARCHAR, fiscal_end DATE, report_date DATE,
                report_time VARCHAR, eps_estimate DOUBLE, eps_actual DOUBLE,
                source VARCHAR, published_ts TIMESTAMP)
        """)
        # (1) published BEFORE the fiscal period it describes — impossible.
        con.execute("INSERT INTO fund_earnings_event VALUES "
                    "('AAA', DATE '2026-06-30', DATE '2026-07-30', 'amc', 1.0, 1.1, "
                    "'x', TIMESTAMP '2026-01-01 00:00:00')")
        # (2) the same identity at the same publication instant, twice.
        for _ in range(2):
            con.execute("INSERT INTO fund_earnings_event VALUES "
                        "('BBB', DATE '2026-03-31', DATE '2026-04-30', 'amc', 1.0, 1.1, "
                        "'x', TIMESTAMP '2026-05-01 00:00:00')")
        # (3) a NULL publication instant — expected as a warning, not a failure.
        con.execute("INSERT INTO fund_earnings_event VALUES "
                    "('CCC', DATE '2026-03-31', DATE '2026-04-30', 'amc', 1.0, 1.1, 'x', NULL)")
        con.close()

        results, violations, warnings = run(db)
        e = next(r for r in results if r["fact"] == "earnings")

        checks = [
            ("published-before-valid detected", e["published_before_valid"] == 1),
            ("duplicate vintage detected", e["duplicate_vintage"] == 1),
            ("null publication counted as a warning", e["null_published"] == 1),
            ("run() reports a non-zero violation count", violations == 2),
            ("warnings are not counted as violations", warnings == 1),
        ]
        ok = True
        for label, passed in checks:
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
            ok &= passed

    print(f"\nself-test {'PASSED — the audit has teeth' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the audit catches known-bad rows (uses a temp DB)")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    results, violations, warnings = run(DB_PATH)
    if args.json:
        print(json.dumps({
            "db": str(DB_PATH), "violations": violations,
            "warnings": warnings, "facts": results,
            "blocked": [n for n, s in FACTS.items() if not s.is_point_in_time],
        }, indent=1))
    else:
        render(results, violations, warnings)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
