#!/usr/bin/env python3
"""Make the fundamentals store bitemporal and append-only.

WHAT THIS FIXES

Every fact table was written with INSERT OR REPLACE on a primary key that did
not include the publication instant. A revision therefore overwrote the
observation it revised:

  - an earnings restatement destroyed the originally-reported figure
  - the routine pre-print -> post-print transition destroyed the pre-announcement
    consensus view of that quarter
  - a 13F-A amendment overwrote the original 13F IN PLACE, taking the original's
    earlier filing_date with it -- after which even a correct
    `filing_date < as_of` filter UNDERSTATES what was knowable at the time

No amount of filtering fixes that. "What did I know on March 3" is unanswerable
against a row that has since been overwritten. History has to survive first.

WHAT IT DOES

For each point-in-time fact in sovereign/pit/spec.py:
  1. add `observed_at` (when WE fetched it -- operational, never used in an
     as-of predicate; it is deliberately named differently from published_ts
     because conflating the two is a classic way to reintroduce leakage)
  2. rebuild the table WITHOUT its PRIMARY KEY, so two vintages of the same
     identity can coexist
  3. leave every existing row untouched

Step 2 is the load-bearing one. The PK is what forced the collision that
INSERT OR REPLACE then resolved by deleting history.

SAFETY
  --dry-run   report only, touch nothing (default is to ASK before writing)
  A timestamped backup of the DB is taken before any write.
  Idempotent: re-running is a no-op once tables are already PK-free.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from sovereign.pit.spec import FACTS  # noqa: E402
from sovereign.pit.store import DB_PATH  # noqa: E402


def inspect_table(con, table: str) -> dict | None:
    try:
        info = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    except duckdb.Error:
        return None
    if not info:
        return None
    return {
        "cols": [r[1] for r in info],
        "types": {r[1]: r[2] for r in info},
        "pk": [r[1] for r in info if r[5]],
        "rows": con.execute(f"SELECT count(*) FROM {table}").fetchone()[0],
    }


def rebuild_without_pk(con, table: str, meta: dict) -> None:
    """Recreate the table with identical columns and no PRIMARY KEY."""
    cols, types = meta["cols"], meta["types"]
    tmp = f"{table}__pit_tmp"
    ddl = ", ".join(f'"{c}" {types[c]}' for c in cols)
    con.execute(f"DROP TABLE IF EXISTS {tmp}")
    con.execute(f"CREATE TABLE {tmp} ({ddl})")
    quoted = ", ".join(f'"{c}"' for c in cols)
    con.execute(f"INSERT INTO {tmp} ({quoted}) SELECT {quoted} FROM {table}")
    con.execute(f"DROP TABLE {table}")
    con.execute(f"ALTER TABLE {tmp} RENAME TO {table}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    ap.add_argument("--yes", action="store_true", help="apply without prompting")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"no store at {DB_PATH} — nothing to migrate")
        return 0

    con = duckdb.connect(str(DB_PATH), read_only=True)
    plan: list[tuple[str, dict, bool, bool]] = []
    try:
        for name, spec in sorted(FACTS.items()):
            if not spec.is_point_in_time:
                continue
            meta = inspect_table(con, spec.table)
            if meta is None:
                continue
            needs_pk_drop = bool(meta["pk"])
            needs_observed = "observed_at" not in meta["cols"]
            plan.append((name, meta, needs_pk_drop, needs_observed))
    finally:
        con.close()

    print(f"store: {DB_PATH}\n")
    print(f"{'fact':<16} {'rows':>8}  {'primary key':<44} action")
    print("-" * 100)
    todo = 0
    for name, meta, drop_pk, add_col in plan:
        acts = []
        if drop_pk:
            acts.append("drop PK")
        if add_col:
            acts.append("add observed_at")
        if acts:
            todo += 1
        pk = ",".join(meta["pk"]) or "(none)"
        print(f"{name:<16} {meta['rows']:>8}  {pk[:44]:<44} {' + '.join(acts) or 'already append-only'}")

    if not todo:
        print("\nnothing to do — store is already append-only")
        return 0

    print(f"\n{todo} table(s) need migrating.")
    print("Existing rows are preserved; only the PRIMARY KEY constraint is removed")
    print("so that a restatement can coexist with the observation it revises.")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    if not args.yes:
        try:
            if input("\napply? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1
        except EOFError:
            print("\nnon-interactive; re-run with --yes to apply")
            return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = DB_PATH.with_suffix(f".pre-pit-{stamp}.db")
    shutil.copy2(DB_PATH, backup)
    print(f"backup: {backup}")

    con = duckdb.connect(str(DB_PATH))
    try:
        for name, meta, drop_pk, add_col in plan:
            spec = FACTS[name]
            before = meta["rows"]
            if add_col:
                con.execute(f"ALTER TABLE {spec.table} ADD COLUMN observed_at TIMESTAMP")
            if drop_pk:
                fresh = inspect_table(con, spec.table)
                rebuild_without_pk(con, spec.table, fresh)
            after = con.execute(f"SELECT count(*) FROM {spec.table}").fetchone()[0]
            if after != before:
                raise RuntimeError(
                    f"{spec.table}: row count changed {before} -> {after} during migration; "
                    f"restore from {backup}"
                )
            print(f"  {name:<16} ok ({after} rows preserved)")
    finally:
        con.close()

    print("\ndone — writes are now append-only; a revision adds a vintage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
