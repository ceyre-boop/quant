"""The structural walls around sovereign/pit/.

Timestamps and strict inequalities (test_pit_clock.py, test_pit_reader.py)
only matter if nothing can route AROUND the door they guard. This file is
that second layer: it proves via AST/text inspection, not convention, that

  1. sovereign/pit/ itself cannot reach into execution-path code,
  2. every FactSpec is internally consistent (a blocked fact really is
     blocked; a point-in-time fact really has a publication column),
  3. no file outside sovereign/pit/ issues a raw SQL SELECT against a
     PIT-registered table except the small, named, shrinking allowlist of
     pre-existing readers this layer has not migrated yet,
  4. `as_of` cannot be called with zero arguments.

DO NOT "FIX" A FAILURE HERE BY RELAXING THE TEST. A failure means either a
real new bypass was introduced (fix the caller, route it through
sovereign.pit.view instead) or the allowlist needs a line removed because a
caller was migrated (shrink it, never grow it for a new caller). Growing the
allowlist for a NEW file defeats the entire point of building this layer.
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from sovereign.pit.clock import as_of
from sovereign.pit.spec import FACTS

ROOT = Path(__file__).resolve().parent.parent
PIT_DIR = ROOT / "sovereign" / "pit"

FORBIDDEN_ROOTS = {"ict", "ict_engine", "execution", "backtester", "training"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return {n.split(".")[0] for n in names}


# ── 1. sovereign/pit/ imports nothing from the execution path ──────────────

def test_pit_package_imports_nothing_forbidden():
    py_files = sorted(PIT_DIR.rglob("*.py"))
    assert py_files, "no python files found under sovereign/pit/"
    for path in py_files:
        roots = _imported_roots(path)
        bad = roots & FORBIDDEN_ROOTS
        assert not bad, f"{path.relative_to(ROOT)} imports forbidden root(s): {sorted(bad)}"


# ── 2. every FactSpec is internally consistent ──────────────────────────────

def test_point_in_time_facts_have_published_col_and_no_blocked_reason():
    for name, spec in FACTS.items():
        if spec.is_point_in_time:
            assert spec.published_col is not None, f"{name}: point-in-time fact missing published_col"
            assert spec.blocked_reason is None, (
                f"{name}: has a published_col but ALSO a blocked_reason — "
                f"pick one, a fact cannot be both readable and blocked"
            )


def test_blocked_facts_have_no_published_col_and_a_dated_fix_reason():
    for name, spec in FACTS.items():
        if not spec.is_point_in_time:
            assert spec.published_col is None, f"{name}: blocked fact must not carry a published_col"
            assert spec.blocked_reason, f"{name}: blocked fact must state why (blocked_reason)"
            assert "FIX:" in spec.blocked_reason, (
                f"{name}: blocked_reason must name the concrete fix, not just the "
                f"problem — a blocked fact with no stated fix is an invisible gap"
            )


def test_every_fact_declares_a_nonempty_identity():
    for name, spec in FACTS.items():
        assert spec.identity, f"{name}: identity tuple must be non-empty (used to collapse vintages)"


# ── 3. no bypass: no raw SELECT against a PIT table outside sovereign/pit/ ─

#: Pre-existing readers that predate this layer and have not been migrated
#: onto sovereign.pit.view() yet. THIS LIST IS MIGRATION DEBT.
#: It must only ever SHRINK (a caller moves onto the pit reader) — a new
#: caller of a PIT-registered table belongs behind sovereign.pit.view(),
#: not in this list. Adding a new entry here re-opens exactly the hole this
#: layer was built to close.
ALLOWLIST = {
    "sovereign/fundamentals/panel.py",
    "sovereign/fundamentals/providers/free.py",
    "scripts/harvest_13f_bulk.py",
    "scripts/harvest_fundamentals.py",
    # The migration tool's whole job is schema surgery on these tables — it
    # rebuilds them without their primary keys. It cannot go through the reader
    # by definition, and it is not a research read path.
    "scripts/migrate_pit.py",
    # The auditor MUST see the raw tables, including the rows the reader
    # deliberately refuses (NULL publication instants). Reading through the
    # as-of door would hide exactly what it exists to count, and it feeds no
    # research path — it only reports.
    "scripts/audit_pit.py",
    # sovereign/fundamentals/store.py was here and has been REMOVED: its writes
    # now route through sovereign.pit.store.append(), so it no longer issues a
    # raw SELECT against a point-in-time table. That is the debt being paid off,
    # which is exactly what this list is for.
    #
    # ── newly discovered when the scan widened beyond sovereign/ + scripts/ ──
    # scripts/migrate_acceptance_ts.py: the one-time backfill that adds
    # published_ts to fund_insider_txn / fund_institution_holding and fills it
    # from EDGAR acceptanceDateTime (see sovereign/pit/spec.py's insider and
    # institutions repoint). Like scripts/migrate_pit.py it does schema
    # surgery — SELECTs to find what needs backfilling and UPDATEs to set the
    # new column — and is not a research read path. Pre-existing debt as of
    # this scan, not new code introduced by widening it.
    "scripts/migrate_acceptance_ts.py",
}

#: Table names declared anywhere in the spec (point-in-time AND blocked —
#: a blocked fact's table is exactly as sensitive, since nothing should be
#: reading it as-of from outside the layer either).
PIT_TABLES = {spec.table for spec in FACTS.values()}


def _selects_pit_table(text: str, table: str) -> bool:
    """Text-level SELECT scan, mirroring test_isolation.py's approach of
    parsing file TEXT rather than importing the module under test. A regex
    is deliberately used instead of a full SQL parser: this only needs to
    catch 'SELECT ... FROM <table>' / 'SELECT ... <table>' fragments built
    as f-strings or plain strings, and a parser would miss those too."""
    pattern = re.compile(
        rf"SELECT\b[^;\"']*\bFROM\s+{re.escape(table)}\b", re.IGNORECASE | re.DOTALL
    )
    return bool(pattern.search(text))


#: A SELECT whose table name is interpolated at runtime, e.g.
#:     f"SELECT max(fetched_at) FROM {table} WHERE ticker = ?"
#: (scripts/harvest_fundamentals.py:150).
#:
#: This is the text scan's blind spot and it must be treated as a HIT, not a
#: miss. Reading the name from a variable is exactly how a bypass would look,
#: and calling it "no PIT access" would have quietly de-allowlisted a file that
#: really does read these tables — opening the hole this suite exists to close.
_DYNAMIC_TABLE_SELECT = re.compile(
    r"SELECT\b[^;\"']*\bFROM\s*\{", re.IGNORECASE | re.DOTALL
)


def _selects_any_pit_table(text: str) -> bool:
    """True if the file reads a PIT table by name OR builds the name at runtime."""
    return (
        any(_selects_pit_table(text, t) for t in PIT_TABLES)
        or bool(_DYNAMIC_TABLE_SELECT.search(text))
    )


#: Directories scanned for a bypass. Originally just sovereign/ and scripts/
#: (where the fundamentals harvesters live); widened to every layer that can
#: plausibly hold a research or execution read path, since a raw SELECT
#: against a PIT table is exactly as dangerous wherever it is written.
SCANNED_ROOT_DIRS = (
    "sovereign/",
    "scripts/",
    "research/",
    "backtester/",
    "training/",
    "ict/",
    "execution/",
    "experience/",
)

#: Append-only is the other half of the point-in-time contract (see
#: sovereign/pit/store.py's module docstring): a DELETE, UPDATE, or
#: INSERT OR REPLACE against a PIT table destroys a vintage that the reader
#: promised would survive. Nothing previously guarded that outside
#: sovereign/pit/ itself.
_DESTRUCTIVE_PATTERNS = {
    "DELETE FROM": lambda t: re.compile(rf"DELETE\s+FROM\s+{re.escape(t)}\b", re.IGNORECASE),
    "UPDATE": lambda t: re.compile(rf"UPDATE\s+{re.escape(t)}\b", re.IGNORECASE),
    "INSERT OR REPLACE INTO": lambda t: re.compile(
        rf"INSERT\s+OR\s+REPLACE\s+INTO\s+{re.escape(t)}\b", re.IGNORECASE
    ),
}


def _destructive_hits(text: str, table: str) -> list[str]:
    return [label for label, mk in _DESTRUCTIVE_PATTERNS.items() if mk(table).search(text)]


def test_no_file_outside_pit_selects_a_pit_table_unless_allowlisted():
    assert PIT_TABLES, "spec.py declared no facts — fixture is broken"
    offenders = []
    for py in sorted(ROOT.rglob("*.py")):
        rel = py.relative_to(ROOT).as_posix()
        if rel.startswith("sovereign/pit/"):
            continue
        if rel.startswith(".venv") or "/.venv" in rel or "__pycache__" in rel:
            continue
        if rel in ALLOWLIST:
            continue
        if not rel.startswith(SCANNED_ROOT_DIRS):
            continue
        try:
            text = py.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for table in PIT_TABLES:
            if _selects_pit_table(text, table):
                offenders.append(f"{rel} selects {table!r} directly")
            for label in _destructive_hits(text, table):
                offenders.append(f"{rel} issues {label} {table!r} outside sovereign/pit/ — breaks append-only")
        # A runtime-interpolated table name is the scan's blind spot, and it is
        # also exactly what a bypass looks like. Flag it unless the file is
        # already known debt — otherwise "FROM {table}" is a free pass.
        if _DYNAMIC_TABLE_SELECT.search(text) and "fundamentals" in text:
            offenders.append(
                f"{rel} builds a SELECT table name at runtime near fundamentals "
                f"code — the text scan cannot verify which table, so route it "
                f"through sovereign.pit.view() or allowlist it explicitly"
            )
    assert not offenders, (
        "Files bypass sovereign.pit — either a raw SELECT against a PIT table, "
        "or a destructive statement (DELETE/UPDATE/INSERT OR REPLACE) that "
        "breaks the append-only contract:\n  " + "\n  ".join(offenders) +
        "\nEither route the read through sovereign.pit.view() / the write "
        "through sovereign.pit.store.append(), or — if this is truly "
        "pre-existing debt not yet migrated — add it to ALLOWLIST in this "
        "test with a comment, never silently."
    )


def test_allowlist_entries_actually_exist_and_actually_select_a_pit_table():
    """The allowlist is debt, not decoration: every entry must be a real
    file that really does the thing it's excused for, or it is dead weight
    hiding a shrink that never happened."""
    for rel in ALLOWLIST:
        path = ROOT / rel
        assert path.exists(), f"allowlisted file {rel} no longer exists — remove it from ALLOWLIST"
        text = path.read_text()
        assert _selects_any_pit_table(text), (
            f"{rel} is allowlisted but no longer selects any PIT table — "
            f"remove it from ALLOWLIST, the debt was paid off"
        )


# ── 4. as_of cannot be called with zero arguments ───────────────────────────

def test_as_of_has_no_default_argument():
    sig = inspect.signature(as_of)
    assert len(sig.parameters) == 1
    (param,) = sig.parameters.values()
    assert param.default is inspect.Parameter.empty, "as_of() must not have a default — no implicit 'now'"


def test_as_of_called_with_no_args_raises_type_error():
    with pytest.raises(TypeError):
        as_of()  # type: ignore[call-arg]
