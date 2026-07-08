"""Isolation wall (spec §4, HARD): research/political_alpha/ imports nothing from
the live system. AST-based — pattern copied from
tests/test_sentiment_board.py::test_sentiment_does_not_import_ict (:153), extended
to parse file TEXT so no module here is ever imported to be checked.

Run: python3 -m pytest research/political_alpha/tests/ -q
(NOT auto-discovered by the repo suite — pytest.ini testpaths=tests.)
"""

import ast
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_ROOTS = {
    "sovereign", "ict", "ict_engine", "config", "audit", "scripts",
    "layer1", "layer2", "layer3", "execution", "entry_engine",
}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return {n.split(".")[0] for n in names}


def test_political_alpha_imports_nothing_live():
    py_files = sorted(MODULE_ROOT.rglob("*.py"))
    assert py_files, "no python files found under research/political_alpha/"
    for path in py_files:
        roots = _imported_roots(path)
        bad = roots & FORBIDDEN_ROOTS
        assert not bad, f"{path.relative_to(MODULE_ROOT)} imports forbidden root(s): {sorted(bad)}"


def test_no_oanda_no_launchd_strings():
    """Belt-and-suspenders for spec §8: no OANDA client, no launchd/plist writes."""
    for path in sorted(MODULE_ROOT.rglob("*.py")):
        if path == Path(__file__):        # this file names the phrases it forbids
            continue
        src = path.read_text().lower()
        for phrase in ("oandapyv20", "oanda_bridge", "launchctl", "libraryagents"):
            assert phrase not in src, f"{path.name} contains forbidden phrase {phrase!r}"
