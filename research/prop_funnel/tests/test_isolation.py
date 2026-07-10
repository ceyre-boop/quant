"""Isolation wall (TICK-022, HARD): research/prop_funnel/ touches nothing live.

Pattern: research/political_alpha/tests/test_isolation.py, extended with a
sovereign WHITELIST — this module deliberately reuses sovereign.propfirm /
sovereign.risk.monte_carlo_prop / sovereign.discovery / sovereign.reporting
(read-only engines), and nothing else under sovereign.

Run: python3 -m pytest research/prop_funnel/tests/ -q
(NOT auto-discovered by the repo suite — pytest.ini testpaths=tests.)
"""

import ast
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_ROOTS = {
    "ict", "ict_engine", "config", "audit", "scripts",
    "layer1", "layer2", "layer3", "execution", "entry_engine", "orchestrator",
}

SOVEREIGN_WHITELIST = (
    "sovereign.propfirm",
    "sovereign.risk.monte_carlo_prop",
    "sovereign.discovery",
    "sovereign.reporting",
)


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_prop_funnel_imports_nothing_live():
    py_files = sorted(MODULE_ROOT.rglob("*.py"))
    assert py_files, "no python files found under research/prop_funnel/"
    for path in py_files:
        names = _imported_names(path)
        roots = {n.split(".")[0] for n in names}
        bad = roots & FORBIDDEN_ROOTS
        assert not bad, f"{path.relative_to(MODULE_ROOT)} imports forbidden root(s): {sorted(bad)}"
        for n in names:
            if n == "sovereign" or n.startswith("sovereign."):
                allowed = any(n == w or n.startswith(w + ".") for w in SOVEREIGN_WHITELIST)
                assert allowed, (f"{path.relative_to(MODULE_ROOT)} imports {n!r} — outside the "
                                 f"sovereign whitelist {SOVEREIGN_WHITELIST}")


def test_no_oanda_no_launchd_strings():
    """Belt-and-suspenders: no OANDA client, no launchd/plist writes anywhere here."""
    for path in sorted(MODULE_ROOT.rglob("*.py")):
        if path == Path(__file__):        # this file names the phrases it forbids
            continue
        src = path.read_text().lower()
        for phrase in ("oandapyv20", "oanda_bridge", "launchctl", "libraryagents"):
            assert phrase not in src, f"{path.name} contains forbidden phrase {phrase!r}"


def test_writes_stay_inside_output_dir():
    """No python file references the live output paths this ticket must not touch."""
    forbidden_write_targets = (
        "data/risk/prop_monte_carlo.json",   # only via the monkeypatched OUT
        "data/futures/",
        "data/propfirm/",
        "hypothesis_ledger",
    )
    for path in sorted(MODULE_ROOT.rglob("*.py")):
        if path == Path(__file__):
            continue
        src = path.read_text()
        for target in forbidden_write_targets:
            if target == "data/risk/prop_monte_carlo.json" and path.name in ("parity.py", "test_parity.py"):
                continue  # both name it to document/verify the monkeypatch — read-only guard, never a write
            assert target not in src, f"{path.name} references write-restricted path {target!r}"
