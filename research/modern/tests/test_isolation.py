"""Isolation wall (HYP-090, HARD): research/modern/ is a read-only study.

Whitelist covers the canonical backtest surface it must replay; everything
live-adjacent is forbidden. Extra guard: this study must never resemble
sovereign/monthly_reopt.py (the live parameter re-optimizer) — no
config/parameters.yml references at all.

Run: python3 -m pytest research/modern/tests/ -q
(NOT auto-discovered — pytest.ini testpaths=tests.)
"""

import ast
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_ROOTS = {
    "ict", "ict_engine", "config", "audit", "scripts",
    "layer1", "layer2", "layer3", "execution", "entry_engine", "orchestrator",
}

SOVEREIGN_WHITELIST = (
    "sovereign.forex.fast_backtester",
    "sovereign.forex.forex_backtester",
    "sovereign.forex.exit_machine",
    "sovereign.forex.data_fetcher",
    "sovereign.forex.signal_engine",
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


def test_modern_imports_nothing_live():
    py_files = sorted(MODULE_ROOT.rglob("*.py"))
    assert py_files, "no python files under research/modern/"
    for path in py_files:
        names = _imported_names(path)
        roots = {n.split(".")[0] for n in names}
        bad = roots & FORBIDDEN_ROOTS
        assert not bad, f"{path.relative_to(MODULE_ROOT)} imports forbidden root(s): {sorted(bad)}"
        for n in names:
            if n == "sovereign" or n.startswith("sovereign."):
                ok = any(n == w or n.startswith(w + ".") for w in SOVEREIGN_WHITELIST)
                assert ok, (f"{path.relative_to(MODULE_ROOT)} imports {n!r} — outside whitelist "
                            f"{SOVEREIGN_WHITELIST}")


def test_no_oanda_no_launchd_no_live_config_strings():
    # preregister_hyp090.py is the hash-locked law text — it NAMES the forbidden
    # paths in order to forbid them (same exemption class as this test file).
    exempt = {Path(__file__).name, "preregister_hyp090.py"}
    for path in sorted(MODULE_ROOT.rglob("*.py")):
        if path.name in exempt:
            continue
        src = path.read_text().lower()
        for phrase in ("oandapyv20", "oanda_bridge", "launchctl", "libraryagents",
                       "parameters.yml", "monthly_reopt"):
            assert phrase not in src, f"{path.name} contains forbidden phrase {phrase!r}"


def test_write_targets_stay_inside_study_outputs():
    """Only the prereg writer + the P4 report may name the prereg/ledger paths."""
    allowed_ledger_writers = {"preregister_hyp090.py", "report.py", "_lib.py", "test_prereg_gate.py"}
    for path in sorted(MODULE_ROOT.rglob("*.py")):
        if path.name in allowed_ledger_writers or path == Path(__file__):
            continue
        src = path.read_text()
        for target in ("hypothesis_ledger", "data/research/preregister"):
            assert target not in src, f"{path.name} references restricted path {target!r}"
