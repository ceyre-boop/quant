"""Isolation guardrails for research/tsmom_hyp091 (HYP-091, TICK-027).

Asserts the study (a) imports nothing from the execution path or the parallel
session's research/tsmom/, and (b) writes only under data/research/tsmom_hyp091/.
Reading sovereign READ-ONLY (Sharpe/DSR utils, price loader, differentials, v015
CSV) is permitted — this is an isolated-OUTPUT track (prop_funnel-style), not the
strict full-isolation of political_alpha_v2.

    python3 -m pytest research/tsmom_hyp091/test_isolation.py -q
"""
from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parent
OUT_REL = "data/research/tsmom_hyp091"

# Modules that must NEVER be imported by this study (execution path / parallel dir).
FORBIDDEN_PREFIXES = (
    "ict.pipeline", "ict.orchestrator", "ict.ict_veto_ledger", "ict_engine",
    "research.tsmom.",            # the parallel session's package
    "sovereign.forex.forex_exit_manager", "sovereign.execution",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_no_forbidden_imports():
    offenders = {}
    for py in PKG.glob("*.py"):
        bad = {imp for imp in _imports(py)
               if any(imp == pre.rstrip(".") or imp.startswith(pre) for pre in FORBIDDEN_PREFIXES)}
        if bad:
            offenders[py.name] = sorted(bad)
    assert not offenders, f"forbidden imports: {offenders}"


def test_writes_only_under_out_dir():
    """Every string literal that looks like a data write-path must be under OUT_REL."""
    suspicious = {}
    for py in PKG.glob("*.py"):
        if py.name == "test_isolation.py":
            continue  # this file names paths as test literals, not writes
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if v.startswith("data/") and OUT_REL not in v and "preregister" not in v \
                        and "proof/backtest_trades_v015" not in v and "swap_calibration" not in v \
                        and "hypothesis_ledger" not in v:
                    suspicious.setdefault(py.name, []).append(v)
    # preregister/ledger/read-only inputs are explicitly allow-listed above.
    assert not suspicious, f"data paths outside {OUT_REL}: {suspicious}"


def test_out_dir_constant_is_isolated():
    from research.tsmom_hyp091._lib import OUT_DIR, ROOT
    assert str(OUT_DIR.relative_to(ROOT)) == OUT_REL
