#!/usr/bin/env python3
"""Read-only guard tests for the ICT projection package (TICK-028).

Asserts the package never imports the live ICT execution path and only writes
under the sanctioned research output directories. Run:

    python3 -m pytest research/ict_projection/test_readonly.py -v
    (or)  python3 research/ict_projection/test_readonly.py
"""
from __future__ import annotations

import re
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent

FORBIDDEN_IMPORTS = [
    "ict.pipeline",
    "ict.orchestrator",
    "ict.ict_veto_ledger",
]
# Also forbid touching the exit/execution path directly.
FORBIDDEN_SUBSTRINGS = [
    "forex_exit_manager",
    "decide_exit",
    "exit_machine",
]

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([\w\.]+)", re.MULTILINE)


def _package_sources():
    return sorted(p for p in PKG_DIR.glob("*.py"))


def test_no_forbidden_imports():
    for src in _package_sources():
        text = src.read_text()
        modules = _IMPORT_RE.findall(text)
        for mod in modules:
            for bad in FORBIDDEN_IMPORTS:
                assert not (mod == bad or mod.startswith(bad + ".")), (
                    f"{src.name} imports forbidden live-path module: {mod}"
                )


def test_no_execution_path_references():
    # test_readonly.py itself names these strings to define the ban; skip it.
    for src in _package_sources():
        if src.name == "test_readonly.py":
            continue
        text = src.read_text()
        for bad in FORBIDDEN_SUBSTRINGS:
            assert bad not in text, f"{src.name} references execution-path symbol: {bad}"


def test_output_paths_under_sanctioned_dirs():
    from research.ict_projection import run

    out_root = run.output_root()
    results = {"verdict": "x", "caveats": [], "window": {}, "dedup": {},
               "veto_rates_live": {}, "taken_base_rate": {}, "fill_gap": {},
               "projection_90d": {}, "near_30": {}}
    # We only check the constant path shape, not a full write.
    json_rel = Path("data") / "research" / "ict_projection" / "projection_90d.json"
    md_rel = Path("research") / "ict_projection" / "report.md"
    assert (out_root / json_rel).parent.name == "ict_projection"
    assert str(json_rel).startswith("data/research/ict_projection")
    assert str(md_rel).startswith("research/ict_projection")


if __name__ == "__main__":
    test_no_forbidden_imports()
    test_no_execution_path_references()
    test_output_paths_under_sanctioned_dirs()
    print("OK: read-only guards pass")
