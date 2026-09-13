import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ALLOWED = ("research.formal", "backtester.luld", "backtester.realistic_fills", "z3", "numpy", "pandas", "pytest")
STDLIB = {"json", "sys", "os", "pathlib", "dataclasses", "datetime", "typing", "__future__", "math"}


def test_no_execution_path_imports():
    for f in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for n in names:
                assert n.split(".")[0] in STDLIB or n.startswith(ALLOWED), f"{f.name}: forbidden import {n}"
