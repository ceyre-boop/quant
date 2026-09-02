"""AST import guard: research/hyp111 must not import anything from the execution path."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ALLOWED_PREFIXES = ("research.hyp111", "research.modern._lib", "sovereign.discovery", "numpy", "pandas",
                    "requests", "pytest")
STDLIB = {"json", "time", "sys", "os", "pathlib", "dataclasses", "datetime", "hashlib", "argparse",
          "shutil", "tempfile", "math", "itertools", "collections", "__future__", "typing", "ast"}


def test_no_execution_path_imports():
    for f in PKG.glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                top = n.split(".")[0]
                assert top in STDLIB or n.startswith(ALLOWED_PREFIXES), f"{f.name}: forbidden import {n}"
