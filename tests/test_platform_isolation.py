"""Isolation test for the neutral platform/ connective layer.

CLAUDE.md NON-NEGOTIABLE #1: ict/ and ict-engine/ must never import sovereign/.
The platform/ package sits ABOVE both walls and is importable from BOTH sides,
so it must import NEITHER — otherwise importing it from ict/ would drag in
sovereign/ transitively and breach the wall.

This test asserts isolation BOTH DIRECTIONS:
  (A) static — no platform/*.py source contains an import of ict or sovereign.
  (B) runtime — importing platform.* pulls neither 'ict' nor 'sovereign'
      (nor their submodules) into sys.modules.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLATFORM = REPO / "platform"

FORBIDDEN_ROOTS = {"ict", "ict_engine", "sovereign", "layer1", "layer2", "layer3"}


def _imported_roots(py: Path) -> set[str]:
    """Top-level package names imported by a source file, via AST (robust to
    comments/strings, unlike substring matching)."""
    tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # absolute imports only
                roots.add(node.module.split(".")[0])
    return roots


def test_platform_source_imports_neither_ict_nor_sovereign():
    """(A) Static scan of every platform/*.py file."""
    files = sorted(PLATFORM.glob("*.py"))
    assert files, "platform/ has no python files — nothing to check"
    for py in files:
        roots = _imported_roots(py)
        bad = roots & FORBIDDEN_ROOTS
        assert not bad, f"Isolation violated: {py.name} imports {sorted(bad)}"


def test_importing_platform_pulls_neither_side_at_runtime():
    """(B) Runtime: import platform.* in a clean interpreter and assert neither
    'ict' nor 'sovereign' (nor submodules) ended up in sys.modules."""
    code = (
        "import sys\n"
        "import platform.regime_client\n"
        "import platform.regime_contract\n"
        "bad = [m for m in sys.modules\n"
        "       if m.split('.')[0] in {'ict','ict_engine','sovereign','layer1','layer2','layer3'}]\n"
        "assert not bad, 'platform import pulled in: %r' % bad\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"runtime isolation failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
