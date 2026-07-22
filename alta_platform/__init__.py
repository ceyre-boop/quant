"""alta_platform — Alta Investments shared, neutral connective layer.

This package is the "nervous system" that connects every strategy to one live
regime read. It is deliberately NEUTRAL:

  * It imports NOTHING from ``ict/`` and NOTHING from ``sovereign/``.
  * It communicates ONLY by data contract — reading and writing JSON files.
    A file read is not an import, so the ``ict/`` <-> ``sovereign/`` isolation
    wall (CLAUDE.md NON-NEGOTIABLE #1) is never touched by this code.

Because it imports neither side, it is importable from BOTH ``ict/`` and
``sovereign/`` without creating a cross-layer dependency. That is the whole
point: one place both walls can read from.

Contents:
  * ``regime_client``   — the reader: ``get_regime(strategy) -> RegimeRead``.
  * ``regime_contract`` — the writer's data-model + classifiers.

The writer script lives at ``scripts/build_system_regime.py``.

Naming: this package was originally named ``platform``, which SHADOWED Python's
stdlib ``platform`` module (used by ``uuid``, and therefore ``pytest``). It was
renamed to ``alta_platform`` on 2026-07-22 so the stdlib name is never masked;
the transparent-superset workaround that renaming required is no longer needed.
"""

__all__ = ["regime_client", "regime_contract"]
