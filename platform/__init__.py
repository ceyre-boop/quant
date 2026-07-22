"""platform — Alta Investments shared, neutral connective layer.

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
"""

# --------------------------------------------------------------------------
# stdlib-shadow guard (IMPORTANT — do not remove)
# --------------------------------------------------------------------------
# The spec (specs/system_regime_contract.md) mandates this package be named
# `platform` and imported as `from platform.regime_client import get_regime`.
# But `platform` is ALSO a Python standard-library module (platform.system(),
# platform.python_version(), ...). With the repo root on sys.path (as pytest and
# the live scripts put it), THIS package shadows the stdlib one, and anything
# that does `import platform; platform.system()` — notably stdlib `uuid`, and
# therefore pytest — breaks with `AttributeError: module 'platform' has no
# attribute 'system'`.
#
# Rather than rename away from the spec, we make this package a transparent
# SUPERSET of the stdlib module: we load the real stdlib `platform` by file
# path and copy its public attributes into our namespace. `platform.system()`
# keeps working; `platform.regime_client` is ours. One name, both behaviours.
def _absorb_stdlib_platform() -> None:
    import importlib.util
    import sysconfig
    import os as _os

    stdlib_dir = sysconfig.get_paths().get("stdlib", "")
    real = _os.path.join(stdlib_dir, "platform.py")
    if not _os.path.isfile(real):
        return
    spec = importlib.util.spec_from_file_location("_stdlib_platform", real)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return
    g = globals()
    for name in dir(mod):
        if name.startswith("__"):
            continue
        g.setdefault(name, getattr(mod, name))


_absorb_stdlib_platform()

__all__ = ["regime_client", "regime_contract"]
