#!/usr/bin/env python3
"""Scratch proof: a strategy reading the contract via platform.regime_client.
Run AFTER scripts/build_system_regime.py. Shows carry = STAND_ASIDE (narrowing)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alta_platform.regime_client import get_regime

r = get_regime("carry")
print(f"carry -> verdict={r.verdict} favorable={r.favorable} "
      f"size_multiplier={r.size_multiplier} stale={r.stale}")
print(f"reason: {r.reason}")
