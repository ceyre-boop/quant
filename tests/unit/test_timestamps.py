"""normalize_timestamp — regression for the naive-ISO parse bug.

Naive ISO-T timestamps (e.g. '2026-06-05T12:11:21', no tz) were not in _PARSE_FMTS, so they
returned unchanged with a warning. That caused the pulse's false '0 new entries' and an URGENT
anomaly (oracle_pulse RED). These assert they normalize to canonical UTC.
"""
from __future__ import annotations

import pytest

from sovereign.utils.timestamps import normalize_timestamp


@pytest.mark.parametrize("raw,expected", [
    ("2026-06-05T12:11:21", "2026-06-05T12:11:21+00:00"),      # ISO-T naive (the bug)
    ("2026-06-05T12:11:21.123456", "2026-06-05T12:11:21+00:00"),  # ISO-T naive + frac
    ("2026-06-05 12:11:21", "2026-06-05T12:11:21+00:00"),      # space-separated naive
    ("2026-06-08T12:00:10+00:00", "2026-06-08T12:00:10+00:00"),  # already canonical
    ("2026-06-05", "2026-06-05T00:00:00+00:00"),              # bare date
])
def test_normalize(raw, expected):
    out = normalize_timestamp(raw)
    assert out == expected, f"{raw!r} -> {out!r}"
    assert "+00:00" in out                                    # never a naive pass-through


def test_empty_safe():
    assert normalize_timestamp("") == ""
