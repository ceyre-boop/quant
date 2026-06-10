"""Econ calendar artifact tests — counts per year, schema, known dates."""
import json
from pathlib import Path

import pytest

CAL_PATH = Path(__file__).resolve().parents[2] / "data" / "es_nq" / "econ_calendar_2018_2026.json"


@pytest.fixture(scope="module")
def cal():
    if not CAL_PATH.exists():
        pytest.skip("calendar not built — run scripts/build_econ_calendar_es_nq.py")
    return json.loads(CAL_PATH.read_text())


def _count(cal, year, event):
    return sum(1 for d, v in cal.items()
               if not d.startswith("_") and d[:4] == year and event in v["events"])


def test_fomc_counts(cal):
    for yr in ("2018", "2019", "2021", "2022", "2023", "2024", "2025"):
        assert _count(cal, yr, "FOMC") == 8, yr
    assert _count(cal, "2020", "FOMC") == 9   # incl. Mar 3 + Mar 15 emergency actions


def test_cpi_nfp_counts_within_band(cal):
    for yr in (str(y) for y in range(2018, 2026)):
        assert 10 <= _count(cal, yr, "CPI") <= 14, yr
        assert 10 <= _count(cal, yr, "NFP") <= 14, yr


def test_known_event_dates(cal):
    assert "FOMC" in cal["2022-06-15"]["events"]   # 75bp hike
    assert "FOMC" in cal["2023-03-22"]["events"]
    assert "FOMC" in cal["2024-09-18"]["events"]   # 50bp cut


def test_schema(cal):
    assert "_meta" in cal
    for d, v in cal.items():
        if d.startswith("_"):
            continue
        assert len(d) == 10 and d[4] == "-" and d[7] == "-"
        assert isinstance(v["events"], list) and v["events"]
        assert set(v["events"]) <= {"FOMC", "CPI", "NFP"}
