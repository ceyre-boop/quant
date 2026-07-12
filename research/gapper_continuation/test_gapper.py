"""HYP-092 module tests — run explicitly: python3 -m pytest research/gapper_continuation/ -q
(Not part of tests/ — the 40-known-failure suite baseline is untouched.)
"""
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
from stage2_run import (classify, et_start, process_candidate,  # noqa: E402
                        SLICE_START, SLICE_END)

ET = ZoneInfo("America/New_York")


def bar(t_utc, o, h, l, c, v, vw=None):
    return {"t": t_utc, "o": o, "h": h, "l": l, "c": c, "v": v,
            "vw": vw if vw is not None else (h + l + c) / 3}


def rth_bar(day, hh, mm, o, h, l, c, v, vw=None):
    """Build a bar from an ET wall-clock time (DST-safe via zoneinfo)."""
    ts = datetime(2000, 1, 1).replace(year=int(day[:4]), month=int(day[5:7]),
                                      day=int(day[8:10]), hour=hh, minute=mm,
                                      tzinfo=ET)
    return bar(ts.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ"),
               o, h, l, c, v, vw)


def test_dst_conversion_edt_and_est():
    # July (EDT): 13:30Z == 09:30 ET ; January (EST): 14:30Z == 09:30 ET
    assert et_start(bar("2025-07-15T13:30:00Z", 1, 1, 1, 1, 1)).hour == 9
    assert et_start(bar("2026-01-15T14:30:00Z", 1, 1, 1, 1, 1)).hour == 9


def slice_of(bars):
    return [b for b in bars if SLICE_START <= et_start(b) <= SLICE_END]


def make_day(day, closes, vols, post=True):
    """12 slice bars 09:30..10:25 with given closes/vols + optional PM bars."""
    bars = []
    times = [(9, 30), (9, 35), (9, 40), (9, 45), (9, 50), (9, 55),
             (10, 0), (10, 5), (10, 10), (10, 15), (10, 20), (10, 25)]
    prev_c = closes[0]
    for (hh, mm), c, v in zip(times, closes, vols):
        o = prev_c
        h, l = max(o, c) * 1.001, min(o, c) * 0.999
        bars.append(rth_bar(day, hh, mm, o, h, l, c, v))
        prev_c = c
    if post:
        bars.append(rth_bar(day, 10, 30, prev_c, prev_c * 1.01, prev_c * 0.99,
                            prev_c * 1.005, 50000))
        bars.append(rth_bar(day, 15, 55, 5.0, 5.1, 4.9, 5.05, 50000))
    return bars


def test_cutoff_excludes_1030_bar():
    bars = make_day("2025-08-01", [3.0 + i * 0.05 for i in range(12)], [100000] * 12)
    s = slice_of(bars)
    assert len(s) == 12
    assert all(et_start(b) <= SLICE_END for b in s)


def test_clean_continuation_reads_cont():
    # steadily rising closes, rising into 10:25, volume on up bars
    closes = [3.0, 3.1, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.7]
    bars = make_day("2025-08-01", closes, [100000] * 12)
    read, votes, P, H, L, vwap, vsum = classify(slice_of(bars))
    assert read == "CONT", (read, votes)


def test_clean_exhaustion_reads_ex():
    # climax early, then fading below VWAP with lower highs
    closes = [4.0, 4.8, 4.6, 4.4, 4.2, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4]
    vols = [900000, 1500000, 400000, 300000, 200000, 150000,
            120000, 100000, 90000, 60000, 50000, 40000]
    bars = make_day("2025-08-01", closes, vols)
    read, votes, *_ = classify(slice_of(bars))
    assert read == "EX", (read, votes)


def test_mixed_reads_unc():
    # genuinely conflicting: P>=VWAP + higher lows (C=2) BUT lower highs +
    # climax-fade (E=2) with down-volume dominant and P below mid-range
    closes = [3.2, 5.0, 4.6, 4.4, 4.2, 4.1, 4.05, 4.0, 3.95, 3.9, 3.92, 3.9]
    vols = [5000000, 300000, 300000, 350000, 300000, 250000,
            200000, 180000, 150000, 130000, 110000, 100000]
    bars = make_day("2025-08-01", closes, vols)
    read, votes, *_ = classify(slice_of(bars))
    assert read == "UNC", (read, votes)


def test_determinism():
    closes = [3.0, 3.1, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.7]
    bars = make_day("2025-08-01", closes, [100000] * 12)
    a = classify(slice_of(bars))
    b = classify(slice_of(bars))
    assert a == b


def test_process_candidate_outcome_origin_is_1030_open():
    closes = [3.0, 3.1, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.9]
    bars = make_day("2025-08-01", closes, [100000] * 12, post=True)
    daily = [{"t": "2025-07-31T04:00:00Z", "c": 2.5, "o": 2.5, "h": 2.6,
              "l": 2.4, "v": 1000000, "vw": 2.5}]
    from collections import defaultdict
    counters = defaultdict(int)
    row = process_candidate("TEST", bars, daily, counters)
    assert row is not None, dict(counters)
    # entry = open of the 10:30 bar (prev close chain -> 3.9), NOT 10:25 close
    assert abs(row["entry_open_1030"] - 3.9) < 1e-9
    # outcome end = close of last bar before 16:00 (the 15:55 bar, 5.05)
    assert abs(row["close_eod"] - 5.05) < 1e-9
    assert abs(row["outcome_pct"] - (5.05 / 3.9 - 1)) < 5.1e-5  # csv rounds 4dp
    # gain filter used prev daily close 2.5: P=3.9 => +56% passes 30%
    assert row["gain_1030"] > 0.30


def test_sparse_day_excluded():
    closes = [3.0, 3.1, 3.2, 3.25]
    times = [(9, 30), (9, 40), (9, 50), (10, 0)]  # only 4 bars, last < 10:15
    bars = []
    prev_c = 3.0
    for (hh, mm), c in zip(times, closes):
        bars.append(rth_bar("2025-08-01", hh, mm, prev_c, max(prev_c, c),
                            min(prev_c, c), c, 100000))
        prev_c = c
    daily = [{"t": "2025-07-31T04:00:00Z", "c": 2.0, "o": 2, "h": 2, "l": 2,
              "v": 500000, "vw": 2.0}]
    from collections import defaultdict
    counters = defaultdict(int)
    assert process_candidate("TEST", bars, daily, counters) is None
    assert counters["unreadable_sparse"] == 1


def test_isolation_no_sovereign_or_ict_imports():
    here = Path(__file__).parent
    needles = ["from " + "sovereign", "import " + "sovereign",
               "from " + "ict", "import " + "ict"]
    for py in here.glob("*.py"):
        if py.name == Path(__file__).name:
            continue  # this file contains the needle strings
        src = py.read_text()
        for n in needles:
            assert n not in src, f"{py.name} contains '{n}'"
