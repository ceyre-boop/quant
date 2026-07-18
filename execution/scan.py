"""Candidate screening and the two frozen signal filters.

Ported from `research/yield_frontier/live_shadow.py:103` (HYP-093) and
`research/gapper/hyp107_shadow.py:115` (HYP-107). Filter logic is reproduced
exactly; the constants now come from `execution.config.FROZEN` so the two legs
can no longer drift apart the way the forked scripts did.

The `passes_*` functions are PURE — no I/O — so they unit-test against fixtures
without network access, and every rejection returns a reason string so a
zero-signal day is auditable rather than mysterious.

A KNOWN FORK, PRESERVED DELIBERATELY
------------------------------------
The two legs screen the movers list at different thresholds:
    HYP-107 -> percent_change >= 30
    HYP-093 -> percent_change >= 40
This is not a bug to tidy. Both are frozen values from their respective preregs,
and unifying them would silently change one leg's event set. The harness screens
at the LOWER threshold and applies each leg's own floor afterwards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from typing import Any

from execution import alpaca
from execution.config import frozen

ET = alpaca.ET
UTC = alpaca.UTC


@dataclass
class Candidate:
    """One screened symbol with everything both filters need."""
    symbol: str
    day: date
    prev_close: float
    bars: list[dict] = field(default_factory=list, repr=False)

    # HYP-107 inputs (09:30 bar)
    open_0930: float | None = None
    vol_0930: int | None = None
    overnight_gap: float | None = None
    log_vol: float | None = None

    # HYP-093 inputs (09:30-10:25 window)
    price_1025: float | None = None
    cum_vol_1025: int | None = None
    gain_1025: float | None = None

    excluded: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "ticker": self.symbol, "prev_close": self.prev_close,
            "overnight_gap": self.overnight_gap, "log_vol": self.log_vol,
            "first_min_vol": self.vol_0930,
            "price_1025": self.price_1025, "gain_1025": self.gain_1025,
            "cum_vol_1025": self.cum_vol_1025,
        }


def symbol_shape_ok(sym: str) -> bool:
    """Exclude warrants / rights / units, matching both shadows verbatim."""
    sc = frozen("screener")
    return (sym.isalpha() and len(sym) <= sc["max_symbol_len"]
            and not (len(sym) == sc["max_symbol_len"]
                     and sym[-1] in sc["excluded_last_chars"]))


def bar_at(bars: list[dict], hh: int, mm: int) -> dict | None:
    """First bar at or after HH:MM ET."""
    tgt = dtime(hh, mm)
    for b in bars:
        if alpaca.et_t(b) >= tgt:
            return b
    return None


def bar_exactly(bars: list[dict], hh: int, mm: int) -> dict | None:
    """Bar covering exactly HH:MM ET, or None if that minute did not print."""
    tgt = dtime(hh, mm)
    for b in bars:
        if alpaca.et_t(b) == tgt:
            return b
    return None


def _parse_et(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm))


def has_mna_headline(symbol: str, day: date) -> bool:
    """True if a merger/acquisition headline landed before 10:30 ET.

    Window is prior-session 16:00 ET through 10:30 ET, matching live_shadow.
    """
    sc = frozen("screener")
    s = alpaca.et_dt(day - timedelta(days=1), dtime(16, 0))
    e = alpaca.et_dt(day, dtime(10, 30))
    articles = alpaca.news(symbol, s, e)
    blob = " ".join(a.get("headline", "").lower() for a in articles)
    return any(k in blob for k in sc["buckets_mna"])


ARCHIVE_CSV = (alpaca.ROOT / "data" / "research" / "gapper"
               / "per_candidate_enriched.csv")


def archived_symbols(day: date) -> list[str]:
    """Gapper symbols recorded for `day` in the historical candidate archive.

    REPLAY CANNOT USE THE LIVE SCREENER. `alpaca.movers()` returns whatever is
    moving RIGHT NOW; asking it about a past session silently returns today's
    universe, which produces a plausible-looking but meaningless replay. This is
    the archived universe those events were actually drawn from.
    """
    import csv as _csv
    if not ARCHIVE_CSV.exists():
        return []
    out: list[str] = []
    with open(ARCHIVE_CSV, newline="") as fh:
        for row in _csv.DictReader(fh):
            if row.get("date") == str(day):
                sym = (row.get("ticker") or "").strip().upper()
                if sym and symbol_shape_ok(sym):
                    out.append(sym)
    return sorted(set(out))


def scan_universe(day: date, *, lag_minutes: int | None = None,
                  check_news: bool = True,
                  symbols: list[str] | None = None) -> list[Candidate]:
    """Screen for candidates on `day` and build Candidates.

    `symbols` overrides the universe source. Live runs leave it None and use the
    movers screener; replay runs MUST pass an archived universe, because the
    screener has no historical mode (see `archived_symbols`).

    Screens at the LOWER of the two legs' movers thresholds; each leg applies
    its own floor in its `passes_*` function.
    """
    cap = frozen("capture")
    lag = lag_minutes if lag_minutes is not None else cap["sip_lag_minutes"]
    sc = frozen("screener")
    c107 = frozen("hyp107")
    c093 = frozen("hyp093")

    if symbols is not None:
        symbols = [s for s in symbols if symbol_shape_ok(s)]
    else:
        floor = min(c107["movers_pct_change_min"], c093["movers_pct_change_min"])
        gainers = alpaca.movers(sc["movers_top"])
        symbols = [m["symbol"] for m in gainers
                   if m.get("percent_change", 0) >= floor
                   and symbol_shape_ok(m["symbol"])]

    out: list[Candidate] = []
    for sym in symbols:
        bars = alpaca.minute_bars(sym, day, lag_minutes=lag)
        pc = alpaca.daily_prev_close(sym, day)
        if not bars or not pc or pc <= 0:
            continue

        cand = Candidate(symbol=sym, day=day, prev_close=float(pc), bars=bars)

        b0930 = bar_at(bars, 9, 30)
        if b0930:
            cand.open_0930 = float(b0930["o"])
            cand.vol_0930 = int(b0930["v"])
            cand.overnight_gap = cand.open_0930 / pc - 1
            cand.log_vol = math.log10(cand.vol_0930 + 1)

        m_start = _parse_et(c093["measure_start_et"])
        m_end = _parse_et(c093["measure_end_et"])
        window = [b for b in bars if m_start <= alpaca.et_t(b) <= m_end]
        if window:
            cand.price_1025 = float(window[-1]["c"])
            cand.cum_vol_1025 = int(sum(b["v"] for b in window))
            cand.gain_1025 = cand.price_1025 / pc - 1
            cand._window_bars = window          # type: ignore[attr-defined]

        if check_news and has_mna_headline(sym, day):
            cand.excluded = "mna_headline"

        out.append(cand)
    return out


def passes_hyp107(c: Candidate) -> tuple[bool, str]:
    """Frozen HYP-107 LONG filter — all inputs available at 09:31.

    overnight_gap >= gap_floor  AND  overnight_gap <= og_max
    AND log10(first-minute volume + 1) <= logvol_max
    """
    cfg = frozen("hyp107")
    if c.excluded:
        return False, c.excluded
    if c.overnight_gap is None or c.log_vol is None:
        return False, "missing_0930_bar"
    if c.overnight_gap < cfg["gap_floor"]:
        return False, f"gap_below_floor_{c.overnight_gap:.4f}"
    if c.overnight_gap > cfg["og_max"]:
        return False, f"gap_above_og_max_{c.overnight_gap:.4f}"
    if c.log_vol > cfg["logvol_max"]:
        return False, f"logvol_above_max_{c.log_vol:.4f}"
    return True, "pass"


def passes_hyp093(c: Candidate) -> tuple[bool, str]:
    """Frozen HYP-093 SHORT filter (prereg c5b10616).

        P >= qual_gain * prev_close   (1.30x)
        P >= price_min                ($2.00)
        cumulative volume >= vol_min  (500k)
        gain >= gain_min              (0.50)

    Plus the structural window guards: at least `min_bars` bars in the
    09:30-10:25 window and a last bar no earlier than 10:15.
    """
    cfg = frozen("hyp093")
    if c.excluded:
        return False, c.excluded
    if c.price_1025 is None or c.gain_1025 is None or c.cum_vol_1025 is None:
        return False, "missing_measure_window"

    window = getattr(c, "_window_bars", [])
    if len(window) < cfg["min_bars"]:
        return False, f"too_few_bars_{len(window)}"
    if window and alpaca.et_t(window[-1]) < _parse_et(cfg["min_last_bar_et"]):
        return False, "window_ends_too_early"

    if c.price_1025 < cfg["qual_gain"] * c.prev_close:
        return False, f"below_qual_gain_{c.price_1025:.4f}"
    if c.price_1025 < cfg["price_min"]:
        return False, f"below_price_min_{c.price_1025:.4f}"
    if c.cum_vol_1025 < cfg["vol_min"]:
        return False, f"below_vol_min_{c.cum_vol_1025}"
    if c.gain_1025 < cfg["gain_min"]:
        return False, f"below_gain_min_{c.gain_1025:.4f}"
    return True, "pass"
