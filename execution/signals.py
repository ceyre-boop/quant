"""Layer 3 — ranked GO / NO-GO signal list.

Reuses the frozen, hash-locked filters in `execution/scan.py` (TICK-038). No
thresholds are defined here and none may be: `execution/config.py` is the single
source and `verify_frozen_hash()` fails startup on drift.

WHY NO_GO ROWS ARE KEPT
-----------------------
Every candidate is emitted with its decision AND its reason, including the ones
that failed. A zero-signal day must be auditable — "nothing qualified" and "the
scanner silently returned nothing" look identical in a file that only records
passes, and this repo has already been bitten by exactly that class of bug
(Reddit reporting success with zero posts, recorded downstream as data).

RANKING
-------
GO rows are ranked, but ranking is NOT a conviction score and must not be read as
one. HYP-107 is ordered by how far inside its frozen gap/volume band a candidate
sits; HYP-093 by cumulative volume. Both are descriptive orderings of an already
binary pass, not a prediction of which will pay. Nothing downstream sizes on rank.

THIS MODULE DOES NOT IMPORT `execution.bias`, AND MUST NOT.
`tests/unit/test_bias_isolation.py` enforces it. See ARCHITECTURE.md:284-286.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from execution import borrow, scan
from execution.config import FROZEN_HASH, frozen, verify_frozen_hash
from sovereign.utils.timestamps import canonical_timestamp

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "signals"
UTC = timezone.utc


@dataclass
class Signal:
    signal_id: str
    date: str
    ticker: str
    hypothesis: str
    side: str
    decision: str                    # GO | NO_GO
    reason: str
    rank: int | None = None
    rank_metric: float | None = None
    entry_et: str = ""
    exit_et: str = ""
    features: dict[str, Any] = field(default_factory=dict)
    frozen_hash: str = FROZEN_HASH

    def to_json(self) -> dict:
        return asdict(self)


def _sid(day: date | str, ticker: str, hyp: str) -> str:
    return f"{day}:{ticker}:{hyp}"


def _rank_107(c) -> float:
    """Headroom inside the frozen band. Descriptive ordering, not conviction."""
    cfg = frozen("hyp107")
    gap_room = (cfg["og_max"] - (c.overnight_gap or 0.0)) / max(cfg["og_max"], 1e-9)
    vol_room = (cfg["logvol_max"] - (c.log_vol or 0.0)) / max(cfg["logvol_max"], 1e-9)
    return round((gap_room + vol_room) / 2.0, 6)


def build_signals(day: date, *, replay: bool = False,
                  check_news: bool = True) -> list[Signal]:
    """Score every candidate against both frozen filters."""
    verify_frozen_hash()
    c107, c093 = frozen("hyp107"), frozen("hyp093")

    symbols = None
    if replay:
        symbols = scan.archived_symbols(day)
        if not symbols:
            return []

    cands = scan.scan_universe(day, check_news=check_news, symbols=symbols)
    locate = borrow.load_locate(day)
    out: list[Signal] = []

    for c in cands:
        ok, reason = scan.passes_hyp107(c)
        out.append(Signal(
            signal_id=_sid(day, c.symbol, "HYP-107"), date=str(day), ticker=c.symbol,
            hypothesis="HYP-107", side="LONG",
            decision="GO" if ok else "NO_GO", reason=reason,
            rank_metric=_rank_107(c) if ok else None,
            entry_et=c107["entry_bar_et"], exit_et=c107["exit_bar_et"],
            features={"overnight_gap": c.overnight_gap, "log_vol": c.log_vol,
                      "first_min_vol": c.vol_0930},
        ))

    for c in cands:
        ok, reason = scan.passes_hyp093(c)
        decision, why = ("GO", reason) if ok else ("NO_GO", reason)
        if ok:
            # Borrow is part of the GO decision for a short: an unborrowable name
            # is not a tradeable signal, and recording it as GO would overstate
            # the opportunity set.
            allowed, breason = borrow.borrow_ok(c.symbol, locate)
            if not allowed:
                decision, why = "NO_GO", breason
        out.append(Signal(
            signal_id=_sid(day, c.symbol, "HYP-093"), date=str(day), ticker=c.symbol,
            hypothesis="HYP-093", side="SHORT",
            decision=decision, reason=why,
            rank_metric=float(c.cum_vol_1025) if decision == "GO" else None,
            entry_et=c093["entry_bar_et"], exit_et=c093["exit_bar_et"],
            features={"gain_1025": c.gain_1025, "price_1025": c.price_1025,
                      "cum_vol_1025": c.cum_vol_1025},
        ))

    for hyp in ("HYP-107", "HYP-093"):
        gos = [s for s in out if s.hypothesis == hyp and s.decision == "GO"]
        gos.sort(key=lambda s: s.rank_metric or 0.0, reverse=True)
        for i, s in enumerate(gos, 1):
            s.rank = i
    return out


def write_signals(day: date, signals: list[Signal],
                  out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    gos = [s for s in signals if s.decision == "GO"]
    doc = {
        "date": str(day),
        "generated_at": canonical_timestamp(),
        "frozen_hash": FROZEN_HASH,
        "n_candidates": len({s.ticker for s in signals}),
        "n_go": len(gos),
        "n_no_go": len(signals) - len(gos),
        "signals": [s.to_json() for s in signals],
        "note": ("NO_GO rows are retained with reasons so a zero-signal day is "
                 "auditable. Rank is a descriptive ordering of a binary pass, "
                 "NOT a conviction score — nothing downstream sizes on it."),
    }
    p = out_dir / f"signals_{day}.json"
    p.write_text(json.dumps(doc, indent=2, default=str))
    return p


def load_signals(day: date | str, out_dir: Path | None = None) -> list[dict]:
    p = (out_dir or OUT_DIR) / f"signals_{day}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("signals", [])


def go_list(day: date | str, out_dir: Path | None = None) -> list[dict]:
    return [s for s in load_signals(day, out_dir) if s["decision"] == "GO"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Layer 3 — ranked GO/NO-GO signals")
    ap.add_argument("--day", default=None)
    ap.add_argument("--replay", action="store_true",
                    help="source the universe from the archive (past dates)")
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    day = date.fromisoformat(args.day) if args.day else datetime.now(UTC).date()
    sigs = build_signals(day, replay=args.replay, check_news=not args.no_news)
    p = write_signals(day, sigs, Path(args.out) if args.out else None)

    gos = [s for s in sigs if s.decision == "GO"]
    print(f"{day}: {len(gos)} GO / {len(sigs) - len(gos)} NO_GO "
          f"across {len({s.ticker for s in sigs})} candidates")
    for s in sorted(gos, key=lambda s: (s.hypothesis, s.rank or 0)):
        print(f"  GO    #{s.rank} {s.hypothesis} {s.side:<5} {s.ticker:<6} "
              f"{s.entry_et}->{s.exit_et}  metric={s.rank_metric}")
    shown = 0
    for s in sigs:
        if s.decision == "NO_GO" and shown < 8:
            print(f"  NO_GO   {s.hypothesis} {s.ticker:<6} {s.reason}")
            shown += 1
    if sigs and shown == 0 and not gos:
        print("  (no candidates screened)")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
