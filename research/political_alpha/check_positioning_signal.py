"""Phase 3 — pre-announcement rr25 positioning + manipulation flags (HYP-085, §7-P3).

Pinned formulas (plan §Pinned formulas (c)/(d)/(e), locked with the prereg):
  - chain calendar: the statement maps onto the CHAIN symbol's own session calendar
    with the us_etf rule, so both pre-window EODs precede the statement in every
    branch (a pre-open statement's D0-1 close is still strictly earlier).
  - pre_rr25_move = rr25(D0-1) - rr25(D0-3)   [EOD only; never widened to find data]
  - direction: expected_bull for the option underlying — native ETF chain: sign of
    the tagged ETF's post_return; FXE proxy (all forex rows, spec-pinned): USD-leg
    map — a USD-bearish resolution (EURUSD/GBPUSD/AUDUSD up, USDJPY/DXY down) is
    FXE-bullish. Zero move counts as NOT directional.
  - volume leg: put/call volume ratio on the same ~30d expiry;
    pre_pcr_move = pcr(D0-1) - pcr(D0-3); falling PCR = call-tilted = bullish.
  - condition (2) = rr25_directional OR volume_directional (spec §2 states the
    disjunction); manipulation_signal = post_big_move AND condition(2).
    DESCRIPTIVE — the study's p-value comes only from Phase 4's bootstrap.

Data availability is recorded, never synthesized: FXE is the confirmed feasible
chain; sector-ETF chains are probed per instrument (spec §3 feasibility).

STOP CONDITION: ThetaTerminal unreachable -> exit(2), print the remedy, write nothing.

Run:  python3 research/political_alpha/check_positioning_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402


def _sign(x: float) -> int:
    return int(np.sign(x)) if np.isfinite(x) else 0


def main() -> int:
    events = _lib.read_jsonl(_lib.DATA_DIR / "trump_events.jsonl")
    study = {(r["event_id"], r["instrument_tagged"]): r
             for r in _lib.read_jsonl(_lib.DATA_DIR / "event_study_results.jsonl")}
    if not events or not study:
        print("STOP: run Phase 1 and Phase 2 first.")
        return 2

    client = _lib.ThetaClient()
    if not client.alive():
        print("STOP: ThetaTerminal is not reachable at "
              f"{client.base_url} — start ThetaTerminal v3 (serves on :25503), then re-run "
              "this phase. Phases 1-2 outputs are unaffected. Nothing was written.")
        return 2

    # probe native ETF chains once (spec: record a gap where absent, never synthesize)
    chain_ok: dict[str, bool] = {}
    for etf, sym in _lib.ETF_CHAIN.items():
        try:
            chain_ok[sym] = len(client.expirations(sym)) > 0
        except Exception:
            chain_ok[sym] = False
    print(f"Phase 3 — chain availability: FXE=True (spec-pinned) native={chain_ok}")

    px_cache: dict[str, pd.DataFrame] = {}

    def px(symbol: str) -> pd.DataFrame:
        if symbol not in px_cache:
            px_cache[symbol] = _lib.fetch_daily(symbol)
        return px_cache[symbol]

    rr_cache: dict[tuple[str, str], dict | None] = {}

    def rr25_at(symbol: str, day: pd.Timestamp) -> dict | None:
        key = (symbol, str(day.date()))
        if key not in rr_cache:
            frame = px(symbol)
            spot = float(frame.loc[day, "Close"]) if (not frame.empty and day in frame.index) else float("nan")
            try:
                rr_cache[key] = _lib.rr25_for_day(client, symbol, day, spot)
            except RuntimeError as exc:            # circuit breaker — abort loudly
                raise SystemExit(f"STOP: {exc}")
        return rr_cache[key]

    rows = []
    for e in events:
        inst = e["instrument_tagged"]
        s = study.get((e["event_id"], inst), {})
        post_big = s.get("big_move")
        direction = s.get("direction")
        out = {
            "event_id": e["event_id"], "instrument_tagged": inst,
            "rr25_source": "unavailable", "chain_symbol": None,
            "pre_rr25_move": None, "pre_pcr_move": None,
            "pre_rr25_directional": None, "pre_volume_directional": None,
            "pre_directional": None,
            "post_big_move": post_big, "manipulation_signal": None,
            "positioning_available": False, "gap_reason": "",
        }

        # chain symbol per spec: forex rows -> FXE proxy; ETF rows -> native if served
        if _lib.ASSET_CLASS[inst] == "fx":
            symbol, source = _lib.FXE_SYMBOL, "FXE_proxy"
        elif chain_ok.get(_lib.ETF_CHAIN[inst]):
            symbol, source = _lib.ETF_CHAIN[inst], "native"
        else:
            out["gap_reason"] = "chain_not_served_on_tier"
            rows.append(out)
            continue
        out["chain_symbol"], out["rr25_source"] = symbol, source

        frame = px(symbol)
        if frame.empty:
            out["gap_reason"] = "no_underlying_price_data"
            rows.append(out)
            continue
        d0 = _lib.map_t0(e["timestamp_utc"], frame.index, "us_etf")
        if d0 is None:
            out["gap_reason"] = "chain_calendar_beyond_data"
            rows.append(out)
            continue
        pos = frame.index.get_loc(d0)
        if pos < 3:
            out["gap_reason"] = "pre_window_beyond_data"
            rows.append(out)
            continue
        d_m1, d_m3 = frame.index[pos - 1], frame.index[pos - 3]

        r1, r3 = rr25_at(symbol, d_m1), rr25_at(symbol, d_m3)
        if r1 is None or r3 is None:
            out["gap_reason"] = "rr25_missing_pre_window"
            rows.append(out)
            continue
        out["positioning_available"] = True
        out["pre_rr25_move"] = round(r1["rr25"] - r3["rr25"], 6)

        def pcr(read: dict) -> float | None:
            return (read["put_volume"] / read["call_volume"]) if read["call_volume"] > 0 else None

        p1, p3 = pcr(r1), pcr(r3)
        if p1 is not None and p3 is not None:
            out["pre_pcr_move"] = round(p1 - p3, 6)

        # expected_bull for the option underlying, from the resolved post direction
        if direction in ("up", "down"):
            move = +1 if direction == "up" else -1
            if source == "native":
                expected_bull = move
            else:                                   # FXE (EUR): USD-bearish == FXE-bullish
                usd_move = _lib.USD_SIGN[inst] * move
                expected_bull = -usd_move
            out["pre_rr25_directional"] = bool(_sign(out["pre_rr25_move"]) == expected_bull)
            if out["pre_pcr_move"] is not None:
                out["pre_volume_directional"] = bool(_sign(out["pre_pcr_move"]) == -expected_bull)
            legs = [x for x in (out["pre_rr25_directional"], out["pre_volume_directional"])
                    if x is not None]
            out["pre_directional"] = bool(any(legs)) if legs else None
        else:
            out["gap_reason"] = "post_direction_undefined"

        if post_big is not None and out["pre_directional"] is not None:
            out["manipulation_signal"] = bool(post_big and out["pre_directional"])
        rows.append(out)

    _lib.write_jsonl(_lib.DATA_DIR / "manipulation_flags.jsonl", rows)

    avail = [r for r in rows if r["positioning_available"]]
    flags = [r for r in rows if r["manipulation_signal"]]
    src_counts = pd.Series([r["rr25_source"] for r in rows]).value_counts().to_dict()
    gaps = pd.Series([r["gap_reason"] for r in rows if r["gap_reason"]]).value_counts().to_dict()
    print(f"  rows: {len(rows)}  positioning_available: {len(avail)}  "
          f"manipulation_signal=True: {len(flags)}")
    print(f"  rr25_source: {src_counts}")
    print(f"  gaps: {gaps if gaps else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
