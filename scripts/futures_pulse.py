#!/usr/bin/env python3
"""
The Big Move Oracle pulse — recurring intraday forecast of the day's ONE big move.

Every 15 min during RTH, fuses structure + CVD + regime + VIX-implied move + catalysts into
a single Sonnet-4.6 BigMoveForecast, logs it, feeds the dashboard, and pings Telegram only
when the call materially changes. Cognition only — places NO orders.

Usage:
    python3.13 scripts/futures_pulse.py --once --instrument MES --source ib
    python3.13 scripts/futures_pulse.py --loop --instrument MES        # every 15 min, RTH only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.futures import big_move_oracle as bmo          # noqa: E402
from sovereign.futures import telegram_gateway as tg          # noqa: E402

ET = ZoneInfo("America/New_York")
PULSE_LOG = ROOT / "data" / "futures" / "big_move_pulse.jsonl"
DASH_FEED = ROOT / "data" / "agent" / "big_move.json"
G, R, Y, BD, DM, RS = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[2m", "\033[0m"


def _in_rth(now=None) -> bool:
    et = (now or datetime.now(timezone.utc)).astimezone(ET)
    if et.weekday() >= 5:
        return False
    hm = et.hour * 60 + et.minute
    return 9 * 60 + 30 <= hm <= 16 * 60


def _last_logged(instrument: str) -> dict | None:
    if not PULSE_LOG.exists():
        return None
    last = None
    for line in PULSE_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if r.get("instrument") == instrument:
                last = r
        except Exception:
            pass
    return last


def _summary(fc: bmo.BigMoveForecast) -> str:
    dc = G if fc.direction == "LONG" else (R if fc.direction == "SHORT" else Y)
    return (f"{dc}{BD}{fc.direction}{RS} {fc.instrument}  →  draw to {fc.drawn_to_level}  "
            f"(~{fc.expected_move_pts}pt, VIX {fc.vix})\n"
            f"  conv {fc.conviction}/3 · p={fc.stated_probability:.0%} · {fc.trigger_window} · {fc.catalyst}\n"
            f"  {DM}falsifier: {fc.falsifier}{RS}\n  {DM}{fc.reasoning}{RS}")


def _pulse_once(instrument: str, source: str, notify: bool) -> bmo.BigMoveForecast | None:
    ctx = bmo.gather_context(instrument, source=source)
    fc = bmo.forecast(ctx)
    if fc is None:
        print(f"{Y}  [{instrument}] forecast unavailable (API/parse) — context logged only{RS}")
        return None

    print(f"\n{BD}══ BIG MOVE PULSE {datetime.now(ET).strftime('%H:%M ET')} ══{RS}")
    print(_summary(fc))

    rec = fc.to_dict()
    rec["context_errors"] = ctx.get("errors", [])
    PULSE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PULSE_LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")

    # dashboard feed — the proactive, always-current call the user sees
    DASH_FEED.parent.mkdir(parents=True, exist_ok=True)
    DASH_FEED.write_text(json.dumps({
        "updated_at": fc.ts, "instrument": fc.instrument, "direction": fc.direction,
        "expected_move_pts": fc.expected_move_pts, "drawn_to_level": fc.drawn_to_level,
        "trigger_window": fc.trigger_window, "catalyst": fc.catalyst,
        "conviction": fc.conviction, "probability": fc.stated_probability,
        "falsifier": fc.falsifier, "reasoning": fc.reasoning,
        "vix": fc.vix, "implied_move_pts": fc.implied_move_pts,
        "regime": fc.regime_state, "cvd_slope": fc.cvd_slope, "model": fc.model,
    }, indent=2))

    # Telegram only on a MATERIAL change (direction flip or high conviction) — not every pulse
    if notify and tg.enabled():
        prev = _last_logged(instrument)
        flipped = prev and prev.get("direction") != fc.direction
        if (flipped or fc.conviction >= 3) and fc.direction in ("LONG", "SHORT"):
            tag = "🔄 FLIP" if flipped else "⚡ HIGH CONVICTION"
            tg.send(f"{tag} — {fc.instrument} BIG MOVE {fc.direction} → {fc.drawn_to_level} "
                    f"(~{fc.expected_move_pts}pt, conv {fc.conviction}/3, p={fc.stated_probability:.0%})\n"
                    f"{fc.catalyst} | {fc.trigger_window}")
    return fc


def main() -> None:
    ap = argparse.ArgumentParser(description="Big Move Oracle pulse")
    ap.add_argument("--instrument", default="MES", choices=["MES", "MNQ"])
    ap.add_argument("--source", default="ib", choices=["ib", "yf"])
    ap.add_argument("--once", action="store_true", help="single pulse and exit")
    ap.add_argument("--loop", action="store_true", help="every 15 min during RTH")
    ap.add_argument("--interval", type=int, default=900, help="loop seconds (default 900=15min)")
    ap.add_argument("--no-notify", action="store_true", help="never ping Telegram")
    args = ap.parse_args()
    notify = not args.no_notify

    if args.loop:
        print(f"{BD}Big Move Oracle — {args.instrument} every {args.interval//60} min during RTH "
              f"(Ctrl+C to stop){RS}")
        try:
            while True:
                if _in_rth():
                    _pulse_once(args.instrument, args.source, notify)
                else:
                    print(f"{DM}  [{datetime.now(ET).strftime('%H:%M ET')}] outside RTH — idle{RS}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n  Pulse stopped.")
    else:
        _pulse_once(args.instrument, args.source, notify)


if __name__ == "__main__":
    main()
