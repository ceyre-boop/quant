#!/usr/bin/env python3
"""
Futures session PLAYER — watch a whole day's trading fast-forward like a live tape.

Same engine as scripts/futures_replay.py (it animates simulate_session's events), so what
you watch is exactly what the report says — a full RTH session reviews in seconds, not all day.

Usage:
    python3.13 scripts/futures_player.py                       # most recent MES session, ~30s
    python3.13 scripts/futures_player.py --day 2026-06-05 --seconds 20
    python3.13 scripts/futures_player.py --play-all --turbo    # every recent day, instant
    python3.13 scripts/futures_player.py --instrument MNQ --source ib --seconds 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sovereign.futures import bar_feed as bf          # noqa: E402
import futures_replay as fr                           # noqa: E402

G, R, Y, DM, BD, RS = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
BLOCKS = "▁▂▃▄▅▆▇█"


def _spark(vals, width: int = 28) -> str:
    v = vals[-width:]
    if not v:
        return ""
    lo, hi = min(v), max(v)
    if hi == lo:
        return BLOCKS[0] * len(v)
    return "".join(BLOCKS[int((x - lo) / (hi - lo) * (len(BLOCKS) - 1))] for x in v)


class Renderer:
    """Animates simulate_session events into a live-style HUD."""

    def __init__(self, instrument: str, seconds: float, turbo: bool):
        self.instrument = instrument
        self.seconds = seconds
        self.turbo = turbo
        self.sleep = 0.0
        self.equity = [0.0]

    def __call__(self, kind: str, p: dict) -> None:
        if kind == "bar":
            if self.sleep == 0.0 and not self.turbo and p["n"]:
                self.sleep = self.seconds / max(p["n"], 1)
            self._bar(p)
            if not self.turbo:
                time.sleep(self.sleep)
        elif kind == "entry":
            self._entry(p)
        elif kind == "exit":
            self._exit(p)

    def _bar(self, p: dict) -> None:
        ind = p["ind"]
        et = p["ts"].astimezone(bf.ET).strftime("%H:%M")
        bias = p["bias"]
        bc = G if bias == "LONG" else (R if bias == "SHORT" else Y)
        vd = ind.last_price - ind.vwap
        arr = "▲" if vd >= 0 else "▼"
        ema = "↑" if ind.last_price > ind.ema_slow else ("↓" if ind.last_price < ind.ema_fast else "≈")
        eq = p["realized"]
        ec = G if eq >= 0 else R
        pos = f"{Y}●{RS}" if p["in_position"] else " "
        line = (f"[{et} ET] {BD}{self.instrument} {ind.last_price:>9.2f}{RS} "
                f"VWAP {arr}{abs(vd):>4.2f}  RSI {ind.rsi:4.1f} EMA{ema}  "
                f"| {bc}{bias:<7}{RS} | R {p['session_r']:+.1f} | "
                f"{ec}eq ${eq:+7.0f}{RS} {pos} {DM}{p['i']:>3}/{p['n']:<3}{RS} "
                f"{G}{_spark(self.equity)}{RS}")
        print(f"\r{line:<150}", end="", flush=True)

    def _entry(self, t: dict) -> None:
        dc = G if t["direction"] == "LONG" else R
        print(f"\n  ⚡ {dc}{BD}ENTER {t['direction']} {t['contracts']} {self.instrument} "
              f"@ {t['entry']:.2f}{RS}  SL {t['stop']:.2f}  TP {t['target']:.2f}  "
              f"{DM}({t['setup'].lower()}){RS}")

    def _exit(self, t: dict) -> None:
        self.equity.append(self.equity[-1] + t["net_usd"])
        win = t["net_usd"] > 0
        mark = f"{G}✓{RS}" if win else f"{R}✗{RS}"
        col = G if win else R
        print(f"  {mark} {t['exit_reason']:<8} {col}{t['r_realized']:+.2f}R  "
              f"${t['net_usd']:+7.2f}{RS}  | eq ${self.equity[-1]:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Animated fast-forward futures session player")
    ap.add_argument("--instrument", default="MES", choices=["MES", "MNQ"])
    ap.add_argument("--source", default="yf", choices=["yf", "ib"])
    ap.add_argument("--day", default=None, help="ET day YYYY-MM-DD")
    ap.add_argument("--lookback", default="5d")
    ap.add_argument("--bias", default="auto", choices=["auto", "long", "short", "neutral"])
    ap.add_argument("--orb-size", default="safe", choices=["safe", "big"])
    ap.add_argument("--seconds", type=float, default=30.0, help="approx run time per session (default 30)")
    ap.add_argument("--turbo", action="store_true", help="no pacing — instant")
    ap.add_argument("--play-all", action="store_true", help="play every available session back-to-back")
    args = ap.parse_args()

    print(f"Loading {args.instrument} history ({args.source})...", end=" ", flush=True)
    # Always load the full window (not just --day) so the prior session is present for
    # prior-close/bias context; --day only SELECTS which session to play.
    df = bf.load_history(args.instrument, source=args.source, day=None, lookback=args.lookback)
    if df is None or len(df) == 0:
        print("\n  No data. (yfinance 1m only covers ~7 days; try --source ib.)")
        sys.exit(1)
    print("done.")

    all_days = bf.session_days(df)
    if args.day:
        days = [args.day]
    elif args.play_all:
        days = all_days
    else:
        days = all_days[-1:]          # most recent session by default

    renderer = Renderer(args.instrument, args.seconds, args.turbo)
    sessions = []
    prior_close = None
    for day in all_days:              # walk all for prior_close continuity
        day_df = df[df.index.tz_convert(bf.ET).strftime("%Y-%m-%d") == day]
        if len(day_df) < 3:
            prior_close = float(day_df["Close"].iloc[-1]) if len(day_df) else prior_close
            continue
        if day in days:
            bias_dir, key_levels = fr._day_bias(day_df, day, prior_close, args.instrument, args.bias)
            print(f"\n{BD}══ {day}  ·  {args.instrument}  ·  bias {bias_dir} ══{RS}")
            renderer.equity = [0.0]
            sessions.append(fr.simulate_session(day_df, day, bias_dir, key_levels,
                                                args.instrument, args.orb_size, on_event=renderer))
            print()  # drop off the HUD line
        prior_close = float(day_df["Close"].iloc[-1])

    if sessions:
        agg = fr._aggregate(sessions, args.instrument)
        fr._print_report(agg, sessions)


if __name__ == "__main__":
    main()
