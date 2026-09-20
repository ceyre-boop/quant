"""
sovereign/terminal/app.py
=========================
ALTA TERM — the command loop.

Bloomberg's shape, because the shape is right: you type a symbol, you get
everything the desk knows about it, on one screen, now. The difference is that
what this desk knows is 120 sealed hypotheses and a library of 63 historical
episodes, and most of that has never reached a decision.

Read-only. There is no order path in this process — not disabled, absent.

    alta term                 # interactive
    alta term EURUSD          # one shot, prints and exits
    alta term HEALTH
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from sovereign.terminal import screens, sources as S

# Live pulls are cheap but not free, and a broker is not a thing to poll in a
# tight loop. Within this window a repeated command reuses the last reading.
LIVE_TTL_S = 20.0


@dataclass
class Session:
    """Holds the live readings between commands so navigation stays instant."""
    console: Console
    _acct: Optional[S.Source] = None
    _acct_at: float = 0.0
    _quotes: dict[str, S.Source] = field(default_factory=dict)
    _quotes_at: float = 0.0
    _ctx: dict[str, S.Source] = field(default_factory=dict)

    # ── live, TTL-cached ────────────────────────────────────────────────────
    def account(self, force: bool = False) -> S.Source:
        if force or self._acct is None or time.time() - self._acct_at > LIVE_TTL_S:
            self._acct = S.account()
            self._acct_at = time.time()
        return self._acct

    def quotes(self, force: bool = False) -> dict[str, S.Source]:
        if force or not self._quotes or time.time() - self._quotes_at > LIVE_TTL_S:
            self._quotes = {p: S.quote(p) for p in screens.LIVE_PAIRS}
            self._quotes_at = time.time()
        return self._quotes

    def risk(self) -> S.RiskBudget:
        a = self.account()
        nav = a.data["nav"] if a.ok else 0.0
        cur = a.data["currency"] if a.ok else "USD"
        return S.risk_budget(nav, cur)

    def context(self, symbol: str, force: bool = False) -> S.Source:
        """The research packet. Offline by default — the Library's SPY history
        moves on a daily bar, so a stale-cache read costs nothing a live fetch
        would buy, and the packet labels the staleness either way."""
        key = symbol.upper()
        if force or key not in self._ctx:
            self._ctx[key] = S.trade_context(key, offline=True)
        return self._ctx[key]

    def invalidate(self) -> None:
        self._acct = None
        self._quotes = {}
        self._ctx = {}


# ── header ───────────────────────────────────────────────────────────────────

def header(sess: Session) -> RenderableType:
    a = sess.account()
    mode = a.data["mode"] if a.ok else "no broker"
    style = "bold red" if mode == "LIVE" else "cyan"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    left = Text.assemble(("ALTA TERM", "bold white on blue"), ("  ", ""),
                         (mode, style))
    right = Text(f"{now}   read-only", style="dim")
    bar = Text.assemble(left, ("   ", ""), right)
    return Group(bar, Rule(style="blue"))


# ── dispatch ─────────────────────────────────────────────────────────────────

def render(sess: Session, raw: str) -> RenderableType:
    """Map one typed command to one screen. Unknown words are tried as symbols."""
    parts = raw.strip().split()
    if not parts:
        return home(sess)
    cmd, args = parts[0].upper(), parts[1:]

    if cmd in ("HOME", "H"):
        return home(sess)
    if cmd in ("HELP", "?"):
        return Panel(screens.HELP_TEXT.strip(), title="HELP", border_style="blue")
    if cmd == "POS":
        return screens.positions(sess.account(), sess.risk())
    if cmd == "RISK":
        return screens.risk_panel(sess.risk(), sess.account(), S.kill_switch())
    if cmd in ("BOARD", "MKT"):
        a = sess.account()
        pos = ({p["instrument"]: p["long_units"] + p["short_units"]
                for p in a.data["positions"]} if a.ok else {})
        return screens.board_panel(sess.quotes(), pos)
    if cmd == "HEALTH":
        return screens.health(S.loop_health())
    if cmd == "EDGE":
        return screens.edge_panel(sess.context(args[0] if args else "EURUSD"))
    if cmd == "HYP":
        return screens.hypotheses(S.read_json("hyp_ledger"), " ".join(args))
    if cmd == "MACRO":
        return screens.macro(S.read_json("fred"))
    if cmd == "LIB":
        return _library(sess, args[0] if args else "EURUSD")
    if cmd == "DES":
        if not args:
            return Panel(Text("DES needs a symbol, e.g. DES EURUSD", style="yellow"),
                         border_style="yellow")
        return _instrument(sess, args[0])

    # A bare token is treated as a symbol — Bloomberg's habit. A token with
    # trailing words the dispatcher doesn't know is a typo, not a ticker, so it
    # is reported rather than silently looked up as an instrument.
    if not args and cmd.replace("_", "").replace("/", "").isalnum() and len(cmd) <= 12:
        return _instrument(sess, cmd)
    return Panel(Text(f'unknown command "{raw.strip()}" — type HELP', style="yellow"),
                 border_style="yellow")


def home(sess: Session) -> RenderableType:
    return screens.home(sess.account(), sess.risk(), S.kill_switch(),
                        S.loop_health(), sess.quotes(), sess.context("EURUSD"))


def _instrument(sess: Session, symbol: str) -> RenderableType:
    sym = symbol.upper()
    return screens.instrument(sym, S.quote(sym), sess.context(sym),
                              sess.risk(), sess.account())


def _library(sess: Session, symbol: str) -> RenderableType:
    ctx = sess.context(symbol)
    if not ctx.ok:
        return Panel(Text(ctx.note, style="red"), title="LIBRARY", border_style="red")
    L = ctx.data.get("library", {})
    if not L.get("available"):
        return Panel(Text(L.get("reason") or "unavailable", style="yellow"),
                     title="ALEXANDRIAN LIBRARY", border_style="yellow")
    lines: list[RenderableType] = [
        Text.assemble(("regime  ", "dim"), (L["primary_regime"], "bold white")),
        Text.assemble(("match   ", "dim"),
                      (f"{L['similarity']:.3f}",
                       "green" if L["above_floor"] else "bold red"),
                      ("" if L["above_floor"] else
                       f"  below the {L['similarity_floor']:.2f} noise floor — ABSTAIN",
                       "bold red")),
        Text.assemble(("threat  ", "dim"), (f"{L['threat_level']} "
                                            f"({L['threat_score']:.3f})", "yellow")),
        Text.assemble(("sizing  ", "dim"), (f"{L['size_modifier']:.2f}×", "white")),
        Text(""),
        Text(L.get("advisory", ""), style="dim"),
    ]
    from rich.table import Table
    t = Table(box=None, expand=True)
    t.add_column("precedent", ratio=2)
    t.add_column("volume", ratio=2, style="dim")
    t.add_column("date", ratio=1)
    t.add_column("sim", justify="right")
    t.add_column("what followed", ratio=3, style="dim")
    for m in L.get("precedents", []):
        t.add_row(m["label"], m["volume"].replace("VOLUME_", ""), m["date"],
                  f"{m['similarity']:.2f}", m["outcome"])
    src_notes = L.get("source_notes") or []
    foot = Text("  ".join(src_notes), style="yellow" if any(
        "STALE" in n or "failed" in n for n in src_notes) else "dim")
    return Panel(Group(*lines, Text(""), t, foot),
                 title="ALEXANDRIAN LIBRARY — today's tape",
                 border_style="magenta")


# ── loop ─────────────────────────────────────────────────────────────────────

def run(console: Console, first: Optional[str] = None, once: bool = False) -> int:
    sess = Session(console=console)
    pending = first

    while True:
        cmd = pending if pending is not None else ""
        pending = None
        try:
            body = render(sess, cmd)
        except Exception as exc:                    # a bad screen must not kill the terminal
            body = Panel(Text(f"{type(exc).__name__}: {exc}", style="bold red"),
                         title="SCREEN ERROR", border_style="red")

        if not once:
            console.clear()
        console.print(header(sess))
        console.print(body)

        if once:
            return 0

        try:
            entry = console.input("\n[bold blue]ALTA>[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return 0
        if entry.upper() in ("Q", "QUIT", "EXIT"):
            console.print("[dim]bye[/dim]")
            return 0
        if entry.upper() == "R":
            sess.invalidate()
            pending = "HOME"
            continue
        pending = entry


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="alta term",
        description="ALTA TERM — read-only trading terminal over this desk's whole record.")
    ap.add_argument("command", nargs="*",
                    help="optional one-shot command (EURUSD, HEALTH, POS, ...)")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--width", type=int, default=None)
    args = ap.parse_args(argv)

    console = Console(no_color=args.no_color, width=args.width,
                      force_terminal=None if sys.stdout.isatty() else False)
    if args.command:
        return run(console, " ".join(args.command), once=True)
    return run(console)


if __name__ == "__main__":
    raise SystemExit(main())
