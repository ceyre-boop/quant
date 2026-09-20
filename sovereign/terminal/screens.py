"""
sovereign/terminal/screens.py
=============================
Renderers for ALTA TERM. One rule, enforced everywhere: a number is never shown
without its age. A nine-day-old briefing and a live quote must not look alike.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from rich.align import Align
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sovereign.terminal import sources as S

# The four pairs the live v015 portfolio actually trades. AUDNZD is named by the
# constitution but excluded from the universe by HYP-045 (both legs RBA-driven —
# no independent rate differential). USDJPY is ICT NY-AM only, not carry.
LIVE_PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "GBPJPY")

STATUS_STYLE = {
    "LIVE": "bold green", "FRESH": "green", "STALE": "yellow",
    "DEAD": "bold red", "MISSING": "red", "ERROR": "bold red",
}
VERDICT_STYLE = {"CONFIRMED": "bold green", "CLOSED": "red", "OPEN": "yellow"}


def tag(src: S.Source) -> Text:
    """The age chip that travels with every panel."""
    label = "live" if src.status == "LIVE" else f"{src.status.lower()} · {src.age_label}"
    return Text(label, style=STATUS_STYLE.get(src.status, "dim"))


def money(v: float, cur: str = "$") -> str:
    return f"{cur}{v:,.2f}"


def _kv(rows: list[tuple[str, Any, Optional[str]]]) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="left")
    t.add_column(justify="right")
    for label, value, style in rows:
        t.add_row(label, Text(str(value), style=style or ""))
    return t


# ── home ─────────────────────────────────────────────────────────────────────

def account_panel(acct: S.Source) -> Panel:
    if not acct.ok:
        return Panel(Text(acct.note or "unavailable", style="bold red"),
                     title="ACCOUNT", border_style="red")
    d = acct.data
    mode_style = "bold red" if d["mode"] == "LIVE" else "cyan"
    pl_style = "green" if d["unrealized"] >= 0 else "red"
    body = _kv([
        ("mode", d["mode"], mode_style),
        ("NAV", money(d["nav"]), "bold white"),
        ("balance", money(d["balance"]), None),
        ("unrealized", money(d["unrealized"]), pl_style),
        ("lifetime P&L", money(d["realized_lifetime"]),
         "green" if d["realized_lifetime"] >= 0 else "red"),
        ("margin avail", money(d["margin_available"]), None),
        ("open trades", d["open_trades"], "bold yellow" if d["open_trades"] else "dim"),
    ])
    return Panel(body, title="ACCOUNT", subtitle=tag(acct), border_style="cyan")


def risk_panel(rb: S.RiskBudget, acct: S.Source, kill: S.Source) -> Panel:
    frozen = bool(kill.data and kill.data.get("frozen"))
    rows: list[tuple[str, Any, Optional[str]]] = [
        ("per trade (max)", money(rb.per_trade), "bold white"),
        (f"  = {rb.per_trade_pct}% of NAV", "", "dim"),
        ("carry heat (all)", money(rb.carry_heat), "bold white"),
        (f"  = {rb.carry_heat_pct}% of NAV", "", "dim"),
    ]
    used = 0.0
    if acct.ok:
        used = sum(abs(p["unrealized"]) for p in acct.data["positions"] if p["unrealized"] < 0)
    rows.append(("open drawdown", money(used), "red" if used else "dim"))
    rows.append(("kill switch", "FROZEN" if frozen else "open",
                 "bold red" if frozen else "green"))
    body = [_kv(rows)]

    lad = Table.grid(padding=(0, 2))
    lad.add_column(style="dim")
    lad.add_column(justify="right")
    lad.add_row(Text("drawdown ladder", style="bold dim"), "")
    for name, pct, level in rb.ladder_levels():
        lad.add_row(f"  −{pct}% → {name}", Text(money(level), style="yellow"))
    body.append(lad)
    if not rb.parsed_ok:
        body.append(Text(rb.note, style="bold yellow"))
    return Panel(Group(*body), title="RISK BUDGET — NOW",
                 subtitle=Text("from RISK_CONSTITUTION.md", style="dim"),
                 border_style="yellow")


def loops_panel(lh: S.Source) -> Panel:
    d = lh.data
    t = Table(box=None, expand=True, pad_edge=False)
    t.add_column("loop", style="dim")
    t.add_column("state", justify="right")
    t.add_column("age", justify="right")
    for r in d["rows"]:
        age = "—" if r["age_h"] is None else (
            f"{r['age_h']:.0f}h" if r["age_h"] < 48 else f"{r['age_h']/24:.0f}d")
        t.add_row(r["loop"],
                  Text(r["status"], style=STATUS_STYLE.get(r["status"], "dim")),
                  Text(age, style="red" if r["status"] in ("DEAD", "MISSING") else "dim"))
    dead = d["n_dead"]
    border = "red" if dead else "green"
    head = Text(f"{dead} of {d['n_total']} writers stopped",
                style="bold red" if dead else "bold green")
    return Panel(Group(head, t), title="LOOPS", border_style=border)


def board_panel(quotes: dict[str, S.Source], positions: dict[str, float]) -> Panel:
    t = Table(box=None, expand=True, pad_edge=False)
    t.add_column("pair", style="bold", ratio=1)
    t.add_column("bid", justify="right", ratio=1)
    t.add_column("ask", justify="right", ratio=1)
    t.add_column("sprd", justify="right", ratio=1)
    t.add_column("position", justify="right", ratio=1)
    t.add_column("", ratio=4)
    for pair in LIVE_PAIRS:
        q = quotes.get(pair)
        if q is None or not q.ok:
            t.add_row(pair, Text("—", style="red"), Text("—", style="red"),
                      Text("—", style="red"),
                      Text((q.note if q else "not fetched")[:22], style="red"), "")
            continue
        d = q.data
        units = positions.get(S._oanda_symbol(pair), 0.0)
        pos = Text("flat", style="dim") if units == 0 else Text(
            f"{units:+,.0f}", style="green" if units > 0 else "red")
        t.add_row(pair,
                  f"{d['bid']:.5f}" if d["bid"] else "—",
                  f"{d['ask']:.5f}" if d["ask"] else "—",
                  Text(f"{d['spread_pips']:.1f}" if d["spread_pips"] is not None else "—",
                       style="yellow" if (d["spread_pips"] or 0) > 3 else "dim"),
                  pos, "")
    note = Text("v015 universe · AUDNZD excluded (HYP-045)", style="dim")
    return Panel(Group(t, note), title="BOARD — live",
                 subtitle=Text("live · now", style="bold green"), border_style="green")


def edge_panel(ctx: S.Source) -> Panel:
    if not ctx.ok:
        return Panel(Text(ctx.note or "unavailable", style="red"),
                     title="EDGE", border_style="red")
    edges = ctx.data["edges"]
    body: list[RenderableType] = []
    conf = edges.get("confirmed", [])
    if conf:
        t = Table.grid(padding=(0, 1))
        t.add_column()
        for r in conf[:4]:
            t.add_row(Text(f"✓ {r['claim']}", style="green"))
        body.append(t)
    else:
        body.append(Text("nothing on the ledger is CONFIRMED", style="bold red"))
    nclosed = len(edges.get("closed", []))
    body.append(Text(f"\n{nclosed} claims closed · {len(edges.get('open', []))} open",
                     style="dim"))
    return Panel(Group(*body), title="EDGE LEDGER", subtitle=tag(ctx),
                 border_style="cyan")


def home(acct: S.Source, rb: S.RiskBudget, kill: S.Source, lh: S.Source,
         quotes: dict[str, S.Source], ctx: S.Source) -> RenderableType:
    positions: dict[str, float] = {}
    if acct.ok:
        for p in acct.data["positions"]:
            positions[p["instrument"]] = p["long_units"] + p["short_units"]
    top = Columns([account_panel(acct), risk_panel(rb, acct, kill), loops_panel(lh)],
                  expand=True, equal=True)
    return Group(top, board_panel(quotes, positions), edge_panel(ctx))


# ── instrument screen ────────────────────────────────────────────────────────

def instrument(sym: str, q: S.Source, ctx: S.Source, rb: S.RiskBudget,
               acct: S.Source) -> RenderableType:
    out: list[RenderableType] = []

    # price + the position you actually hold
    if q.ok:
        d = q.data
        held = 0.0
        if acct.ok:
            for p in acct.data["positions"]:
                if p["instrument"] == d["instrument"]:
                    held = p["long_units"] + p["short_units"]
        rows = [
            ("bid / ask", f"{d['bid']:.5f} / {d['ask']:.5f}", "bold white"),
            ("spread", f"{d['spread_pips']} pips", None),
            ("tradeable", "yes" if d["tradeable"] else "NO", 
             "green" if d["tradeable"] else "bold red"),
            ("your position", "flat" if held == 0 else f"{held:+,.0f} units",
             "dim" if held == 0 else "bold yellow"),
            ("max risk here", money(rb.per_trade), "bold white"),
        ]
        out.append(Panel(_kv(rows), title=f"{sym} — PRICE",
                         subtitle=tag(q), border_style="green"))
    else:
        out.append(Panel(Text(q.note or "no quote", style="red"),
                         title=f"{sym} — PRICE", border_style="red"))

    if not ctx.ok:
        out.append(Panel(Text(ctx.note or "context unavailable", style="red"),
                         title="CONTEXT", border_style="red"))
        return Group(*out)

    c = ctx.data

    # closed doors first — the whole point
    doors = c.get("closed_doors", [])
    if doors:
        t = Table(box=None, expand=True)
        t.add_column("what", style="white", ratio=3)
        t.add_column("verdict", style="red", ratio=2)
        t.add_column("src", style="dim", ratio=1)
        for d_ in doors[:8]:
            t.add_row(d_["what"], d_["verdict"][:60], d_["source"])
        out.append(Panel(t, title="⛔ CLOSED DOORS — already paid for",
                         border_style="red"))
    else:
        out.append(Panel(Text("none recorded for this instrument — "
                              "absence of a closed door is not evidence of an edge",
                              style="dim"),
                         title="CLOSED DOORS", border_style="dim"))

    # library precedents
    L = c.get("library", {})
    if L.get("available"):
        head = Text.assemble(
            (f"{L['primary_regime']}  ", "bold white"),
            (f"sim {L['similarity']:.3f}", "green" if L.get("above_floor") else "bold red"),
            ("  BELOW NOISE FLOOR — ABSTAIN" if not L.get("above_floor") else "", "bold red"),
        )
        t = Table(box=None, expand=True)
        t.add_column("precedent", ratio=2)
        t.add_column("date", ratio=1)
        t.add_column("sim", justify="right")
        t.add_column("what followed", ratio=3, style="dim")
        for m in L.get("precedents", [])[:5]:
            t.add_row(m["label"], m["date"], f"{m['similarity']:.2f}", m["outcome"])
        out.append(Panel(Group(head, t), title="ALEXANDRIAN LIBRARY",
                         subtitle=Text(f"threat {L['threat_level']} · "
                                       f"size {L['size_modifier']:.2f}×", style="dim"),
                         border_style="magenta" if L.get("above_floor") else "dim"))
    else:
        out.append(Panel(Text(L.get("reason") or "unavailable", style="yellow"),
                         title="ALEXANDRIAN LIBRARY", border_style="dim"))

    # lessons
    lessons = c.get("lessons", {}).get("matched", [])
    if lessons:
        t = Table(box=None, expand=True)
        t.add_column("id", style="dim")
        t.add_column("verdict", ratio=1)
        t.add_column("lesson", ratio=5, style="white")
        for m in lessons[:6]:
            t.add_row(m["id"],
                      Text(m["verdict"][:18],
                           style=VERDICT_STYLE.get(m["door"], "dim")),
                      m["lesson"])
        out.append(Panel(t, title="LESSONS THAT APPLY HERE", border_style="blue"))

    if c.get("warnings"):
        out.append(Panel(Text("\n".join(f"· {w}" for w in c["warnings"]), style="yellow"),
                         title="DEGRADATION", border_style="yellow"))
    return Group(*out)


# ── other screens ────────────────────────────────────────────────────────────

def positions(acct: S.Source, rb: S.RiskBudget) -> RenderableType:
    if not acct.ok:
        return Panel(Text(acct.note, style="red"), title="POSITIONS", border_style="red")
    d = acct.data
    if not d["trades"]:
        return Panel(Group(
            Text("FLAT — no open trades.", style="bold"),
            Text(f"\nAvailable now: {money(rb.per_trade)} on one trade, "
                 f"{money(rb.carry_heat)} across all carry.", style="dim")),
            title="POSITIONS", subtitle=tag(acct), border_style="cyan")
    t = Table(box=None, expand=True)
    for col, just in (("id", "left"), ("instrument", "left"), ("units", "right"),
                      ("entry", "right"), ("unrealized", "right"), ("opened", "left")):
        t.add_column(col, justify=just)
    for tr in d["trades"]:
        t.add_row(str(tr["id"]), tr["instrument"], f"{tr['units']:+,.0f}",
                  f"{tr['price']:.5f}",
                  Text(money(tr["unrealized"]),
                       style="green" if tr["unrealized"] >= 0 else "red"),
                  tr["opened"])
    return Panel(t, title=f"POSITIONS — {len(d['trades'])} open",
                 subtitle=tag(acct), border_style="cyan")


def health(lh: S.Source) -> RenderableType:
    t = Table(box=None, expand=True)
    t.add_column("loop", ratio=2)
    t.add_column("state", ratio=1)
    t.add_column("last write", ratio=2, style="dim")
    t.add_column("file", ratio=3, style="dim")
    for r in lh.data["rows"]:
        t.add_row(r["loop"],
                  Text(r["status"], style=STATUS_STYLE.get(r["status"], "dim")),
                  (r["asof"] or "never").replace("T", " ")[:16], r["path"])
    dead = lh.data["n_dead"]
    msg = (Text(f"\n{dead} writers have stopped. Anything sourced from them is history, "
                f"not state.", style="bold red") if dead else
           Text("\nAll writers current.", style="bold green"))
    return Panel(Group(t, msg), title="SYSTEM HEALTH",
                 border_style="red" if dead else "green")


def hypotheses(src: S.Source, query: str = "", limit: int = 25) -> RenderableType:
    if not src.ok or not isinstance(src.data, list):
        return Panel(Text(src.note or "ledger unreadable", style="red"),
                     title="HYPOTHESES", border_style="red")
    rows = src.data
    if query:
        ql = query.lower()
        rows = [e for e in rows
                if ql in (str(e.get("name", "")) + str(e.get("id", "")) +
                          str(e.get("result", "")) + str(e.get("status", ""))).lower()]
    t = Table(box=None, expand=True)
    t.add_column("id", style="dim", ratio=1)
    t.add_column("status", ratio=1)
    t.add_column("name", ratio=3)
    t.add_column("result", ratio=4, style="dim")
    for e in rows[-limit:]:
        status = str(e.get("status", "?"))
        style = ("green" if status in ("CONFIRMED", "VALID_EDGE", "MEASURED")
                 else "red" if any(k in status for k in
                                   ("REJECT", "NOT_", "FAIL", "GRAVE", "NULL"))
                 else "yellow")
        t.add_row(str(e.get("id", "—")), Text(status, style=style),
                  str(e.get("name", ""))[:60], str(e.get("result", ""))[:90])
    title = f"HYPOTHESES — {len(rows)} match" + ("" if len(rows) == 1 else "es")
    if query:
        title += f' for "{query}"'
    return Panel(t, title=title, subtitle=tag(src), border_style="blue")


def macro(src: S.Source) -> RenderableType:
    if not src.ok:
        return Panel(Text(src.note or "unavailable", style="red"),
                     title="MACRO", border_style="red")
    data = src.data if isinstance(src.data, dict) else {}
    series = data.get("series") or data.get("indicators") or data
    t = Table(box=None, expand=True)
    t.add_column("series", ratio=2)
    t.add_column("value", justify="right")
    t.add_column("detail", ratio=3, style="dim")
    shown = 0
    for k, v in series.items():
        if shown >= 20:
            break
        if isinstance(v, dict):
            val = v.get("value", v.get("latest", "—"))
            detail = str(v.get("date", v.get("as_of", "")))[:40]
        else:
            val, detail = v, ""
        t.add_row(str(k)[:40], str(val)[:18], detail)
        shown += 1
    warn = (Text("\nThis loop has stopped — treat as history.", style="bold red")
            if not src.trustworthy else Text(""))
    return Panel(Group(t, warn), title="MACRO BACKDROP", subtitle=tag(src),
                 border_style="red" if not src.trustworthy else "blue")


def swap(rep, render_text: str) -> RenderableType:
    """The financing verdict — is the broker leaving anything of the premium."""
    head_style = ("bold red" if "NO TRADE" in rep.headline
                  else "yellow" if rep.headline.startswith("PROVISIONAL")
                  else "bold green")
    lines: list[RenderableType] = [
        Text(rep.headline, style=head_style),
        Text(""),
    ]
    t = Table(box=None, expand=True)
    t.add_column("pair", style="bold", ratio=1)
    t.add_column("side", ratio=1)
    t.add_column("financing %/yr", justify="right", ratio=1)
    t.add_column("broker take", justify="right", ratio=1)
    t.add_column("model", justify="right", ratio=1)
    t.add_column("off by", justify="right", ratio=1)
    for v in rep.pairs:
        if v.verdict == "NOT_QUOTED":
            t.add_row(v.pair, Text("—", style="dim"),
                      Text("NOT QUOTED", style="bold yellow"),
                      Text("—", style="dim"), Text("—", style="dim"),
                      Text("—", style="dim"))
            continue
        carry_style = ("green" if v.carry_rate_mean_pct > 0
                       else "red" if v.carry_rate_mean_pct < 0 else "dim")
        off = "—" if v.model_ratio is None else f"{v.model_ratio:.1f}x"
        t.add_row(v.pair, v.carry_side,
                  Text(f"{v.carry_rate_mean_pct:+.3f}", style=carry_style),
                  Text(f"{v.broker_take_mean_pct:+.3f}", style="red"),
                  Text("—" if v.modelled_pct is None else f"{v.modelled_pct:+.3f}",
                       style="dim"),
                  Text(off, style="bold red" if (v.model_ratio or 0) > 3 else "dim"))
    lines.append(t)
    notes = [(v.pair, v.note) for v in rep.pairs if v.note]
    if notes:
        lines.append(Text(""))
        for pair, n in notes:
            lines.append(Text(f"  ! {pair}: {n}", style="yellow"))
    if rep.warnings:
        lines.append(Text(""))
        for w in rep.warnings:
            lines.append(Text(f"  · {w}", style="dim yellow"))
    progress = f"{rep.n_days}/{10} readings"
    sub = Text(f"{rep.mode} · {progress}"
               + (f" · {rep.first_day} → {rep.last_day}" if rep.first_day else ""),
               style="green" if rep.enough_data else "yellow")
    return Panel(Group(*lines), title="SWAP — what the broker actually takes",
                 subtitle=sub,
                 border_style="red" if "NO TRADE" in rep.headline else "yellow")


HELP_TEXT = """
[bold]ALTA TERM[/bold] — read-only. Nothing here can place, modify or close an order.

  [bold cyan]EURUSD[/bold cyan]          any symbol — full pre-trade context for it
  [bold cyan]DES[/bold cyan] <sym>       same thing, explicit
  [bold cyan]HOME[/bold cyan] / [bold cyan]H[/bold cyan]       the one screen: account, risk, loops, board, edge
  [bold cyan]POS[/bold cyan]             open positions and what is left in the budget
  [bold cyan]RISK[/bold cyan]            the constitution's caps against this NAV
  [bold cyan]BOARD[/bold cyan] / [bold cyan]MKT[/bold cyan]    live prices for the v015 universe
  [bold cyan]HEALTH[/bold cyan]          which writers have stopped
  [bold cyan]EDGE[/bold cyan]            the edge ledger
  [bold cyan]HYP[/bold cyan] [<query>]   the hypothesis record, optionally filtered
  [bold cyan]MACRO[/bold cyan]           FRED backdrop
  [bold cyan]SWAP[/bold cyan]            what the broker's financing takes from the premium
  [bold cyan]R[/bold cyan]               refresh live data
  [bold cyan]HELP[/bold cyan] / [bold cyan]?[/bold cyan]       this
  [bold cyan]Q[/bold cyan]               quit

[dim]Every panel carries its own age. live/fresh is state; stale/dead is history.[/dim]
"""
