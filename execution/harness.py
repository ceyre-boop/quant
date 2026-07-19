"""Execution harness — measures whether the gapper edges survive real fills.

WHAT THIS IS
------------
A measurement instrument. It places no orders, moves no capital, and emits NO
funding-readiness verdict. It answers one question: when the frozen filters fire,
what does the fill actually cost at the real quoted bid/ask?

It replaces two forked shadow trackers:
    research/yield_frontier/live_shadow.py   (HYP-093 fade, SHORT)
    research/gapper/hyp107_shadow.py         (HYP-107 runner, LONG)
Both remain running through Phase 1 of the migration; see Plans/.

WHY DEFERRED CAPTURE, NOT REAL TIME
-----------------------------------
This account's SIP entitlement 403s inside a 15-minute recency window (measured
2026-07-18: -13min -> 403, -16min -> 200). Real-time IEX is permitted but
produces ~10% spreads on a ~2%-volume venue and is unusable for cost measurement.
So the harness records signal INTENT live off delayed bars, then captures the true
NBBO at each decision timestamp on a deferred pass past the boundary. For an
instrument that needs accuracy rather than latency this costs nothing.

THE COMPARISON THAT MATTERS
---------------------------
`backtest_expected_return` is produced by calling the SAME
`realistic_fills.realistic_long_return()` the backtests call, on the same bars,
same scenario. Everything except the quote source is held identical, so
`vs_backtest_delta` isolates quote reality from model assumption. Before this
harness existed, neither shadow called that module at all — the delta would have
compared two models to each other.

MODES
-----
  --live              scheduled path (launchd 09:25 ET Mon-Fri)
  --replay DATE       reconstruct a past session end-to-end
  --backfill FILE     price a list of historical events (the fastest way to
                      resolve HYP-107's open execution question at n=57)

NOTE: `data/execution/fills.jsonl` (Jun 30) belongs to an unrelated older system.
This harness writes `fill_log.jsonl`. Do not merge or "tidy" them together.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time as time_mod
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import pandas as pd

from backtester.realistic_fills import realistic_long_return
from execution import alpaca, borrow, halts, quotes, risk, scan, signals as signals_mod
from execution.config import frozen, verify_frozen_hash, FROZEN_HASH
from sovereign.autonomous._common import append_jsonl, make_logger
from sovereign.utils import kill_switch
from sovereign.utils.timestamps import canonical_timestamp

ET = alpaca.ET
UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "execution"
FILL_LOG = "fill_log.jsonl"
SUMMARY_CSV = "daily_summary.csv"
HEARTBEAT = ROOT / "data" / "execution" / "heartbeat.json"

SUMMARY_COLUMNS = [
    "date", "n_signals", "n_filled", "n_skipped",
    "median_net_return", "median_spread_cost", "vs_backtest_delta",
]

_log = make_logger("harness")


# ── Records ───────────────────────────────────────────────────────────────────

@dataclass
class FillRecord:
    """One row of fill_log.jsonl.

    `date` is the ET SESSION date — the single place ET leaks into persistence.
    Every other timestamp is UTC via canonical_timestamp(). This asymmetry is
    deliberate (a session is an ET concept) and is called out here because it
    will otherwise confuse every future reader.
    """
    ticker: str
    date: str
    signal_type: str                    # LONG | SHORT | SKIP_*
    entry_time: str | None = None
    entry_bid: float | None = None
    entry_ask: float | None = None
    entry_fill: float | None = None
    exit_fill: float | None = None
    gross_return: float | None = None
    spread_cost: float | None = None
    net_return: float | None = None
    backtest_expected_return: float | None = None
    # non-breaking additions
    hypothesis: str = ""
    reason: str = ""
    # Layer 3/5 wiring: which signal produced this fill, and what risk said
    signal_id: str = ""
    signal_rank: int | None = None
    risk_action: str = ""
    risk_allowed: bool | None = None
    risk_breached: list[str] = field(default_factory=list)
    risk_size_mult: float | None = None
    wide_quote: bool = False
    luld_band_used: float | None = None
    scenario: str = "base"
    frozen_hash: str = FROZEN_HASH
    capture_lag_s: float | None = None
    entry_mid: float | None = None
    exit_mid: float | None = None
    quote_raw: dict[str, Any] = field(default_factory=dict)
    logged_at: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["logged_at"] = d["logged_at"] or canonical_timestamp()
        return d


def is_skip(signal_type: str) -> bool:
    return signal_type.startswith("SKIP_")


def apply_risk(rec: FillRecord, state: "risk.AccountState",
               risk_fraction: float) -> FillRecord:
    """Route a prospective fill through the ratified risk gates (Layer 5).

    A blocked fill is recorded as SKIP_RISK with the breached articles attached —
    never silently dropped. A risk gate that discards its own refusals leaves no
    evidence that it fired, which is the failure mode the whole feedback layer
    exists to prevent.
    """
    d = risk.check(symbol=rec.ticker, risk_fraction=risk_fraction, state=state)
    rec.risk_action = d.action.value
    rec.risk_allowed = d.allowed
    rec.risk_breached = list(d.breached)
    rec.risk_size_mult = d.size_mult
    if not d.allowed:
        rec.signal_type = "SKIP_RISK"
        rec.reason = d.reason
    return rec


# ── Output ────────────────────────────────────────────────────────────────────

def _existing_keys(out_dir: Path, day: date) -> set[tuple[str, str, str]]:
    """(date, ticker, signal_type) already recorded — for restart idempotency."""
    fp = out_dir / FILL_LOG
    keys: set[tuple[str, str, str]] = set()
    if not fp.exists():
        return keys
    for line in fp.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("date") == str(day):
            keys.add((r.get("date", ""), r.get("ticker", ""), r.get("signal_type", "")))
    return keys


def record_fill(rec: FillRecord, out_dir: Path) -> None:
    """Append one fill row. append_jsonl raises on failure — let it.

    A silently unwritten fill is worse than a crashed harness: the crash is
    visible in logs/harness.err, the missing row is not.
    """
    append_jsonl(out_dir / FILL_LOG, rec.to_json())


def write_daily_summary(day: date, rows: list[FillRecord], out_dir: Path) -> dict:
    """Append or rewrite today's summary row. No readiness column, by design."""
    filled = [r for r in rows if not is_skip(r.signal_type)]
    skipped = [r for r in rows if is_skip(r.signal_type)]

    nets = [r.net_return for r in filled if r.net_return is not None]
    spreads = [r.spread_cost for r in filled if r.spread_cost is not None]
    expected = [r.backtest_expected_return for r in filled
                if r.backtest_expected_return is not None]

    med_net = median(nets) if nets else None
    med_spread = median(spreads) if spreads else None
    delta = (med_net - median(expected)) if (nets and expected) else None

    row = {
        "date": str(day),
        "n_signals": len(rows),
        "n_filled": len(filled),
        "n_skipped": len(skipped),
        # Empty (not 0.0) on a zero-fill day: a zero would read as a measurement.
        "median_net_return": "" if med_net is None else f"{med_net:.6f}",
        "median_spread_cost": "" if med_spread is None else f"{med_spread:.6f}",
        "vs_backtest_delta": "" if delta is None else f"{delta:.6f}",
    }

    fp = out_dir / SUMMARY_CSV
    fp.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if fp.exists():
        with open(fp, newline="") as fh:
            existing = [r for r in csv.DictReader(fh)]

    existing = [r for r in existing if r.get("date") != str(day)]   # rewrite in place
    existing.append(row)
    existing.sort(key=lambda r: r.get("date", ""))

    with open(fp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(existing)
    return row


# ── Backtest comparison ───────────────────────────────────────────────────────

def _bars_frame(bars: list[dict]) -> pd.DataFrame:
    """Alpaca bars -> the frame shape realistic_fills expects."""
    return pd.DataFrame({
        "time": [alpaca.et_t(b).strftime("%H:%M") for b in bars],
        "open": [float(b["o"]) for b in bars],
        "high": [float(b["h"]) for b in bars],
        "low": [float(b["l"]) for b in bars],
        "close": [float(b["c"]) for b in bars],
        "volume": [float(b["v"]) for b in bars],
    })


def backtest_expected(bars: list[dict], entry_et: str, exit_et: str,
                      stop_pct: float, scenario: str = "base") -> float | None:
    """What the backtest model says this trade returns — same call, same module."""
    if not bars:
        return None
    frame = _bars_frame(bars)
    try:
        return realistic_long_return(frame, entry_et, exit_et, stop_pct,
                                     scenario=scenario)
    except Exception as e:              # noqa: BLE001 - model failure is informative
        _log(f"backtest_expected failed: {type(e).__name__}: {e}")
        return None


# ── Leg pricing ───────────────────────────────────────────────────────────────

def price_leg(symbol: str, day: date, *, side: str, hypothesis: str,
              entry_et: str, exit_et: str, bars: list[dict],
              stop_pct: float, scenario: str = "base") -> FillRecord:
    """Capture real quotes at entry/exit and build the fill record."""
    cfg_cap = frozen("capture")
    hh, mm = (int(x) for x in entry_et.split(":"))
    xh, xm = (int(x) for x in exit_et.split(":"))

    # Entry is the OPEN of the entry bar -> the instant the bar starts.
    # Exit is the CLOSE of the exit bar -> the instant the bar ENDS, i.e. one
    # minute later. The frozen specs use b0931["o"] and b1030["c"] respectively
    # (hyp107_shadow.py:147,183), so an exit quote taken at 10:30:00 would price
    # the wrong end of the bar and make vs_backtest_delta meaningless.
    entry_ts = alpaca.et_dt(day, dtime(hh, mm)).astimezone(UTC)
    exit_ts = (alpaca.et_dt(day, dtime(xh, xm)) + timedelta(minutes=1)).astimezone(UTC)

    base = FillRecord(ticker=symbol, date=str(day), signal_type=side,
                      hypothesis=hypothesis, scenario=scenario,
                      entry_time=entry_ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"))

    halted, evidence = halts.halted_at(_bars_frame(bars), dtime(hh, mm))
    if halted:
        base.signal_type = "SKIP_HALT"
        base.reason = evidence
        return base

    eq = quotes.quote_at(symbol, entry_ts,
                         window_seconds=cfg_cap["quote_window_seconds"])
    if eq is None:
        base.signal_type = "SKIP_NO_QUOTE"
        base.reason = f"no_usable_quote_at_entry_{entry_et}"
        return base

    xq = quotes.quote_at(symbol, exit_ts,
                         window_seconds=cfg_cap["quote_window_seconds"])
    if xq is None:
        base.signal_type = "SKIP_NO_QUOTE"
        base.reason = f"no_usable_quote_at_exit_{exit_et}"
        base.entry_bid, base.entry_ask = eq.bid, eq.ask
        return base

    acct = quotes.round_trip(eq, xq, side)
    frame = _bars_frame(bars)
    bands = halts.bands_for(frame)
    idx = [i for i, t in enumerate(frame["time"]) if t == f"{hh:02d}:{mm:02d}"]

    base.entry_bid, base.entry_ask = eq.bid, eq.ask
    base.entry_fill = acct["entry_fill"]
    base.exit_fill = acct["exit_fill"]
    base.entry_mid = acct["entry_mid"]
    base.exit_mid = acct["exit_mid"]
    base.gross_return = acct["gross_return"]
    base.net_return = acct["net_return"]
    base.spread_cost = acct["spread_cost"]
    base.wide_quote = quotes.is_wide(eq) or quotes.is_wide(xq)
    base.luld_band_used = float(bands[idx[0]]) if idx else None
    base.capture_lag_s = (datetime.now(UTC) - entry_ts).total_seconds()
    base.quote_raw = {"entry": eq.to_record(), "exit": xq.to_record()}
    base.backtest_expected_return = backtest_expected(
        bars, f"{hh:02d}:{mm:02d}", f"{xh:02d}:{xm:02d}", stop_pct, scenario)
    base.reason = "filled"
    return base


# ── Session ───────────────────────────────────────────────────────────────────

def _signal_index(day: date, signals_dir: Path | None) -> dict[tuple[str, str], dict]:
    """(ticker, hypothesis) -> GO signal, from the Layer 3 output if present."""
    if signals_dir is None:
        return {}
    return {(s["ticker"], s["hypothesis"]): s
            for s in signals_mod.go_list(day, signals_dir)}


def run_session(day: date, *, out_dir: Path | None = None,
                check_news: bool = True, scenario: str = "base",
                replay: bool = False, signals_dir: Path | None = None,
                account: "risk.AccountState | None" = None) -> list[FillRecord]:
    """Score both legs for one ET session date. Pure measurement.

    `replay=True` sources the universe from the historical archive instead of the
    live movers screener. This is mandatory for past dates: the screener has no
    historical mode and would silently substitute today's movers.
    """
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    c107, c093 = frozen("hyp107"), frozen("hyp093")

    seen = _existing_keys(out_dir, day)
    locate = borrow.load_locate(day)
    sig_index = _signal_index(day, signals_dir)
    if signals_dir is not None:
        _log(f"{day}: Layer 3 signal file supplies {len(sig_index)} GO signal(s)")
    # Flat account state unless supplied. With no live equity feed the honest
    # default is zero drawdown, which means the Art. 3 ladder cannot fire — that
    # is recorded per fill via risk_action so it is visible, not assumed.
    acct = account or risk.AccountState(equity=100_000.0, peak_equity=100_000.0)

    symbols = None
    if replay:
        symbols = scan.archived_symbols(day)
        if not symbols:
            _log(f"{day}: NO ARCHIVED UNIVERSE for this date — refusing to fall "
                 f"back to the live screener, which would replay today's movers "
                 f"against a past session. Nothing scored.")
            write_daily_summary(day, [], out_dir)
            return []
        _log(f"{day}: replay universe from archive: {len(symbols)} symbol(s)")

    cands = scan.scan_universe(day, check_news=check_news, symbols=symbols)
    _log(f"{day}: screened {len(cands)} candidate(s)")

    records: list[FillRecord] = []

    for c in cands:
        ok, reason = scan.passes_hyp107(c)
        if not ok:
            continue
        if sig_index and (c.symbol, "HYP-107") not in sig_index:
            continue                      # Layer 3 is authoritative when present
        if (str(day), c.symbol, "LONG") in seen:
            continue
        rec = price_leg(c.symbol, day, side="LONG", hypothesis="HYP-107",
                        entry_et=c107["entry_bar_et"], exit_et=c107["exit_bar_et"],
                        bars=c.bars, stop_pct=c107["stop_pct"], scenario=scenario)
        sig = sig_index.get((c.symbol, "HYP-107"))
        if sig:
            rec.signal_id, rec.signal_rank = sig["signal_id"], sig.get("rank")
        rec = apply_risk(rec, acct, c107["stop_pct"] * 0.03)
        record_fill(rec, out_dir)
        records.append(rec)

    for c in cands:
        ok, reason = scan.passes_hyp093(c)
        if not ok:
            continue
        if (str(day), c.symbol, "SHORT") in seen:
            continue
        allowed, breason = borrow.borrow_ok(c.symbol, locate)
        if not allowed:
            rec = FillRecord(ticker=c.symbol, date=str(day),
                             signal_type="SKIP_NO_BORROW", hypothesis="HYP-093",
                             reason=breason)
            record_fill(rec, out_dir)
            records.append(rec)
            continue
        if sig_index and (c.symbol, "HYP-093") not in sig_index:
            continue
        rec = price_leg(c.symbol, day, side="SHORT", hypothesis="HYP-093",
                        entry_et=c093["entry_bar_et"], exit_et=c093["exit_bar_et"],
                        bars=c.bars, stop_pct=c093["stop_mult"] - 1.0,
                        scenario=scenario)
        sig = sig_index.get((c.symbol, "HYP-093"))
        if sig:
            rec.signal_id, rec.signal_rank = sig["signal_id"], sig.get("rank")
        rec = apply_risk(rec, acct, c093["notional_w"] * c093["locate_w"])
        record_fill(rec, out_dir)
        records.append(rec)

    row = write_daily_summary(day, records, out_dir)
    _log(f"{day}: {row['n_filled']} filled / {row['n_skipped']} skipped | "
         f"median_net={row['median_net_return'] or 'n/a'} "
         f"median_spread={row['median_spread_cost'] or 'n/a'} "
         f"vs_backtest={row['vs_backtest_delta'] or 'n/a'}")
    return records


def run_backfill(events: Iterable[dict], *, out_dir: Path,
                 scenario: str = "base") -> list[FillRecord]:
    """Price a list of historical events: [{"ticker":..., "date":...}, ...].

    This is execution-cost measurement on already-adjudicated events. It is NOT
    a fresh signal test and adds no multiplicity — it re-prices known events with
    real quotes instead of a bar-range proxy. Label it as such wherever reported.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    c107 = frozen("hyp107")
    records: list[FillRecord] = []

    for ev in events:
        sym = str(ev["ticker"]).upper()
        d = date.fromisoformat(str(ev["date"]))
        side = str(ev.get("side", "LONG")).upper()
        hyp = str(ev.get("hypothesis", "HYP-107"))
        bars = alpaca.minute_bars(sym, d)
        if not bars:
            rec = FillRecord(ticker=sym, date=str(d), signal_type="SKIP_NO_DATA",
                             hypothesis=hyp, reason="no_bars")
            record_fill(rec, out_dir)
            records.append(rec)
            continue
        rec = price_leg(sym, d, side=side, hypothesis=hyp,
                        entry_et=ev.get("entry_et", c107["entry_bar_et"]),
                        exit_et=ev.get("exit_et", c107["exit_bar_et"]),
                        bars=bars, stop_pct=c107["stop_pct"], scenario=scenario)
        record_fill(rec, out_dir)
        records.append(rec)
        _log(f"backfill {sym} {d}: {rec.signal_type} net={rec.net_return} "
             f"spread={rec.spread_cost}")

    filled = [r for r in records if not is_skip(r.signal_type)]
    if filled:
        nets = [r.net_return for r in filled if r.net_return is not None]
        sprs = [r.spread_cost for r in filled if r.spread_cost is not None]
        _log(f"backfill complete: n={len(nets)} median_net={median(nets):.4f} "
             f"median_spread={median(sprs):.4f}")
    return records


# ── Entry point ───────────────────────────────────────────────────────────────

def _write_heartbeat(mode: str) -> None:
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(json.dumps({
        "component": "execution_harness", "mode": mode,
        "updated": canonical_timestamp(), "frozen_hash": FROZEN_HASH}, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gapper execution harness (measurement only)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--live", action="store_true", help="scheduled session run")
    g.add_argument("--replay", metavar="DATE", help="reconstruct a past session")
    g.add_argument("--backfill", metavar="FILE",
                   help="JSON list of {ticker,date[,side,hypothesis]} events")
    ap.add_argument("--out", default=None, help="output dir (default data/execution)")
    ap.add_argument("--scenario", default="base",
                    choices=["optimistic", "base", "pessimistic"])
    ap.add_argument("--no-news", action="store_true",
                    help="skip the M&A headline check (offline/replay use)")
    ap.add_argument("--signals", metavar="DIR", default=None,
                    help="Layer 3 signal directory; when given, only GO signals fill")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else OUT_DIR

    # Order is load-bearing: env -> frozen hash (before any network I/O) ->
    # heartbeat -> kill switch. The kill switch comes AFTER the heartbeat so a
    # freeze reports EXECUTION (FROZEN) rather than a false DOWN.
    alpaca.load_env()
    verify_frozen_hash()

    mode = "live" if args.live else ("replay" if args.replay else "backfill")
    _write_heartbeat(mode)

    if kill_switch.skip_if_frozen("execution_harness", logger=_log):
        return 0                                   # clean exit, not an error

    if args.backfill:
        events = json.loads(Path(args.backfill).read_text())
        run_backfill(events, out_dir=out_dir, scenario=args.scenario)
        return 0

    day = date.fromisoformat(args.replay) if args.replay else datetime.now(ET).date()
    run_session(day, out_dir=out_dir, check_news=not args.no_news,
                scenario=args.scenario, replay=bool(args.replay),
                signals_dir=Path(args.signals) if args.signals else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
