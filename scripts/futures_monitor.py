#!/usr/bin/env python3
"""
Live MES/MNQ paper trading terminal.

Polls price every 5 seconds, detects VWAP reclaim + RSI confirmation,
surfaces approve/skip/modify trade proposal, places bracket orders via
IB Gateway paper account. Sim-only until 150-trade validation.

Usage:
    python3.13 scripts/futures_monitor.py
    python3.13 scripts/futures_monitor.py --instrument MNQ --dry-run
    python3.13 scripts/futures_monitor.py --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Single source of truth for signal logic — shared with scripts/futures_replay.py
from sovereign.futures import scalp_strategy as strat
from sovereign.futures import telegram_gateway as tg
from sovereign.futures import decision_engine as de
from sovereign.futures import reasoning as rsn
from sovereign.futures import regime as regime_mod
from sovereign.futures import volume_profile as vp
from sovereign.futures.config import futures_params

BIAS_LOG  = ROOT / "data" / "futures" / "bias_log.jsonl"
TRADE_LOG = ROOT / "data" / "futures" / "trade_log.jsonl"

TICKER_MAP = {"MES": "ES=F", "MNQ": "NQ=F"}

# ANSI
G  = "\033[92m"   # green
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
BW = "\033[97m"   # bright white
DM = "\033[2m"    # dim
BD = "\033[1m"    # bold
RS = "\033[0m"    # reset


# ── helpers ──────────────────────────────────────────────────────────────────

def _stars(conviction: int) -> str:
    c = max(0, min(3, conviction))
    return "★" * c + "☆" * (3 - c)


def _today() -> str:
    # The futures trading day is ET (matches session_windows + futures_bias.py); using UTC here
    # rolled the day over at 20:00 ET and dropped the day's bias -> NEUTRAL. Anchor to ET.
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _norm_inst(x: str) -> str:
    """Treat the micro and its underlying as the same product: MES==ES, MNQ==NQ.
    (futures_bias.py logs ES/NQ; the monitor/oracle use MES/MNQ — normalize so the monitor
    finds the bias.)"""
    x = (x or "").upper()
    return {"MES": "ES", "MNQ": "NQ"}.get(x, x)


def _load_bias(instrument: str) -> dict:
    today = _today()
    last = None
    if BIAS_LOG.exists():
        with open(BIAS_LOG) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("date") == today and _norm_inst(rec.get("instrument", "")) == _norm_inst(instrument):
                        last = rec
                except Exception:
                    pass
    if last is None:
        print(f"{Y}  [warn] No bias entry for today ({today}). Running in NEUTRAL mode.{RS}")
        return {"bias": "NEUTRAL", "conviction": 0, "key_levels": {}, "instrument": instrument}
    return last


def _check_macro(instrument: str) -> dict | None:
    if not BIAS_LOG.exists():
        return None
    by_date: dict[str, dict] = {}
    with open(BIAS_LOG) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if _norm_inst(rec.get("instrument", "")) == _norm_inst(instrument):
                    d = rec.get("date", "")
                    if d:
                        by_date[d] = rec
            except Exception:
                pass
    sorted_dates = sorted(by_date.keys(), reverse=True)[:5]
    if len(sorted_dates) < 3:
        return None
    last3 = [by_date[d] for d in sorted_dates[:3]]
    dirs = [r.get("bias", "NEUTRAL") for r in last3]
    if len(set(dirs)) != 1 or dirs[0] == "NEUTRAL":
        return None
    avg_conv = sum(r.get("conviction", 0) for r in last3) / 3
    if avg_conv < 2.0:
        return None
    return {"direction": dirs[0], "streak": 3,
            "avg_conviction": round(avg_conv, 2), "dates": sorted_dates[:3]}


def _oracle_invalidation(instrument: str) -> float | None:
    """Today's oracle falsifier price for this instrument (file I/O), or None."""
    oracle_path = ROOT / "data" / "futures" / "oracle_mornings.jsonl"
    if not oracle_path.exists():
        return None
    today = _today()
    try:
        for line in oracle_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("date") == today and _norm_inst(r.get("instrument", "")) == _norm_inst(instrument):
                inv = (r.get("key_levels") or {}).get("invalidation")
                if isinstance(inv, (int, float)):
                    return float(inv)
    except Exception:
        pass
    return None


def _kill_level(bias_dir: str, bias: dict, instrument: str) -> float | None:
    """Soonest of rules invalidation (overnight high/low) and today's oracle falsifier.
    Pure combination lives in scalp_strategy.kill_level; file I/O stays here."""
    return strat.kill_level(bias_dir, bias.get("key_levels", {}),
                            _oracle_invalidation(instrument))


def _fetch_bars(instrument: str, bridge=None, contract=None, require_ib: bool = False):
    """Live 1-min RTH bars. Prefers the connected IB bridge (matches what the replay
    uses); falls back to yfinance if no bridge or the IB call fails — UNLESS require_ib is
    set, in which case it returns empty rather than trade on fallback data (Guard 1)."""
    if bridge is not None and contract is not None:
        try:
            from sovereign.futures.bar_feed import live_session_bars
            df = live_session_bars(bridge, contract)
            if df is not None and len(df) > 0:
                return df
        except Exception:
            pass  # fall through to yfinance (unless IB is required)
    if require_ib:
        import pandas as pd
        return pd.DataFrame()   # no yfinance shadow data when IB is the required source
    import yfinance as yf
    import pandas as pd
    ticker = TICKER_MAP[instrument]
    bars = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
    if bars.empty:
        return bars
    # Flatten MultiIndex if present
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)
    # Filter to today's RTH session (9:30 ET onward)
    et = ZoneInfo("America/New_York")
    now_et = datetime.now(et)
    session_start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    utc_start = session_start.astimezone(ZoneInfo("UTC"))
    bars = bars[bars.index >= utc_start]
    return bars


def _compute_indicators(bars) -> tuple[float, float, float, float, float, float, float]:
    """(last_price, vwap, rsi, curr_volume, avg_volume, ema_fast, ema_slow).
    Delegates to scalp_strategy (single source of truth). Raises ValueError if empty."""
    ind = strat.compute_indicators(bars)
    return (ind.last_price, ind.vwap, ind.rsi, ind.curr_volume,
            ind.avg_volume, ind.ema_fast, ind.ema_slow)


def _check_signal(bias_dir: str, curr_price: float, curr_vwap: float, curr_rsi: float,
                  prev_price: float, prev_vwap: float, prev_rsi: float,
                  last_proposal_time: datetime | None, proposals_count: int,
                  curr_volume: float, avg_volume_20: float,
                  ema8: float, ema21: float, max_trades: int = 3) -> str | None:
    """Delegates to scalp_strategy.micro_signal. CLI --max-trades is honored as an
    extra bound on top of the config cap."""
    if proposals_count >= max_trades:
        return None
    curr = strat.Indicators(curr_price, curr_vwap, curr_rsi, curr_volume,
                            avg_volume_20, ema8, ema21)
    prev = strat.Indicators(prev_price, prev_vwap, prev_rsi, 0.0, 0.0, 0.0, 0.0)
    return strat.micro_signal(bias_dir, curr, prev,
                              now=datetime.now(timezone.utc),
                              last_entry_time=last_proposal_time,
                              trades_taken=proposals_count)


def _print_header(instrument: str, price: float, vwap: float, rsi: float,
                  bias: dict, session_r: float, verbose: bool,
                  ema8: float | None = None, ema21: float | None = None) -> None:
    bias_dir  = bias.get("bias", "NEUTRAL")
    conviction = bias.get("conviction", 0)
    stars = _stars(conviction)
    ts = datetime.now().strftime("%H:%M:%S")

    color = G if bias_dir == "LONG" else (R if bias_dir == "SHORT" else Y)
    price_color = BW if price >= vwap else DM

    vwap_diff = price - vwap
    diff_arrow = "▲" if vwap_diff >= 0 else "▼"
    r_sign = "+" if session_r >= 0 else ""

    line = (f"{color}{BD}[{ts}]  {instrument}  {price_color}{price:.2f}{RS}"
            f"  VWAP {vwap:.2f} ({diff_arrow} {abs(vwap_diff):.2f})"
            f"  RSI {rsi:.1f}"
            f"  |  Bias: {color}{bias_dir} {stars}{RS}"
            f"  |  Session R: {r_sign}{session_r:.2f}")
    if verbose and ema8 is not None and ema21 is not None:
        ema_status = "↑ LONG" if price > ema21 else ("↓ SHORT" if price < ema8 else "≈ CHOP")
        line += f"  |  EMA {ema8:.1f}/{ema21:.1f} {ema_status}"
    print(f"\r{line:<120}", end="", flush=True)


def _compute_stop(bias: dict, direction: str, entry: float) -> float:
    kl = bias.get("key_levels", {})
    return strat.compute_stop(direction, entry,
                              kl.get("overnight_low"), kl.get("overnight_high"))


def _sizing_rationale(session_trades: int, session_r: float) -> str:
    return strat.sizing_rationale(session_trades, session_r)


def _show_proposal(direction: str, instrument: str, entry: float, stop: float, t1: float,
                   rr: float, prev_rsi: float, curr_rsi: float, bias: dict,
                   session_trades: int, session_r: float) -> None:
    rationale = _sizing_rationale(session_trades, session_r)
    ts = datetime.now().strftime("%H:%M:%S")
    r_sign = "+" if session_r >= 0 else ""
    print(f"\n\n{BD}╔══════════════════════════════════════════════════════╗{RS}")
    print(f"  ⚡ PROPOSAL: {direction} {rationale}")
    print(f"  [{ts}] {instrument} {entry:.2f}\n")
    if direction == "LONG":
        print(f"  Entry:  {entry:.2f}  (VWAP reclaim, bias-aligned)")
        print(f"  Stop:   {stop:.2f}  (below overnight low)")
    else:
        print(f"  Entry:  {entry:.2f}  (VWAP rejection, bias-aligned)")
        print(f"  Stop:   {stop:.2f}  (above overnight high)")
    print(f"  T1:     {t1:.2f}  (1:1 R)")
    print(f"  R:R:    {rr:.2f}\n")
    print(f"  Reasoning: Price {'reclaimed' if direction=='LONG' else 'rejected'} VWAP on bias-aligned move.")
    print(f"             Bias: {direction} conviction {bias.get('conviction',0)}. RSI turned {prev_rsi:.1f}→{curr_rsi:.1f}.")
    print(f"  Session:  Trade {session_trades+1} of max 3 | Session R: {r_sign}{session_r:.2f}\n")
    print(f"  → approve [y] / skip [n] / modify [m]: ", end="", flush=True)
    print(f"\n{BD}╚══════════════════════════════════════════════════════╝{RS}", end="")


def _log_trade(record: dict) -> None:
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def _log_session_halted(instrument: str, reason: str) -> None:
    """Write ONE integrity record marking the session stopped (e.g. IB disconnected). Never an entry."""
    _log_trade({
        "ts":           datetime.now(timezone.utc).isoformat(),
        "instrument":   instrument,
        "event":        "SESSION_HALTED",
        "exit_reason":  reason,
        "entry":        None, "stop": None, "target_1": None, "exit": None,
        "size_contracts": 0,
        "data_quality": "SIMULATED",
        "notes":        f"SESSION_HALTED: {reason} — entry generation stopped; no shadow trades logged.",
    })


def _build_record(instrument: str, direction: str, entry: float, stop: float, t1: float,
                  bias: dict, session_trades: int, session_r: float,
                  rationale: str, notes: str, exit_reason: str | None = None,
                  r_realized: float | None = None, setup_type: str = "MICRO",
                  contracts: int = 1, reasoning: dict | None = None,
                  data_quality: str = "SIMULATED") -> dict:
    return {
        "ts":                       datetime.now(timezone.utc).isoformat(),
        "instrument":               instrument,
        "trade_num_in_session":     session_trades + 1,
        "setup_type":               setup_type,
        "bias_direction":           bias.get("bias", "NEUTRAL"),
        "bias_conviction":          bias.get("conviction", 0),
        "direction":                direction,
        "entry":                    entry,
        "stop":                     stop,
        "target_1":                 t1,
        "target_2":                 None,
        "exit":                     None,
        "exit_reason":              exit_reason,
        "r_realized":               r_realized,
        "size_contracts":           contracts,
        "sizing_rationale":         rationale,
        "bias_aligned":             bias.get("bias") == direction,
        "below_proven_bar":         bias.get("conviction", 0) < 2,
        "session_result_so_far_r":  round(session_r, 3),
        # Integrity tag: LIVE_PAPER only when a real IB bracket order was placed; otherwise SIMULATED
        # (dry-run, IB disconnected, learning rep with no fill, REJECTED). The learning loop ignores
        # SIMULATED records by default so fabricated entries never train Oracle.
        "data_quality":             data_quality,
        "reasoning":                reasoning,          # entry-time belief block (learning agent)
        "exit_reasoning":           None,               # filled at close by reasoning.exit_attribution
        "notes":                    notes,
    }


def _execute_decision(d, instrument: str, bias: dict, dry_run: bool, ib_connected: bool,
                      bridge, session_trades: int, session_r: float, proposals_count: int,
                      loss_limit: float) -> tuple[datetime, int, int]:
    """Execute + log a decision_engine EntryDecision (full setup ladder + reasoning block).
    Auto-places the bracket (paper); logs the annotated trade. The single execution path."""
    now = datetime.now(timezone.utc)
    block = _trading_blocked(bridge, instrument, loss_limit)
    if block:
        print(f"\n  {R}⛔ BLOCKED — {block}{RS}")
        return now, session_trades, proposals_count + 1

    entry, stop, t1 = d.entry, d.stop, d.target
    rationale = _sizing_rationale(session_trades, session_r)
    tag = f"{d.setup_type} {d.direction} {d.contracts}x"
    wb = f" | would_have_blocked={d.would_have_blocked}" if d.would_have_blocked else ""
    data_quality = "SIMULATED"
    if not dry_run and ib_connected and bridge:
        try:
            side = "BUY" if d.direction == "LONG" else "SELL"
            contract = bridge.mes_contract() if instrument == "MES" else bridge.mnq_contract()
            results = bridge.bracket_order(contract, side, d.contracts, entry, stop, t1)
            print(f"\n  {G}{BD}⚡ FILLED {tag} {instrument} @ {entry:.2f}{RS}  SL {stop:.2f}  TP {t1:.2f}"
                  f"  conf={d.confidence} R={d.expected_r}  id={results[0].order_id}")
            notes = f"{tag}: {d.confidence} R={d.expected_r} | order_id={results[0].order_id}{wb}"
            data_quality = "LIVE_PAPER"   # a real bracket order hit the paper account
        except Exception as e:
            print(f"\n  {R}Order failed: {type(e).__name__}: {e}{RS}")
            notes = f"{tag}: order failed: {type(e).__name__}: {e}{wb}"
    else:
        why = "dry-run" if dry_run else "IB disconnected"
        print(f"\n  {G}{BD}⚡ [{why}] {tag} {instrument} @ {entry:.2f}{RS}  SL {stop:.2f}  TP {t1:.2f}"
              f"  conf={d.confidence} R={d.expected_r}")
        if d.would_have_blocked:
            print(f"  {DM}learning rep — strict would_have_blocked: {d.would_have_blocked}{RS}")
        notes = f"{tag} ({why}): {d.confidence} R={d.expected_r}{wb}"

    record = _build_record(instrument, d.direction, entry, stop, t1, bias,
                           session_trades, session_r, rationale, notes,
                           setup_type=d.setup_type, contracts=d.contracts,
                           reasoning=rsn.entry_reasoning(d, bias),
                           data_quality=data_quality)
    _log_trade(record)
    return now, session_trades + 1, proposals_count + 1


def _handle_proposal(direction: str, curr_price: float, curr_vwap: float,
                     curr_rsi: float, prev_rsi: float,
                     bias: dict, instrument: str, dry_run: bool,
                     ib_connected: bool, bridge,
                     session_trades: int, session_r: float,
                     proposals_count: int) -> tuple[datetime, int, int]:
    entry = round(curr_price, 2)
    stop  = round(_compute_stop(bias, direction, entry), 2)
    t1    = round(entry + (entry - stop) if direction == "LONG" else entry - (stop - entry), 2)
    rr    = round(abs(t1 - entry) / max(abs(entry - stop), 0.01), 2)
    rationale = _sizing_rationale(session_trades, session_r)

    _show_proposal(direction, instrument, entry, stop, t1, rr,
                   prev_rsi, curr_rsi, bias, session_trades, session_r)

    try:
        answer = input("\n  ").strip().lower()
    except EOFError:
        answer = "n"

    now = datetime.now(timezone.utc)

    if answer == "y":
        print()
        if not dry_run and ib_connected and bridge:
            try:
                order_side = "BUY" if direction == "LONG" else "SELL"
                contract = bridge.mes_contract() if instrument == "MES" else bridge.mnq_contract()
                results = bridge.bracket_order(contract, order_side, 1, entry, stop, t1)
                print(f"  Order placed: {results[0].status} | id={results[0].order_id}")
                notes = f"monitor: approved | order_id={results[0].order_id}"
            except Exception as e:
                print(f"  {R}Order placement failed: {type(e).__name__}: {e}{RS}")
                notes = f"monitor: order placement failed: {type(e).__name__}: {e}"
        elif dry_run:
            print(f"  [DRY-RUN] WOULD PLACE {direction} 1 {instrument} @ {entry:.2f} | stop={stop:.2f} | t1={t1:.2f}")
            notes = "monitor: dry-run approved"
        else:
            print(f"  {Y}[display-only] IB not connected — logged without order.{RS}")
            notes = "monitor: approved (IB disconnected)"

        _log_trade(_build_record(instrument, direction, entry, stop, t1, bias,
                                 session_trades, session_r, rationale, notes))
        return now, session_trades + 1, proposals_count + 1

    elif answer == "m":
        print()
        try:
            raw_e = input("  New entry price: ").strip()
            raw_s = input("  New stop price:  ").strip()
            new_entry = round(float(raw_e), 2)
            new_stop  = round(float(raw_s), 2)
            new_t1    = round(new_entry + (new_entry - new_stop) if direction == "LONG"
                               else new_entry - (new_stop - new_entry), 2)
            new_rr    = round(abs(new_t1 - new_entry) / max(abs(new_entry - new_stop), 0.01), 2)
            _show_proposal(direction, instrument, new_entry, new_stop, new_t1, new_rr,
                           prev_rsi, curr_rsi, bias, session_trades, session_r)
            try:
                confirm = input("\n  ").strip().lower()
            except EOFError:
                confirm = "n"
            if confirm == "y":
                print()
                if not dry_run and ib_connected and bridge:
                    try:
                        order_side = "BUY" if direction == "LONG" else "SELL"
                        contract = (bridge.mes_contract() if instrument == "MES"
                                    else bridge.mnq_contract())
                        results = bridge.bracket_order(contract, order_side, 1,
                                                        new_entry, new_stop, new_t1)
                        print(f"  Order placed: {results[0].status} | id={results[0].order_id}")
                        notes = f"monitor: modified+approved | order_id={results[0].order_id}"
                    except Exception as e:
                        print(f"  {R}Order placement failed: {type(e).__name__}: {e}{RS}")
                        notes = f"monitor: modified, order failed: {type(e).__name__}: {e}"
                elif dry_run:
                    print(f"  [DRY-RUN] WOULD PLACE {direction} 1 {instrument} @ {new_entry:.2f}")
                    notes = "monitor: dry-run modified+approved"
                else:
                    notes = "monitor: modified+approved (IB disconnected)"
                _log_trade(_build_record(instrument, direction, new_entry, new_stop, new_t1,
                                         bias, session_trades, session_r, rationale, notes))
                return now, session_trades + 1, proposals_count + 1
            else:
                print(f"  Skipped after modify.")
                _log_trade(_build_record(instrument, direction, new_entry, new_stop, new_t1,
                                         bias, session_trades, session_r, rationale,
                                         "monitor: skipped after modify", "SKIPPED", 0.0))
                return now, session_trades, proposals_count + 1
        except ValueError:
            print(f"  Invalid prices — skipping.")
            _log_trade(_build_record(instrument, direction, entry, stop, t1, bias,
                                     session_trades, session_r, rationale,
                                     "monitor: skipped (bad modify input)", "SKIPPED", 0.0))
            return now, session_trades, proposals_count + 1

    else:  # n or anything else
        print(f"\n  Skipped.")
        _log_trade(_build_record(instrument, direction, entry, stop, t1, bias,
                                 session_trades, session_r, rationale,
                                 "monitor: skipped", "SKIPPED", 0.0))
        return now, session_trades, proposals_count + 1


def _trading_blocked(bridge, instrument: str, loss_limit: float) -> str | None:
    """Return a block reason if the auto path must NOT place a trade, else None.
    Checks the global kill switch (Track 1) AND the sandbox daily loss limit ($ hard lock)."""
    try:
        from sovereign.utils.kill_switch import trading_frozen
        frz = trading_frozen()
        if frz:
            return f"SYSTEM FROZEN ({frz.get('mode')}): {frz.get('reason', '')}"
    except Exception:
        pass
    try:
        from sovereign.futures import loss_limit as ll
        pnl = ll.session_pnl_usd(bridge, instrument)
        if ll.check_and_lock(pnl, loss_limit):
            return f"DAILY LOSS LIMIT LOCKED — {ll.lock_reason()} (press 'u' or delete data/futures/.session_lock to unlock)"
    except Exception:
        pass
    return None


def _auto_execute(direction: str, curr_price: float, curr_rsi: float, prev_rsi: float,
                  curr_volume: float, avg_volume_20: float, ema8: float, ema21: float,
                  bias: dict, instrument: str, dry_run: bool, ib_connected: bool, bridge,
                  session_trades: int, session_r: float, proposals_count: int,
                  loss_limit: float) -> tuple[datetime, int, int]:
    """Fully-auto micro scalp: place the bracket without a prompt, log with a rule-derived reason."""
    now = datetime.now(timezone.utc)
    block = _trading_blocked(bridge, instrument, loss_limit)
    if block:
        print(f"\n  {R}⛔ AUTO BLOCKED — {block}{RS}")
        return now, session_trades, proposals_count + 1

    entry = round(curr_price, 2)
    stop  = round(_compute_stop(bias, direction, entry), 2)
    t1    = round(entry + (entry - stop) if direction == "LONG" else entry - (stop - entry), 2)
    rationale = _sizing_rationale(session_trades, session_r)
    vol_x = (curr_volume / avg_volume_20) if avg_volume_20 > 0 else 0.0
    reason = (f"VWAP {'reclaim' if direction=='LONG' else 'reject'} + {vol_x:.1f}x vol + "
              f"{'>' if direction=='LONG' else '<'}EMA8/21 + RSI {prev_rsi:.0f}->{curr_rsi:.0f} + bias {direction}")

    data_quality = "SIMULATED"
    if not dry_run and ib_connected and bridge:
        try:
            side = "BUY" if direction == "LONG" else "SELL"
            contract = bridge.mes_contract() if instrument == "MES" else bridge.mnq_contract()
            results = bridge.bracket_order(contract, side, 1, entry, stop, t1)
            print(f"\n  {G}{BD}⚡ AUTO FILLED {direction} 1 {instrument} @ {entry:.2f}{RS}"
                  f"  SL {stop:.2f}  TP {t1:.2f}  | id={results[0].order_id}")
            notes = f"auto: {reason} | order_id={results[0].order_id}"
            data_quality = "LIVE_PAPER"
        except Exception as e:
            print(f"\n  {R}Auto order failed: {type(e).__name__}: {e}{RS}")
            notes = f"auto: order failed: {type(e).__name__}: {e}"
    else:
        tag = "[DRY-RUN] " if dry_run else "[IB disconnected] "
        print(f"\n  {G}{BD}⚡ {tag}WOULD AUTO-PLACE {direction} 1 {instrument} @ {entry:.2f}{RS}"
              f"  SL {stop:.2f}  TP {t1:.2f}")
        print(f"  {DM}reason: {reason}{RS}")
        notes = f"auto ({'dry-run' if dry_run else 'IB disconnected'}): {reason}"

    _log_trade(_build_record(instrument, direction, entry, stop, t1, bias,
                             session_trades, session_r, rationale, notes,
                             data_quality=data_quality))
    return now, session_trades + 1, proposals_count + 1


# ── ORB (opening range breakout) — the macro discretionary gate ───────────────

def _orb_range(bars, orb_minutes: int) -> tuple[float, float] | None:
    """High/low of the first `orb_minutes` 1-min RTH bars. Delegates to scalp_strategy."""
    return strat.orb_range(bars, orb_minutes)


def _handle_orb(direction: str, instrument: str, entry: float, orb_high: float, orb_low: float,
                bias: dict, dry_run: bool, ib_connected: bool, bridge,
                session_trades: int, session_r: float, proposals_count: int,
                loss_limit: float) -> tuple[int, int, bool]:
    """ORB break → big/safe checkbox → place. Returns (session_trades, proposals_count, taken)."""
    now_block = _trading_blocked(bridge, instrument, loss_limit)
    if now_block:
        print(f"\n  {R}⛔ ORB BLOCKED — {now_block}{RS}")
        return session_trades, proposals_count, True   # taken=True so we don't re-prompt
    o = futures_params()["orb"]
    big_ct, small_ct = o["big_contracts"], o["safe_contracts"]
    big_tp  = round(entry + o["big_target_points"] if direction == "LONG"
                    else entry - o["big_target_points"], 2)
    safe_tp = round(entry + o["safe_target_points"] if direction == "LONG"
                    else entry - o["safe_target_points"], 2)
    stop    = round(orb_low if direction == "LONG" else orb_high, 2)

    # Decision channel: phone (headless) if wired, else terminal.
    if tg.two_way_ready():
        print(f"\n  📲 ORB {direction} {instrument} @ {entry:.2f} — asking phone "
              f"(big/small/now/wait/skip)...")
        d = tg.ask(tg.macro_prompt("ORB BREAKOUT", instrument, direction, entry, stop,
                                   big_tp, big_ct, safe_tp, small_ct), timeout_s=300)
        if d is None:
            print(f"  {Y}No reply in time — ORB skipped.{RS}")
            tg.send("⏱ No reply — ORB skipped.")
            return session_trades, proposals_count, True
        if d["action"] == "skip":
            print("  Phone: skip.")
            return session_trades, proposals_count, True
        if d.get("timing") == "wait":
            print("  Phone: wait for retrace — re-arming ORB.")
            tg.send("⏳ Holding for a retrace; I'll re-ask on the next break.")
            return session_trades, proposals_count, False   # re-arm (don't mark taken)
        choice = d["size"]                                   # big | small
    else:
        print(f"\n\n{BD}╔══════════════════════════════════════════════════════╗{RS}")
        print(f"  🌅 ORB BREAKOUT {direction} — {instrument} @ {entry:.2f}  (range {orb_low:.2f}-{orb_high:.2f})")
        print(f"  [B]ig ({big_ct} ct, TP {big_tp:.2f})   [S]afe ({small_ct} ct, TP {safe_tp:.2f})   [W]ait   [n]skip")
        print(f"{BD}╚══════════════════════════════════════════════════════╝{RS}")
        try:
            ans = input("  → ").strip().lower()
        except EOFError:
            ans = "n"
        if ans == "w":
            print("  Waiting for retrace — re-arming.")
            return session_trades, proposals_count, False
        if ans not in ("b", "s"):
            print("  ORB skipped.")
            return session_trades, proposals_count, True
        choice = "big" if ans == "b" else "small"

    size = big_ct if choice == "big" else small_ct
    t1   = big_tp if choice == "big" else safe_tp
    rationale = "press" if choice == "big" else "probe"
    data_quality = "SIMULATED"
    if not dry_run and ib_connected and bridge:
        try:
            side = "BUY" if direction == "LONG" else "SELL"
            contract = bridge.mes_contract() if instrument == "MES" else bridge.mnq_contract()
            results = bridge.bracket_order(contract, side, size, entry, stop, t1)
            print(f"  {G}{BD}ORB {choice.upper()} FILLED {direction} {size} {instrument} @ {entry:.2f}{RS} | id={results[0].order_id}")
            notes = f"ORB macro {rationale} | order_id={results[0].order_id}"
            data_quality = "LIVE_PAPER"
            if tg.enabled():
                tg.send(f"✅ ORB {choice} FILLED {direction} {size} {instrument} @ {entry:.2f} "
                        f"(SL {stop:.2f} TP {t1:.2f})")
        except Exception as e:
            print(f"  {R}ORB order failed: {type(e).__name__}: {e}{RS}")
            notes = f"ORB order failed: {type(e).__name__}: {e}"
    else:
        tag = "[DRY-RUN] " if dry_run else "[IB disconnected] "
        print(f"  {G}{tag}WOULD PLACE ORB {choice.upper()} {direction} {size} {instrument} @ {entry:.2f} SL {stop:.2f} TP {t1:.2f}{RS}")
        notes = f"ORB macro {rationale} ({'dry-run' if dry_run else 'IB disconnected'})"
        if tg.enabled():
            tg.send(f"{tag}WOULD PLACE ORB {choice} {direction} {size} {instrument} @ {entry:.2f}")
    rec = _build_record(instrument, direction, entry, stop, t1, bias,
                        session_trades, session_r, rationale, notes,
                        data_quality=data_quality)
    rec["size_contracts"] = size
    rec["setup"] = "ORB"
    _log_trade(rec)
    return session_trades + 1, proposals_count + 1, True


def _show_macro_proposal(macro: dict) -> None:
    d = macro["direction"]
    dates_str = ", ".join(macro["dates"])
    print(f"\n{BD}╔══════════════════════════════════════════════════════╗{RS}")
    print(f"  🌍 MACRO PROPOSAL: {d} hold")
    print(f"  {macro['streak']}-session streak (avg conviction {macro['avg_conviction']:.1f}/3)")
    print(f"  Dates: {dates_str}\n")
    print(f"  Multi-session directional alignment detected.")
    print(f"  Consider: hold overnight position through tomorrow's session.")
    print(f"  Risk: treat as 2R trade with wide stop at prior week low.\n")
    print(f"  → confirm awareness [y] / dismiss [n]: ", end="", flush=True)
    print(f"\n{BD}╚══════════════════════════════════════════════════════╝{RS}")


# ── main ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Live MES/MNQ paper trading terminal")
    ap.add_argument("--instrument", default="MES", choices=["MES", "MNQ"])
    ap.add_argument("--dry-run",  action="store_true", help="No IB order placement (prints WOULD-PLACE)")
    ap.add_argument("--auto",     action="store_true",
                    help="Auto-execute micro VWAP scalps (no prompt) — the felt loop")
    ap.add_argument("--loss-limit", type=float, default=500.0,
                    help="Hard daily loss limit in $ (locks auto path when hit; default 500)")
    ap.add_argument("--max-trades", type=int, default=20,
                    help="Bounded per-session trade cap (the $ loss limit is the real guard; default 20)")
    ap.add_argument("--orb-minutes", type=int, default=5,
                    help="Opening-range minutes for the ORB macro setup (default 5)")
    ap.add_argument("--verbose",  action="store_true", help="Extra indicator output per tick")
    ap.add_argument("--learning-mode", action="store_true",
                    help="Paper-month default: bypass session-window + regime gates, loosen volume, "
                         "log would_have_blocked. Reps > purity (decision_engine).")
    ap.add_argument("--require-ib", action="store_true",
                    help="Integrity guard: refuse to start (and halt the session) if IB Gateway is not "
                         "connected. No shadow trades on yfinance fallback while disconnected.")
    ap.add_argument("--trace", action="store_true",
                    help="Print every bar's gate outcome (why no trade) — Priority-Zero diagnosis.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    instrument = args.instrument
    dry_run    = args.dry_run

    # ── Paper-only guard ──────────────────────────────────────────────────────
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    ib_port = int(os.environ.get("IB_PORT", "4002"))
    if ib_port == 4001:
        print(f"{R}ERROR: IB_PORT=4001 is the LIVE port. This terminal is paper-only. Exiting.{RS}")
        sys.exit(1)

    print(f"\n{BD}  FUTURES MONITOR — {instrument} {'[DRY-RUN]' if dry_run else ''}{RS}")
    print(f"  Paper port {ib_port} | python3.13 | Ctrl+C to exit\n")

    # ── Load bias ─────────────────────────────────────────────────────────────
    bias = _load_bias(instrument)
    bias_dir = bias.get("bias", "NEUTRAL")
    print(f"  Today's bias: {bias_dir} conviction {bias.get('conviction',0)}/3")
    if bias_dir == "NEUTRAL":
        print(f"  {Y}  NEUTRAL bias — signal detection disabled. Display-only mode.{RS}")
    kill_level = _kill_level(bias_dir, bias, instrument)
    if kill_level is not None:
        side = "above" if bias_dir == "SHORT" else "below"
        print(f"  Kill level: {bias_dir} bias dies {side} {kill_level:.2f} → auto-gates to NEUTRAL")

    # ── IB connection ─────────────────────────────────────────────────────────
    bridge = None
    ib_connected = False
    live_contract = None
    if not dry_run:
        try:
            from sovereign.futures.ib_bridge import IBBridge
            bridge = IBBridge()
            bridge.connect()
            ib_connected = True
            live_contract = bridge.mes_contract() if instrument == "MES" else bridge.mnq_contract()
            print(f"  {G}IB Gateway connected.{RS}")
        except Exception as e:
            print(f"  {Y}[warn] IB connection failed ({type(e).__name__}: {e}). Display-only mode.{RS}")
    else:
        print(f"  [DRY-RUN] Skipping IB connection.")

    # ── Guard 1: integrity halt — refuse to start without IB when --require-ib is set ──
    if not dry_run and args.require_ib and not ib_connected:
        print(f"\n  {R}{BD}⛔ SESSION HALTED — IB Gateway not connected and --require-ib is set.{RS}")
        print(f"  {Y}  Refusing to run: shadow trades on fallback data corrupt the learning loop.{RS}")
        print(f"  {Y}  Start IB Gateway (paper port {ib_port}) and re-run.{RS}")
        _log_session_halted(instrument, "IB_DISCONNECTED")
        sys.exit(1)

    # Which bar feed is live this session
    _bars_src = "IB Gateway 1-min" if (ib_connected and live_contract is not None) else "yfinance fallback"
    print(f"  Bars: {_bars_src}")

    # ── Macro proposal ────────────────────────────────────────────────────────
    macro = _check_macro(instrument)
    if macro:
        _show_macro_proposal(macro)
        try:
            ans = input("\n  ").strip().lower()
        except EOFError:
            ans = "n"
        if ans == "y":
            _log_trade({
                "ts":                       datetime.now(timezone.utc).isoformat(),
                "instrument":               instrument,
                "trade_num_in_session":     0,
                "bias_direction":           macro["direction"],
                "bias_conviction":          int(round(macro["avg_conviction"])),
                "direction":               macro["direction"],
                "entry":                    None,
                "stop":                     None,
                "target_1":                 None,
                "target_2":                 None,
                "exit":                     None,
                "exit_reason":              "MACRO_HOLD_NOTED",
                "r_realized":               None,
                "size_contracts":           0,
                "sizing_rationale":         "macro",
                "bias_aligned":             True,
                "below_proven_bar":         macro["avg_conviction"] < 2,
                "session_result_so_far_r":  0.0,
                "data_quality":             "SIMULATED",  # a noted hold, not an executed fill
                "notes":                    f"macro hold noted: {macro['streak']}-session streak",
            })
            print(f"  Macro hold logged.\n")

    auto       = args.auto
    loss_limit = float(args.loss_limit)
    max_trades = int(args.max_trades)
    orb_minutes = int(args.orb_minutes)
    learning_mode = args.learning_mode
    trace      = args.trace

    # Loss-limit / kill-switch startup status
    from sovereign.futures import loss_limit as ll
    if ll.is_locked():
        print(f"  {R}{BD}⛔ Auto path is LOCKED from a prior session — {ll.lock_reason()}{RS}")
        print(f"  {Y}  Unlock: delete data/futures/.session_lock{RS}")
    print(f"  Mode: {'AUTO' if auto else 'MANUAL'}{' | LEARNING (reps>purity)' if learning_mode else ' | STRICT'}"
          f"{' | TRACE' if trace else ''} | daily loss limit ${loss_limit:.0f} | max {max_trades} trades/session")

    print(f"\n  Setup ladder: ORB / VWAP-MR / micro (decision_engine, one engine == replay)")
    if learning_mode:
        print(f"  {Y}LEARNING MODE: session-window + regime bypassed, volume loosened; "
              f"would_have_blocked logged on every rep.{RS}\n")
    else:
        print(f"  STRICT MODE: full gates (validation-only).\n")

    # ── Main polling loop ─────────────────────────────────────────────────────
    prev_price: float = 0.0
    prev_vwap:  float = 0.0
    prev_rsi:   float = 50.0
    last_proposal_time: datetime | None = None
    proposals_count: int = 0
    session_trades:  int = 0
    session_r:       float = 0.0
    last_bar_ts = None
    orb_high: float | None = None
    orb_low:  float | None = None
    orb_taken: bool = False
    bias_invalidated: bool = False

    try:
        while True:
            try:
                bars = _fetch_bars(instrument, bridge if ib_connected else None, live_contract,
                                   require_ib=args.require_ib)
                if bars.empty:
                    time.sleep(5)
                    continue

                curr_ts = bars.index[-1]
                curr_price, curr_vwap, curr_rsi, curr_volume, avg_volume_20, ema8, ema21 = _compute_indicators(bars)

                _print_header(instrument, curr_price, curr_vwap, curr_rsi,
                              bias, session_r, args.verbose,
                              ema8=ema8, ema21=ema21)

                # Only check signal on new bar (avoid repeat-firing on same data)
                if last_bar_ts is not None and curr_ts == last_bar_ts:
                    time.sleep(5)
                    continue

                # ── Falsifier gate: once price crosses the kill level, the bias is DEAD (latched) ──
                if not bias_invalidated and kill_level is not None and bias_dir in ("LONG", "SHORT"):
                    breached = (curr_price > kill_level) if bias_dir == "SHORT" else (curr_price < kill_level)
                    if breached:
                        bias_invalidated = True
                        print(f"\n  {R}{BD}⚠ BIAS INVALIDATED — {instrument} {curr_price:.2f} crossed kill "
                              f"{kill_level:.2f}. Gating to NEUTRAL for the session.{RS}")
                bias_dir_eff = "NEUTRAL" if bias_invalidated else bias_dir

                # ── ORB range capture (once per session) ──
                if orb_high is None:
                    rng = _orb_range(bars, orb_minutes)
                    if rng is not None and len(bars) >= orb_minutes:
                        orb_high, orb_low = rng

                # ── Guard 1: mid-session IB-disconnect halt — stop, don't shadow-log ──
                if (not dry_run and (args.require_ib or learning_mode) and ib_connected
                        and bridge is not None and not bridge.is_connected()):
                    print(f"\n  {R}{BD}⛔ SESSION HALTED — IB Gateway disconnected mid-session.{RS}")
                    print(f"  {Y}  Entry generation stopped. No shadow trades. Reconnect + restart to resume.{RS}")
                    _log_session_halted(instrument, "IB_DISCONNECTED")
                    break

                # ── ONE decision engine: ORB / VWAP-MR / micro + full telemetry (== replay) ──
                if proposals_count < max_trades and (bias_dir_eff in ("LONG", "SHORT") or learning_mode):
                    prev_ind = None
                    if len(bars) > 1:
                        try:
                            prev_ind = strat.compute_indicators(bars.iloc[:-1])
                        except Exception:
                            prev_ind = None
                    try:
                        regime_ctx = regime_mod.classify_session(bars)
                    except Exception:
                        regime_ctx = None
                    try:
                        profile_ctx = vp.compute_profile(bars)
                    except Exception:
                        profile_ctx = None
                    eff_bias = dict(bias); eff_bias["bias"] = bias_dir_eff
                    decision = de.evaluate_entry(
                        bars, bias=eff_bias, ts=datetime.now(timezone.utc), instrument=instrument,
                        prev_ind=prev_ind,
                        orb_levels=((orb_high, orb_low) if orb_high is not None else None),
                        orb_taken=orb_taken, regime=regime_ctx, prior_profile=profile_ctx,
                        last_entry_time=last_proposal_time, trades_taken=proposals_count,
                        learning_mode=learning_mode,
                        oracle_invalidation=_oracle_invalidation(instrument),
                    )
                    if decision is not None and decision.rejected_reason:
                        # Integrity guard (Guard 2): a structurally-impossible entry
                        # (stop==entry / fabricated R / wrong-side stop). Log it as REJECTED with the
                        # bad values preserved, but NEVER execute or count it as a trade.
                        print(f"\n  {R}⛔ REJECTED entry — {decision.rejected_reason}{RS}")
                        _log_trade(_build_record(
                            instrument, decision.direction, decision.entry, decision.stop,
                            decision.target, bias, session_trades, session_r,
                            _sizing_rationale(session_trades, session_r),
                            f"REJECTED: {decision.rejected_reason} (expected_r={decision.expected_r})",
                            exit_reason="REJECTED:INVALID_STOP", setup_type=decision.setup_type,
                            contracts=decision.contracts, data_quality="SIMULATED"))
                    elif decision is not None:
                        print()  # newline after header
                        if decision.setup_type == "ORB":
                            orb_taken = True
                        last_proposal_time, session_trades, proposals_count = _execute_decision(
                            decision, instrument, bias, dry_run, ib_connected, bridge,
                            session_trades, session_r, proposals_count, loss_limit,
                        )
                    elif trace:
                        print(f"\n  {DM}[trace] {instrument} no entry @ {curr_price:.2f} | "
                              f"bias={bias_dir_eff} window_ok={strat.in_trade_window(datetime.now(timezone.utc))} "
                              f"orb={'set' if orb_high is not None else 'pending'} vol={curr_volume:.0f}/"
                              f"{avg_volume_20:.0f} learning={learning_mode}{RS}")

                prev_price, prev_vwap, prev_rsi = curr_price, curr_vwap, curr_rsi
                last_bar_ts = curr_ts

            except ValueError:
                pass  # empty bars or computation error — skip tick
            except Exception as e:
                if args.verbose:
                    print(f"\n  [warn] tick error: {type(e).__name__}: {e}")

            time.sleep(5)

    except KeyboardInterrupt:
        r_sign = "+" if session_r >= 0 else ""
        print(f"\n\n  {BD}Session complete.{RS}")
        print(f"  Proposals: {proposals_count} | Trades logged: {session_trades} | Session R: {r_sign}{session_r:.2f}")
        if ib_connected and bridge:
            try:
                bridge.disconnect()
            except Exception:
                pass
        sys.exit(0)


if __name__ == "__main__":
    main()
