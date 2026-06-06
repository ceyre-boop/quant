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
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_bias(instrument: str) -> dict:
    today = _today()
    last = None
    if BIAS_LOG.exists():
        with open(BIAS_LOG) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("date") == today and rec.get("instrument", "").upper() == instrument.upper():
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
                if rec.get("instrument", "").upper() == instrument.upper():
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


def _fetch_bars(instrument: str):
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
    """Returns (last_price, vwap, rsi, curr_volume, avg_volume_20, ema8, ema21). Raises ValueError if bars empty."""
    if bars.empty:
        raise ValueError("empty bars")
    last_price = float(bars["Close"].iloc[-1].item())
    bars = bars.copy()
    bars["typical"] = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    bars["cum_tp_vol"] = (bars["typical"] * bars["Volume"]).cumsum()
    bars["cum_vol"]    = bars["Volume"].cumsum()
    vwap = float((bars["cum_tp_vol"] / bars["cum_vol"]).iloc[-1].item())
    delta    = bars["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs       = avg_gain / avg_loss
    rsi      = float((100 - (100 / (1 + rs))).iloc[-1].item())
    curr_volume   = float(bars["Volume"].iloc[-1].item())
    avg_volume_20 = float(bars["Volume"].tail(20).mean())
    ema8  = float(bars["Close"].ewm(span=8,  adjust=False).mean().iloc[-1].item())
    ema21 = float(bars["Close"].ewm(span=21, adjust=False).mean().iloc[-1].item())
    return last_price, vwap, rsi, curr_volume, avg_volume_20, ema8, ema21


def _check_signal(bias_dir: str, curr_price: float, curr_vwap: float, curr_rsi: float,
                  prev_price: float, prev_vwap: float, prev_rsi: float,
                  last_proposal_time: datetime | None, proposals_count: int,
                  curr_volume: float, avg_volume_20: float,
                  ema8: float, ema21: float) -> str | None:
    if bias_dir not in ("LONG", "SHORT"):
        return None
    if proposals_count >= 3:
        return None
    if last_proposal_time is not None:
        elapsed = (datetime.now(timezone.utc) - last_proposal_time).total_seconds()
        if elapsed < 300:
            return None
    # Volume confirmation gate — signal bar must have above-average participation
    if avg_volume_20 > 0 and curr_volume < 1.5 * avg_volume_20:
        return None
    # EMA position filter — price must be on the correct structural side
    above_both = curr_price > ema8 and curr_price > ema21
    below_both = curr_price < ema8 and curr_price < ema21
    long_signal  = (prev_price < prev_vwap and curr_price >= curr_vwap
                    and prev_rsi < 50 and curr_rsi >= 50)
    short_signal = (prev_price > prev_vwap and curr_price <= curr_vwap
                    and prev_rsi > 50 and curr_rsi <= 50)
    if long_signal  and bias_dir == "LONG"  and above_both:
        return "LONG"
    if short_signal and bias_dir == "SHORT" and below_both:
        return "SHORT"
    return None


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
    if direction == "LONG":
        return float(kl.get("overnight_low") or entry * 0.999)
    return float(kl.get("overnight_high") or entry * 1.001)


def _sizing_rationale(session_trades: int, session_r: float) -> str:
    if session_trades == 0:
        return "probe"
    return "press" if session_r > 0 else "reduce"


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


def _build_record(instrument: str, direction: str, entry: float, stop: float, t1: float,
                  bias: dict, session_trades: int, session_r: float,
                  rationale: str, notes: str, exit_reason: str | None = None,
                  r_realized: float | None = None) -> dict:
    return {
        "ts":                       datetime.now(timezone.utc).isoformat(),
        "instrument":               instrument,
        "trade_num_in_session":     session_trades + 1,
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
        "size_contracts":           1,
        "sizing_rationale":         rationale,
        "bias_aligned":             True,
        "below_proven_bar":         bias.get("conviction", 0) < 2,
        "session_result_so_far_r":  round(session_r, 3),
        "notes":                    notes,
    }


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
    ap.add_argument("--dry-run",  action="store_true", help="No IB order placement")
    ap.add_argument("--verbose",  action="store_true", help="Extra indicator output per tick")
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

    # ── IB connection ─────────────────────────────────────────────────────────
    bridge = None
    ib_connected = False
    if not dry_run:
        try:
            from sovereign.futures.ib_bridge import IBBridge
            bridge = IBBridge()
            bridge.connect()
            ib_connected = True
            print(f"  {G}IB Gateway connected.{RS}")
        except Exception as e:
            print(f"  {Y}[warn] IB connection failed ({type(e).__name__}: {e}). Display-only mode.{RS}")
    else:
        print(f"  [DRY-RUN] Skipping IB connection.")

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
                "notes":                    f"macro hold noted: {macro['streak']}-session streak",
            })
            print(f"  Macro hold logged.\n")

    print(f"\n  Watching for VWAP reclaim + RSI confirmation in bias direction...")
    print(f"  Max 3 proposals per session | 5-minute cooldown between proposals\n")

    # ── Main polling loop ─────────────────────────────────────────────────────
    prev_price: float = 0.0
    prev_vwap:  float = 0.0
    prev_rsi:   float = 50.0
    last_proposal_time: datetime | None = None
    proposals_count: int = 0
    session_trades:  int = 0
    session_r:       float = 0.0
    last_bar_ts = None

    try:
        while True:
            try:
                bars = _fetch_bars(instrument)
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

                if prev_price > 0:  # have at least one prior tick
                    signal = _check_signal(
                        bias_dir, curr_price, curr_vwap, curr_rsi,
                        prev_price, prev_vwap, prev_rsi,
                        last_proposal_time, proposals_count,
                        curr_volume, avg_volume_20, ema8, ema21,
                    )
                    if signal:
                        print()  # newline after header
                        last_proposal_time, session_trades, proposals_count = _handle_proposal(
                            signal, curr_price, curr_vwap, curr_rsi, prev_rsi,
                            bias, instrument, dry_run, ib_connected, bridge,
                            session_trades, session_r, proposals_count,
                        )

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
