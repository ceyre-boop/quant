#!/usr/bin/env python3
"""
Nightly ES/NQ replay — "backtest like it's live tonight."

Drives the EXACT live scalp + ORB logic (sovereign/futures/scalp_strategy.py) over
historical 1-min bars, simulates bracket fills with the REAL cost model, and enforces
the prop-firm trailing-drawdown / daily-loss / consistency rules. Writes a morning-ready
report so you walk in tomorrow knowing how the machine would have traded.

What you backtest here is what futures_monitor.py trades — same module, same params.

Usage:
    python3.13 scripts/futures_replay.py                         # MES, yfinance, last 5d
    python3.13 scripts/futures_replay.py --instrument MNQ --source yf
    python3.13 scripts/futures_replay.py --day 2026-06-06 --bias auto
    python3.13 scripts/futures_replay.py --source ib --lookback 5d
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.futures import scalp_strategy as strat          # noqa: E402
from sovereign.futures import bar_feed as bf                   # noqa: E402
from sovereign.futures import regime as regime_mod             # noqa: E402
from sovereign.futures import volume_profile as vp             # noqa: E402
from sovereign.futures import cvd as cvd_mod                   # noqa: E402
from sovereign.futures.config import futures_params, contract_spec, round_turn_cost_usd  # noqa: E402

BIAS_LOG = ROOT / "data" / "futures" / "bias_log.jsonl"
REPORT_DIR = ROOT / "data" / "futures"


# ── bias for a replay day ─────────────────────────────────────────────────────

def _logged_bias(day: str, instrument: str) -> dict | None:
    norm = {"MES": "ES", "MNQ": "NQ"}.get(instrument.upper(), instrument.upper())
    if not BIAS_LOG.exists():
        return None
    found = None
    for line in BIAS_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        rinst = {"MES": "ES", "MNQ": "NQ"}.get(r.get("instrument", "").upper(), r.get("instrument", "").upper())
        if r.get("date") == day and rinst == norm:
            found = r
    return found


def _day_bias(day_df, day: str, prior_close: float | None,
              instrument: str, mode: str) -> tuple[str, dict]:
    """Return (bias_dir, key_levels). mode: auto|long|short|neutral."""
    if mode in ("long", "short"):
        return mode.upper(), {}
    if mode == "neutral":
        return "NEUTRAL", {}
    # auto: prefer the logged bias for that date
    logged = _logged_bias(day, instrument)
    if logged and logged.get("bias") in ("LONG", "SHORT"):
        return logged["bias"], logged.get("key_levels", {})
    # proxy: first RTH bar vs prior session close
    if prior_close is not None and len(day_df) > 0:
        first_close = float(day_df["Close"].iloc[0])
        return ("LONG" if first_close >= prior_close else "SHORT"), {}
    return "NEUTRAL", {}


# ── fill simulation ───────────────────────────────────────────────────────────

def _check_exit(pos: dict, high: float, low: float) -> tuple[float | None, str | None]:
    """Conservative same-bar resolution: stop is assumed to fill before target."""
    if pos["direction"] == "LONG":
        if low <= pos["stop"]:
            return pos["stop"], "STOPPED"
        if high >= pos["target"]:
            return pos["target"], "T1_HIT"
    else:
        if high >= pos["stop"]:
            return pos["stop"], "STOPPED"
        if low <= pos["target"]:
            return pos["target"], "T1_HIT"
    return None, None


def simulate_session(day_df, day: str, bias_dir: str, key_levels: dict,
                     instrument: str, orb_size: str, on_event=None,
                     setups=None, regime: dict | None = None,
                     cvd_gate: bool = False, scale_in: bool = False,
                     prior_profile: dict | None = None) -> dict:
    """Run the live logic over one RTH session. Returns trades + per-day stats.

    `setups` (set, default {"orb","micro"}) chooses which strategies may trade — pass a
    single setup to judge it in isolation (Increment 3: ORB and VWAP-MR are never blended
    when validating). `regime` (optional) gates each setup to its favorable regime.
    All entries also pass the time-of-day window gate.

    Increment 4 (all opt-in, telemetry-logged regardless): `cvd_gate` blocks entries whose
    CVD proxy doesn't confirm; `scale_in` adds contracts into confirmed winners (loser stays
    1 lot); `prior_profile` (POC/VAH/VAL) drives confluence scoring. None of these change
    the entry SIGNAL — only gating, scoring, and sizing on top.

    `on_event(kind, payload)` (optional) animates the SAME engine the batch report uses."""
    p = futures_params()
    spec = contract_spec(instrument)
    dpp = spec["dollars_per_point"]
    tick = spec["tick"]
    orb_cfg = p["orb"]
    dd_model = p["prop"]["drawdown_model"]
    tol_price = p["volume_profile"]["confluence_tol_ticks"] * tick
    sz = p["sizing"]
    daily_loss = p["prop"]["daily_loss_limit_usd"]
    if setups is None:
        setups = {"orb", "micro"}

    def _emit(kind, payload):
        if on_event is not None:
            on_event(kind, payload)

    def _may_fire(setup_key: str, ts) -> bool:
        if setup_key not in setups:
            return False
        if not strat.in_trade_window(ts):
            return False
        if regime is not None:
            ok, _why = regime_mod.setup_allowed(setup_key, regime)
            if not ok:
                return False
        return True

    feed = bf.ReplayBarFeed(day_df, warmup=2)
    n_bars = len(feed)
    trades: list[dict] = []
    realized = 0.0
    hwm = 0.0
    max_dd = 0.0                 # realized-equity drawdown (EOD/static models)
    peak_intra = 0.0
    intraday_dd = 0.0           # intraday drawdown incl. open-position heat (Apex trailing)
    day_low_equity = 0.0

    position: dict | None = None
    prev_ind = None
    last_entry_time = None
    trades_taken = 0
    session_r = 0.0
    orb_hi = orb_lo = None
    orb_taken = False

    def _close(pos, exit_price, reason, ts):
        nonlocal realized, hwm, max_dd, day_low_equity, session_r, last_entry_time, trades_taken
        long = pos["direction"] == "LONG"
        pts = (exit_price - pos["entry"]) if long else (pos["entry"] - exit_price)
        gross = pts * dpp * pos["contracts"]
        cost = round_turn_cost_usd(instrument, pos["contracts"])
        net = gross - cost
        # static 1-lot counterfactual (off the tier-1 entry) for the scale-in comparison
        pts1 = (exit_price - pos["base_entry"]) if long else (pos["base_entry"] - exit_price)
        net_1lot = pts1 * dpp * 1 - round_turn_cost_usd(instrument, 1)
        r = strat.compute_r(pos["entry"], pos["stop"], exit_price, pos["direction"])
        realized += net
        hwm = max(hwm, realized)
        max_dd = max(max_dd, hwm - realized)
        day_low_equity = min(day_low_equity, realized)
        session_r += r
        last_entry_time = ts
        trades_taken += 1
        trades.append({
            "day": day, "instrument": instrument, "setup": pos["setup"],
            "direction": pos["direction"], "entry": round(pos["entry"], 2),
            "stop": round(pos["stop"], 2), "target": round(pos["target"], 2),
            "kill_level": round(pos["stop"], 2), "exit": round(exit_price, 2),
            "exit_reason": reason, "contracts": pos["contracts"],
            "gross_usd": round(gross, 2), "cost_usd": round(cost, 2), "net_usd": round(net, 2),
            "net_1lot_usd": round(net_1lot, 2),
            "confluence": pos.get("confluence", 0), "cvd_slope": pos.get("cvd_slope"),
            "cvd_confirmed": pos.get("cvd_confirmed"),
            "r_realized": r, "entry_ts": pos["entry_ts"], "exit_ts": ts.isoformat(),
        })
        _emit("exit", trades[-1])

    def _make_position(setup_label, direction, entry, stop, target, base_ct, window, ts):
        """Build a position with confluence + CVD telemetry. Applies the CVD gate (opt-in).
        Returns the position dict, or None if the gate blocks it."""
        cstate = cvd_mod.cvd_state(window)
        confirmed = cvd_mod.cvd_confirms(setup_label, direction, cstate)
        if cvd_gate and confirmed is False:        # None (unknown) is allowed through
            return None
        return {
            "direction": direction, "entry": entry, "base_entry": entry, "stop": stop,
            "target": target, "contracts": base_ct, "setup": setup_label,
            "entry_ts": ts.isoformat(),
            "confluence": vp.confluence_score(entry, prior_profile, tol_price),
            "cvd_slope": (round(cstate["slope"], 2) if cstate else None),
            "cvd_confirmed": confirmed, "cvd_state": cstate,
        }

    for i, (ts, window) in enumerate(feed.stream(), start=1):
        bar = window.iloc[-1]
        high, low, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        try:
            ind = strat.compute_indicators(window)
        except ValueError:
            continue

        _emit("bar", {"ts": ts, "i": i, "n": n_bars, "ind": ind, "bias": bias_dir,
                      "realized": realized, "session_r": session_r,
                      "in_position": position is not None})

        # Intraday equity incl. open-position heat (the honest Apex trailing-DD model).
        if dd_model == "trailing_intraday" and position is not None:
            ct = position["contracts"]
            if position["direction"] == "LONG":
                fav = (high - position["entry"]) * dpp * ct
                adv = (low - position["entry"]) * dpp * ct
            else:
                fav = (position["entry"] - low) * dpp * ct
                adv = (position["entry"] - high) * dpp * ct
            peak_intra = max(peak_intra, realized + fav)
            intraday_dd = max(intraday_dd, peak_intra - (realized + adv))
        else:
            peak_intra = max(peak_intra, realized)
            intraday_dd = max(intraday_dd, peak_intra - realized)

        # 1) manage open position on this bar
        if position is not None:
            ex, reason = _check_exit(position, high, low)
            if ex is not None:
                _close(position, ex, reason, ts)
                position = None

        # 2) capture ORB range
        if orb_hi is None and len(window) >= orb_cfg["minutes"]:
            rng = strat.orb_range(window, orb_cfg["minutes"])
            if rng:
                orb_hi, orb_lo = rng

        # 3) new entries only when flat
        if position is None:
            # ORB macro (once per session, bias-aligned)
            if not orb_taken and orb_hi is not None and bias_dir in ("LONG", "SHORT") and _may_fire("orb", ts):
                d = strat.orb_break(bias_dir, close, orb_hi, orb_lo, ind)
                if d:
                    contracts = orb_cfg["big_contracts"] if orb_size == "big" else orb_cfg["safe_contracts"]
                    tgt_pts = orb_cfg["big_target_points"] if orb_size == "big" else orb_cfg["safe_target_points"]
                    stop = orb_lo if d == "LONG" else orb_hi
                    target = close + tgt_pts if d == "LONG" else close - tgt_pts
                    orb_taken = True
                    pos = _make_position("ORB", d, close, stop, target, contracts, window, ts)
                    if pos:
                        position = pos
                        _emit("entry", position)
            # VWAP mean-reversion (fade; no bias needed — regime decides when allowed)
            if position is None and _may_fire("vwap_mr", ts):
                sig = strat.vwap_mr_signal(window, ind, now=ts,
                                           last_entry_time=last_entry_time, trades_taken=trades_taken)
                if sig:
                    bands = strat.vwap_bands(window)
                    stop, target = strat.vwap_mr_levels(sig, close, bands, instrument)
                    pos = _make_position("VWAP_MR", sig, close, stop, target, 1, window, ts)
                    if pos:
                        position = pos
                        _emit("entry", position)
            # legacy micro scalp (VWAP reclaim + RSI, bias-aligned) — still ships in monitor
            if position is None and prev_ind is not None and bias_dir in ("LONG", "SHORT") and _may_fire("micro", ts):
                sig = strat.micro_signal(bias_dir, ind, prev_ind, now=ts,
                                         last_entry_time=last_entry_time, trades_taken=trades_taken)
                if sig:
                    stop = strat.compute_stop(sig, close, key_levels.get("overnight_low"),
                                              key_levels.get("overnight_high"))
                    target = strat.target_from_rr(sig, close, stop)
                    pos = _make_position("MICRO", sig, close, stop, target, 1, window, ts)
                    if pos:
                        position = pos
                        _emit("entry", position)

        # 3b) scale-in into a confirmed winner (opt-in; loser NEVER grows beyond 1 lot)
        elif scale_in and position["contracts"] < sz["max_contracts"]:
            long = position["direction"] == "LONG"
            fav_ticks = ((close - position["base_entry"]) if long else (position["base_entry"] - close)) / tick
            dd_room = (daily_loss + realized) > sz["dd_buffer_ticks"] * tick * dpp
            cstate = cvd_mod.cvd_state(window)
            confirmed = cvd_mod.cvd_confirms(position["setup"], position["direction"], cstate)
            add = 0
            if fav_ticks >= sz["tier2_trigger_ticks"] and confirmed and dd_room:
                if position["contracts"] == 1:
                    add = 1                                  # tier 2
                elif (position["contracts"] == 2
                      and position.get("confluence", 0) >= sz["tier3_min_confluence"]
                      and cvd_mod.is_strong(cstate)):
                    add = 1                                  # tier 3 (full confluence + strong CVD)
            if add:
                newct = position["contracts"] + add
                position["entry"] = (position["entry"] * position["contracts"] + close * add) / newct
                position["contracts"] = newct
                _emit("scale", {"ts": ts, "contracts": newct, "add_price": round(close, 2)})

        prev_ind = ind

    # EOD: close any runner at last close
    if position is not None:
        last_close = float(day_df["Close"].iloc[-1])
        _close(position, last_close, "EOD", day_df.index[-1])

    binding_dd = intraday_dd if dd_model == "trailing_intraday" else max_dd
    return {
        "day": day, "bias": bias_dir, "trades": trades, "setups": sorted(setups),
        "net_usd": round(realized, 2), "max_drawdown_usd": round(max_dd, 2),
        "intraday_dd_usd": round(intraday_dd, 2), "binding_dd_usd": round(binding_dd, 2),
        "day_low_equity_usd": round(day_low_equity, 2), "n_trades": len(trades),
    }


# ── aggregate + prop verdict ──────────────────────────────────────────────────

def _aggregate(sessions: list[dict], instrument: str) -> dict:
    p = futures_params()["prop"]
    all_trades = [t for s in sessions for t in s["trades"]]
    n = len(all_trades)
    net_total = round(sum(s["net_usd"] for s in sessions), 2)
    wins = [t for t in all_trades if t["net_usd"] > 0]
    losses = [t for t in all_trades if t["net_usd"] <= 0]
    gross_win = sum(t["net_usd"] for t in wins)
    gross_loss = abs(sum(t["net_usd"] for t in losses)) or 1e-9
    total_r = round(sum(t["r_realized"] for t in all_trades), 3)

    # Increment 4: confluence → outcome, CVD confirmation split, scale-in vs static
    from collections import defaultdict
    conf_bkt = defaultdict(list)
    for t in all_trades:
        conf_bkt[t.get("confluence", 0)].append(t["r_realized"])
    by_confluence = {c: {"n": len(rs), "avg_r": round(sum(rs) / len(rs), 3)}
                     for c, rs in sorted(conf_bkt.items())}

    def _wr(ts_):
        w = sum(1 for t in ts_ if t["net_usd"] > 0)
        return {"n": len(ts_), "win_rate": round(w / len(ts_), 2)} if ts_ else {"n": 0, "win_rate": 0.0}
    cvd_split = {
        "confirmed": _wr([t for t in all_trades if t.get("cvd_confirmed") is True]),
        "unconfirmed": _wr([t for t in all_trades if t.get("cvd_confirmed") is False]),
        "unknown": {"n": sum(1 for t in all_trades if t.get("cvd_confirmed") is None)},
    }
    net_1lot_total = round(sum(t.get("net_1lot_usd", t["net_usd"]) for t in all_trades), 2)

    # run-level trailing drawdown on the cumulative realized curve
    cum = 0.0
    hwm = 0.0
    run_dd = 0.0
    for s in sessions:
        cum += s["net_usd"]
        hwm = max(hwm, cum)
        run_dd = max(run_dd, hwm - cum)

    # Apex-style trailing-intraday DD includes open-position heat (computed per session).
    dd_model = futures_params()["prop"]["drawdown_model"]
    intra_dd = max((s.get("intraday_dd_usd", 0.0) for s in sessions), default=0.0)
    binding_dd = intra_dd if dd_model == "trailing_intraday" else run_dd

    # win rate by ET hour bucket — makes the time-of-day gate data-driven
    from collections import defaultdict
    buckets = defaultdict(lambda: [0, 0])
    for t in all_trades:
        try:
            h = datetime.fromisoformat(t["entry_ts"]).astimezone(bf.ET).hour
        except Exception:
            continue
        buckets[h][1] += 1
        if t["net_usd"] > 0:
            buckets[h][0] += 1
    by_hour = {h: {"wins": w, "n": nn, "win_rate": round(w / nn, 2)}
               for h, (w, nn) in sorted(buckets.items())}

    day_nets = [s["net_usd"] for s in sessions]
    worst_day = min(day_nets) if day_nets else 0.0
    best_day = max(day_nets) if day_nets else 0.0

    hit_target = net_total >= p["profit_target_usd"]

    # Hard rules (blow the account anytime): trailing DD (binding model) + daily loss.
    breaches = []
    if binding_dd >= p["trailing_drawdown_usd"]:
        breaches.append(f"TRAILING DD ${binding_dd:.0f} >= ${p['trailing_drawdown_usd']:.0f} ({dd_model})")
    if worst_day <= -p["daily_loss_limit_usd"]:
        breaches.append(f"DAILY LOSS ${worst_day:.0f} <= -${p['daily_loss_limit_usd']:.0f}")

    # Structural (not a P&L breach) — strategy/account mismatch, flag loudly.
    structural = []
    setups_used = {t["setup"] for t in all_trades}
    if dd_model == "trailing_intraday" and "VWAP_MR" in setups_used:
        structural.append("VWAP_MR under trailing-intraday DD — mean-reversion carries unrealized "
                          "heat that this account model punishes; ORB is the structural fit.")

    # Consistency only matters at payout — soft warning until the target is hit.
    consistency_ok = True
    warnings_ = []
    if net_total > 0 and best_day / net_total > p["consistency_pct"]:
        consistency_ok = False
        msg = f"CONSISTENCY: best day {best_day/net_total:.0%} > {p['consistency_pct']:.0%}"
        (breaches if hit_target else warnings_).append(msg)

    verdict = "PASS" if (hit_target and not breaches) else ("BREACH" if breaches else "IN_PROGRESS")

    return {
        "instrument": instrument,
        "sessions": len(sessions),
        "n_trades": n,
        "net_total_usd": net_total,
        "win_rate": round(len(wins) / n, 3) if n else 0.0,
        "profit_factor": round(min(gross_win / gross_loss, 20.0), 3) if losses else None,
        "total_r": total_r,
        "avg_r": round(total_r / n, 3) if n else 0.0,
        "run_trailing_dd_usd": round(run_dd, 2),
        "intraday_dd_usd": round(intra_dd, 2),
        "binding_dd_usd": round(binding_dd, 2),
        "dd_model": dd_model,
        "by_hour": by_hour,
        "by_confluence": by_confluence,
        "cvd_split": cvd_split,
        "net_dynamic_usd": net_total,
        "net_1lot_usd": net_1lot_total,
        "best_day_usd": round(best_day, 2),
        "worst_day_usd": round(worst_day, 2),
        "prop_profit_target_usd": p["profit_target_usd"],
        "prop_trailing_dd_usd": p["trailing_drawdown_usd"],
        "prop_progress_pct": round(net_total / p["profit_target_usd"] * 100, 1) if p["profit_target_usd"] else 0.0,
        "prop_breaches": breaches,
        "structural_warnings": structural,
        "warnings": warnings_,
        "consistency_ok": consistency_ok,
        "verdict": verdict,
    }


def _print_report(agg: dict, sessions: list[dict]) -> None:
    G, R, Y, BD, RS = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"
    v = agg["verdict"]
    vc = G if v == "PASS" else (R if v == "BREACH" else Y)
    print(f"\n{BD}{'═'*64}{RS}")
    print(f"  {BD}REPLAY REPORT — {agg['instrument']}  |  {agg['sessions']} session(s){RS}")
    print(f"{BD}{'═'*64}{RS}")
    if agg.get("untrusted"):
        print(f"  {Y}{BD}⚠ UNTRUSTED — volume profile / CVD need real volume; yfinance is "
              f"telemetry only. Use --source databento (or ib) and n≥30/condition before believing this.{RS}")
    pf = "∞" if agg["profit_factor"] is None else f"{agg['profit_factor']:.2f}"
    print(f"  Trades: {agg['n_trades']}   Win rate: {agg['win_rate']:.0%}   PF: {pf}")
    print(f"  Net P&L (after costs): {G if agg['net_total_usd']>=0 else R}${agg['net_total_usd']:.2f}{RS}"
          f"   Total R: {agg['total_r']:+.2f}   Avg R: {agg['avg_r']:+.3f}")
    print(f"  Drawdown: realized ${agg['run_trailing_dd_usd']:.2f}  |  "
          f"intraday(MAE) ${agg['intraday_dd_usd']:.2f}  →  binding ${agg['binding_dd_usd']:.2f} "
          f"({agg['dd_model']})")
    print(f"  Best/Worst day: ${agg['best_day_usd']:.2f} / ${agg['worst_day_usd']:.2f}")
    if agg.get("by_hour"):
        hrs = "  ".join(f"{h:02d}:00 {d['win_rate']:.0%}({d['n']})" for h, d in agg["by_hour"].items())
        print(f"  Win rate by ET hour: {hrs}")
    if agg.get("by_confluence"):
        conf = "  ".join(f"{c}:{d['avg_r']:+.2f}R(n{d['n']})" for c, d in agg["by_confluence"].items())
        print(f"  Avg R by confluence (0–3): {conf}")
    cs = agg.get("cvd_split", {})
    if cs:
        c_, u_, k_ = cs.get("confirmed", {}), cs.get("unconfirmed", {}), cs.get("unknown", {})
        print(f"  CVD: confirmed {c_.get('win_rate',0):.0%}(n{c_.get('n',0)})  "
              f"unconfirmed {u_.get('win_rate',0):.0%}(n{u_.get('n',0)})  unknown n{k_.get('n',0)}")
    if agg.get("net_1lot_usd") is not None and agg.get("net_dynamic_usd") != agg.get("net_1lot_usd"):
        print(f"  Sizing: static 1-lot ${agg['net_1lot_usd']:.2f}  vs  "
              f"dynamic scale-in ${agg['net_dynamic_usd']:.2f}")
    for sw in agg.get("structural_warnings", []):
        print(f"  {R}{BD}⚠ STRUCTURAL: {sw}{RS}")
    print(f"\n  {BD}Prop check ({futures_params()['prop']['firm']}):{RS}")
    print(f"    Profit target: ${agg['prop_profit_target_usd']:.0f}  "
          f"(progress {agg['prop_progress_pct']:.0f}%)")
    print(f"    Trailing DD limit: ${agg['prop_trailing_dd_usd']:.0f}")
    if agg["prop_breaches"]:
        for b in agg["prop_breaches"]:
            print(f"    {R}✗ {b}{RS}")
    else:
        print(f"    {G}✓ no hard-rule breaches{RS}")
    for w in agg.get("warnings", []):
        print(f"    {Y}⚠ {w} (soft — only blocks at payout){RS}")
    print(f"\n  {BD}Verdict: {vc}{v}{RS}")
    print(f"\n  Per session:")
    for s in sessions:
        print(f"    {s['day']}  bias {s['bias']:<7}  trades {s['n_trades']:>2}  "
              f"net ${s['net_usd']:+8.2f}  maxDD ${s['max_drawdown_usd']:.2f}")
    print(f"{BD}{'═'*64}{RS}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Nightly ES/NQ replay (backtest like live)")
    ap.add_argument("--instrument", default="MES", choices=["MES", "MNQ"])
    ap.add_argument("--source", default="yf", choices=["yf", "ib", "databento"],
                    help="yf=yfinance (synthetic vol), ib=IB Gateway bars, databento=real CME 1-min "
                         "(GLBX.MDP3, trustworthy volume; needs DATABENTO_API_KEY)")
    ap.add_argument("--day", default=None, help="Single ET day YYYY-MM-DD (else all available)")
    ap.add_argument("--lookback", default="5d", help="yfinance lookback window (1m caps ~7d)")
    ap.add_argument("--bias", default="auto", choices=["auto", "long", "short", "neutral"])
    ap.add_argument("--orb-size", default="safe", choices=["safe", "big"])
    ap.add_argument("--setup", default=None, choices=["orb", "micro", "vwap_mr", "all"],
                    help="trade one setup in isolation (default = orb+micro, matches monitor)")
    ap.add_argument("--regime-gate", action="store_true",
                    help="only fire each setup in its favorable regime (the router)")
    ap.add_argument("--cvd-gate", action="store_true",
                    help="block entries whose CVD proxy doesn't confirm (Increment 4)")
    ap.add_argument("--scale-in", action="store_true",
                    help="add contracts into confirmed winners; loser stays 1 lot (Increment 4)")
    ap.add_argument("--no-log", action="store_true", help="Don't write the JSON report")
    args = ap.parse_args()

    setmap = {"orb": {"orb"}, "micro": {"micro"}, "vwap_mr": {"vwap_mr"},
              "all": {"orb", "micro", "vwap_mr"}}
    setups = setmap.get(args.setup)        # None => simulate default {orb, micro}

    print(f"Loading {args.instrument} 1-min history via {args.source}...", end=" ", flush=True)
    # Always load the full window so the prior session is present for prior-close/bias
    # context; --day only SELECTS which session(s) to replay.
    df = bf.load_history(args.instrument, source=args.source, day=None, lookback=args.lookback)
    if df is None or len(df) == 0:
        print(f"\n  No data returned. (yfinance 1m only covers ~7 days; try --source ib.)")
        sys.exit(1)
    print(f"done. {len(df)} bars.")

    all_days = bf.session_days(df)
    days = [args.day] if args.day else all_days
    sessions = []
    prior_close = None
    prior_ranges: list[float] = []
    prior_day_df = None                   # for the prior-day volume profile (confluence)
    for day in all_days:                  # walk all for prior_close continuity
        day_df = df[df.index.tz_convert(bf.ET).strftime("%Y-%m-%d") == day]
        if len(day_df) < 3:
            prior_close = float(day_df["Close"].iloc[-1]) if len(day_df) else prior_close
            continue
        day_range = float(day_df["High"].max() - day_df["Low"].min())
        if day in days:
            bias_dir, key_levels = _day_bias(day_df, day, prior_close, args.instrument, args.bias)
            regime = None
            if args.regime_gate:
                adr_used = regime_mod.adr_used_from_ranges(day_range, prior_ranges)
                regime = regime_mod.classify_session(day_df, vix=None, adr_used_pct=adr_used)
            prior_profile = vp.compute_profile(prior_day_df) if prior_day_df is not None else None
            s = simulate_session(day_df, day, bias_dir, key_levels, args.instrument,
                                 args.orb_size, setups=setups, regime=regime,
                                 cvd_gate=args.cvd_gate, scale_in=args.scale_in,
                                 prior_profile=prior_profile)
            if regime:
                s["regime"] = regime["trend_state"]
            sessions.append(s)
        prior_close = float(day_df["Close"].iloc[-1])
        prior_ranges.append(day_range)
        prior_day_df = day_df

    if not sessions:
        print("  No complete sessions to replay.")
        sys.exit(1)

    agg = _aggregate(sessions, args.instrument)
    agg["untrusted"] = (args.source not in ("ib", "databento"))   # VP/CVD need real volume (IB or Databento)
    _print_report(agg, sessions)

    if not args.no_log:
        tag = args.day or f"{days[0]}_to_{days[-1]}"
        out = REPORT_DIR / f"replay_report_{args.instrument}_{tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "instrument": args.instrument, "source": args.source,
            "bias_mode": args.bias, "orb_size": args.orb_size,
            "summary": agg, "sessions": sessions,
        }
        out.write_text(json.dumps(report, indent=2))
        print(f"  Report → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
