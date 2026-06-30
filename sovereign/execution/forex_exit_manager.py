"""sovereign/execution/forex_exit_manager.py — L2 daily exit manager (SHADOW MODE).

The backtest captures most of its Sharpe in the exit layer (trailing / time / reversal / cb_refresh),
which the live leg currently does NOT manage: `forex_live_scan` only OPENS trades. This manager closes
that gap by stepping the SAME deterministic `decide_exit` (Step 1) once per trading day over every open
broker trade — so live == backtest by construction, not by re-implementation.

    ┌─ SHADOW_MODE = True  → logs the would-act decision to a JSONL shadow log; NO broker writes.
    └─ SHADOW_MODE = False → calls bridge.set_stop / bridge.close_trade (LIVE).

Flipping the single boolean below is the ENTIRE "go live" action — gated on Step 5 approval AND on the
Step 2 set_stop round-trip actually passing on a market-open practice account (still outstanding: the
Step 2 test SKIPPED on MARKET_HALTED). Until then this file must run shadow-only.

Trailing fidelity = EXIT_MACHINE_DESIGN.md §2 Option C: each day, amend a PLAIN broker stop to the
backtest's daily-close ATR-trail price (best − mult·ATR·best). `stop_price` fed to `decide_exit` is the
FIXED initial ATR stop and is NEVER moved (backtest parity); the ratcheting trail is tracked separately
as `last_stop` and only ever drives the would-amend action — it never feeds back into the exit decision.

Parity is proven offline by tests/test_forex_exit_manager.py (replay-match vs fast_backtester's ledger).
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol

# Make the repo root importable when run by absolute path under launchd (sys.path[0] = execution/).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sovereign.forex.exit_machine import (
    BarContext, ExitConfig, ExitDecision, PositionState, decide_exit,
)

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════════════════════
#  THE TOGGLE — flipping this to False is the entire "go live" action (Step 5, param_change-gated).
# ════════════════════════════════════════════════════════════════════════════════════════════════
SHADOW_MODE = True

# ── Canonical v015 config — mirrored from sovereign/forex/forex_backtester.py ────────────────────
STOP_ATR_MULT = 2.0
DEFAULT_TRAILING_ATR_MULT = 1.25
STRICT_MODE = False           # canonical → DONCHIAN never fires; donchian_exit_low stays NaN
ENABLE_CB_REFRESH = True      # = (not strict_mode) in the backtester
HOLD_DAYS = 60                # PAIR_HOLD_OVERRIDES is empty in v015 → every pair holds 60d
STOP_PCT = 0.04               # fallback only (unused while stop_atr_mult > 0)

# Per-pair trailing overrides, keyed by OANDA instrument (config uses =X form: GBPUSD=X 2.0 etc.)
PAIR_TRAILING_OVERRIDES: dict[str, float] = {
    "GBP_USD": 2.0,
    "AUD_USD": 1.0,
    "EUR_USD": 1.25,
    "AUD_NZD": 1.25,
}

# ── File paths ───────────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
STATE_PATH = ROOT / "data" / "exec" / "exit_manager_state.json"
SHADOW_LOG_PATH = ROOT / "data" / "exec" / "exit_manager_shadow.jsonl"
STATE_VERSION = 1


def cfg_for_pair(pair: str) -> ExitConfig:
    """ExitConfig for an OANDA instrument under the canonical v015 config."""
    return ExitConfig(
        stop_atr_mult=STOP_ATR_MULT,
        trailing_atr_mult=PAIR_TRAILING_OVERRIDES.get(pair, DEFAULT_TRAILING_ATR_MULT),
        strict_mode=STRICT_MODE,
        enable_cb_refresh=ENABLE_CB_REFRESH,
    )


# ── Action vocabulary ──────────────────────────────────────────────────────────────────────────
class Action(str, Enum):
    HOLD = "HOLD"              # no decision, trail did not tighten — do nothing
    AMEND_STOP = "AMEND_STOP"  # HOLD but the ATR-trail tightened → would set_stop(trail)
    CLOSE = "CLOSE"            # decide_exit fired a non-HOLD exit → would close_trade


# ── Per-trade persisted state ────────────────────────────────────────────────────────────────────
@dataclass
class TradeState:
    """Everything the manager must remember about one open trade between daily runs.

    `stop_price` is the FIXED initial ATR stop fed to decide_exit (never moved — backtest parity).
    `last_stop` is the most recent stop the manager WOULD have set on the broker (the ratcheting
    Option-C trail); it starts equal to stop_price and only the would-amend action moves it.
    """
    trade_id: str
    pair: str
    direction: int          # +1 long / -1 short
    entry_price: float
    entry_date: str         # ISO timestamp from the broker
    stop_price: float       # initial ATR stop — decide_exit reads this, manager NEVER moves it
    last_stop: float        # ratcheting trail stop the manager would hold on the broker
    best_price: float
    worst_price: float
    hold_count: int
    hold_limit: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TradeState":
        return cls(**d)


@dataclass(frozen=True)
class MarketBar:
    """The latest COMPLETED daily bar's inputs for one pair (same shape the backtester steps over)."""
    pair: str
    date: str
    close: float
    atr_pct: float
    signal: int            # today's carry signal (-1/0/+1)
    hold_today: int        # the signal's hold_days for this bar (canonical = 60)


@dataclass(frozen=True)
class StepResult:
    decision: ExitDecision     # what decide_exit returned (HOLD or an exit reason)
    action: Action             # what the manager WOULD do on the broker
    amend_to: Optional[float]  # the trail price for an AMEND_STOP (else None)
    trail_price: float         # the computed daily-close ATR trail (informational, always present)
    reentry_signal: int        # re-entry signal from decide_exit (handled by next scan, not here)
    new_state: TradeState      # state advanced for this bar (best/worst/hold_count, and last_stop)


class StateError(RuntimeError):
    """Raised when the state file is missing or corrupt — the manager HALTS, never silently re-inits."""


# ── Pure decision core (broker-free; the parity-bearing function the test drives) ─────────────────

def init_trade_state(
    trade_id: str,
    pair: str,
    direction: int,
    entry_price: float,
    entry_atr_pct: float,
    hold_days_at_entry: int,
    entry_date: str,
    cfg: ExitConfig,
) -> TradeState:
    """Reconstruct the backtester's entry state for a freshly-seen open trade.

    Mirrors fast_backtester._simulate_forex_core's entry block EXACTLY:
        entry_atr = max(atr%@entry-signal-bar, 1e-6)
        stop_dist = entry * stop_atr_mult * entry_atr   (or entry * STOP_PCT if mult <= 0)
        stop      = entry − dist (long) / entry + dist (short)
        best = worst = entry; hold_count = 0; hold_limit = max(hold_days_at_entry, 1)
    `entry_atr_pct` is the ATR% as-of the entry SIGNAL bar (the bar whose close preceded the
    next-open fill) — the same value the backtester used when it sized the stop.
    """
    entry_atr = max(entry_atr_pct, 1e-6)
    if cfg.stop_atr_mult > 0:
        stop_dist = entry_price * cfg.stop_atr_mult * entry_atr
    else:
        stop_dist = entry_price * STOP_PCT
    stop_price = entry_price - stop_dist if direction == 1 else entry_price + stop_dist
    return TradeState(
        trade_id=trade_id, pair=pair, direction=int(direction), entry_price=float(entry_price),
        entry_date=entry_date, stop_price=float(stop_price), last_stop=float(stop_price),
        best_price=float(entry_price), worst_price=float(entry_price),
        hold_count=0, hold_limit=max(int(hold_days_at_entry), 1),
    )


def step_trade(state: TradeState, bar: MarketBar, cfg: ExitConfig) -> StepResult:
    """One daily step for one trade. Calls the shared decide_exit and maps the result to a broker
    action — WITHOUT moving state.stop_price (parity) and without touching the broker (that is the
    caller's job, and only when not in shadow)."""
    ps = PositionState(
        direction=state.direction, stop_price=state.stop_price,
        best_price=state.best_price, worst_price=state.worst_price,
        hold_count=state.hold_count, hold_limit=state.hold_limit,
    )
    bc = BarContext(
        close=bar.close, atr_pct=bar.atr_pct, signal=int(bar.signal),
        hold_today=int(bar.hold_today), donchian_exit_low=math.nan,  # strict_mode=False → unused
    )
    res = decide_exit(ps, bc, cfg)

    # The daily-close ATR trail (Option C) computed from the bar's updated best/worst.
    atr = max(bar.atr_pct, 1e-6)
    if state.direction == 1:
        trail_price = res.state.best_price - (cfg.trailing_atr_mult * atr * res.state.best_price)
    else:
        trail_price = res.state.worst_price + (cfg.trailing_atr_mult * atr * res.state.worst_price)

    # Advance state: best/worst/hold_count from decide_exit; stop_price UNCHANGED (parity).
    new_state = replace(
        state,
        best_price=res.state.best_price, worst_price=res.state.worst_price,
        hold_count=res.state.hold_count,
    )

    if res.decision != ExitDecision.HOLD:
        action, amend_to = Action.CLOSE, None
    else:
        # Would-amend only if the trail TIGHTENS the stop currently held on the broker (last_stop).
        if cfg.trailing_atr_mult > 0 and (
            (state.direction == 1 and trail_price > state.last_stop)
            or (state.direction == -1 and trail_price < state.last_stop)
        ):
            action, amend_to = Action.AMEND_STOP, float(trail_price)
            new_state = replace(new_state, last_stop=float(trail_price))  # ratchet the would-hold stop
        else:
            action, amend_to = Action.HOLD, None

    return StepResult(
        decision=res.decision, action=action, amend_to=amend_to,
        trail_price=float(trail_price), reentry_signal=res.reentry_signal, new_state=new_state,
    )


# ── State persistence ─────────────────────────────────────────────────────────────────────────────

def init_state_file(path: Path = STATE_PATH) -> dict:
    """Explicitly create an empty state file (the ONLY sanctioned way to (re)initialize). Refuses to
    clobber an existing file — that would be the silent re-init the constraints forbid."""
    if path.exists():
        raise StateError(f"refusing to overwrite existing state file {path} — inspect it manually")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"version": STATE_VERSION, "updated_at": datetime.now(timezone.utc).isoformat(), "trades": {}}
    _atomic_write(path, state)
    logger.info("[exit_manager] initialized empty state file %s", path)
    return state


def load_state(path: Path = STATE_PATH) -> dict:
    """Load state. HALTS LOUDLY (StateError) if missing or corrupt — never silently re-initializes."""
    if not path.exists():
        raise StateError(
            f"state file {path} is MISSING. The manager refuses to silently re-initialize — a missing "
            f"state file means lost best_price/hold_count for live trades. Investigate, then run --init "
            f"only if a fresh start is truly intended."
        )
    try:
        with open(path) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise StateError(f"state file {path} is CORRUPT ({exc}). Halting — human intervention required.")
    if not isinstance(state, dict) or "trades" not in state or not isinstance(state["trades"], dict):
        raise StateError(f"state file {path} has an unexpected shape. Halting — human intervention required.")
    return state


def _atomic_write(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["version"] = STATE_VERSION
    _atomic_write(path, state)


# ── Market data provider interface ────────────────────────────────────────────────────────────────

class MarketProvider(Protocol):
    def market_bar(self, pair: str) -> MarketBar: ...
    def entry_atr_pct(self, pair: str, entry_date: str) -> float: ...


# ── Reconciliation ─────────────────────────────────────────────────────────────────────────────────

def reconcile(state: dict, open_trades: list[dict], provider: MarketProvider) -> dict:
    """Drop state for trades no longer open; initialize state for newly-seen open trades.

    Newly-seen = a broker trade with no persisted state. We reconstruct its entry state from broker
    fields (entry price, side, open time) + the ATR% as-of its entry date from the provider, with the
    canonical hold_limit (60d — v015 has no per-pair hold overrides). This per-TRADE init is expected
    reconciliation; it is NOT the forbidden whole-FILE silent re-init (that only applies to load_state).
    """
    open_by_id = {str(t.get("id") or t.get("tradeID")): t for t in open_trades}
    trades = state.setdefault("trades", {})

    for tid in [t for t in trades if t not in open_by_id]:
        logger.info("[exit_manager] reconcile: trade %s no longer open — dropping its state", tid)
        trades.pop(tid)

    for tid, bt in open_by_id.items():
        if tid in trades:
            continue
        pair = bt["instrument"]
        units = float(bt.get("currentUnits", 0))
        direction = 1 if units > 0 else -1
        entry_price = float(bt["price"])
        entry_date = bt.get("openTime", "")
        atr_at_entry = provider.entry_atr_pct(pair, entry_date)
        ts = init_trade_state(
            trade_id=tid, pair=pair, direction=direction, entry_price=entry_price,
            entry_atr_pct=atr_at_entry, hold_days_at_entry=HOLD_DAYS,
            entry_date=entry_date, cfg=cfg_for_pair(pair),
        )
        trades[tid] = ts.to_dict()
        logger.info("[exit_manager] reconcile: new trade %s (%s %s) — initialized state, stop=%.5f",
                    tid, "LONG" if direction == 1 else "SHORT", pair, ts.stop_price)
    return state


# ── Shadow log ──────────────────────────────────────────────────────────────────────────────────────

def _write_shadow_log(entry: dict, path: Path = SHADOW_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Daily runner ──────────────────────────────────────────────────────────────────────────────────

def run_daily(
    bridge,
    provider: MarketProvider,
    *,
    shadow: bool = SHADOW_MODE,
    state_path: Path = STATE_PATH,
    shadow_log_path: Path = SHADOW_LOG_PATH,
) -> list[dict]:
    """One daily management pass over all open broker trades. Returns the per-trade log entries.

    SHADOW (default): every decision is logged; bridge.set_stop / bridge.close_trade are NEVER called.
    LIVE (shadow=False): the SAME decisions are executed on the broker. The toggle is the only thing
    standing between the two paths — there is no other behavioral difference.
    """
    state = load_state(state_path)                  # halts loudly if missing/corrupt
    open_trades = bridge.get_open_trades()
    state = reconcile(state, open_trades, provider)

    run_ts = datetime.now(timezone.utc).isoformat()
    log_entries: list[dict] = []

    for bt in open_trades:
        tid = str(bt.get("id") or bt.get("tradeID"))
        ts = TradeState.from_dict(state["trades"][tid])
        cfg = cfg_for_pair(ts.pair)

        try:
            bar = provider.market_bar(ts.pair)
        except Exception as exc:
            entry = {
                "run_ts": run_ts, "mode": "SHADOW" if shadow else "LIVE", "trade_id": tid,
                "pair": ts.pair, "action": "SKIP", "reason": f"market_data_unavailable: {exc}",
            }
            log_entries.append(entry)
            _write_shadow_log(entry, shadow_log_path)
            logger.warning("[exit_manager] %s: market data unavailable — skipped (%s)", tid, exc)
            continue

        res = step_trade(ts, bar, cfg)
        state["trades"][tid] = res.new_state.to_dict()

        entry = {
            "run_ts": run_ts,
            "mode": "SHADOW" if shadow else "LIVE",
            "trade_id": tid,
            "pair": ts.pair,
            "direction": "LONG" if ts.direction == 1 else "SHORT",
            "bar_date": bar.date,
            "close": bar.close,
            "atr_pct": round(bar.atr_pct, 6),
            "signal": bar.signal,
            "hold_count": res.new_state.hold_count,
            "hold_limit": ts.hold_limit,
            "best_price": round(res.new_state.best_price, 5),
            "worst_price": round(res.new_state.worst_price, 5),
            "initial_stop": round(ts.stop_price, 5),
            "decision": res.decision.name,
            "action": res.action.value,
            "would_amend_stop_to": round(res.amend_to, 5) if res.amend_to is not None else None,
            "trail_price": round(res.trail_price, 5),
            "reentry_signal": res.reentry_signal,
        }
        log_entries.append(entry)
        _write_shadow_log(entry, shadow_log_path)

        if not shadow:
            # ── LIVE PATH — the ONLY place broker writes happen. Unreachable while SHADOW_MODE=True. ──
            if res.action == Action.CLOSE:
                bridge.close_trade(tid)
            elif res.action == Action.AMEND_STOP and res.amend_to is not None:
                bridge.set_stop(tid, res.amend_to)

    save_state(state, state_path)
    return log_entries


# ── Live market provider (OANDA candles for price/ATR; signal engine for the carry signal) ─────────

class LiveMarketProvider:
    """Builds MarketBars from real OANDA daily candles (close, ATR%) and the live carry signal.

    Close & ATR% come from broker candles (never revise, no yfinance drift). The signal is the
    canonical macro carry signal from ForexSignalEngine. Any failure raises — run_daily turns that
    into a logged SKIP rather than guessing a decision.
    """

    def __init__(self, bridge, lookback_days: int = 120):
        self._bridge = bridge
        self._lookback = lookback_days

    def _candles(self, pair: str, end_date: Optional[str] = None):
        import pandas as pd
        from datetime import timedelta
        end = datetime.now(timezone.utc) if not end_date else datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        start = end - timedelta(days=self._lookback)
        # bridge.get_historical_candles accepts EUR_USD / EURUSD / EURUSD=X
        return self._bridge.get_historical_candles(
            pair, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), granularity="D"
        )

    def entry_atr_pct(self, pair: str, entry_date: str) -> float:
        from sovereign.forex.signal_engine import ForexSignalEngine
        df = self._candles(pair, entry_date)
        atr = ForexSignalEngine._compute_atr_pct(df["Close"], df)
        return float(atr.iloc[-1]) if atr is not None and len(atr) else 1e-6

    def market_bar(self, pair: str) -> MarketBar:
        from sovereign.forex.signal_engine import ForexSignalEngine, SignalConfig
        from sovereign.forex.data_fetcher import ForexDataFetcher
        from sovereign.forex.entry_engine import CBEventTrigger
        from sovereign.forex.pair_universe import PAIR_CONFIG, CB_TO_COUNTRY

        df = self._candles(pair)
        if df is None or len(df) < 20:
            raise RuntimeError(f"insufficient OANDA candles for {pair}")
        close = df["Close"]
        atr = ForexSignalEngine._compute_atr_pct(close, df)
        atr_pct = float(atr.iloc[-1]) if atr is not None and len(atr) else 1e-6

        # Canonical carry signal for the latest completed bar (engine built as the backtester does).
        ysym = pair.replace("_", "") + "=X"          # EUR_USD → EURUSD=X (PAIR_CONFIG key)
        pc = PAIR_CONFIG.get(ysym)
        if pc is None:
            raise RuntimeError(f"no PAIR_CONFIG for {pair}")
        engine = ForexSignalEngine(
            fetcher=ForexDataFetcher(), cb_trigger=CBEventTrigger(),
            config=SignalConfig(hold_days=HOLD_DAYS, signal_threshold=0.15, strict_mode=False),
        )
        frame = engine.build_signal_frame(
            prices=df,
            base_country=CB_TO_COUNTRY[pc.base_central_bank],
            quote_country=CB_TO_COUNTRY[pc.quote_central_bank],
            start=str(close.index[0].date()), end=str(close.index[-1].date()), pair=ysym,
        )
        last = frame.iloc[-1]
        return MarketBar(
            pair=pair, date=str(close.index[-1].date()), close=float(close.iloc[-1]),
            atr_pct=atr_pct, signal=int(round(float(last["signal"]))),
            hold_today=int(last.get("hold_days", HOLD_DAYS)),
        )


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    ap = argparse.ArgumentParser(description="L2 forex exit manager (shadow by default)")
    ap.add_argument("--init", action="store_true", help="create an empty state file (sanctioned re-init)")
    ap.add_argument("--run", action="store_true", help="run one daily shadow pass against the live broker")
    ap.add_argument("--status", action="store_true", help="print current persisted state")
    args = ap.parse_args()

    if args.init:
        init_state_file()
        return
    if args.status:
        print(json.dumps(load_state(), indent=2))
        return
    if args.run:
        if not SHADOW_MODE:
            raise SystemExit("SHADOW_MODE is False — live runs are gated on Step 5 (param_change). Aborting.")
        from sovereign.execution.oanda_bridge import OandaBridge
        bridge = OandaBridge()
        entries = run_daily(bridge, LiveMarketProvider(bridge), shadow=True)
        print(json.dumps(entries, indent=2))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
