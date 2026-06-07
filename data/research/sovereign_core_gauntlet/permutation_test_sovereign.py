"""
Stage 1 — Costed permutation test for the Sovereign Core (THE KILL-GATE).

QUESTION
--------
Does the Sovereign Core's entry *timing + direction* beat random entries fired at the
SAME frequency, through the SAME costed mark-to-market, on clean frozen prices?

This is the analogue of scripts/permutation_test_forex.py. If the real signal can't
beat random entries (p >= 0.10), the system has no demonstrated edge and the gauntlet
STOPS here — no walk-forward, no holdout.

METHOD (faithful to the "as-is" serving path — the five gates the README says BLOCK)
-----------------------------------------------------------------------------------
  1. Router/FLAT      — router.classify(); Hurst dead-zone + SPY bear filter
  2. Specialist/NEUTRAL — specialists[regime].predict()
  3. Risk/ATR         — risk_engine.compute() ATR safety gate
  4. Risk/EV          — risk_engine.compute() expected-value gate
  5. Hard constraints — one position per symbol at a time (max-positions analogue)

  REAL: run the trained pipeline once over the in-sample window per symbol. On a signal,
        enter at the NEXT bar open (T+1 anchor), exit on ATR stop / target / time barrier.
        Mark each trade to market daily, apply round-trip costs, compute per-symbol
        annualized Sharpe, then a sqrt(n)-weighted portfolio Sharpe (matches forex).

  NULL (N perms): per symbol, place the SAME number of entries at random bars with random
        sign, reuse the real hold durations, mark to market with the SAME costs, aggregate
        identically. p_value = P(null_portfolio_sharpe >= real_portfolio_sharpe).

REPRODUCIBILITY
---------------
  - Reads ONLY data/cache/equity/*.parquet (frozen by freeze_sovereign_dataset.py).
  - yfinance is stubbed to serve frozen, point-in-time SPY to the router bear filter,
    so a single REAL pass makes ZERO live network calls.
  - Models trained on IN-SAMPLE ONLY (default 2015-2022). Holdout never touched here.

Usage:
  python3 scripts/permutation_test_sovereign.py --perms 1000 --seed 7
  python3 scripts/permutation_test_sovereign.py --perms 200            # quick smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.ERROR)
for _lib in ("yfinance", "urllib3", "requests", "peewee", "sovereign", "layer1", "contracts"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

EQUITY_DIR = ROOT / "data" / "cache" / "equity"
OUT_PATH = ROOT / "data" / "research" / "permutation_test_sovereign.json"

IN_SAMPLE_END = "2022-12-31"   # holdout (2023-2024) is never touched in this stage
IN_SAMPLE_START = "2015-01-01"
LOOKBACK = 90                  # bars required before first record (matches _build_records_from_df)
MAX_HOLD = 20                  # time-barrier exit (trading days)
TRADING_DAYS = 252

# Round-trip cost in fraction of notional: 2x slippage (0.1%/side) + 2x commission
# (0.05%/side) = 0.30% — the SovereignBacktest engine's own cost assumptions, round-trip.
COST_ROUNDTRIP = 2 * 0.001 + 2 * 0.0005


# ─── frozen, point-in-time yfinance stub (neutralizes the router bear filter) ──────── #

_FROZEN: dict[str, pd.DataFrame] = {}
_SIM_TS: pd.Timestamp | None = None


class _FrozenTicker:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def history(self, period: str = "60d", auto_adjust: bool = True, **_):
        df = _FROZEN.get(self.symbol)
        if df is None or _SIM_TS is None:
            return pd.DataFrame()
        # Point-in-time: only bars up to and including the current sim date.
        window = df.loc[:_SIM_TS]
        n = 60
        if isinstance(period, str) and period.endswith("d"):
            try:
                n = int(period[:-1])
            except ValueError:
                n = 60
        window = window.tail(n)
        # Router expects title-case OHLC columns from yfinance.
        return window.rename(columns={"close": "Close", "open": "Open",
                                      "high": "High", "low": "Low", "volume": "Volume"})


def _install_yf_stub():
    stub = types.ModuleType("yfinance")
    stub.Ticker = _FrozenTicker  # type: ignore[attr-defined]

    def _download(symbol, *a, **k):  # pragma: no cover - defensive
        return _FrozenTicker(symbol if isinstance(symbol, str) else symbol[0]).history()
    stub.download = _download  # type: ignore[attr-defined]
    sys.modules["yfinance"] = stub


# ─── data / features ──────────────────────────────────────────────────────────────── #

def _load_frozen(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(EQUITY_DIR / f"{symbol}.parquet")
    return df.sort_index()


def _atr14(df: pd.DataFrame) -> np.ndarray:
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    prev = np.empty_like(close)
    prev[0] = close[0]
    prev[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr[13] = tr[:14].mean()
        for i in range(14, len(tr)):
            atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    return atr


# ─── annualized Sharpe from a daily mark-to-market return series ───────────────────── #

def _annualized_sharpe(daily_returns: np.ndarray) -> float:
    r = daily_returns[~np.isnan(daily_returns)]
    if r.size < 2 or r.std(ddof=1) < 1e-12:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _portfolio_sharpe(per_symbol: dict[str, tuple[float, int]]) -> float:
    """sqrt(n)-weighted mean of per-symbol Sharpes (matches holdout_validation_v014)."""
    items = [(s, n) for (s, n) in per_symbol.values() if n > 0 and not np.isnan(s)]
    if not items:
        return 0.0
    w = [np.sqrt(n) for _, n in items]
    return float(sum(s * wi for (s, _), wi in zip(items, w)) / sum(w))


def _mtm_daily_returns(closes: np.ndarray, trades: list[dict], n_bars: int) -> np.ndarray:
    """
    Build a per-bar mark-to-market return series from a list of trades.

    Each trade: {entry_idx, exit_idx, direction}. Entry/exit indices index into `closes`
    (entry at entry_idx OPEN already handled by caller via entry price); here we MTM on
    close-to-close while in the position and book round-trip cost on the entry bar.
    """
    daily = np.zeros(n_bars)
    held = np.zeros(n_bars, dtype=bool)
    for t in trades:
        ei, xi, d = t["entry_idx"], t["exit_idx"], t["direction"]
        # close-to-close MTM from entry bar to exit bar
        for b in range(ei + 1, xi + 1):
            if b < n_bars and closes[b - 1] > 0:
                daily[b] += d * (closes[b] - closes[b - 1]) / closes[b - 1]
                held[b] = True
        # book round-trip cost on the entry bar
        if ei < n_bars:
            daily[ei] -= COST_ROUNDTRIP
            held[ei] = True
    # Only days with exposure count toward the Sharpe (matches a per-trade strategy).
    out = np.where(held, daily, np.nan)
    return out


# ─── REAL serving pass ─────────────────────────────────────────────────────────────── #

def _real_trades_for_symbol(orch, symbol: str, df: pd.DataFrame, records: list,
                            signal_mode: bool = True) -> list[dict]:
    """
    Walk in-sample bars; emit actual trades.

    signal_mode=False ("live"): faithful five-gate path including the EV/Kelly sizing
        throttle. NOTE: run cold this emits ZERO trades — the Hoeffding lower-bound on a
        cold n_trades=20 makes every signal negative-EV (documented finding).
    signal_mode=True ("signal"): tests whether the MODELS (router regime + specialist
        direction) have edge, independent of the cold-start sizing throttle. Keeps the
        ATR volatility gate (a signal-quality filter); drops only EV/Kelly sizing.
        Stop/target use the system's own ATR multiples from config.
    """
    global _SIM_TS
    from contracts.types import Direction
    from config.loader import params as _params

    atr_stop_mult = float(_params.get("risk", {}).get("atr_stop_multiplier", 1.5))
    atr_tp_mult = float(_params.get("risk", {}).get("atr_target_multiplier", 3.0))
    atr_gate = _params.get("atr_gate", {})

    closes = df["close"].to_numpy(float)
    opens = df["open"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    atr = _atr14(df)
    index = df.index

    # records align to bars from LOOKBACK onward (built by _build_records_from_df)
    rec_by_idx = {}
    rec_list_start = LOOKBACK
    for k, rec in enumerate(records):
        rec_by_idx[rec_list_start + k] = rec

    trades: list[dict] = []
    open_pos = None  # dict or None — one position per symbol (hard-constraint analogue)
    n = len(df)

    for i in range(LOOKBACK, n - 1):  # need i+1 for next-bar-open entry
        _SIM_TS = index[i]

        # manage open position first
        if open_pos is not None:
            exit_idx, exit_price = _maybe_exit(open_pos, i, highs, lows, closes)
            if exit_idx is not None:
                open_pos["exit_idx"] = exit_idx
                open_pos["exit_price"] = exit_price
                trades.append(open_pos)
                open_pos = None
            else:
                continue  # still holding; no new entry while in a position

        rec = rec_by_idx.get(i)
        if rec is None:
            continue

        # 1. Router
        router_out = orch.router.classify(rec)
        if router_out.regime == "FLAT" or router_out.specialist_to_run is None:
            continue

        # 2. Specialist
        specialist = orch.specialists.get(router_out.specialist_to_run)
        if specialist is None:
            continue
        bias = specialist.predict(rec)
        if bias.direction == Direction.NEUTRAL:
            continue

        # 3. ATR volatility gate (applies in both modes — it's a signal-quality filter)
        entry_ref = closes[i]
        atr_i = atr[i] if not np.isnan(atr[i]) else entry_ref * 0.02
        if (atr_i / entry_ref) * 100 > atr_gate.get(symbol, 4.0):
            continue

        d = 1 if bias.direction == Direction.LONG else -1

        if signal_mode:
            # Stop/target from the system's own ATR multiples (direction-aware)
            stop = entry_ref - d * atr_stop_mult * atr_i
            tp = entry_ref + d * atr_tp_mult * atr_i
        else:
            # 4. EV/Kelly sizing throttle (faithful "live" path)
            try:
                risk_out = orch.risk.compute(
                    bias=bias, router=router_out, account_equity=100_000.0,
                    atr=atr_i, entry_price=entry_ref,
                )
            except Exception:
                continue
            if risk_out.position_size <= 0 or not risk_out.ev_positive:
                continue
            stop = float(risk_out.stop_price)
            tp = float(risk_out.tp1_price)

        # ENTER at next-bar open (T+1 anchor)
        open_pos = {
            "entry_idx": i + 1,
            "entry_price": float(opens[i + 1]),
            "direction": d,
            "stop": stop,
            "tp": tp,
            "regime": router_out.regime,
        }

    # close any dangling position at the last bar
    if open_pos is not None:
        open_pos["exit_idx"] = n - 1
        open_pos["exit_price"] = float(closes[n - 1])
        trades.append(open_pos)

    _SIM_TS = None
    return trades


def _maybe_exit(pos, i, highs, lows, closes):
    """Check stop / target / time-barrier exit at bar i. Returns (exit_idx, exit_price) or (None, None)."""
    ei = pos["entry_idx"]
    if i <= ei:
        return None, None
    d = pos["direction"]
    stop, tp = pos["stop"], pos["tp"]
    hi, lo, cl = highs[i], lows[i], closes[i]
    if d == 1:
        if stop > 0 and lo <= stop:
            return i, stop
        if tp > 0 and hi >= tp:
            return i, tp
    else:
        if stop > 0 and hi >= stop:
            return i, stop
        if tp > 0 and lo <= tp:
            return i, tp
    if i - ei >= MAX_HOLD:
        return i, cl
    return None, None


# ─── NULL: random entries, same count, reuse hold durations ───────────────────────── #

def _null_symbol_sharpe(rng, closes: np.ndarray, real_trades: list[dict],
                        first_idx: int, last_idx: int) -> tuple[float, int]:
    n_trades = len(real_trades)
    if n_trades == 0:
        return 0.0, 0
    durations = [max(1, t["exit_idx"] - t["entry_idx"]) for t in real_trades]
    n_bars = len(closes)
    null_trades = []
    # sample non-overlapping-ish random entries (simple: independent, allow overlap —
    # MTM sums exposure, matching the real series construction)
    lo, hi = first_idx, max(first_idx + 1, last_idx)
    entries = rng.integers(lo, hi, size=n_trades)
    signs = rng.choice([-1, 1], size=n_trades)
    for e, dur, s in zip(entries, durations, signs):
        xi = min(e + dur, n_bars - 1)
        null_trades.append({"entry_idx": int(e), "exit_idx": int(xi), "direction": int(s)})
    daily = _mtm_daily_returns(closes, null_trades, n_bars)
    return _annualized_sharpe(daily), n_trades


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--mode", choices=["signal", "live"], default="signal",
                    help="signal=test model edge (skip cold-start EV throttle); "
                         "live=faithful five-gate path (emits zero trades cold)")
    args = ap.parse_args()
    signal_mode = args.mode == "signal"

    _install_yf_stub()

    manifest = json.loads((EQUITY_DIR / "manifest.json").read_text())
    symbols = args.symbols or list(manifest["symbols"].keys())

    # Freeze SPY for the bear-filter stub (must be present)
    for s in set(symbols) | {"SPY"}:
        _FROZEN[s] = _load_frozen(s)

    from train_core import _build_records_from_df
    from sovereign.orchestrator import SovereignOrchestrator

    # ── Build in-sample records per symbol, train models ──────────────────────── #
    insample = {}
    all_records = []
    for s in symbols:
        df = _FROZEN[s]
        df_is = df.loc[IN_SAMPLE_START:IN_SAMPLE_END]
        insample[s] = df_is
        recs = _build_records_from_df(s, df_is)
        all_records.extend(recs)

    print(f"Training on {len(all_records)} in-sample records ({IN_SAMPLE_START}..{IN_SAMPLE_END})")
    orch = SovereignOrchestrator(mode="paper")
    orch.train(all_records)

    # ── REAL pass ─────────────────────────────────────────────────────────────── #
    real_per_symbol: dict[str, tuple[float, int]] = {}
    real_trades_by_symbol: dict[str, list[dict]] = {}
    total_trades = 0
    for s in symbols:
        df_is = insample[s]
        recs = _build_records_from_df(s, df_is)
        trades = _real_trades_for_symbol(orch, s, df_is, recs, signal_mode=signal_mode)
        closes = df_is["close"].to_numpy(float)
        daily = _mtm_daily_returns(closes, trades, len(closes))
        sharpe = _annualized_sharpe(daily)
        real_per_symbol[s] = (sharpe, len(trades))
        real_trades_by_symbol[s] = trades
        total_trades += len(trades)
        print(f"  {s}: {len(trades)} trades | Sharpe {sharpe:+.3f}")

    real_sharpe = _portfolio_sharpe(real_per_symbol)
    print(f"\nREAL portfolio Sharpe (sqrt-n weighted): {real_sharpe:+.4f} | {total_trades} trades")

    if total_trades == 0:
        verdict = "NO_TRADES — system produced zero entries in-sample; cannot test edge."
        _write(args, symbols, real_sharpe, [], total_trades, None, verdict)
        print(verdict)
        return 0

    # ── NULL distribution ─────────────────────────────────────────────────────── #
    rng = np.random.default_rng(args.seed)
    null_sharpes = np.empty(args.perms)
    closes_by_symbol = {s: insample[s]["close"].to_numpy(float) for s in symbols}
    for p in range(args.perms):
        per_symbol = {}
        for s in symbols:
            closes = closes_by_symbol[s]
            per_symbol[s] = _null_symbol_sharpe(
                rng, closes, real_trades_by_symbol[s], LOOKBACK, len(closes) - 1
            )
        null_sharpes[p] = _portfolio_sharpe(per_symbol)
        if (p + 1) % max(1, args.perms // 10) == 0:
            print(f"  null {p + 1}/{args.perms}")

    p_value = float((np.sum(null_sharpes >= real_sharpe) + 1) / (args.perms + 1))

    verdict = (
        "EDGE_DEMONSTRATED (p<0.05) — proceed to walk-forward" if p_value < 0.05
        else "AMBIGUOUS (0.05<=p<0.10) — weak; treat as unproven"
        if p_value < 0.10 else
        "NO_EDGE (p>=0.10) — random entries match it; STOP. Gauntlet ends."
    )
    _write(args, symbols, real_sharpe, null_sharpes, total_trades, p_value, verdict,
           real_per_symbol)

    print(f"\np-value: {p_value:.4f}")
    print(f"null Sharpe: mean {null_sharpes.mean():+.3f} | 95th pct {np.percentile(null_sharpes, 95):+.3f}")
    print(f"VERDICT: {verdict}")
    return 0


def _write(args, symbols, real_sharpe, null_sharpes, total_trades, p_value, verdict,
           per_symbol=None):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((EQUITY_DIR / "manifest.json").read_text())
    ns = np.asarray(null_sharpes, dtype=float)
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "1_permutation_kill_gate",
        "mode": args.mode,
        "dataset_hash": manifest.get("dataset_hash"),
        "data_source": "yfinance (frozen)",
        "in_sample": [IN_SAMPLE_START, IN_SAMPLE_END],
        "symbols": symbols,
        "n_perms": int(args.perms),
        "seed": int(args.seed),
        "cost_roundtrip": COST_ROUNDTRIP,
        "max_hold": MAX_HOLD,
        "real_sharpe": round(float(real_sharpe), 4),
        "total_trades": int(total_trades),
        "per_symbol": {k: {"sharpe": round(v[0], 4), "trades": v[1]}
                       for k, v in (per_symbol or {}).items()},
        "p_value": p_value,
        "null_distribution": {
            "mean": round(float(ns.mean()), 4) if ns.size else None,
            "std": round(float(ns.std()), 4) if ns.size else None,
            "pct95": round(float(np.percentile(ns, 95)), 4) if ns.size else None,
            "max": round(float(ns.max()), 4) if ns.size else None,
        },
        "verdict": verdict,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
