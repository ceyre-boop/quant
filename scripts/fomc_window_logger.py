#!/usr/bin/env python3
"""FOMC-window execution logger (TICK-056 companion) — PURE OBSERVATION, NO ORDERS.

Over a configurable window around a timestamp (default 2:00pm ET 2026-07-29, ±N min),
samples the MT5 DEMO terminal at high frequency for:
  - bid / ask / spread per configured symbol
  - (if demo positions are open) fill price + realized slippage vs the intended price
    recorded by the bridge in data/execution/mt5_routed.jsonl

Writes data/execution/fomc_window_<date>.jsonl (one JSON object per sample tick).

It PLACES NO ORDERS. It reads the SAME MT5 connection the bridge uses (MT5Connector +
demo guard) so the measured spread/slippage is directly comparable to the backtest's
assumed per-trade cost. No silent mocking: if the MetaTrader5 package / terminal /
demo creds are absent, it fails loud with exact remediation.

Usage
  python scripts/fomc_window_logger.py                       # defaults (FOMC 2026-07-29)
  python scripts/fomc_window_logger.py --center "2026-07-29T14:00" --tz America/New_York \
      --window-min 15 --interval-sec 1.0 --symbols EURUSD,GBPUSD
  python scripts/fomc_window_logger.py --dry-run             # print window + exit, no connect
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

import yaml

# Run-as-script support: ensure the repo root (parent of scripts/) is importable so
# `sovereign.execution.mt5` resolves whether invoked as a module or a bare script.
_REPO_ROOT_FOR_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT_FOR_PATH not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_PATH)

from sovereign.execution.mt5.connector import Connector, ConnectorError, MT5Connector
from sovereign.execution.mt5.guard import assert_routable

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "mt5.yml"

DEFAULT_CENTER = "2026-07-29T14:00"      # FOMC statement, 2:00pm ET
DEFAULT_TZ = "America/New_York"
DEFAULT_WINDOW_MIN = 15
DEFAULT_INTERVAL_SEC = 1.0
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "GBPJPY", "USDJPY"]


class LoggerError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Window / config (pure, unit-testable)                                        #
# --------------------------------------------------------------------------- #

def parse_center(center: str, tz_name: str) -> datetime:
    """Parse a naive/aware ISO timestamp into a tz-aware datetime in tz_name."""
    if ZoneInfo is None:
        raise LoggerError("zoneinfo unavailable — need Python 3.9+ for tz handling")
    try:
        tz = ZoneInfo(tz_name)
    except Exception as e:
        raise LoggerError(f"unknown timezone '{tz_name}': {e}")
    dt = datetime.fromisoformat(center)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt


def compute_window(center: datetime, window_min: int) -> tuple[datetime, datetime]:
    delta = timedelta(minutes=window_min)
    return center - delta, center + delta


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------- #
# Routed-ledger index → intended prices for slippage (pure)                    #
# --------------------------------------------------------------------------- #

def load_intended_prices(routed_ledger: Path) -> dict:
    """Map (symbol, magic) -> most-recent intended request_price from the bridge's
    idempotency ledger. Used to compute realized slippage on open positions."""
    index: dict = {}
    if not routed_ledger.exists():
        return index
    for line in routed_ledger.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (rec.get("symbol"), rec.get("magic"))
        # keep latest by routed_at (lexical ISO sort is chronological)
        prev = index.get(key)
        if prev is None or str(rec.get("routed_at", "")) >= str(prev.get("routed_at", "")):
            index[key] = rec
    return index


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def position_slippage(position: Any, intended_price: Optional[float]) -> Optional[dict]:
    """Realized slippage of an open position's fill vs its intended price.

    Signed so that a POSITIVE value is adverse (worse fill) for both sides:
      BUY  filled above intended  → adverse  → positive
      SELL filled below intended  → adverse  → positive
    """
    if intended_price is None:
        return None
    fill = _attr(position, "price_open")
    if fill is None:
        return None
    ptype = _attr(position, "type")  # 0 == BUY, 1 == SELL (MT5)
    raw = fill - intended_price
    adverse = raw if ptype == 0 else -raw
    return {
        "intended_price": intended_price,
        "fill_price": fill,
        "slippage_price_raw": raw,
        "slippage_price_adverse": adverse,
    }


# --------------------------------------------------------------------------- #
# Sampling                                                                     #
# --------------------------------------------------------------------------- #

def build_sample(
    connector: Connector,
    symbols: list[str],
    intended_index: dict,
    *,
    now_iso: str,
) -> dict:
    """One observation tick across all symbols + open positions. No orders."""
    quotes = []
    for sym in symbols:
        try:
            tick = connector.symbol_tick(sym)
        except ConnectorError as e:
            quotes.append({"symbol": sym, "error": str(e)})
            continue
        if tick is None:
            quotes.append({"symbol": sym, "error": "no tick (market closed?)"})
            continue
        point = None
        try:
            info = connector.symbol_info(sym)
            point = _attr(info, "point")
        except Exception:
            point = None
        spread_price = tick.ask - tick.bid
        spread_points = (spread_price / point) if point else None
        quotes.append({
            "symbol": sym,
            "bid": tick.bid,
            "ask": tick.ask,
            "spread_price": spread_price,
            "spread_points": spread_points,
            "time_msc": tick.time_msc,
        })

    positions_out = []
    for pos in connector.positions_get():
        sym = _attr(pos, "symbol")
        magic = _attr(pos, "magic")
        intended_rec = intended_index.get((sym, magic))
        intended_price = _attr(intended_rec, "request_price") if intended_rec else None
        slip = position_slippage(pos, intended_price)
        positions_out.append({
            "ticket": _attr(pos, "ticket"),
            "symbol": sym,
            "magic": magic,
            "type": _attr(pos, "type"),
            "volume": _attr(pos, "volume"),
            "price_open": _attr(pos, "price_open"),
            "price_current": _attr(pos, "price_current"),
            "slippage": slip,
        })

    return {
        "sampled_at": now_iso,
        "quotes": quotes,
        "open_positions": positions_out,
    }


def run(
    connector: Connector,
    cfg: dict,
    *,
    center: datetime,
    window_min: int,
    interval_sec: float,
    symbols: list[str],
    out_path: Path,
    max_samples: Optional[int] = None,
    clock=time.time,
    sleep=time.sleep,
) -> int:
    """Connect, verify DEMO, and sample the window into out_path (jsonl)."""
    start, end = compute_window(center, window_min)
    print(f"FOMC window: {start.isoformat()} → {end.isoformat()} "
          f"(center {center.isoformat()}, ±{window_min}m, every {interval_sec}s)")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Output : {out_path}")

    connector.initialize()
    account = connector.account_info()
    # Same guard the bridge uses. The logger only observes, but we refuse to sample
    # anything that isn't the demo terminal — no accidental live-account observation.
    unlock_path = REPO_ROOT / cfg.get("paths", {}).get("live_unlock", "data/execution/mt5_LIVE_UNLOCK.json")
    mode_name = assert_routable(account, unlock_path=unlock_path)
    print(f"Account: login={_attr(account, 'login')} server={_attr(account, 'server')} [{mode_name}]")

    routed_ledger = REPO_ROOT / cfg.get("paths", {}).get("routed_ledger", "data/execution/mt5_routed.jsonl")
    intended_index = load_intended_prices(routed_ledger)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = start.timestamp()
    end_epoch = end.timestamp()
    now = clock()
    if now < start_epoch:
        wait = start_epoch - now
        print(f"Waiting {wait:.0f}s for window start…")
        sleep(wait)

    n = 0
    with out_path.open("a") as f:
        while clock() <= end_epoch:
            now_iso = datetime.fromtimestamp(clock(), tz=timezone.utc).isoformat()
            sample = build_sample(connector, symbols, intended_index, now_iso=now_iso)
            f.write(json.dumps(sample) + "\n")
            f.flush()
            n += 1
            if max_samples is not None and n >= max_samples:
                break
            sleep(interval_sec)

    print(f"Done. Wrote {n} samples → {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Entry                                                                        #
# --------------------------------------------------------------------------- #

def build_connector(cfg: dict) -> Connector:
    import os
    conn = cfg.get("connection", {})
    login = os.environ.get("ALTA_MT5_LOGIN")
    return MT5Connector(
        login=int(login) if login else None,
        password=os.environ.get("ALTA_MT5_PASSWORD"),
        server=os.environ.get("ALTA_MT5_SERVER") or conn.get("server"),
        terminal_path=conn.get("terminal_path"),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FOMC-window execution logger (no orders)")
    p.add_argument("--center", default=DEFAULT_CENTER, help="ISO timestamp center (default FOMC 2:00pm ET)")
    p.add_argument("--tz", default=DEFAULT_TZ, help="timezone for a naive --center")
    p.add_argument("--window-min", type=int, default=DEFAULT_WINDOW_MIN, help="± minutes around center")
    p.add_argument("--interval-sec", type=float, default=DEFAULT_INTERVAL_SEC, help="sampling period")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma-separated symbols")
    p.add_argument("--out", default=None, help="output jsonl path (default data/execution/fomc_window_<date>.jsonl)")
    p.add_argument("--dry-run", action="store_true", help="print window and exit; no connection")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None, connector: Optional[Connector] = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config()
        center = parse_center(args.center, args.tz)
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

        date_tag = center.strftime("%Y-%m-%d")
        out_path = Path(args.out) if args.out else (
            REPO_ROOT / "data" / "execution" / f"fomc_window_{date_tag}.jsonl"
        )

        if args.dry_run:
            start, end = compute_window(center, args.window_min)
            print(f"DRY-RUN — window {start.isoformat()} → {end.isoformat()}")
            print(f"  symbols={symbols} interval={args.interval_sec}s out={out_path}")
            print("  (no connection attempted; places no orders)")
            return 0

        conn = connector if connector is not None else build_connector(cfg)
        try:
            return run(
                conn, cfg,
                center=center,
                window_min=args.window_min,
                interval_sec=args.interval_sec,
                symbols=symbols,
                out_path=out_path,
            )
        finally:
            try:
                conn.shutdown()
            except Exception:
                pass
    except (LoggerError, ConnectorError) as e:
        print(f"ABORT: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
