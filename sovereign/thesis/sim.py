"""Simulated control account for a sealed thesis.

Buys the sealed basket at the official open of the entry date with a fixed dollar amount and
never touches it again. Written in the same fill-row format as the broker files, so
`reconcile` measures it exactly like a real account. The control answers: what did the
sealed basket, held untouched, do with the same money?

Network only through the injected open-price loader."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILLS = ROOT / "data" / "thesis" / "fills"
OpenLoader = Callable[[str], pd.Series]


def load_open(ticker: str) -> pd.Series:
    """Daily opens, adjusted consistently with prices.load_daily (auto_adjust=True)."""
    import yfinance as yf  # lazy
    h = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
    if h.empty:
        raise RuntimeError(f"yfinance returned nothing for {ticker}")
    s = h["Open"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.name = ticker
    return s.astype(float)


def sim_account_id(thesis_id: str) -> str:
    return f"SIM-{thesis_id}"


def sim_fills(record: dict[str, Any], dollars: float, entry_date: str,
              open_loader: OpenLoader) -> list[dict[str, Any]]:
    """One DEPOSIT row plus one BUY per leg at that day's open. Raises if any open is missing
    (the day has not traded yet) rather than silently using another price."""
    acct = sim_account_id(record["id"])
    d = pd.Timestamp(entry_date).normalize()
    ts = d.strftime("%Y-%m-%dT13:30:00+00:00")  # 09:30 ET on the dot
    rows = [{"ts": ts, "type": "CASH", "symbol": None, "side": None, "qty": None,
             "price": None, "net": float(dollars), "subtype": "DEPOSIT", "account": acct}]
    for leg in record["basket"]:
        s = open_loader(leg["ticker"])
        if d not in s.index:
            raise LookupError(f"no open for {leg['ticker']} on {entry_date}; market not traded yet?")
        px = float(s.loc[d])
        alloc = float(dollars) * float(leg["weight"])
        rows.append({"ts": ts, "type": "TRADE", "symbol": leg["ticker"], "side": "BUY",
                     "qty": round(alloc / px, 6), "price": px, "net": -round(alloc, 2),
                     "subtype": "SIM_OPEN", "account": acct})
    return rows


def write_sim_fills(record: dict[str, Any], dollars: float, entry_date: str,
                    open_loader: OpenLoader, fills_dir: Path = FILLS, force: bool = False) -> Path:
    acct = sim_account_id(record["id"])
    f = fills_dir / f"{acct}.jsonl"
    if f.exists() and not force:
        raise FileExistsError(f"{f} exists; a control is written once (use --force to redo)")
    rows = sim_fills(record, dollars, entry_date, open_loader)
    meta = {"_meta": True, "account": acct, "kind": "simulated_control", "thesis": record["id"],
            "hash_lock": record.get("hash_lock"), "dollars": dollars, "entry_date": entry_date,
            "fill_rule": "official open of entry_date, weights from sealed basket, never touched",
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    fills_dir.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(json.dumps(r) for r in [meta, *rows]) + "\n")
    return f
