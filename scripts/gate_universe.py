#!/usr/bin/env python3
"""
gate_universe.py — Petrules Gate universe builder.

Builds the daily scan universe: S&P 500 + Russell 2000 + 50 major ETFs + 4 carry
forex pairs (~2,550 instruments) and writes it to data/agent/gate_universe.json.

Free data only:
  - S&P 500 : Wikipedia constituents table (always current)
  - Russell 2000 : iShares IWM holdings CSV
  - ETFs / carry pairs : hardcoded in config/gate_params.yml

Refresh weekly. Called by the scanner on first run / when the universe file is
older than universe.refresh_days.

DISCIPLINE: standalone research script. No sovereign/ or ict/ imports.
No silent failures — if a source is unreachable, the caller decides; this module
raises so the scanner can write the honest error JSON.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    import requests
except ImportError:  # pragma: no cover - requests is a repo dependency
    requests = None

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "gate_params.yml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _http_get(url: str, headers: dict | None = None, timeout: int = 30) -> str:
    if requests is None:
        raise RuntimeError("requests library not available")
    resp = requests.get(url, headers=headers or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_sp500(cfg: dict) -> list[str]:
    """S&P 500 tickers from the Wikipedia constituents table."""
    url = cfg["universe"]["sp500_wikipedia"]
    headers = {"User-Agent": cfg["edgar"]["user_agent"]}
    html = _http_get(url, headers=headers)
    # Parse the first wikitable's first column of <td> ticker links without pandas.
    import re

    tickers: list[str] = []
    # Rows in the constituents table look like: <td><a ...>MMM</a></td>
    for m in re.finditer(r'<td[^>]*>\s*<a[^>]*>([A-Z][A-Z.\-]{0,6})</a>', html):
        sym = m.group(1).strip().replace(".", "-")
        if sym and sym not in tickers:
            tickers.append(sym)
    if len(tickers) < 400:
        raise RuntimeError(
            f"S&P 500 parse returned only {len(tickers)} tickers — layout changed?"
        )
    return tickers


def fetch_russell2000(cfg: dict) -> list[str]:
    """Russell 2000 tickers from the iShares IWM holdings CSV."""
    url = cfg["universe"]["russell2000_iwm_csv"]
    headers = {"User-Agent": cfg["edgar"]["user_agent"]}
    raw = _http_get(url, headers=headers)
    # The IWM CSV has a preamble; the holdings table starts at the "Ticker" header row.
    lines = raw.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.lower().lstrip('"').startswith("ticker"):
            start = i
            break
    if start is None:
        raise RuntimeError("IWM holdings CSV — could not locate Ticker header row")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    tickers: list[str] = []
    for row in reader:
        sym = (row.get("Ticker") or "").strip().strip('"')
        asset = (row.get("Asset Class") or "").strip().strip('"')
        if not sym or sym == "-":
            continue
        if asset and asset.lower() != "equity":
            continue
        sym = sym.replace(".", "-")
        if sym not in tickers:
            tickers.append(sym)
    if len(tickers) < 1000:
        raise RuntimeError(
            f"Russell 2000 parse returned only {len(tickers)} tickers — CSV changed?"
        )
    return tickers


def build_universe(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    sp500 = fetch_sp500(cfg)
    russell = fetch_russell2000(cfg)
    etfs = list(cfg["universe"]["etfs"])
    carry = list(cfg["universe"]["carry_pairs"])

    # De-dup across equity buckets while preserving provenance.
    equities: list[str] = []
    for sym in sp500 + russell:
        if sym not in equities:
            equities.append(sym)

    combined = equities + [e for e in etfs if e not in equities] + carry
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {
            "sp500": len(sp500),
            "russell2000": len(russell),
            "etfs": len(etfs),
            "carry_pairs": len(carry),
            "total": len(combined),
        },
        "sp500": sp500,
        "russell2000": russell,
        "etfs": etfs,
        "carry_pairs": carry,
        "symbols": combined,
    }


def universe_is_stale(path: Path, refresh_days: int) -> bool:
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text())
        built = datetime.fromisoformat(data["built_at"])
    except Exception:
        return True
    age = datetime.now(timezone.utc) - built
    return age.days >= refresh_days


def write_universe(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    out = REPO_ROOT / cfg["paths"]["universe"]
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build_universe(cfg)
    out.write_text(json.dumps(data, indent=2))
    return out


def main() -> int:
    cfg = load_config()
    try:
        out = write_universe(cfg)
    except Exception as exc:  # honest failure — no fabricated universe
        print(f"[gate_universe] FAILED: {exc}", file=sys.stderr)
        return 1
    data = json.loads(out.read_text())
    print(f"[gate_universe] wrote {out} — {data['counts']['total']} instruments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
