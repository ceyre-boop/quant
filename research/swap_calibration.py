"""Swap/financing calibration: OANDA's actual financing rates vs the model table.

READ-ONLY (HTTP GET only). Pulls per-instrument financing rates from the OANDA v3
instruments endpoint for the 4 live carry pairs, compares against
SWAP_RATES_ANNUAL in sovereign/forex/forex_backtester.py, and writes
data/research/swap_calibration.json. Changes NOTHING — the table update is
TICK-024, gated (it re-baselines every backtest number incl. the 0.6886 anchor).

Motivated by the 2026-07-11 live-position triage: trade #227 (short EUR_USD)
EARNS ≈ +0.42%/yr financing while the model books short-EURUSD at −0.10%/yr.

Run: python3 research/swap_calibration.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "research" / "swap_calibration.json"

PAIR_MAP = {  # yfinance name -> OANDA instrument
    "EURUSD=X": "EUR_USD",
    "GBPUSD=X": "GBP_USD",
    "USDJPY=X": "USD_JPY",
    "AUDUSD=X": "AUD_USD",
}


def _env() -> dict:
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def _get(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    assert req.get_method() == "GET"
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main() -> None:
    from sovereign.forex.forex_backtester import SWAP_RATES_ANNUAL

    env = _env()
    token = env.get("OANDA_API_KEY") or os.environ.get("OANDA_API_KEY")
    account = env.get("OANDA_ACCOUNT_ID") or os.environ.get("OANDA_ACCOUNT_ID")
    base = env.get("OANDA_BASE_URL", "https://api-fxpractice.oanda.com")
    if not token or not account:
        raise SystemExit("FATAL: OANDA credentials not found in .env — nothing to calibrate")

    instruments = ",".join(PAIR_MAP.values())
    data = _get(f"{base}/v3/accounts/{account}/instruments?instruments={instruments}", token)

    rows = {}
    mismatches = []
    for inst in data.get("instruments", []):
        oanda_name = inst["name"]
        yf_name = next(k for k, v in PAIR_MAP.items() if v == oanda_name)
        fin = inst.get("financing", {})
        long_rate = float(fin.get("longRate", "nan"))
        short_rate = float(fin.get("shortRate", "nan"))
        model = SWAP_RATES_ANNUAL.get(yf_name, {})
        row = {}
        for side, oanda_rate in (("LONG", long_rate), ("SHORT", short_rate)):
            model_rate = model.get(side)
            sign_mismatch = (model_rate is not None and oanda_rate == oanda_rate
                             and (oanda_rate > 0) != (model_rate > 0)
                             and abs(oanda_rate) > 1e-6 and abs(model_rate) > 1e-6)
            row[side] = {"oanda_annual": oanda_rate, "model_annual": model_rate,
                         "delta_annual": (round(oanda_rate - model_rate, 6)
                                          if model_rate is not None else None),
                         "sign_mismatch": bool(sign_mismatch)}
            if sign_mismatch:
                mismatches.append(f"{yf_name} {side}: oanda {oanda_rate:+.4f} vs "
                                  f"model {model_rate:+.4f}")
        rows[yf_name] = row

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "OANDA v3 /accounts/{id}/instruments financing.longRate/shortRate (practice)",
        "empirical_anchor": ("trade #227 short EUR_USD earned +1.1122 USD financing over "
                             "2026-07-03..07-11 (~+0.42%/yr on notional) — every day a credit"),
        "model_source": "sovereign/forex/forex_backtester.py SWAP_RATES_ANNUAL",
        "pairs": rows,
        "sign_mismatches": mismatches,
        "note": ("CALIBRATION ONLY — updating SWAP_RATES_ANNUAL is TICK-024 (gated: feeds "
                 "_apply_costs -> every backtest -> the 0.6886 reconcile anchor; needs "
                 "param_change_log rationale + full re-reconcile + band re-anchoring decision)"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'pair':10s} {'side':5s} {'OANDA':>9s} {'model':>9s} {'delta':>9s}  sign")
    for pair, row in rows.items():
        for side in ("LONG", "SHORT"):
            r = row[side]
            print(f"{pair:10s} {side:5s} {r['oanda_annual']:>+9.4f} "
                  f"{(r['model_annual'] if r['model_annual'] is not None else float('nan')):>+9.4f} "
                  f"{(r['delta_annual'] if r['delta_annual'] is not None else float('nan')):>+9.4f}  "
                  f"{'MISMATCH' if r['sign_mismatch'] else 'ok'}")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
