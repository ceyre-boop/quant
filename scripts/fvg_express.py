"""
fvg_express.py — FVG-only data-collection trades for G2 accumulation.

Runs every 60 min during London (02:00–05:00 UTC) and NY AM (12:00–15:00 UTC).
0.25% risk per trade. Three gates must all pass before placing a trade:
  1. FVG active: unmitigated fair value gap in the last 10 bars
  2. Regime TRENDING or MOMENTUM (external or internal)
  3. Pair not blacked out (FOMC, NFP, CPI, etc.)

Logs every scan to data/ledger/fvg_express_scans.jsonl.
Usage:
    python3 scripts/fvg_express.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fvg_express")

SCAN_LOG = ROOT / "data" / "ledger" / "fvg_express_scans.jsonl"
RISK_PCT  = 0.0025   # 0.25%
TP_R      = 2.0      # TP at 2R
FVG_PAIRS = ["GBPUSD", "EURUSD", "AUDUSD", "AUDNZD", "USDJPY"]

_OANDA_MAP = {
    "GBPUSD": "GBP_USD",
    "EURUSD": "EUR_USD",
    "AUDUSD": "AUD_USD",
    "AUDNZD": "AUD_NZD",
    "USDJPY": "USD_JPY",
}
_YF_MAP = {pair: f"{pair}=X" for pair in FVG_PAIRS}


# ─── FVG stop level ──────────────────────────────────────────────────────────

def _find_fvg_stop(df, direction: str) -> float:
    """Return price level at the FVG zone boundary (= structural stop)."""
    h = df["High"]
    l = df["Low"]
    for i in range(len(df) - 1, max(len(df) - 11, 1), -1):
        if direction == "LONG" and float(l.iloc[i]) > float(h.iloc[i - 2]):
            return float(h.iloc[i - 2])
        if direction == "SHORT" and float(h.iloc[i]) < float(l.iloc[i - 2]):
            return float(l.iloc[i - 2])
    # Fallback to ATR-based stop when no FVG zone found
    atr = float((h - l).tail(14).mean())
    entry = float(df["Close"].iloc[-1])
    return (entry - atr) if direction == "LONG" else (entry + atr)


# ─── Gate checks ─────────────────────────────────────────────────────────────

def _check_fvg(df) -> int:
    """Return +1 (bullish FVG), -1 (bearish FVG), or 0 (none)."""
    from sovereign.intelligence.indicator_library import ind_fvg
    return ind_fvg(df).state


def _check_regime() -> str:
    """Return 'TRENDING', 'MOMENTUM', or 'OTHER'."""
    try:
        from sovereign.intelligence.regime_confidence import score_regime_confidence
        conf = score_regime_confidence()
        if conf.external_regime == "TRENDING":
            return "TRENDING"
        if conf.internal_regime == "MOMENTUM":
            return "MOMENTUM"
        return "OTHER"
    except Exception as e:
        log.debug("regime_confidence unavailable: %s", e)
        return "OTHER"


def _check_blackout(pair: str) -> bool:
    """Return True if pair is blacked out (high-impact news event)."""
    try:
        from ict.daily_bias import DailyBiasEngine
        biases = DailyBiasEngine().get_biases()
        return biases.get(pair, {}).get("blackout", False)
    except Exception as e:
        log.debug("DailyBiasEngine unavailable: %s", e)
        return False   # no bias data → don't block


# ─── Trade placement ─────────────────────────────────────────────────────────

def _place_trade(pair: str, direction: str, df, dry_run: bool) -> dict:
    oanda_pair = _OANDA_MAP[pair]
    entry      = float(df["Close"].iloc[-1])
    stop       = _find_fvg_stop(df, direction)
    risk_dist  = abs(entry - stop)
    if risk_dist == 0:
        return {"status": "VETOED", "reason": "ZERO_RISK_DISTANCE"}
    tp = entry + TP_R * risk_dist if direction == "LONG" else entry - TP_R * risk_dist

    if dry_run:
        log.info("[DRY RUN] %s %s entry=%.5f stop=%.5f tp=%.5f", pair, direction, entry, stop, tp)
        return {"status": "DRY_RUN", "entry": entry, "stop": stop, "tp": tp}

    from sovereign.execution.oanda_bridge import OandaBridge
    bridge = OandaBridge()
    units = bridge.compute_units(oanda_pair, entry, stop, RISK_PCT)
    if units == 0:
        return {"status": "VETOED", "reason": "ZERO_UNITS"}
    return bridge.place_trade(oanda_pair, direction, units, stop, tp)


# ─── Scan loop ───────────────────────────────────────────────────────────────

def scan(dry_run: bool = False) -> list[dict]:
    import yfinance as yf

    SCAN_LOG.parent.mkdir(parents=True, exist_ok=True)
    regime = _check_regime()
    results = []

    for pair in FVG_PAIRS:
        ts = datetime.now(timezone.utc).isoformat()
        entry: dict = {"pair": pair, "timestamp": ts, "dry_run": dry_run}

        try:
            df = yf.Ticker(_YF_MAP[pair]).history(period="30d", interval="1d", auto_adjust=True)
            if len(df) < 10:
                entry.update({"action": "SKIP", "reason": "INSUFFICIENT_DATA"})
                results.append(entry)
                _log_scan(entry)
                continue

            fvg_state = _check_fvg(df)
            entry["fvg_state"] = fvg_state
            entry["regime"]    = regime

            if fvg_state == 0:
                entry.update({"action": "SKIP", "reason": "NO_FVG"})
                results.append(entry)
                _log_scan(entry)
                continue

            direction = "LONG" if fvg_state == 1 else "SHORT"
            entry["direction"] = direction

            if regime not in ("TRENDING", "MOMENTUM"):
                entry.update({"action": "SKIP", "reason": "REGIME_NOT_ALIGNED"})
                results.append(entry)
                _log_scan(entry)
                continue

            blackout = _check_blackout(pair)
            entry["blackout"] = blackout
            if blackout:
                entry.update({"action": "SKIP", "reason": "BLACKOUT"})
                results.append(entry)
                _log_scan(entry)
                continue

            # All three gates passed — place trade
            result = _place_trade(pair, direction, df, dry_run)
            entry["action"]   = "TRADE"
            entry["trade_id"] = result.get("trade_id")
            entry["status"]   = result.get("status")
            log.info("FVG express: %s %s → %s", pair, direction, result.get("status"))

        except Exception as e:
            log.warning("fvg_express scan failed for %s: %s", pair, e)
            entry.update({"action": "ERROR", "error": str(e)})

        results.append(entry)
        _log_scan(entry)

    return results


def _log_scan(entry: dict) -> None:
    with SCAN_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FVG express scanner — G2 data collection")
    parser.add_argument("--dry-run", action="store_true", help="Check gates, skip trade placement")
    args = parser.parse_args()

    results = scan(dry_run=args.dry_run)
    trades = [r for r in results if r.get("action") == "TRADE"]
    skips  = [r for r in results if r.get("action") == "SKIP"]
    print(f"\nFVG Express: {len(trades)} trade(s) | {len(skips)} skip(s) | {len(FVG_PAIRS)} pairs scanned")
    for r in results:
        status = r.get("status") or r.get("reason") or r.get("action", "?")
        print(f"  {r['pair']:8} {r.get('direction','----'):5} fvg={r.get('fvg_state', 0):+d}  → {status}")
