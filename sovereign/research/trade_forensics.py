"""
Trade Forensics Engine — v004 post-mortem analysis.

For every historical trade, reconstruct the full entry context (macro score,
rate differential, IRP deviation, momentum, CB trigger state, ATR regime,
hold days, exit reason) and assign:

  outcome_r:   R-multiple (P&L / avg_stop_size) — continuous
  grade:       1=marginal  2=solid  3=exceptional  (for wins AND losses)
  failure_mode: WHY the loss happened (for losers)
  win_driver:   WHY the win worked (for winners)

Output:
  data/research/trade_forensics.json   — one record per trade, full context
  data/research/failure_clusters.json  — failure mode frequency + avg R
  data/research/win_drivers.json       — win driver frequency + avg R
  data/research/combat_rules.json      — auto-generated veto/boost rules

Run:
  python3 sovereign/research/trade_forensics.py
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
TRADES_FILE  = ROOT / "logs" / "forex_backtest_trades.json"
OUT_DIR      = ROOT / "data" / "research"
OUT_FORENSICS = OUT_DIR / "trade_forensics.json"
OUT_FAILURES  = OUT_DIR / "failure_clusters.json"
OUT_WINS      = OUT_DIR / "win_drivers.json"
OUT_COMBAT    = OUT_DIR / "combat_rules.json"

# Approximate full-stop size per pair in pct (used to convert pnl→R)
# Average stop across backtest = ~1.0–1.5% depending on ATR
AVG_STOP_PCT = {
    "EURUSD=X": 0.0080,
    "GBPUSD=X": 0.0095,
    "USDJPY=X": 0.0090,
    "AUDUSD=X": 0.0085,
    "USDCAD=X": 0.0075,
    "GBPJPY=X": 0.0120,
    "AUDNZD=X": 0.0065,
}

FALLBACK_RATES = {
    "US": 2.0, "EU": 0.5, "UK": 1.5, "JP": -0.1,
    "AU": 3.5, "CA": 2.5, "NZ": 3.0, "CH": -0.5,
}
FALLBACK_CPI = {
    "US": 2.5, "EU": 1.5, "UK": 2.0, "JP": 0.5,
    "AU": 2.5, "CA": 2.0, "NZ": 2.5, "CH": 0.5,
}

CB_TO_COUNTRY = {
    "FED": "US", "ECB": "EU", "BOE": "UK", "BOJ": "JP",
    "RBA": "AU", "BOC": "CA", "RBNZ": "NZ", "SNB": "CH",
}

PAIR_COUNTRIES = {
    "EURUSD=X": ("EU", "US"),
    "GBPUSD=X": ("UK", "US"),
    "USDJPY=X": ("US", "JP"),
    "AUDUSD=X": ("AU", "US"),
    "USDCAD=X": ("US", "CA"),
    "GBPJPY=X": ("UK", "JP"),
    "AUDNZD=X": ("AU", "NZ"),
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _to_r(pnl_pct: float, pair: str) -> float:
    stop = AVG_STOP_PCT.get(pair, 0.009)
    return round(pnl_pct / stop, 3)


def _grade_win(r: float) -> int:
    if r >= 2.0:
        return 3   # exceptional: hit 2R or better
    if r >= 0.8:
        return 2   # solid: meaningful win
    return 1        # marginal: just above zero


def _grade_loss(r: float) -> int:
    if r > -0.5:
        return 1   # marginal: scratched out, held on too long
    if r > -1.1:
        return 2   # solid stop: took the loss cleanly at stop
    return 3        # exceptional loss: blew past stop (gap, reversal)


def _get_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Handle both flat and multi-level yfinance column layouts."""
    if name in df.columns:
        col = df[name]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col.squeeze()
    # multi-level: find matching top-level name
    matches = [c for c in df.columns if (c[0] if isinstance(c, tuple) else c) == name]
    if matches:
        return df[matches[0]].squeeze()
    return df.iloc[:, 0].squeeze()


def _atr_pct(prices: pd.DataFrame, date: pd.Timestamp, window: int = 14) -> float:
    hist = prices.loc[:date].tail(window + 1)
    if len(hist) < 5:
        return 0.01
    high  = _get_col(hist, "High")
    low   = _get_col(hist, "Low")
    close = _get_col(hist, "Close")
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.mean())
    mid = float(close.iloc[-1])
    return float(atr / mid) if mid > 0 else 0.01


def _momentum_pct(prices: pd.DataFrame, date: pd.Timestamp, lookback: int = 63) -> float:
    hist = prices.loc[:date]
    close = _get_col(hist, "Close")
    if len(close) < lookback:
        return 0.0
    return float(close.iloc[-1] / close.iloc[-lookback] - 1)


def _real_rate_diff(
    base_country: str,
    quote_country: str,
    date: pd.Timestamp,
    rate_cache: Dict,
    cpi_cache: Dict,
) -> Tuple[float, float, float]:
    """Returns (rate_diff_nominal, rate_diff_real, macro_score 0-1)."""
    b_rate = rate_cache.get(base_country, pd.Series(dtype=float))
    q_rate = rate_cache.get(quote_country, pd.Series(dtype=float))
    b_cpi  = cpi_cache.get(base_country, pd.Series(dtype=float))
    q_cpi  = cpi_cache.get(quote_country, pd.Series(dtype=float))

    def _val(series, fallback):
        if len(series) and date >= series.index[0]:
            return float(series.asof(date))
        return fallback

    b_r = _val(b_rate, FALLBACK_RATES.get(base_country, 2.0))
    q_r = _val(q_rate, FALLBACK_RATES.get(quote_country, 2.0))
    b_c = _val(b_cpi,  FALLBACK_CPI.get(base_country,   2.0))
    q_c = _val(q_cpi,  FALLBACK_CPI.get(quote_country,  2.0))

    nom_diff  = b_r - q_r
    real_diff = (b_r - b_c) - (q_r - q_c)
    # macro_score is [-1,1]; normalize to [0,1] for grading
    ms = float(np.clip(real_diff / 4.0, -1, 1))
    return nom_diff, real_diff, ms


# ── Failure mode classifier ────────────────────────────────────────────────

def _classify_failure(
    trade: dict,
    ctx: dict,
) -> str:
    """Single most important reason this loss happened."""
    direction = trade["direction"]   # 1=long -1=short
    exit_r    = trade["exit_reason"]
    real_diff = ctx["real_rate_diff"]
    momentum  = ctx["momentum_63d"]
    atr_pct   = ctx["atr_14d_pct"]
    hold_days = trade["hold_days"]
    macro_vs_dir = ctx["macro_vs_direction"]  # 1=aligned, -1=against, 0=neutral

    if macro_vs_dir == -1:
        return "MACRO_AGAINST"        # traded against the rate/CPI signal
    if exit_r == "reversal" and hold_days <= 3:
        return "PREMATURE_REVERSAL"   # signal flipped almost immediately
    if exit_r == "time" and hold_days >= 15:
        return "HELD_TOO_LONG"        # dragged to time exit, no momentum
    if exit_r == "trailing_stop" and abs(momentum) < 0.01:
        return "LOW_MOMENTUM_ENTRY"   # entered flat market, trailing caught
    if atr_pct < 0.006:
        return "LOW_VOLATILITY"       # ATR compressed, not enough range to profit
    if exit_r in ("stop", "trailing_stop") and abs(real_diff) < 0.5:
        return "WEAK_RATE_SIGNAL"     # rate differential too narrow to sustain
    if momentum * direction < -0.01:
        return "COUNTER_MOMENTUM"     # traded against price momentum
    return "UNEXPLAINED"


def _classify_win_driver(
    trade: dict,
    ctx: dict,
) -> str:
    """Single most important reason this win worked."""
    direction = trade["direction"]
    exit_r    = trade["exit_reason"]
    real_diff = ctx["real_rate_diff"]
    momentum  = ctx["momentum_63d"]
    macro_vs_dir = ctx["macro_vs_direction"]
    r = ctx["outcome_r"]

    if macro_vs_dir == 1 and r >= 2.0:
        return "MACRO_ALIGNED_STRONG"   # macro + direction + big win
    if exit_r == "trailing_stop" and r >= 1.5:
        return "TRAILING_CAPTURED_TREND"
    if macro_vs_dir == 1 and r >= 0.8:
        return "MACRO_ALIGNED_CLEAN"
    if momentum * direction > 0.015 and r >= 1.0:
        return "MOMENTUM_CONFIRMED"
    if abs(real_diff) >= 2.0:
        return "STRONG_RATE_DIVERGENCE"
    if exit_r == "reversal" and r >= 0.5:
        return "TIMELY_EXIT"
    return "MARGINAL_WIN"


# ── Data loaders ──────────────────────────────────────────────────────────

def _load_rate_and_cpi_cache() -> Tuple[Dict, Dict]:
    """Load cached macro data from the ForexDataFetcher."""
    try:
        from sovereign.forex.data_fetcher import ForexDataFetcher
        fetcher = ForexDataFetcher()
        countries = set()
        for base, quote in PAIR_COUNTRIES.values():
            countries.add(base)
            countries.add(quote)

        rate_cache: Dict[str, pd.Series] = {}
        cpi_cache:  Dict[str, pd.Series] = {}
        for country in countries:
            try:
                rates = fetcher.get_rate_history(country, start="2014-01-01")
                if rates is not None and len(rates):
                    rate_cache[country] = rates
            except Exception:
                pass
            try:
                cpi = fetcher.get_cpi_history(country, start="2014-01-01")
                if cpi is not None and len(cpi):
                    cpi_cache[country] = cpi
            except Exception:
                pass
        return rate_cache, cpi_cache
    except Exception as exc:
        print(f"  [warn] Could not load macro data: {exc}")
        return {}, {}


def _load_prices(pair: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        # Use cache if available
        cache_path = ROOT / "data" / "cache" / f"{pair.replace('=X','')}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            return df
        df = yf.download(pair, start="2014-01-01", end="2026-05-01", progress=False)
        if df is not None and len(df) > 100:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass
    return None


# ── Main engine ───────────────────────────────────────────────────────────

def run_forensics(verbose: bool = True) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Loading trade history…")
    raw = json.loads(TRADES_FILE.read_text())

    if verbose:
        print("Loading macro data (rates + CPI)…")
    rate_cache, cpi_cache = _load_rate_and_cpi_cache()

    all_records: List[Dict] = []
    failure_counts: Dict[str, List[float]] = defaultdict(list)
    win_driver_counts: Dict[str, List[float]] = defaultdict(list)

    for pair, trades in raw.items():
        if verbose:
            print(f"  {pair}: {len(trades)} trades", end="", flush=True)

        base_country, quote_country = PAIR_COUNTRIES.get(pair, ("US", "US"))
        prices = _load_prices(pair)

        for trade in trades:
            entry_date = pd.Timestamp(trade["entry_date"])
            r = _to_r(trade["pnl_pct"], pair)
            is_win = trade["pnl_pct"] > 0

            # Reconstruct entry context
            atr = _atr_pct(prices, entry_date) if prices is not None else 0.009
            mom = _momentum_pct(prices, entry_date) if prices is not None else 0.0
            nom_diff, real_diff, macro_score = _real_rate_diff(
                base_country, quote_country, entry_date, rate_cache, cpi_cache
            )

            # Did the macro signal agree with the trade direction?
            macro_sign = int(np.sign(real_diff)) if abs(real_diff) > 0.2 else 0
            macro_vs_dir = macro_sign * trade["direction"]  # 1=aligned -1=against 0=neutral

            # Quarter of year for seasonality
            quarter = f"Q{(entry_date.month - 1) // 3 + 1}"
            day_of_week = entry_date.strftime("%A")

            ctx: Dict[str, Any] = {
                "pair": pair,
                "entry_date": str(trade["entry_date"]),
                "exit_date": str(trade["exit_date"]),
                "direction": trade["direction"],
                "direction_label": "LONG" if trade["direction"] == 1 else "SHORT",
                "pnl_pct": round(trade["pnl_pct"], 6),
                "outcome_r": r,
                "hold_days": trade["hold_days"],
                "exit_reason": trade["exit_reason"],
                "atr_14d_pct": round(atr, 5),
                "momentum_63d": round(mom, 5),
                "nom_rate_diff": round(nom_diff, 3),
                "real_rate_diff": round(real_diff, 3),
                "macro_score_01": round((macro_score + 1) / 2, 3),
                "macro_vs_direction": macro_vs_dir,
                "quarter": quarter,
                "day_of_week": day_of_week,
                "base_country": base_country,
                "quote_country": quote_country,
            }

            if is_win:
                ctx["outcome"] = "WIN"
                ctx["grade"] = _grade_win(r)
                ctx["win_driver"] = _classify_win_driver(trade, ctx)
                ctx["failure_mode"] = None
                win_driver_counts[ctx["win_driver"]].append(r)
            else:
                ctx["outcome"] = "LOSS"
                ctx["grade"] = _grade_loss(r)
                ctx["failure_mode"] = _classify_failure(trade, ctx)
                ctx["win_driver"] = None
                failure_counts[ctx["failure_mode"]].append(r)

            all_records.append(ctx)

        if verbose:
            wins_here = sum(1 for t in trades if t["pnl_pct"] > 0)
            print(f" → {wins_here}W/{len(trades)-wins_here}L")

    # ── Save forensics ────────────────────────────────────────────────────
    OUT_FORENSICS.write_text(json.dumps(all_records, indent=2))
    if verbose:
        print(f"\nSaved {len(all_records)} trade records → {OUT_FORENSICS}")

    # ── Failure cluster report ────────────────────────────────────────────
    failure_report = []
    for mode, r_list in sorted(failure_counts.items(), key=lambda x: -len(x[1])):
        failure_report.append({
            "failure_mode": mode,
            "count": len(r_list),
            "pct_of_losses": round(len(r_list) / max(sum(1 for t in all_records if t["outcome"]=="LOSS"), 1), 3),
            "avg_r": round(np.mean(r_list), 3),
            "worst_r": round(min(r_list), 3),
            "total_r_lost": round(sum(r_list), 2),
        })
    OUT_FAILURES.write_text(json.dumps(failure_report, indent=2))

    # ── Win driver report ─────────────────────────────────────────────────
    win_report = []
    for driver, r_list in sorted(win_driver_counts.items(), key=lambda x: -len(x[1])):
        win_report.append({
            "win_driver": driver,
            "count": len(r_list),
            "pct_of_wins": round(len(r_list) / max(sum(1 for t in all_records if t["outcome"]=="WIN"), 1), 3),
            "avg_r": round(np.mean(r_list), 3),
            "best_r": round(max(r_list), 3),
            "total_r_won": round(sum(r_list), 2),
        })
    OUT_WINS.write_text(json.dumps(win_report, indent=2))

    # ── Combat rules (auto-generated) ────────────────────────────────────
    combat_rules = _generate_combat_rules(all_records, failure_report, win_report)
    OUT_COMBAT.write_text(json.dumps(combat_rules, indent=2))

    return {
        "total_trades": len(all_records),
        "wins": sum(1 for t in all_records if t["outcome"] == "WIN"),
        "losses": sum(1 for t in all_records if t["outcome"] == "LOSS"),
        "failure_modes": failure_report,
        "win_drivers": win_report,
        "combat_rules": combat_rules,
    }


def _generate_combat_rules(
    records: List[Dict],
    failures: List[Dict],
    wins: List[Dict],
) -> List[Dict]:
    """
    Auto-generate actionable veto/boost rules from failure + win patterns.
    Each rule has:
      type:      VETO | BOOST | SIZE_CUT | SIZE_BOOST
      condition: human-readable trigger
      evidence:  trade count, avg R improvement, source failure_mode or win_driver
      priority:  1 (highest) to 3
    """
    rules = []

    # Rule from each failure mode with meaningful sample size
    for f in failures:
        mode = f["failure_mode"]
        count = f["count"]
        avg_r = f["avg_r"]
        if count < 5:
            continue

        if mode == "MACRO_AGAINST":
            rules.append({
                "id": "C-001",
                "type": "VETO",
                "condition": "real_rate_diff sign opposes trade direction",
                "detail": "Block new entries where real rate differential favors the opposite direction. "
                          "This is the macro engine's core signal — trading against it is anti-edge.",
                "evidence": {"trades_affected": count, "avg_r_when_broken": avg_r, "total_r_lost": f["total_r_lost"]},
                "priority": 1,
            })
        elif mode == "LOW_MOMENTUM_ENTRY":
            rules.append({
                "id": "C-002",
                "type": "SIZE_CUT",
                "condition": "63-day momentum < 0.5% AND ATR < 0.7%",
                "detail": "Reduce position size to 0.5× when entering a flat, low-volatility market. "
                          "Trailing stops are premature in compression; wait for expansion.",
                "evidence": {"trades_affected": count, "avg_r_when_broken": avg_r, "total_r_lost": f["total_r_lost"]},
                "priority": 2,
            })
        elif mode == "COUNTER_MOMENTUM":
            rules.append({
                "id": "C-003",
                "type": "VETO",
                "condition": "63-day momentum opposes trade direction by > 1%",
                "detail": "Macro signal may be right directionally but momentum is still against us. "
                          "Wait for momentum crossover or reduce size aggressively.",
                "evidence": {"trades_affected": count, "avg_r_when_broken": avg_r, "total_r_lost": f["total_r_lost"]},
                "priority": 2,
            })
        elif mode == "HELD_TOO_LONG":
            rules.append({
                "id": "C-004",
                "type": "VETO",
                "condition": "exit_reason == time AND hold_days > 14",
                "detail": "Trades reaching the time stop without profit or reversal signal are wasting capital. "
                          "Hard exit at day 12 if no meaningful P&L.",
                "evidence": {"trades_affected": count, "avg_r_when_broken": avg_r, "total_r_lost": f["total_r_lost"]},
                "priority": 3,
            })
        elif mode == "WEAK_RATE_SIGNAL":
            rules.append({
                "id": "C-005",
                "type": "VETO",
                "condition": "|real_rate_diff| < 0.5%",
                "detail": "When rate differential is too narrow (<50bp real), the macro edge disappears. "
                          "Skip these setups or wait for CB divergence to widen.",
                "evidence": {"trades_affected": count, "avg_r_when_broken": avg_r, "total_r_lost": f["total_r_lost"]},
                "priority": 2,
            })
        elif mode == "LOW_VOLATILITY":
            rules.append({
                "id": "C-006",
                "type": "VETO",
                "condition": "ATR_14d_pct < 0.6%",
                "detail": "Compressed volatility means the pair cannot deliver the range needed to hit targets. "
                          "This is separate from the ICT ADR filter — apply at signal generation too.",
                "evidence": {"trades_affected": count, "avg_r_when_broken": avg_r, "total_r_lost": f["total_r_lost"]},
                "priority": 2,
            })

    # Boost rules from strong win drivers
    for w in wins:
        driver = w["win_driver"]
        count = w["count"]
        avg_r = w["avg_r"]
        if count < 5:
            continue

        if driver == "MACRO_ALIGNED_STRONG":
            rules.append({
                "id": "B-001",
                "type": "SIZE_BOOST",
                "condition": "real_rate_diff aligns with direction AND |real_rate_diff| >= 2% AND momentum confirms",
                "detail": "Strong macro alignment with momentum confirmation is the system's highest-confidence setup. "
                          "Scale to 1.5× base size.",
                "evidence": {"trades_affected": count, "avg_r_won": avg_r, "total_r_won": w["total_r_won"]},
                "priority": 1,
            })
        elif driver == "TRAILING_CAPTURED_TREND":
            rules.append({
                "id": "B-002",
                "type": "BOOST",
                "condition": "exit_type would be trailing_stop AND ATR_14d > 0.9%",
                "detail": "High-ATR trending markets let trailing stops run. "
                          "Widen trailing stop multiplier from 1.0× to 1.3× ATR in these conditions.",
                "evidence": {"trades_affected": count, "avg_r_won": avg_r, "total_r_won": w["total_r_won"]},
                "priority": 2,
            })
        elif driver == "STRONG_RATE_DIVERGENCE":
            rules.append({
                "id": "B-003",
                "type": "SIZE_BOOST",
                "condition": "|real_rate_diff| >= 2.5%",
                "detail": "When real rate divergence is extreme (>250bp), the directional edge is highest. "
                          "Scale to 1.25× base size.",
                "evidence": {"trades_affected": count, "avg_r_won": avg_r, "total_r_won": w["total_r_won"]},
                "priority": 2,
            })

    # Compute summary stats
    total_r_lost = sum(f["total_r_lost"] for f in failures)
    recoverable = sum(
        f["total_r_lost"] for f in failures
        if f["failure_mode"] in ("MACRO_AGAINST", "LOW_MOMENTUM_ENTRY", "COUNTER_MOMENTUM", "WEAK_RATE_SIGNAL")
    )

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "total_r_lost_to_failures": round(total_r_lost, 2),
        "potentially_recoverable_r": round(recoverable, 2),
        "recovery_pct_of_losses": round(abs(recoverable / total_r_lost) * 100, 1) if total_r_lost else 0,
        "rules": rules,
    }


# ── Summary printer ───────────────────────────────────────────────────────

def print_summary(result: Dict) -> None:
    wins = result["wins"]
    losses = result["losses"]
    total = result["total_trades"]
    wr = wins / total * 100 if total else 0

    print(f"\n{'='*60}")
    print(f"TRADE FORENSICS SUMMARY — {total} trades ({wr:.1f}% win rate)")
    print(f"{'='*60}")

    print(f"\n── FAILURE MODES (losses only) ──")
    for f in result["failure_modes"]:
        bar = "█" * min(int(f["pct_of_losses"] * 20), 20)
        print(f"  {f['failure_mode']:<25} {f['count']:>4} trades  "
              f"avg {f['avg_r']:>6.2f}R  total {f['total_r_lost']:>7.2f}R  {bar}")

    print(f"\n── WIN DRIVERS (wins only) ──")
    for w in result["win_drivers"]:
        bar = "█" * min(int(w["pct_of_wins"] * 20), 20)
        print(f"  {w['win_driver']:<28} {w['count']:>4} trades  "
              f"avg {w['avg_r']:>5.2f}R  total {w['total_r_won']:>7.2f}R  {bar}")

    c = result["combat_rules"]
    print(f"\n── COMBAT RULES GENERATED ──")
    print(f"  Total R lost to avoidable failures: {c['total_r_lost_to_failures']:.2f}R")
    print(f"  Potentially recoverable:            {c['potentially_recoverable_r']:.2f}R  ({c['recovery_pct_of_losses']:.0f}%)")
    print(f"  Rules generated: {len(c['rules'])}")
    for r in c["rules"]:
        tag = "🔴 VETO" if r["type"] in ("VETO",) else "🟡 SIZE" if "SIZE" in r["type"] else "🟢 BOOST"
        print(f"    [{r['id']}] {tag} — {r['condition']}")
    print()


if __name__ == "__main__":
    result = run_forensics(verbose=True)
    print_summary(result)
    print(f"Files written:")
    print(f"  {OUT_FORENSICS}")
    print(f"  {OUT_FAILURES}")
    print(f"  {OUT_WINS}")
    print(f"  {OUT_COMBAT}")
