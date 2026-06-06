"""
Oracle Tier 1 — PULSE CHECK
sovereign/oracle/pulse_check.py

Runs every 2 hours. No LLM call. Pure computation.
Detects anomalies in decision logs, updates running stats, writes to dashboard.

Cost: $0.00
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from sovereign.utils.timestamps import canonical_timestamp

ROOT           = Path(__file__).resolve().parents[2]
PULSE_DIR      = ROOT / "data" / "oracle" / "pulses"
PULSE_STATE    = ROOT / "data" / "oracle" / ".pulse_state.json"
DECISION_LOG_DIR = ROOT / "data" / "decision_logs"
MESSAGES_PATH  = ROOT / "data" / "agent" / "messages_to_colin.json"
HEALTH_PATH    = ROOT / "data" / "agent" / "health.json"
LEDGER_PATH    = ROOT / "data" / "agent" / "hypothesis_ledger.json"
INDICATORS_DIR = ROOT / "data" / "indicators"
G2_PROGRESS_PATH = ROOT / "data" / "agent" / "g2_progress.json"
G2_TARGET = 8

log = logging.getLogger("oracle.pulse")

_PRICE_PAIRS = {
    'GBPUSD': 'GBPUSD=X',
    'EURUSD': 'EURUSD=X',
    'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X',
    'AUDNZD': 'AUDNZD=X',
}


# ─── Live price fetch ─────────────────────────────────────────────────────────

def _fetch_live_prices() -> dict:
    """Fetch current price snapshot for ICT pairs. Returns {} on any failure — never blocks pulse."""
    import yfinance as yf
    prices = {}
    for pair, ticker in _PRICE_PAIRS.items():
        try:
            hist = yf.Ticker(ticker).history(period='1d', interval='5m')
            if len(hist) == 0:
                continue
            prices[pair] = {
                'current':    round(float(hist['Close'].iloc[-1]), 5),
                'high_today': round(float(hist['High'].max()), 5),
                'low_today':  round(float(hist['Low'].min()), 5),
                'bars_today': len(hist),
            }
        except Exception:
            continue
    return prices


# ─── State helpers ────────────────────────────────────────────────────────────

def _load_pulse_state() -> dict:
    if PULSE_STATE.exists():
        try:
            return json.loads(PULSE_STATE.read_text())
        except Exception:
            pass
    return {}


def _save_pulse_state(last_pulse_time: Optional[str] = None, last_micro_time: Optional[str] = None) -> None:
    state = _load_pulse_state()
    if last_pulse_time is not None:
        state["last_pulse_time"] = last_pulse_time
    if last_micro_time is not None:
        state["last_micro_time"] = last_micro_time
    PULSE_STATE.write_text(json.dumps(state, indent=2))


def _get_last_pulse_time() -> datetime:
    state = _load_pulse_state()
    ts = state.get("last_pulse_time")
    if ts:
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    return datetime.now(timezone.utc) - timedelta(hours=24)


# ─── Decision log loader ──────────────────────────────────────────────────────

def _load_entries_since(cutoff: datetime) -> list[dict]:
    """Load all decision log entries with entry_timestamp >= cutoff."""
    if not DECISION_LOG_DIR.exists():
        return []
    entries = []
    for log_file in sorted(DECISION_LOG_DIR.glob("decisions_*.jsonl")):
        try:
            for line in log_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ts_str = rec.get("entry_timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= cutoff:
                        entries.append(rec)
                except Exception:
                    entries.append(rec)
        except Exception:
            continue
    return entries


# ─── rest_mode escalation ────────────────────────────────────────────────────

def _check_rest_mode_escalation() -> list[dict]:
    """Flag hypotheses validated but not yet implemented or explicitly rejected."""
    if not LEDGER_PATH.exists():
        return []
    try:
        data = json.loads(LEDGER_PATH.read_text())
        hyps = []
        if isinstance(data, dict):
            hyps = data.get("hypotheses", []) + data.get("ledger", [])
        else:
            hyps = data
    except Exception:
        return []

    issues = []
    today = datetime.now(timezone.utc).date()
    seen = set()
    for h in hyps:
        hid = h.get("id", "")
        if hid in seen:
            continue
        seen.add(hid)
        if not h.get("rest_mode"):
            continue
        if h.get("status") in ("REJECTED", "DEPLOYED"):
            continue

        rest_date_str = h.get("rest_mode_set") or h.get("closed") or h.get("date")
        if not rest_date_str:
            issues.append({
                "type": "REST_MODE_NO_DATE",
                "priority": "IMPORTANT",
                "message": f"{hid} is validated + rest_mode but has no rest_mode_set date — add it.",
            })
            continue

        try:
            rest_date = datetime.strptime(rest_date_str[:10], "%Y-%m-%d").date()
            days = (today - rest_date).days
        except Exception:
            continue

        name_snippet = h.get("name", "")[:50]
        if days >= 14:
            issues.append({
                "type": "REST_MODE_EXPIRED",
                "priority": "URGENT",
                "message": (
                    f"{hid} ({name_snippet}) validated and unimplemented for {days} days. "
                    f"Implement it or explicitly reject it. Validated knowledge has an expiry date."
                ),
            })
        elif days >= 7:
            issues.append({
                "type": "REST_MODE_WEEK_TWO",
                "priority": "IMPORTANT",
                "message": f"{hid} validated {days} days ago, still in rest_mode. Week 2 — decision needed.",
            })
        elif days >= 2:
            issues.append({
                "type": "REST_MODE_PENDING",
                "priority": "FYI",
                "message": f"{hid} validated {days} days ago, awaiting implementation.",
            })

    return issues


# ─── Anomaly detection ────────────────────────────────────────────────────────

def _count_consecutive_losses(r_values: list[float]) -> int:
    count = 0
    for r in reversed(r_values):
        if r < 0:
            count += 1
        else:
            break
    return count


def _detect_anomalies(recent_24h: list[dict]) -> list[dict]:
    anomalies = []

    # Consecutive losses
    outcomes_seq = [
        e["r_realized"] for e in recent_24h
        if e.get("r_realized") is not None and e.get("outcome") not in (None, "OPEN")
    ]
    if len(outcomes_seq) >= 3:
        n_consec = _count_consecutive_losses(outcomes_seq)
        if n_consec >= 3:
            anomalies.append({
                "type": "CONSECUTIVE_LOSSES",
                "priority": "URGENT",
                "message": f"{n_consec} consecutive losses detected in last 24h — review system health",
            })

    # Commitment score stuck (all entries share same value)
    scores = [
        e.get("commitment_score") for e in recent_24h
        if e.get("commitment_score") is not None
    ]
    if len(scores) >= 2 and len(set(scores)) == 1:
        anomalies.append({
            "type": "COMMITMENT_SCORE_STUCK",
            "priority": "IMPORTANT",
            "message": (
                f"All {len(scores)} entries have commitment_score={scores[0]}. "
                "Detector may be defaulting — check ict.pipeline logs for 'Commitment detector failed'."
            ),
        })

    # Stale entries (bars_since_signal > 5)
    stale = [e for e in recent_24h if (e.get("bars_since_signal") or 0) > 5]
    if stale:
        anomalies.append({
            "type": "STALE_ENTRY",
            "priority": "FYI",
            "message": f"{len(stale)} entries with bars_since_signal > 5 — entries may be chasing stale FVGs",
        })

    # Old open trades (entry > 7 days ago, no outcome)
    cutoff_7d = datetime.now(timezone.utc) - timedelta(days=7)
    missing_outcome = []
    for e in recent_24h:
        if e.get("outcome") is not None:
            continue
        ts_str = e.get("entry_timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff_7d:
                missing_outcome.append(e.get("pair", "?"))
        except Exception:
            pass
    if missing_outcome:
        anomalies.append({
            "type": "MISSING_OUTCOMES",
            "priority": "IMPORTANT",
            "message": f"{len(missing_outcome)} trades >7d old with no outcome: {', '.join(missing_outcome[:5])}",
        })

    return anomalies


# ─── Stats ────────────────────────────────────────────────────────────────────

def _compute_running_stats(outcomes: list[float]) -> dict:
    if not outcomes:
        return {}
    wins = [r for r in outcomes if r > 0]
    return {
        "trades_24h": len(outcomes),
        "win_rate_24h": round(len(wins) / len(outcomes), 3),
        "avg_r_24h": round(sum(outcomes) / len(outcomes), 3),
        "best_r": max(outcomes),
        "worst_r": min(outcomes),
    }


# ─── Health + messages ────────────────────────────────────────────────────────

def _update_health_json(pulse: dict) -> None:
    try:
        health = json.loads(HEALTH_PATH.read_text()) if HEALTH_PATH.exists() else {}
        if "components" not in health:
            health["components"] = {}
        n_anomalies = len(pulse.get("anomalies", []))
        urgent = any(a["priority"] == "URGENT" for a in pulse.get("anomalies", []))
        important_plus = [a for a in pulse.get("anomalies", []) if a.get("priority") in ("URGENT", "IMPORTANT")]
        status = "RED" if urgent else ("YELLOW" if important_plus else "GREEN")
        health["components"]["oracle_pulse"] = {
            "status": status,
            "detail": f"{pulse['new_entries_since_last']} new entries, {n_anomalies} anomalies",
            "last_pulse": pulse["timestamp"],
        }
        HEALTH_PATH.write_text(json.dumps(health, indent=2))
    except Exception as e:
        log.warning("Failed to update health.json: %s", e)


def _write_messages(anomalies: list[dict]) -> None:
    try:
        data = json.loads(MESSAGES_PATH.read_text()) if MESSAGES_PATH.exists() else {}
        if "messages" not in data:
            data["messages"] = []
        emoji_map = {"URGENT": "🔴", "IMPORTANT": "🟡", "FYI": "🟢"}
        for a in anomalies:
            priority = a.get("priority", "FYI")
            data["messages"].insert(0, {
                "id": f"pulse-{canonical_timestamp()[:16].replace(':', '').replace('-', '')}",
                "priority": priority,
                "emoji": emoji_map.get(priority, "🟢"),
                "text": f"[PULSE] {a['type']}: {a['message']}",
                "timestamp": canonical_timestamp(),
                "read": False,
                "source": "oracle_pulse",
            })
        data["messages"] = data["messages"][:50]
        data["last_updated"] = canonical_timestamp()
        MESSAGES_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.warning("Failed to write messages: %s", e)


# ─── TV regime alignment check ────────────────────────────────────────────────

def _check_regime_alignment() -> list[dict]:
    """Flag when internal and external (TradingView) regimes disagree."""
    try:
        from sovereign.intelligence.regime_confidence import score_regime_confidence
        conf = score_regime_confidence()
    except Exception as exc:
        log.debug("regime_confidence unavailable: %s", exc)
        return []

    if conf.external_regime == "UNKNOWN":
        return []  # no TV signals → no anomaly

    if not conf.agreement:
        return [{
            "type": "REGIME_DISAGREEMENT",
            "priority": "IMPORTANT",
            "message": (
                f"Internal regime ({conf.internal_regime}) conflicts with "
                f"TradingView ({conf.external_regime}). "
                f"Sizing reduced to {conf.sizing_multiplier:.0%}. {conf.reason}"
            ),
        }]

    return []


# ─── Indicator consensus ──────────────────────────────────────────────────────

def _compute_indicator_consensus_for_pulse() -> dict:
    """
    Runs all 30 indicators on latest 90d of daily data for all PRICE_PAIRS.
    Writes data/indicators/live_snapshot.json. Returns {} on any error.
    """
    try:
        import yfinance as yf
        from sovereign.intelligence.indicator_library import compute_all_indicators
        from sovereign.intelligence.indicator_consensus import score_indicator_consensus

        result = {}
        for pair, ticker in _PRICE_PAIRS.items():
            try:
                df = yf.Ticker(ticker).history(period="90d", interval="1d", auto_adjust=True)
                if len(df) < 30:
                    continue
                consensus = score_indicator_consensus(pair, df)
                result[pair] = {
                    "bullish": consensus.bullish_count,
                    "bearish": consensus.bearish_count,
                    "neutral": consensus.neutral_count,
                    "direction": consensus.direction,
                    "conviction": round(consensus.conviction, 3),
                    "hit_rate": consensus.historical_hit_rate,
                    "matching_green": len(consensus.matching_green_long),
                    "matching_green_short": len(consensus.matching_green_short),
                    "top_bullish": consensus.top_bullish[:5],
                    "top_bearish": consensus.top_bearish[:5],
                    "snapshot": consensus.snapshot,
                }
            except Exception as pair_exc:
                log.debug("indicator_consensus(%s) failed: %s", pair, pair_exc)
                continue

        snap_path = INDICATORS_DIR / "live_snapshot.json"
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(json.dumps({
            "timestamp": canonical_timestamp(),
            "pairs": result,
        }, indent=2))
        return result
    except Exception as exc:
        log.warning("_compute_indicator_consensus_for_pulse failed: %s", exc)
        return {}


# ─── G2 progress tracker ─────────────────────────────────────────────────────

def _update_g2_progress() -> dict:
    """
    Count closed OANDA trades, compute win_rate / avg_r, estimate G2 completion.
    Writes data/agent/g2_progress.json. Returns {} on any error.
    """
    try:
        from sovereign.execution.oanda_bridge import OandaBridge
        bridge = OandaBridge()
        closed = bridge.get_closed_trades(limit=100)
        total = len(closed)

        wins = [t for t in closed if float(t.get("realizedPL", 0)) > 0]
        win_rate = round(len(wins) / total, 3) if total else None

        # Avg R from fills cross-reference
        avg_r = _avg_r_from_fills(closed)

        first_date  = closed[-1]["openTime"][:10]  if closed else None
        latest_date = closed[0].get("closeTime", closed[0].get("openTime", ""))[:10] if closed else None

        est_completion = _estimate_g2_date(total, first_date)

        existing: dict = {}
        if G2_PROGRESS_PATH.exists():
            try:
                existing = json.loads(G2_PROGRESS_PATH.read_text())
            except Exception:
                pass

        progress = {
            **existing,                                      # preserve g2a_status, ict_fills, latest_fill
            "total_closed":            total,
            "target":                  G2_TARGET,
            "pct_complete":            round(total / G2_TARGET, 3),
            "win_rate":                win_rate,
            "avg_r":                   avg_r,
            "first_trade_date":        first_date,
            "latest_trade_date":       latest_date,
            "estimated_g2_completion": est_completion,
            "last_updated":            canonical_timestamp(),
        }
        G2_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        G2_PROGRESS_PATH.write_text(json.dumps(progress, indent=2))
        log.info("G2 progress: %d/%d (%.0f%%)", total, G2_TARGET, total / G2_TARGET * 100)
        return progress
    except Exception as exc:
        log.warning("_update_g2_progress failed: %s", exc)
        return {}


def _avg_r_from_fills(closed_oanda: list[dict]) -> Optional[float]:
    """Cross-reference OANDA closed trades with oanda_fills.jsonl to get R-multiples."""
    fills_path = ROOT / "data" / "ledger" / "oanda_fills.jsonl"
    if not fills_path.exists() or not closed_oanda:
        return None
    try:
        fill_ids = {str(t.get("id", "")) for t in closed_oanda}
        r_values = []
        for line in fills_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec.get("trade_id", "")) in fill_ids and rec.get("r_realized") is not None:
                r_values.append(float(rec["r_realized"]))
        return round(sum(r_values) / len(r_values), 3) if r_values else None
    except Exception:
        return None


def _estimate_g2_date(total: int, first_date: Optional[str]) -> Optional[str]:
    """Project G2 completion date based on current pace."""
    if total < 2 or first_date is None:
        return None
    try:
        start = datetime.strptime(first_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        elapsed_days = (datetime.now(timezone.utc) - start).days
        if elapsed_days <= 0:
            return None
        rate_per_day = total / elapsed_days
        days_remaining = (G2_TARGET - total) / rate_per_day
        est = datetime.now(timezone.utc) + timedelta(days=days_remaining)
        return est.strftime("%Y-%m-%d")
    except Exception:
        return None


# ─── Decision-outcome closure (P1: close the Oracle learning loop) ─────────────

FILLS_PATH = ROOT / "data" / "ledger" / "oanda_fills.jsonl"


def _load_all_decisions() -> list[dict]:
    """Every decision record across all monthly logs (any outcome)."""
    return _load_entries_since(datetime(1970, 1, 1, tzinfo=timezone.utc))


def _load_fills() -> list[dict]:
    """Load oanda_fills.jsonl (one fill per line). Returns [] on any failure."""
    if not FILLS_PATH.exists():
        return []
    fills = []
    for line in FILLS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            fills.append(json.loads(line))
        except Exception:
            continue
    return fills


def _entry_ts(rec: dict) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(rec.get("entry_timestamp", "")).replace("Z", "+00:00"))
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    except Exception:
        return None


def _classify_realized(realized_pl: float) -> str:
    if realized_pl > 0:
        return "WIN"
    if realized_pl < 0:
        return "LOSS"
    return "BREAKEVEN"


def _backfill_decision_outcomes() -> int:
    """Match closed OANDA trades to OPEN decision records and back-fill outcomes.

    R-multiple is computed from entry/close/stop. Trades without a sane stop
    (e.g. test fills) are skipped rather than recorded with a fabricated R —
    the Oracle must not learn from garbage. Returns count back-filled.
    """
    try:
        from sovereign.execution.oanda_bridge import OandaBridge
        from sovereign.intelligence.decision_logger import update_outcome
        closed = OandaBridge().get_closed_trades(limit=100)
    except Exception as exc:
        log.warning("backfill: cannot fetch closed trades: %s", exc)
        return 0

    fills_by_id = {str(f.get("trade_id", "")): f for f in _load_fills()}
    n_backfilled = 0

    for trade in closed:
        try:
            pair = str(trade.get("instrument", "")).replace("_", "")
            if not pair:
                continue
            fill = fills_by_id.get(str(trade.get("id", "")))
            entry = float(trade.get("price") or (fill or {}).get("fill_price") or 0.0)
            close_px = float(trade.get("averageClosePrice") or 0.0)
            stop = float((fill or {}).get("stop_price") or 0.0)
            units = float(trade.get("initialUnits") or 0.0)
            direction = 1.0 if units >= 0 else -1.0

            risk = abs(entry - stop)
            # Reject insane/missing stops (e.g. EURUSD stop at 1.0 → 14% risk).
            if entry <= 0 or close_px <= 0 or stop <= 0 or risk <= 0 or risk > 0.5 * entry:
                log.info("backfill: skip %s trade %s — no sane stop (entry=%s stop=%s)",
                         pair, trade.get("id"), entry, stop)
                continue

            r_realized = round((close_px - entry) / risk * direction, 3)
            outcome = _classify_realized(float(trade.get("realizedPL") or 0.0))
            open_time = str(trade.get("openTime", ""))[:19]

            if update_outcome(pair=pair, entry_timestamp=open_time, outcome=outcome,
                              r_realized=r_realized,
                              exit_timestamp=trade.get("closeTime"), system="ICT"):
                n_backfilled += 1
                log.info("backfill: %s %s r=%+.2f", pair, outcome, r_realized)
        except Exception as exc:
            log.warning("backfill: trade %s failed: %s", trade.get("id"), exc)

    # Venue-aware: also backfill Tradovate (CME futures) outcomes. Fully guarded — a no-op
    # without TRADOVATE creds, and can never break the OANDA backfill above.
    try:
        n_backfilled += _backfill_tradovate_outcomes()
    except Exception as exc:
        log.warning("backfill: tradovate venue skipped: %s", exc)

    return n_backfilled


def _backfill_tradovate_outcomes() -> int:
    """Backfill outcomes for the Tradovate (CME futures) venue.

    UNTESTED pending demo creds — early-returns 0 when TRADOVATE_ACCOUNT_ID is absent, so on a
    machine without Tradovate configured this is a pure no-op. The fill-schema parsing below is
    provisional and must be validated against a real demo account before it's relied upon.
    """
    import os
    if not os.environ.get("TRADOVATE_ACCOUNT_ID"):
        return 0
    try:
        from sovereign.execution.tradovate_bridge import TradovateBridge
        from sovereign.intelligence.decision_logger import update_outcome
        closed = TradovateBridge().get_closed_trades(limit=100)
    except Exception as exc:
        log.warning("tradovate backfill: cannot fetch fills: %s", exc)
        return 0

    n = 0
    for f in closed:
        try:
            symbol = str(f.get("symbol") or f.get("contract") or "")
            pnl = float(f.get("pnl") or f.get("realizedPnl") or 0.0)
            ts = str(f.get("timestamp") or f.get("tradeDate") or "")[:19]
            if not symbol or not ts:
                continue
            outcome = _classify_realized(pnl)
            if update_outcome(pair=symbol, entry_timestamp=ts, outcome=outcome,
                              r_realized=0.0, exit_timestamp=f.get("timestamp"),
                              system="FUTURES"):
                n += 1
        except Exception as exc:
            log.warning("tradovate backfill: fill failed: %s", exc)
    return n


def _norm_pair(p: str) -> str:
    """Normalise pair across the system's formats: GBPUSD=X, EUR_USD, GBPUSD → GBPUSD."""
    return str(p).replace("=X", "").replace("_", "").upper()


def _expire_stale_decisions(max_age_hours: int = 48) -> int:
    """Mark OPEN decisions >max_age_hours old with no matching fill as EXPIRED.

    Most logged decisions are scan signals that never became OANDA trades; leaving
    them OPEN starves the Oracle's stats. EXPIRED (r=0) is the truthful terminal
    state for a signal that never filled.

    Edits the monthly files directly (one rewrite per file) rather than routing
    through update_outcome: decision records can carry an entry_timestamp from a
    different month than the file they live in (e.g. a replayed signal dated in
    January logged in May), which would mis-route update_outcome's month lookup.
    Returns count expired.
    """
    if not DECISION_LOG_DIR.exists():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    fills = _load_fills()
    # (normalised pair, YYYY-MM-DD) for fills that became real trades.
    fill_keys = {
        (_norm_pair(f.get("pair", "")), str(f.get("timestamp", ""))[:10])
        for f in fills
    }
    exit_ts = canonical_timestamp()
    n_expired = 0

    def _has_fill(pnorm: str, day) -> bool:
        for fp, fd in fill_keys:
            if fp != pnorm:
                continue
            try:
                if abs((datetime.strptime(fd, "%Y-%m-%d").date() - day).days) <= 2:
                    return True
            except Exception:
                continue
        return False

    for log_file in DECISION_LOG_DIR.glob("decisions_*.jsonl"):
        try:
            lines = log_file.read_text().splitlines()
        except Exception:
            continue
        out, changed = [], False
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                out.append(line)
                continue
            if rec.get("outcome") in (None, "OPEN"):
                ts = _entry_ts(rec)
                if ts is not None and ts < cutoff and not _has_fill(_norm_pair(rec.get("pair", "")), ts.date()):
                    rec["outcome"] = "EXPIRED"
                    rec["r_realized"] = 0.0
                    rec["exit_timestamp"] = exit_ts
                    n_expired += 1
                    changed = True
            out.append(json.dumps(rec, default=str))
        if changed:
            log_file.write_text("\n".join(out) + "\n")

    return n_expired


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_pulse() -> dict:
    """Run pulse check. No LLM call. Writes pulse report, updates health.json."""
    last = _get_last_pulse_time()
    new_entries = _load_entries_since(last)
    recent_24h = _load_entries_since(datetime.now(timezone.utc) - timedelta(hours=24))
    live_prices = _fetch_live_prices()

    outcomes = [
        e["r_realized"] for e in recent_24h
        if e.get("r_realized") is not None and e.get("outcome") not in (None, "OPEN")
    ]
    anomalies = _detect_anomalies(recent_24h) + _check_rest_mode_escalation() + _check_regime_alignment()
    indicator_consensus = _compute_indicator_consensus_for_pulse()
    g2_progress = _update_g2_progress()

    # Close the learning loop: back-fill closed trades, expire never-filled signals.
    n_backfilled = _backfill_decision_outcomes()
    n_expired = _expire_stale_decisions()

    # Write state BEFORE check_all_loops reads it — prevents false DOWN alarm where
    # health reads the 2h-old timestamp because state wasn't updated yet.
    pulse_timestamp = canonical_timestamp()
    _save_pulse_state(last_pulse_time=pulse_timestamp)

    # Watch the loops: detect any improvement-machinery loop that has gone silent.
    loop_health = {}
    try:
        from sovereign.oracle.loop_health import check_all_loops
        loop_health = check_all_loops()
    except Exception as exc:
        log.warning("loop_health check failed: %s", exc)

    pulse = {
        "timestamp": pulse_timestamp,
        "new_entries_since_last": len(new_entries),
        "new_outcomes_since_last": sum(1 for e in new_entries if e.get("outcome")),
        "outcomes_backfilled": n_backfilled,
        "decisions_expired": n_expired,
        "loops_down": loop_health.get("down", []),
        "anomalies": anomalies,
        "running_stats": _compute_running_stats(outcomes),
        "live_prices": live_prices,
        "indicator_consensus": indicator_consensus,
        "g2_progress": g2_progress,
    }

    PULSE_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = pulse["timestamp"][:16].replace(":", "").replace("-", "").replace("T", "")
    (PULSE_DIR / f"pulse_{ts_slug}.json").write_text(json.dumps(pulse, indent=2))

    _update_health_json(pulse)
    if anomalies:
        _write_messages(anomalies)

    return pulse


if __name__ == "__main__":
    result = run_pulse()
    print(f"Pulse complete. New entries: {result['new_entries_since_last']}, "
          f"Outcomes: {result['new_outcomes_since_last']}, "
          f"Backfilled: {result.get('outcomes_backfilled', 0)}, "
          f"Expired: {result.get('decisions_expired', 0)}, "
          f"Anomalies: {len(result['anomalies'])}")
    for a in result["anomalies"]:
        print(f"  [{a['priority']}] {a['type']}: {a['message']}")
    if result["running_stats"]:
        s = result["running_stats"]
        print(f"  24h stats: {s['trades_24h']} trades | WR={s['win_rate_24h']:.0%} | avgR={s['avg_r_24h']:+.3f}")
