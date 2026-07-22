#!/usr/bin/env python3
"""
build_paper_accounts.py — the missing observability plumbing.

Turns the strategies' REAL forward shadow ledgers into the paper-account JSON the
dashboard reads, so the prop panel stops showing a phantom $0 and starts showing the
actual paper run — flat on no-signal days, moving on signal days, no real money at risk.

Writes:
  data/agent/prop_account_balance.json   <- the dashboard's prop panel (Undertow shadow)
  data/agent/carry_paper_account.json    <- carry paper account (data ready; dashboard
                                             needs a second panel to show it)

DISCIPLINE (same as the rest of the nervous system):
  - Reads only what is actually logged. constitutional_day_ret is the measured shadow
    return at constitutional sizing — used as-is. It does NOT fabricate F2+F3-scaled
    numbers it cannot derive; scaling to F2+F3 needs per-signal closed outcomes the daily
    ledger doesn't carry yet (noted below).
  - Fail loud: if a source is missing/stale, status = DEGRADED with a reason. Never a
    confident figure from absent data.
  - Never raises.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHADOW = REPO / "data" / "research" / "yield_frontier" / "shadow" / "shadow_daily.jsonl"
CARRY_EQUITY = REPO / "data" / "agent" / "equity_curve_live.jsonl"
OUT_PROP = REPO / "data" / "agent" / "prop_account_balance.json"
OUT_CARRY = REPO / "data" / "agent" / "carry_paper_account.json"

# Dashboard frame (matches dashboard/index.html: ACCOUNT / TARGET / MAX_DD).
BASE = 200_000.0
TARGET = 10_000.0
MAX_DD = 10_000.0
NOW = datetime.now(timezone.utc)


def build_undertow() -> dict:
    """Undertow paper shadow → prop_account_balance.json (the dashboard's prop panel)."""
    if not SHADOW.exists():
        return _degraded("shadow ledger not found at " + str(SHADOW.relative_to(REPO)))
    rows = []
    for line in SHADOW.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return _degraded("shadow ledger present but empty")

    rows.sort(key=lambda r: r.get("date", ""))
    equity = BASE
    peak = BASE
    prev = BASE
    n_signals_total = 0
    for r in rows:
        ret = float(r.get("constitutional_day_ret", 0.0) or 0.0)
        n_signals_total += int(r.get("n_signals", 0) or 0)
        prev = equity
        equity *= (1.0 + ret)
        peak = max(peak, equity)

    last = rows[-1]
    today_ret = float(last.get("constitutional_day_ret", 0.0) or 0.0)
    today_pnl = equity - equity / (1.0 + today_ret) if today_ret else 0.0
    total_pnl = equity - BASE
    drawdown_used = max(0.0, peak - equity)

    # Freshness: is the shadow logging current? (last date vs today)
    last_date = last.get("date", "")
    status = "OK"
    reason = ""
    try:
        age_days = (NOW.date() - datetime.fromisoformat(last_date).date()).days
        if age_days > 4:
            status, reason = "STALE", f"shadow last logged {last_date} ({age_days}d ago)"
    except Exception:
        status, reason = "DEGRADED", "could not parse last shadow date"

    return {
        "account": "Undertow paper shadow (HYP-093)",
        "label_note": "account framing ($200K/$10K) is cosmetic, inherited from the "
                      "dashboard; the Undertow is an own-capital strategy. Relabel freely.",
        "sizing_note": "returns are at CONSTITUTIONAL sizing (what the shadow logs). "
                       "F2+F3 scaling needs per-signal closed outcomes the daily ledger "
                       "does not carry yet — a shadow-ledger enhancement, not faked here.",
        "generated_at": NOW.isoformat(),
        "status": status,
        "status_reason": reason,
        "base": round(BASE, 2),
        "target": TARGET,
        "max_dd": MAX_DD,
        "balance": round(equity, 2),
        "total_pnl": round(total_pnl, 2),
        "today_pnl": round(today_pnl, 2),
        "drawdown_used": round(drawdown_used, 2),
        "days_trading": len(rows),
        "signals_to_date": n_signals_total,
        "last_signal_date": next((r["date"] for r in reversed(rows)
                                  if int(r.get("n_signals", 0) or 0) > 0), None),
        "source": str(SHADOW.relative_to(REPO)),
    }


def build_carry() -> dict:
    """Carry paper account from the live equity curve (data ready; dashboard needs a
    second panel to display it)."""
    if not CARRY_EQUITY.exists():
        return _degraded("carry equity curve not found", account="Carry paper (OANDA practice)")
    pts = []
    for line in CARRY_EQUITY.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pts.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not pts:
        return _degraded("carry equity curve empty", account="Carry paper (OANDA practice)")

    def _eq(p):
        for k in ("equity", "nav", "balance", "value"):
            if k in p:
                return float(p[k])
        return None

    first = next((_eq(p) for p in pts if _eq(p) is not None), None)
    last = next((_eq(p) for p in reversed(pts) if _eq(p) is not None), None)
    if first is None or last is None:
        return _degraded("carry equity points have no recognizable equity field",
                         account="Carry paper (OANDA practice)")
    return {
        "account": "Carry paper (OANDA practice)",
        "generated_at": NOW.isoformat(),
        "status": "OK",
        "balance": round(last, 2),
        "total_pnl": round(last - first, 2),
        "n_points": len(pts),
        "source": str(CARRY_EQUITY.relative_to(REPO)),
        "dashboard_note": "not yet shown — the dashboard prop panel reads only "
                          "prop_account_balance.json; add a second panel to surface this.",
    }


def _degraded(reason: str, account: str = "Undertow paper shadow (HYP-093)") -> dict:
    return {
        "account": account, "generated_at": NOW.isoformat(),
        "status": "DEGRADED", "status_reason": reason,
        "base": round(BASE, 2), "balance": round(BASE, 2),
        "total_pnl": 0.0, "today_pnl": 0.0, "drawdown_used": 0.0, "days_trading": 0,
    }


def main() -> int:
    prop = build_undertow()
    carry = build_carry()
    OUT_PROP.write_text(json.dumps(prop, indent=2))
    OUT_CARRY.write_text(json.dumps(carry, indent=2))
    print(f"wrote {OUT_PROP.relative_to(REPO)}  [{prop['status']}]  "
          f"balance ${prop.get('balance'):,.2f}  pnl ${prop.get('total_pnl'):,.2f}  "
          f"days {prop.get('days_trading')}  signals {prop.get('signals_to_date')}")
    print(f"wrote {OUT_CARRY.relative_to(REPO)}  [{carry['status']}]  "
          f"balance ${carry.get('balance', 0):,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
