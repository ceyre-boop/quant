"""
Reasoning + attribution templates for the MES/MNQ learning agent.

Pure, null-safe sentence assembly from ACTUAL signal state — no hardcoded narratives, no I/O.
  - entry_reasoning(decision, bias)  -> the `reasoning` block logged at trade time
  - exit_attribution(record, exit_ctx) -> the `exit_reasoning` block logged at trade close

The point: every trade carries the system's belief at entry and its causal read at exit, so the
nightly/weekly review can compare "what I expected" vs "what happened" and compound understanding.
"""
from __future__ import annotations

from typing import Optional


def _why_this_direction(d, bias: dict) -> str:
    """One sentence built from the setup + live values."""
    parts = []
    setup = d.setup_type
    if setup == "ORB":
        kl = d.key_levels or {}
        ref = kl.get("orb_high") if d.direction == "LONG" else kl.get("orb_low")
        parts.append(f"ORB {'broke above' if d.direction == 'LONG' else 'broke below'} the "
                     f"opening range {'high' if d.direction == 'LONG' else 'low'} "
                     f"{ref if ref is not None else '?'}")
    elif setup == "VWAP_MR":
        parts.append(f"price stretched to the {'lower' if d.direction == 'LONG' else 'upper'} "
                     f"VWAP band and is fading back to the mean (VWAP {d.vwap})")
    elif setup == "MICRO":
        parts.append(f"VWAP reclaim with RSI crossing {'up through' if d.direction == 'LONG' else 'down through'} "
                     f"level (RSI {d.rsi}, VWAP {d.vwap})")
    bdir = bias.get("bias", "NEUTRAL")
    if bdir == d.direction:
        parts.append(f"macro bias is {bdir} (conviction {bias.get('conviction', 0)})")
    elif bdir in ("LONG", "SHORT"):
        parts.append(f"macro bias is {bdir} — counter-bias mean-reversion")
    if d.cvd_confirmed is True:
        parts.append(f"CVD confirms (slope {d.cvd_slope}, {d.cvd_quality})")
    elif d.cvd_confirmed is False:
        parts.append(f"CVD does NOT confirm (slope {d.cvd_slope})")
    if d.confluence:
        parts.append(f"entry within tolerance of {d.confluence} volume level(s)")
    return ". ".join(p[0].upper() + p[1:] for p in parts) + "." if parts else "No structured reason."


def entry_reasoning(d, bias: dict) -> dict:
    """The `reasoning` block for trade_log, assembled from an EntryDecision."""
    return {
        "setup_type": d.setup_type,
        "why_this_direction": _why_this_direction(d, bias),
        "key_levels": d.key_levels,
        "confluence_score": d.confluence,
        "cvd_slope": d.cvd_slope,
        "cvd_quality": d.cvd_quality,
        "cvd_confirmed": d.cvd_confirmed,
        "regime": d.regime_state,
        "adr_used_pct": d.adr_used_pct,
        "time_gate": d.time_gate,
        "would_have_blocked": list(d.would_have_blocked or []),
        "confidence": d.confidence,
        "expected_target": d.target,
        "expected_r": d.expected_r,
        "falsifier": d.falsifier_text or (f"kill {d.falsifier}" if d.falsifier is not None else None),
        "learning_mode": d.learning_mode,
    }


def _post_trade_hypothesis(entry_block: Optional[dict], exit_type: str, r_realized: Optional[float]) -> str:
    """Templated causal read from entry conditions vs the outcome."""
    eb = entry_block or {}
    won = (r_realized or 0) > 0
    cvd = eb.get("cvd_confirmed")
    setup = eb.get("setup_type", "?")
    if won:
        if cvd is True:
            return f"{setup} win with CVD confirmation at entry — consistent with order-flow-backed signal."
        if cvd is False:
            return f"{setup} won despite CVD not confirming — possible luck/regime tailwind; watch the sample."
        return f"{setup} win; CVD unknown at entry (thin volume) — outcome not attributable to order flow."
    # loss
    if cvd is False:
        return (f"{setup} stopped out and CVD did NOT confirm entry — stop-out consistent with an "
                f"unconfirmed signal. Pattern: {setup} without CVD confirmation = higher failure rate.")
    if cvd is True:
        return (f"{setup} stopped out despite CVD confirming — possible adverse regime or news-driven move; "
                f"not a signal-quality failure.")
    return f"{setup} stopped out with CVD unknown (thin volume) — cannot attribute; flag for review."


def exit_attribution(record: dict, exit_ctx: dict) -> dict:
    """The `exit_reasoning` block. record = the trade record (has `reasoning`); exit_ctx carries
    exit_type, exit_price, r_realized, and optional market-context-at-exit fields."""
    eb = record.get("reasoning") or {}
    exit_type = exit_ctx.get("exit_type", "UNKNOWN")
    r = exit_ctx.get("r_realized")
    falsifier = eb.get("falsifier")
    what = exit_ctx.get("what_happened")
    if not what:
        if exit_type == "STOP_LOSS":
            what = f"Price hit stop. {('Falsifier triggered. ' if falsifier else '')}r_realized {r}."
        elif exit_type in ("TARGET", "TAKE_PROFIT", "T1"):
            what = f"Price reached target {eb.get('expected_target')}. r_realized {r}."
        else:
            what = f"Exit ({exit_type}). r_realized {r}."
    return {
        "exit_type": exit_type,
        "exit_price": exit_ctx.get("exit_price"),
        "r_realized": r,
        "what_happened": what,
        "market_context_at_exit": exit_ctx.get("market_context_at_exit", {}),
        "post_trade_hypothesis": _post_trade_hypothesis(eb, exit_type, r),
    }
