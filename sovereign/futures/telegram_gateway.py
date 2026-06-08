"""Telegram gateway for futures MACRO approvals (sandbox-local).

Micros run headless; only macro (ORB / multi-session hold) decisions ping your phone.
Dedicated bot (not the PAI bot) so its getUpdates stream is ours alone — no fight with
Pulse's poller. Two-way: we send an alert, then long-poll for YOUR reply.

Vocabulary (free-form, any order): big | small | now | wait | skip
  - size:   big / small        (default small)
  - timing: now / wait         (default now;  "wait" = defer, re-arm for a retrace)
  - skip:   pass on the trade

stdlib only (urllib) + dotenv. No forex/ICT/intelligence imports.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
_API = "https://api.telegram.org/bot{token}/{method}"
VALID = ("big", "small", "now", "wait", "skip")


def _load_env() -> tuple[Optional[str], Optional[str]]:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    token = os.environ.get("TELEGRAM_FUTURES_BOT_TOKEN") or None
    chat = os.environ.get("TELEGRAM_FUTURES_CHAT_ID") or None
    return token, chat


def enabled() -> bool:
    """Can we send? (token present)."""
    return _load_env()[0] is not None


def two_way_ready() -> bool:
    """Can we send AND attribute replies to you? (token + chat id)."""
    t, c = _load_env()
    return bool(t and c)


def _call(method: str, params: dict, timeout: int = 35) -> dict:
    token, _ = _load_env()
    if not token:
        return {"ok": False, "error": "no token"}
    url = _API.format(token=token, method=method)
    data = urllib.parse.urlencode(params).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def send(text: str) -> bool:
    """One-way notice to your chat."""
    _, chat = _load_env()
    if not chat:
        return False
    return _call("sendMessage", {"chat_id": chat, "text": text}).get("ok", False)


def _latest_offset() -> int:
    res = _call("getUpdates", {"timeout": 0}).get("result", [])
    return max((u["update_id"] for u in res), default=0)


def parse_reply(text: str) -> Optional[dict]:
    """Parse a free-form reply into a decision, or None if nothing recognized."""
    words = {w.strip(".,!").lower() for w in (text or "").split()}
    if not words & set(VALID):
        return None
    if "skip" in words:
        return {"action": "skip"}
    size = "big" if "big" in words else ("small" if "small" in words else "small")
    timing = "wait" if "wait" in words else "now"
    return {"action": "enter", "size": size, "timing": timing}


def ask(text: str, timeout_s: int = 300) -> Optional[dict]:
    """Send `text`, then long-poll for YOUR reply. Returns the parsed decision, or
    None on timeout (caller should treat None as 'skip — no answer in time')."""
    token, chat = _load_env()
    if not (token and chat):
        return None
    baseline = _latest_offset()
    send(text)
    deadline = time.time() + timeout_s
    offset = baseline + 1
    while time.time() < deadline:
        poll = min(30, max(1, int(deadline - time.time())))
        res = _call("getUpdates", {"timeout": poll, "offset": offset}, timeout=poll + 5).get("result", [])
        for u in res:
            offset = u["update_id"] + 1
            m = u.get("message") or u.get("edited_message") or {}
            if str(m.get("chat", {}).get("id")) != str(chat):
                continue                       # only accept YOUR chat
            decision = parse_reply(m.get("text", ""))
            if decision:
                return decision
    return None


def macro_prompt(setup: str, instrument: str, direction: str, entry: float,
                 stop: float, big_tp: float, big_ct: int,
                 small_tp: float, small_ct: int) -> str:
    """Standard ORB macro message body."""
    return (
        f"\U0001F305 {setup} {direction} — {instrument} @ {entry:.2f}\n"
        f"stop {stop:.2f}\n"
        f"big  = {big_ct}ct, TP {big_tp:.2f}\n"
        f"small= {small_ct}ct, TP {small_tp:.2f}\n\n"
        f"Reply: big / small / now / wait / skip"
    )
