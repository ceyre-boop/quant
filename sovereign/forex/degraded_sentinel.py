"""
Fail-loud DEGRADED sentinel for live forex data-fetch fallbacks (TICK-025).

When a live yfinance OHLCV fetch fails, the carry/Oracle scan silently falls
back to a degraded input (a dropped pair, a stub ATR of 0.001) and keeps
running — so conviction is computed on bad data with no surface indication that
anything is wrong ("the system is wrong when it silently succeeds"). This helper
makes that failure first-class: it drops a sentinel file under ``sentinel/`` and
logs at WARNING every time a fallback fires.

Observability ONLY. It never changes what the caller selects or returns, never
raises (a monitoring side-effect must not break the trading path), and is not
imported by the backtester or the v015 reconcile anchor.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo-root/sentinel — sibling of data/ and logs/. Absolute, cwd-independent, so
# it resolves the same whether launched by launchd or run by hand.
# parents[2] = <repo>/sovereign/forex/.. -> <repo>
SENTINEL_DIR = Path(__file__).resolve().parents[2] / "sentinel"


def _safe_pair(pair: str) -> str:
    """yfinance ticker -> filename-safe pair symbol ('USDJPY=X' -> 'USDJPY')."""
    cleaned = (pair or "UNKNOWN").upper().replace("=X", "")
    return re.sub(r"[^A-Z0-9]", "", cleaned) or "UNKNOWN"


def flag_degraded(pair: str, reason: str, source: str = "yfinance") -> None:
    """Record a DEGRADED data-input event: WARNING log + sentinel file.

    Writes ``sentinel/DEGRADED_{source}_{pair}.txt`` (overwriting any prior flag
    for the same pair so the file always holds the most recent failure) and logs
    the pair + fallback source at WARNING level. Best-effort and exception-safe
    by contract.
    """
    ts = datetime.now(timezone.utc).isoformat()
    logger.warning(
        "DEGRADED %s data for %s — falling back to synthetic/degraded input: %s",
        source, pair, reason,
    )
    try:
        SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
        path = SENTINEL_DIR / f"DEGRADED_{source}_{_safe_pair(pair)}.txt"
        path.write_text(
            f"timestamp={ts}\n"
            f"pair={pair}\n"
            f"source={source}\n"
            f"reason={reason}\n"
        )
    except Exception as exc:  # never let monitoring break the pipeline
        logger.warning("Failed to write DEGRADED sentinel for %s: %s", pair, exc)
