"""Frozen signal parameters for the execution harness, with a hash lock.

WHY THIS FILE EXISTS
--------------------
Until now the frozen thresholds lived as module constants in two forked shadow
scripts (`research/gapper/hyp107_shadow.py:38-39` and
`research/yield_frontier/live_shadow.py:34-37`). Module constants are editable
without ceremony, which is the wrong property for values that are supposed to be
pre-registered and unchangeable. Here they are single-sourced and sha256-locked:
`verify_frozen_hash()` runs before any network I/O, so drifting a threshold
without a matching `data/agent/param_change_log.jsonl` entry is a hard startup
failure rather than a silent re-fit.

PROVENANCE — every value below is copied VERBATIM from the sealed source.
  HYP-107 : research/gapper/hyp107_shadow.py (frozen at commit 48303cd)
  HYP-093 : research/yield_frontier/live_shadow.py (prereg c5b10616)

Do not "tidy" these numbers. They are not tunable.

NOTE ON THE HYP-093 GAIN FLOOR
------------------------------
The frozen filter is `gain >= 0.50` AND `price >= 1.30 * prev_close`. Task briefs
have described this as "still >= 100% above prior close" — that is NOT the sealed
spec and would be a materially tighter filter producing a different event set.
The prereg governs.
"""
from __future__ import annotations

import copy
import hashlib
import json


class FrozenConfigError(RuntimeError):
    """Raised when the frozen parameter block does not match its recorded hash."""


# ── The frozen block ──────────────────────────────────────────────────────────
FROZEN: dict = {
    "hyp107": {
        # research/gapper/hyp107_shadow.py:38-43 — commit 48303cd
        "og_max": 0.577,          # overnight_gap = 09:30 open / prev_close - 1
        "logvol_max": 5.854,      # log10(09:30-minute volume + 1)
        "gap_floor": 0.30,        # honest pre-selection: gapped >= 30% at open
        "stop_pct": 0.25,         # descriptive only (would-a-stop-have-hit flag)
        "movers_pct_change_min": 30.0,   # NOTE: hyp107 uses 30, hyp093 uses 40
        "filter_bar_et": "09:30",        # first-minute bar the filter reads
        "entry_bar_et": "09:31",         # entry = OPEN of this bar
        "exit_bar_et": "10:30",          # exit = CLOSE of this bar
        "side": "LONG",
        "baseline_median": 0.054,        # backtest holdout gross median
        "source_tag": "shadow_hyp107",
    },
    "hyp093": {
        # research/yield_frontier/live_shadow.py:34-37 — prereg c5b10616
        "gain_min": 0.50,         # P / prev_close - 1 >= 0.50
        "qual_gain": 1.30,        # P >= 1.30 * prev_close
        "price_min": 2.00,
        "vol_min": 500_000,       # cumulative 09:30-10:25 share volume
        "stop_mult": 1.30,        # short stop at entry * 1.30
        "slip": 0.005,
        "locate_w": 0.50,
        "notional_w": 0.0125,
        "borrow_apr": {"0.5": 2.00, "1.0": 4.00, "1.5": 6.00},
        "movers_pct_change_min": 40.0,
        "measure_start_et": "09:30",
        "measure_end_et": "10:25",       # P = close of last bar <= 10:25
        "min_bars": 8,
        "min_last_bar_et": "10:15",
        "entry_bar_et": "10:30",         # entry = OPEN of this bar
        "exit_bar_et": "15:45",
        "side": "SHORT",
        "source_tag": "shadow_gapper",
    },
    "screener": {
        "movers_top": 50,
        "max_symbol_len": 5,
        "excluded_last_chars": "WRU",    # warrants / rights / units
        "buckets_mna": [
            "merger", "acquisition", "acquire", "buyout", "takeover",
            "definitive agreement", "letter of intent",
            "strategic alternatives", "going private",
        ],
    },
    "capture": {
        # SIP quote recency boundary measured empirically 2026-07-18:
        #   -13 min -> HTTP 403 "subscription does not permit querying recent SIP data"
        #   -16 min -> HTTP 200
        # Deferred capture must therefore sit beyond 15 minutes. 16 is the
        # measured boundary; 1 extra minute of margin absorbs clock skew.
        "sip_lag_minutes": 17,
        "quote_window_seconds": 30,      # search back this far for a quote at ts
        "max_quote_age_seconds": 5.0,    # older than this at a decision instant
        "wide_quote_pct": 0.25,          # flag (do NOT drop) above this spread
        "stale_clean_lookback_seconds": 10,
    },
}

# sha256 of json.dumps(FROZEN, sort_keys=True, separators=(",", ":"))
FROZEN_HASH = "66907c7906d7212ebcf3b5ec6772bb30c380207a153be5b675deb965790612a3"


def compute_hash(block: dict | None = None) -> str:
    """sha256 over the canonical JSON encoding of the frozen block."""
    payload = json.dumps(block if block is not None else FROZEN,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_frozen_hash() -> None:
    """Raise FrozenConfigError if FROZEN has drifted from its recorded hash.

    Called at harness startup BEFORE any network I/O. A mismatch means someone
    edited a pre-registered threshold; the remedy is to revert the edit, not to
    update the hash. If the change is intentional, it requires a
    data/agent/param_change_log.jsonl entry first (CLAUDE.md NON-NEGOTIABLE #4).
    """
    actual = compute_hash()
    if actual != FROZEN_HASH:
        raise FrozenConfigError(
            "Frozen parameter drift detected.\n"
            f"  expected: {FROZEN_HASH}\n"
            f"  actual:   {actual}\n"
            "A pre-registered threshold changed. Revert the edit — do not update "
            "FROZEN_HASH to match. Intentional changes require a "
            "data/agent/param_change_log.jsonl entry first."
        )


def frozen(leg: str) -> dict:
    """Return a deep copy of one frozen sub-block ('hyp107' | 'hyp093' | ...)."""
    if leg not in FROZEN:
        raise KeyError(f"unknown frozen block {leg!r}; have {sorted(FROZEN)}")
    return copy.deepcopy(FROZEN[leg])


if __name__ == "__main__":   # `python -m execution.config` prints the live hash
    print(compute_hash())
