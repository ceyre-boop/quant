"""
DRAFT — ADR Early-Session Filter for the ICT pipeline
=====================================================
STATUS: DRAFT ONLY. Not wired to any live code. Not imported anywhere.
        Do NOT apply until the validation + gate conditions at the bottom of
        this header are met.

WHAT THIS IS
------------
A *pre-session* ADR (Average Daily Range) exhaustion filter. The existing gate
in `ict/pipeline.py` (`ICTPipeline._adr_exhaustion`, lines ~767–813, fired at
~line 249) vetoes a single setup AFTER the whole pipeline has been evaluated for
both directions. This draft proposes a cheaper, earlier check: if the day's
range is already mostly consumed *at the moment the NY PM session opens*, skip
the entire pipeline evaluation for that pair for the rest of the session.

WHY (the data that motivates it)
--------------------------------
  • 54% of ICT vetoes are ADR exhaustion.
  • 75% of those fire at 14:00–15:00 ET — i.e. AFTER the NY PM open (13:30 ET).
  • Average ADR consumed when vetoed: 161% (the hard veto fires at ~85%).
  → By the time NY PM opens, the range is frequently already blown. Evaluating
    the full pipeline only to veto at the very end is wasted work, and every
    such pass also burns a yfinance call + logs a veto that drowns the ledger.

EXPECTED IMPACT
---------------
  • Cuts ~75% of ADR-exhausted NY PM vetoes per pair (the 14:00–15:00 ET cluster)
    by short-circuiting BEFORE the per-direction pipeline.evaluate() loop runs.
  • Same protective effect as today's hard veto, just earlier and once-per-pair
    instead of twice (LONG + SHORT) per scan.

RISK
----
  • Threshold too aggressive ⇒ skips days where the range legitimately extends
    LATE (London→NY continuation, high-impact NY data prints). Those are exactly
    the days the live gate already loosens by 1.10× during London/NY_PM overlap
    (see pipeline.py:252). This draft defaults to a CONSERVATIVE 0.80 (below the
    0.85 hard veto) measured *as of session open only*, so a late expansion after
    the open is NOT counted against the pair — it can still trade.
  • Fail-open by design: any data error / insufficient history ⇒ should_skip=False.
    A bad yfinance pull must never silently kill a tradeable session.

SUGGESTED VALIDATION (before wiring live)
-----------------------------------------
  Backtest against the last 60 days of the veto ledger:
    data/ledger/ict_veto_ledger_2026_06.jsonl   (raw vetoes)
    data/ledger/ict_veto_outcomes_2026_06.jsonl (first-touch ±1R outcomes,
                                                  from scripts/label_veto_outcomes.py)
  For each ADR-exhaustion veto, replay this filter at the NY PM open and confirm:
    (a) it would have skipped the same setups the live gate vetoed (true positives), and
    (b) it does NOT skip any veto whose labeled outcome would have been a WIN
        (false-positive skips = lost edge — this is the number to minimize).
  Sweep threshold ∈ {0.75, 0.80, 0.85} and report skip-rate vs. missed-winner rate.

DO NOT APPLY UNTIL (gate — same rule as the sentiment gate)
-----------------------------------------------------------
  • ICT has 10+ CLOSED trades on record (statistical floor before changing gates).
  • A logged param_change rationale exists (CLAUDE.md NON-NEGOTIABLE #4:
    "No live parameter changes without logging").
  • The validation backtest above PASSes (missed-winner rate acceptable).

CALCULATION PARITY NOTE
-----------------------
This mirrors `ICTPipeline._adr_exhaustion` exactly EXCEPT the numerator:
  • ADR denominator: mean of the last 20 daily ranges EXCLUDING today
    (pipeline.py: `daily_ranges.iloc[-_ADR_LOOKBACK:-1].mean()`).
  • Numerator (the ONLY difference): the existing gate uses today's FULL range;
    this draft uses today's range built from intraday bars *up to and including
    session_open_time only* — i.e. how much was consumed BEFORE NY PM opened.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, date
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Mirror the live constants from ict/pipeline.py:82-84 so this draft stays in
# lockstep with the real gate. If those change, change these (or, on wiring,
# import them / read config/ict_params.yml::pipeline instead of redefining).
_ADR_LOOKBACK = 20            # days for ADR computation (pipeline.py:84)
_ADR_HARD_VETO = 0.85         # live hard-veto threshold (pipeline.py:82)
_DEFAULT_PRE_SESSION_THRESHOLD = 0.80  # this filter's default — deliberately
#                                        below the 0.85 hard veto: if 80% of the
#                                        range is already gone *before* the
#                                        session even opens, NY PM has no fuel.


def _fetch_intraday(pair: str) -> pd.DataFrame:
    """
    Fetch the trailing intraday OHLCV for `pair`, normalized to match the
    orchestrator's DataFrame shape (capitalized columns, UTC DatetimeIndex).

    Mirrors ict/orchestrator.py:230-235 exactly so a df produced there can be
    passed straight into check_pre_session_adr() without re-fetching.
    """
    import yfinance as yf  # local import — keeps this draft importable w/o yf

    df = yf.download(pair, period="5d", interval="1h", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.capitalize)[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def check_pre_session_adr(
    pair: str,
    session_open_time: datetime,
    threshold: float = _DEFAULT_PRE_SESSION_THRESHOLD,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[float, bool]:
    """
    Decide whether the ADR is already too consumed at `session_open_time` to be
    worth scanning `pair` this session.

    Parameters
    ----------
    pair : str
        Symbol as used by the orchestrator (e.g. "GBPUSD=X"). Used only for the
        self-fetch path and for logging.
    session_open_time : datetime
        The session open instant (NY PM = 13:30 ET = 17:30 UTC). Consumption is
        measured from intraday bars with timestamp <= this value. If tz-naive,
        it is assumed to be UTC.
    threshold : float
        Skip if consumed_pct >= threshold. Defaults to 0.80.
    df : pd.DataFrame, optional
        Pre-fetched trailing intraday OHLCV (capitalized cols, UTC index). When
        omitted the function fetches its own via _fetch_intraday(pair). Pass the
        orchestrator's already-downloaded df to avoid a duplicate yfinance call.

    Returns
    -------
    (consumed_pct, should_skip)
        consumed_pct : float  — today's range as of session_open_time / 20-day ADR
                                 (0.0 on any data error — fail-open).
        should_skip  : bool   — True iff consumed_pct >= threshold AND the
                                 computation was valid. Always False on error /
                                 insufficient data (never skip on bad data).
    """
    # --- normalize session_open_time to tz-aware UTC -----------------------
    if session_open_time.tzinfo is None:
        session_open_time = session_open_time.replace(tzinfo=timezone.utc)
    else:
        session_open_time = session_open_time.astimezone(timezone.utc)
    session_date: date = session_open_time.date()

    # --- get data (fail-open on any problem) -------------------------------
    if df is None:
        df = _fetch_intraday(pair)
    if df is None or df.empty or len(df) < 2:
        logger.debug("pre-session ADR[%s]: insufficient data — not skipping", pair)
        return 0.0, False

    try:
        work = df.copy()
        work.index = pd.to_datetime(work.index, utc=True)

        # --- ADR denominator: mean of last 20 daily ranges EXCLUDING today --
        # (identical to pipeline.py:787-798)
        daily_hi = work["High"].resample("D").max().dropna()
        daily_lo = work["Low"].resample("D").min().dropna()
        daily_ranges = (daily_hi - daily_lo).dropna()
        if len(daily_ranges) < 2:
            logger.debug("pre-session ADR[%s]: insufficient daily history", pair)
            return 0.0, False

        adr = float(daily_ranges.iloc[-_ADR_LOOKBACK:-1].mean())
        if adr <= 0:
            logger.debug("pre-session ADR[%s]: ADR <= 0", pair)
            return 0.0, False

        # --- numerator: today's range AS OF session_open_time only ----------
        # THIS is the only departure from the live gate. We restrict to bars on
        # session_date whose timestamp is at/before the open, so a range that
        # expands AFTER the open is not held against the pair.
        today_mask = (work.index.date == session_date) & (work.index <= session_open_time)
        today_bars = work.loc[today_mask]
        if today_bars.empty:
            # No bars yet today at/before the open → nothing consumed → trade.
            logger.debug("pre-session ADR[%s]: no pre-open bars — not skipping", pair)
            return 0.0, False

        today_rng = float(today_bars["High"].max() - today_bars["Low"].min())
        consumed_pct = round(today_rng / adr, 3)

        should_skip = consumed_pct >= threshold
        logger.info(
            "pre-session ADR[%s]: %.0f%% consumed at %s (ADR=%.5f, thr=%.0f%%) → %s",
            pair, consumed_pct * 100, session_open_time.isoformat(),
            adr, threshold * 100, "SKIP" if should_skip else "scan",
        )
        return consumed_pct, should_skip

    except Exception as e:  # fail-open: a compute error must not kill the session
        logger.debug("pre-session ADR[%s] computation failed: %s — not skipping", pair, e)
        return 0.0, False


# =============================================================================
# PROPOSED INSERTION POINT — ict/orchestrator.py
# =============================================================================
# Target: ICTOrchestrator.scan_once(), inside `for pair in self.pairs:`.
# Insert AFTER the data-sufficiency guard (line 239, the `continue`) and BEFORE
# `atr = compute_atr(df)` (line 241). This short-circuits the pair BEFORE the
# expensive per-direction pipeline.evaluate() loop (orchestrator.py:244-257),
# which is the whole point — skip the pipeline entirely, not just veto at the end.
#
# `df`, `pair`, `clean`, and `now` are all already in scope at this point, and
# `df` is already normalized (capitalized cols, UTC index) — so we pass it in to
# avoid a second yfinance download.
#
# ---------------------------------------------------------------------------
# BEFORE (ict/orchestrator.py:237-241, exact current text):
# ---------------------------------------------------------------------------
#   237                if len(df) < 30:
#   238                    logger.warning("%s: insufficient data", clean)
#   239                    continue
#   240
#   241                atr     = compute_atr(df)
#
# ---------------------------------------------------------------------------
# AFTER (insert the new block at line 240, between the `continue` and `atr=`):
# ---------------------------------------------------------------------------
#   237                if len(df) < 30:
#   238                    logger.warning("%s: insufficient data", clean)
#   239                    continue
#   240
#   240+               # ── Pre-session ADR filter (DRAFT — gated; see drafts/) ──
#   240+               # Skip the whole pipeline for this pair if the day's range
#   240+               # was already exhausted by the time NY PM opened (17:30 UTC).
#   240+               # Only applies to the NY PM session; London is untouched.
#   240+               if session.kill_zone_name == 'NY_PM':
#   240+                   ny_pm_open = now.replace(hour=17, minute=30, second=0,
#   240+                                            microsecond=0)
#   240+                   consumed_pct, skip = check_pre_session_adr(
#   240+                       pair, ny_pm_open, threshold=0.80, df=df,
#   240+                   )
#   240+                   if skip:
#   240+                       logger.info("%s: pre-session ADR %.0f%% consumed "
#   240+                                   "at NY PM open — skipping scan",
#   240+                                   clean, consumed_pct * 100)
#   240+                       # NOTE on wiring: record this as a skip in the veto
#   240+                       # ledger (ict/ict_veto_ledger.py) with a directional
#   240+                       # entry/stop so it is LABELABLE — 72% of current gate
#   240+                       # vetoes are unlabelable because they lack direction
#   240+                       # (see memory: Veto Outcome Labeling). Then `continue`.
#   240+                       continue
#   241                atr     = compute_atr(df)
#
# ---------------------------------------------------------------------------
# IMPORT (top of ict/orchestrator.py, alongside the other ict imports):
#   from scripts.drafts.adr_session_filter_draft import check_pre_session_adr
# On promotion this function should move into ict/pipeline.py (next to
# _adr_exhaustion) or a small ict/adr_filter.py module, NOT live under scripts/.
# It reads no sovereign/ state, so it is ICT-isolation-safe either way
# (CLAUDE.md NON-NEGOTIABLE #1).
# =============================================================================


if __name__ == "__main__":
    # Tiny self-check harness (DRAFT). Run:  python3 scripts/drafts/adr_session_filter_draft.py
    # Builds a synthetic 21-day hourly frame where today consumes 90% of ADR
    # before the open, and asserts the filter would skip at threshold 0.80.
    import numpy as np

    logging.basicConfig(level=logging.INFO)

    idx = pd.date_range("2026-06-09", periods=21 * 24, freq="h", tz="UTC")
    # 20 prior days each with a clean 1.0000-unit range, then "today" partial.
    base = pd.DataFrame(index=idx)
    base["Open"] = 100.0
    base["Close"] = 100.0
    # Per-day high/low giving a ~0.0100 daily range for the prior 20 days.
    day_of = pd.Series(idx.date, index=idx)
    base["High"] = 100.0050
    base["Low"] = 99.9950
    # Make "today" (the last calendar day) already span 0.0090 (~90% of 0.0100 ADR)
    today = sorted(set(idx.date))[-1]
    open_t = datetime(today.year, today.month, today.day, 17, 30, tzinfo=timezone.utc)
    todays_pre_open = (day_of == today) & (idx <= open_t)
    base.loc[todays_pre_open, "High"] = 100.0090
    base.loc[todays_pre_open, "Low"] = 100.0000

    pct, skip = check_pre_session_adr("TEST=X", open_t, threshold=0.80, df=base)
    print(f"\nself-check: consumed={pct:.0%}  should_skip={skip}")
    assert skip is True, "expected skip when ~90% of ADR consumed before open"
    print("self-check PASSED (draft logic only — NOT a live validation)")
