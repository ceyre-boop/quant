#!/usr/bin/env python3
"""scripts/update_sentiment.py — daily runner for the sentiment board-state pipeline.

Refreshes the three feeders (NewsAPI rolling-24h, FRED macro from 2015, VIX from 2015) and rebuilds the
fused per-pair board, all through ONE shared DuckDB connection. Idempotent — safe to call twice (every
write is a delete-then-insert keyed upsert; the board is a full rebuild).

    python3 scripts/update_sentiment.py              # daily refresh + rebuild + coverage report
    python3 scripts/update_sentiment.py --backfill   # same, explicitly pulling full FRED/VIX history

Scheduled alongside forex_live_scan via scripts/com.alta.sentiment_update.plist (Mon–Fri).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for lib in ("yfinance", "peewee", "urllib3", "requests", "fredapi"):
    logging.getLogger(lib).setLevel(logging.ERROR)

from config.loader import params
from sovereign.sentiment import (
    store, news_feed, macro_feed, vix_feed, gdelt_feed, surprise_feed, cot_feed, vrp_feed,
    options_surface_feed, board_state,
)

HEARTBEAT = ROOT / "logs" / ".heartbeat_sentiment"

# ── Per-feed timeout ─────────────────────────────────────────────────────────
# WHY: this job hung indefinitely and starved the thing it exists to produce.
# Measured 2026-07-18 with the DB lock free:
#     news 5.8s · macro 3.0s · vix 0.6s · surprise 5.3s · cot 4.0s   (~19s total)
#     gdelt >90s · vrp >90s · surface >90s                            (unbounded)
# One observed run survived 1h47m still holding an EXCLUSIVE DuckDB write lock,
# which also blocks every reader of sentiment.db for the whole session.
#
# The three hangers are known-degraded upstream, not transient: gdelt has ingested
# ZERO rows in its lifetime (burst-throttled free tier) and vrp/surface both need
# a ThetaTerminal that is down. Their absence is already modelled downstream — the
# board carries NULL vrp_*/rr25/bf25 and says so in the coverage report.
#
# So each feed gets a hard wall-clock budget. A feed that exceeds it is abandoned,
# logged loudly, and the pipeline CONTINUES — because board_state.rebuild() at the
# end is the output the 08:00 scan actually reads, and a fresh board missing three
# degraded features beats no board at all. Feeds are idempotent delete-then-insert
# upserts, so an interrupted feed is repaired by the next run rather than corrupting.
FEED_TIMEOUT_S = 60


class FeedTimeout(BaseException):
    """A feed exceeded its wall-clock budget and was abandoned.

    Inherits BaseException, NOT Exception, and that is load-bearing.

    `sovereign/sentiment/gdelt_feed.py:58` retries inside `except Exception as exc:`.
    A timeout raised as an Exception is caught by that handler and the retry loop
    simply continues — which is exactly what happened on the first attempt at this
    fix: the alarm fired, the feed swallowed it, and the job still ran past 9
    minutes. Subclassing BaseException makes the timeout uncatchable by ordinary
    handlers, the same reason KeyboardInterrupt and SystemExit do it.
    """


def _run_feed(name: str, fn, timeout_s: int = FEED_TIMEOUT_S):
    """Run one feed under a hard timeout. Never propagates — returns {} on failure.

    Returns (coverage, status) where status is "ok" | "timeout" | "error".
    """
    import signal

    def _alarm(signum, frame):
        raise FeedTimeout(f"{name} exceeded {timeout_s}s")

    prev = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    t0 = time.time()
    try:
        cov = fn()
        return (cov if cov is not None else {}), "ok"
    except FeedTimeout:   # BaseException — deliberately not catchable by the feeds
        print(f"[sentiment] ⚠️  {name}: ABANDONED after {timeout_s}s — feed is "
              f"unbounded, continuing so the board still rebuilds", flush=True)
        return {}, "timeout"
    except Exception as exc:                                 # noqa: BLE001
        print(f"[sentiment] ⚠️  {name}: FAILED after {time.time()-t0:.1f}s — "
              f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)
        return {}, "error"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def main() -> dict:
    ap = argparse.ArgumentParser(description="Sentiment board-state daily updater")
    ap.add_argument("--backfill", action="store_true", help="pull full FRED/VIX history from sentiment.macro_start")
    ap.add_argument("--fixture", action="store_true",
                    help="options surface from data/fixtures/thetadata (LOUD test-only mode — never real)")
    args = ap.parse_args()

    # Heartbeat FIRST (loop_health monitoring), before any network call.
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(datetime.now(timezone.utc).isoformat())

    start = params["sentiment"].get("macro_start", "2015-01-01")
    con = store.connect()
    try:
        print(f"[sentiment] updating  (backfill={args.backfill}, macro_start={start}, "
              f"feed_timeout={FEED_TIMEOUT_S}s)")
        t_start = time.time()
        feed_status: dict[str, str] = {}

        news_cov, feed_status["news"] = _run_feed(
            "news", lambda: news_feed.update(con=con))
        gdelt_cov, feed_status["gdelt"] = _run_feed(
            "gdelt", lambda: gdelt_feed.update(con=con))
        macro_cov, feed_status["macro"] = _run_feed(
            "macro", lambda: macro_feed.update(con=con, start=start))
        vix_cov, feed_status["vix"] = _run_feed(
            "vix", lambda: vix_feed.update(con=con, start=start))
        surprise_cov, feed_status["surprise"] = _run_feed(
            "surprise", lambda: surprise_feed.update(con=con, start=start))
        cot_cov, feed_status["cot"] = _run_feed(
            "cot", lambda: cot_feed.update(con=con))
        vrp_cov, feed_status["vrp"] = _run_feed(
            "vrp", lambda: vrp_feed.update(con=con))
        surf_cov, feed_status["surface"] = _run_feed(
            "surface", lambda: options_surface_feed.update(con=con, fixture=args.fixture))

        # ALWAYS rebuild — this is the artifact the 08:00 forex scan reads, and it
        # must not be hostage to a degraded optional feed.
        board_rows = board_state.rebuild(con=con)
        elapsed = time.time() - t_start
        degraded = [k for k, v in feed_status.items() if v != "ok"]
        print(f"[sentiment] feeds complete in {elapsed:.1f}s — "
              f"{len(feed_status) - len(degraded)}/{len(feed_status)} ok"
              + (f", DEGRADED: {', '.join(degraded)}" if degraded else ""))

        # ── coverage report (required deliverable) ──
        news_probe = news_feed.earliest_article_date()
        print("\n── COVERAGE ─────────────────────────────────────────────")
        print("NEWS (NewsAPI):")
        print(f"   today's per-pair article counts: { {p: c['n_articles'] for p, c in news_cov.items()} }")
        print(f"   earliest-available probe: {news_probe}")
        print("FRED macro:")
        for sid, c in macro_cov.items():
            span = f"{c.get('start')}→{c.get('end')}" if c.get("rows") else "NO DATA"
            print(f"   {sid:14} rows={c.get('rows'):>5}  {span}")
        print("VIX:")
        vspan = f"{vix_cov.get('start')}→{vix_cov.get('end')}" if vix_cov.get("rows") else "NO DATA"
        print(f"   ^VIX           rows={vix_cov.get('rows'):>5}  {vspan}")
        print("GDELT texture (tone, ~2017+):")
        for pair, c in gdelt_cov.items():
            span = f"{c.get('start')}→{c.get('end')}" if c.get("rows") else "NO DATA"
            print(f"   {pair:14} rows={c.get('rows'):>5}  {span}")
        print("ECON SURPRISE (release_innovation — NOT consensus):")
        for sid, c in surprise_cov.items():
            if sid == "_daily":
                print(f"   daily econ_surprise_z rows={c.get('rows'):>5}  {c.get('start')}→{c.get('end')}")
            else:
                print(f"   {sid:14} releases={c.get('releases'):>4}  {c.get('start')}→{c.get('end')}")
        print("COT positioning (CFTC, Friday-published, 1986+):")
        for pair, c in cot_cov.items():
            span = f"{c.get('start')}→{c.get('end')}" if c.get("rows") else "NO DATA"
            print(f"   {pair:14} weeks={c.get('rows'):>5}  {span}")
        print("VRP (ThetaData FX-ETF options, weekly iv_atm−rv_trailing):")
        if not vrp_cov:
            print("   NOT REACHABLE — ThetaTerminal down / key unset (feature skipped, board carries NULL vrp_*)")
        for pair, c in vrp_cov.items():
            span = f"{c.get('start')}→{c.get('end')}" if c.get("rows") else f"NO DATA ({c.get('note','')})"
            src = c.get("iv_source", {})
            print(f"   {pair:6} [{c.get('symbol','?'):3}] obs={c.get('rows'):>4}  {span}"
                  f"  iv_src={src}  earliest_exp={c.get('earliest_expiry','?')}")
        board_pct = con.execute(
            "SELECT 100.0*AVG(CASE WHEN vrp_signal IS NOT NULL THEN 1 ELSE 0 END) FROM sentiment_board_state").fetchone()[0]
        print(f"   board rows with vrp_signal: {board_pct:.1f}%")
        print("OPTIONS SURFACE (ThetaData FX-ETF, weekly rr25/bf25/term):")
        if not surf_cov:
            print("   NOT REACHABLE — ThetaTerminal down (feature skipped, board carries NULL rr25/bf25/atm_term_slope)")
        for key, c in surf_cov.items():
            tag = "⚠️ FIXTURE — NOT REAL DATA  " if key.startswith("FIXTURE:") else ""
            span = f"{c.get('start')}→{c.get('end')}" if c.get("rows") else f"NO DATA ({c.get('note','')})"
            print(f"   {tag}{key:14} [{c.get('symbol','?'):3}] obs={c.get('rows'):>4}  {span}"
                  f"  rr25_nonnull={c.get('rr25_nonnull', 0)}")
        print(f"BOARD: {board_rows} rows (sentiment_board_state)")
        # ── automated look-ahead audit (ALFRED standard) — every run, fail loud AND HALT ──
        # L2 SENTINEL: a look-ahead violation means the fused board carries future information.
        # That is a data-integrity failure, not a warning — exiting 0 here lets the scheduler
        # record a silent success and lets downstream (live scan / oracle) consume a
        # contaminated board. On ANY violation we purge the just-rebuilt board so no reader can
        # pick up leaked values, then sys.exit(1) so the run is recorded as FAILED.
        from scripts.audit_look_ahead import audit as _la_audit
        la = _la_audit(con)
        la_viol = sum(r["violations"] for r in la)
        print(f"LOOK-AHEAD AUDIT: {la_viol} violations across "
              f"{sum(r['total'] for r in la)} provenance-checked rows"
              + ("" if la_viol == 0 else "  ⚠️ VIOLATIONS — DO NOT TRUST THE BOARD"))
        if la_viol:
            for r in la:
                if r["violations"]:
                    print(f"   ✗ {r['table']}.{r['check']}: {r['violations']}/{r['total']}")
            # Purge the contaminated board BEFORE exiting so downstream can never read leaked
            # values. The board is a full rebuild each run, so this leaves it empty (fail-safe),
            # never a stale/leaked mix. DuckDB autocommits, so the purge persists on close().
            con.execute("DELETE FROM sentiment_board_state")
            print(f"DB: {store.DB_PATH}")
            print(f"[sentiment] ABORT — {la_viol} look-ahead violation(s) detected; "
                  f"board purged, exiting 1 (L2 sentinel).")
            sys.exit(1)
        print(f"DB: {store.DB_PATH}")
        # ── positioning-board dashboard export (TICK-007 Step 1, additive, display-only) ──
        # Guarded: the export must never fail the feeder run. Reuses this run's already-open
        # `con` (read-write) so it never takes a second lock on the same DuckDB file. Feeds no
        # live gate — Step 2 (dashboard panel) is a separate session.
        try:
            from scripts.export_positioning_board import export as _export_positioning_board
            _pb = _export_positioning_board(con=con)
            print(f"POSITIONING BOARD EXPORT: as_of={_pb['as_of']} stale={_pb['stale']} "
                  f"pairs={len(_pb['pairs'])}")
        except Exception as exc:
            print(f"[sentiment] positioning-board export failed (non-fatal): {exc}")
        return {"board_rows": board_rows, "news": news_cov, "gdelt": gdelt_cov, "macro": macro_cov,
                "vix": vix_cov, "surprise": surprise_cov, "cot": cot_cov, "vrp": vrp_cov,
                "options_surface": surf_cov, "vrp_board_pct": board_pct, "news_probe": news_probe,
                "look_ahead_violations": la_viol}
    finally:
        con.close()


if __name__ == "__main__":
    main()
