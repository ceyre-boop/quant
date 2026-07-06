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
        print(f"[sentiment] updating  (backfill={args.backfill}, macro_start={start})")
        news_cov = news_feed.update(con=con)
        gdelt_cov = gdelt_feed.update(con=con)            # GDELT texture (5s-rate-limited; ~2017+)
        macro_cov = macro_feed.update(con=con, start=start)
        vix_cov = vix_feed.update(con=con, start=start)
        surprise_cov = surprise_feed.update(con=con, start=start)  # release-innovation spine
        cot_cov = cot_feed.update(con=con)                         # CFTC COT positioning (Friday-published)
        vrp_cov = vrp_feed.update(con=con)                         # VRP feature (ThetaData FX-ETF options)
        surf_cov = options_surface_feed.update(con=con, fixture=args.fixture)  # RR25/BF25/term structure
        board_rows = board_state.rebuild(con=con)

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
        return {"board_rows": board_rows, "news": news_cov, "gdelt": gdelt_cov, "macro": macro_cov,
                "vix": vix_cov, "surprise": surprise_cov, "cot": cot_cov, "vrp": vrp_cov,
                "options_surface": surf_cov, "vrp_board_pct": board_pct, "news_probe": news_probe,
                "look_ahead_violations": la_viol}
    finally:
        con.close()


if __name__ == "__main__":
    main()
