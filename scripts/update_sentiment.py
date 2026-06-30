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
from sovereign.sentiment import store, news_feed, macro_feed, vix_feed, gdelt_feed, surprise_feed, board_state

HEARTBEAT = ROOT / "logs" / ".heartbeat_sentiment"


def main() -> dict:
    ap = argparse.ArgumentParser(description="Sentiment board-state daily updater")
    ap.add_argument("--backfill", action="store_true", help="pull full FRED/VIX history from sentiment.macro_start")
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
        print(f"BOARD: {board_rows} rows (sentiment_board_state)")
        print(f"DB: {store.DB_PATH}")
        return {"board_rows": board_rows, "news": news_cov, "gdelt": gdelt_cov, "macro": macro_cov,
                "vix": vix_cov, "surprise": surprise_cov, "news_probe": news_probe}
    finally:
        con.close()


if __name__ == "__main__":
    main()
