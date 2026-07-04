# R1 — Sensor retargeting audit (2026-07-03, read-only; DB locked by vrp_feed during scan)

## GDELT state (E3-critical)
- Feeder: sovereign/sentiment/gdelt_feed.py — free keyless DOC API, **5s pacing enforced
  (gdelt_feed.py:48; config parameters.yml:185)**, 3 retries, coverage config 2017→present,
  4 pairs × 2 modes (~40s wall-clock per full pass; refetches full range each run).
- **No throttle errors in logs** (no 429/503 anywhere). The 2026-06-30 failure was an
  unpaced 8-call burst — the paced path has not been observed failing.
- Cheapest unblock: plain `python3 scripts/update_sentiment.py` after the vrp_feed lock
  frees (idempotent delete-then-upsert). If retry-3 exhausts: board carries NULL
  honestly; re-run later. NULL-count verification pending DB unlock.
- Board table: sentiment_board_state (27 required columns, store.py:158-180).

## Scheduler finding (headline)
`scripts/com.alta.sentiment_update.plist` (Mon–Fri 07:45) EXISTS but is NOT loaded
(absent from launchctl + ~/Library/LaunchAgents). Heartbeat last 2026-06-30 13:29 —
**the sensory board only updates when a human runs it.** 3 days stale at audit time.

## Sensor verdicts
- [FRED macro_feed] → 3 board columns live, 2015→present · LEAVE
- [GDELT] → 3 board columns + HYP-080 substrate · add resume-cursor (1-2 lines
  gdelt_feed.py:80) to stop full-range refetch each run · RETARGET (efficiency only)
- [AV NEWS_SENTIMENT news_feed] → news_score live; 48h rolling window is design;
  25/day quota accepted; premium = the upgrade path · LEAVE
- [Reddit + com.alta.cache.refresh] → hourly LOADED job → 0 posts (HTTP 403, no creds),
  0 board join, 0 consumers · ATTIC-CANDIDATE (Colin: job keep/kill — we touch nothing)
- [AV technical endpoints] → dead key, no feeder · ATTIC-CANDIDATE (.env hygiene)
- [Tiingo] → dead key (deliberately excluded for data quality) · ATTIC-CANDIDATE
- [Polygon] → not a sentiment sensor (only clawd_trading GEX) · ATTIC-CANDIDATE for
  sentiment .env; belongs to the equity-engine ruling batch
- [ThetaData surface+vrp feeds] → 7 board columns, graceful-skip design, prereg-frozen ·
  LEAVE
- [OANDA] → execution/fills layer, not a sensor — removed from this audit's scope
- [OpenWeather] → dead key, zero usage · ATTIC-CANDIDATE
- [Firebase] → keys present, zero sentiment consumption (and the fed page is dead per
  06-30 audit) · ATTIC-CANDIDATE

## Board columns with no hypothesis
NONE — all 27 required columns map to preregistered hypotheses or the macro/VIX regime
backbone. (Honest negative: the board is NOT over-built.)

## Headlines
1. The sensory organ has no live schedule — the sentiment_update plist was never
   loaded; the board is 3 days stale. Cheapest rewiring in the whole audit: install
   the existing plist (1 launchctl load, additive job).
2. GDELT is likely NOT blocked — paced path never observed failing; the "blocker" was
   one unpaced burst. Verify NULLs after today's rebuild.
3. Five dead credentials (.env hygiene) + one ornamental loaded job (Reddit refresh)
   → Colin's ruling list; no board column is orphaned.
