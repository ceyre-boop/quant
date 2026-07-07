# Health diagnosis — 2026-07-06 (H1/H4, diagnose-before-fix per TICK-008)

Read-only scout + synthesis. Root cause named per organ; fixes are TICKETED, not
applied unilaterally (live-organ + RED-1-family boundaries).

## The Day-2 "dead trifecta" is ALIVE — the picture inverted

| Job | Day-2 reading | Today's evidence | Verdict |
|---|---|---|---|
| health.responder | dead since Jun-14 | log fresh (30-min cadence, Jul-6 20:57) | HEALTHY — Day-2 staleness resolved (Jul-1 plist restoration + subsequent reload/reboot took effect) |
| hypothesis.generator | stale Jun-15 | fired Jul-6 03:06, no errors | HEALTHY |
| research.factory | stale Jun-15 | outputs Jul-6 20:01; `config/autonomous.yml::live=false` | HEALTHY — correctly idle-by-design (dry-run) |
| oracle.reflect | stale Jun-28 (yfinance errors) | oracle_cycle.log Jul-6 04:04, completes (numpy warnings only) | HEALTHY |
| ict_scanner | "silent 2h" URGENT Jul-5 22:50 | heartbeat fresh Jul-6 21:26; Jul-5 22:50 UTC = Sunday post-open → alarm was CORRECT per its market-hours mask | HEALTHY; alarm legitimate |

**Genuine faults (2):**
1. **stray_tripwire WatchPaths INERT since Jun-16** — plist loaded, script exits 0
   manually, but the launchd filesystem watch on `/Users/taboost/quant` has not fired
   through three weeks of commits. Fix (TICK-008 batch, Colin applies): unload/reload
   the plist, or replace WatchPaths with a 15-min StartInterval fallback.
2. **OUTCOME_LOOP_STALL (Jul-5 22:50 URGENT) = hybrid artifact, verdict (a)+(c):**
   - Monitor: `sovereign/oracle/pulse_check.py::_backfill_decision_outcomes` (runs
     inside oracle_cycle).
   - **MAJOR component — probe overcount:** ~7 of the 21 "closed OANDA trades" are
     AUD_NZD/USD_CAD probe/sentinel fills (the invariant guard's I2/I3 class) that
     never had decision records; pulse_check does NOT pre-filter forbidden pairs
     before matching/alarming.
   - **Minor component — genuine timestamp skew:** signal-time (day N) vs fill-time
     (day N+1) falls outside the Tier-2 same-UTC-date window (evidence:
     AUDUSD Jun-15 12:00 fill vs Jun-16 12:01 decision). The Jul-1 matcher did NOT
     regress; the window is simply one day too narrow for signal-dated entries.
   - Decision-log truth: 44 Jul records, 43 with outcomes, 1 legitimately OPEN.
   - **Fixes (named, NOT applied):** (i) pre-filter closed trades against the
     invariants-spec forbidden-pair list before match/alarm; (ii) widen the match
     fallback to ±1 UTC day / adjacent month. Both live in the same contamination
     family as the RED-1 Blue change (`reflect_cycle` source-exclusion) →
     **one Colin review batch ships all three** (this doc + Blue-Team note are the
     review package). Until then the alarm keeps over-counting — known, tolerable,
     honest.

## Quick facts
- Loaded invariant-guard plist is byte-identical to the tracked copy (sha 5e1905cf…).
- `com.alta.sentiment_update` remains the ONLY unloaded organ (no logs exist — never
  fired). Colin's one-liner stands: `cp scripts/com.alta.sentiment_update.plist
  ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.alta.sentiment_update.plist
  && python3 scripts/plist_watchdog.py --rebaseline "loaded sentiment_update"`.
- TICK-008's remaining build scope (staleness engine with per-job deadlines + weekend
  mask + /tmp log redirects diff) stays ticketed — today's evidence says it guards
  against RECURRENCE, not a current outage; priority drops accordingly.
