# Retargeting master table — 2026-07-03

Seven read-only scouts, one question per organ: what decision does it feed, what could
it feed. Detail + evidence: `R1_sensors.md` · `R2_library_vault.md` · `R3_memory_organ.md`
· `R4_factory_harness.md` · `R5_oracle_stack.md` · `R6_dashboard_parity.md` ·
`R7_schedulers_health.md`. Synthesis + ranking: `docs/REWIRING.md`.

## Verdict counts
- **RETARGET: 16** — sentiment schedule (→TICK-013, Colin's hand) · Library→review slice
  (→TICK-005) · review forensics feeds ×4: oracle health / ledger results / vetoes /
  audit parity (→TICK-006) · board→dashboard export+panel (→TICK-007) · health-staleness
  wiring + log redirects (→TICK-008) · Numba re-enable (→TICK-009) · journal context
  preservation (→TICK-010) · lesson-velocity + briefing-macro ports (→TICK-006) ·
  successor fast-engine harness (→TICK-012) · GDELT resume-cursor + vault regen hook
  (→TICK-014) · suite-must-not-trade (DONE today).
- **LEAVE: 17** — FRED/AV-news/ThetaData feeds · board schema (0 orphan columns) · exit
  engine (frozen, mid-audit) · factory/train label deferral (Article 6 by design) ·
  research_factory dry-run · cpcv/discovery · family runner v1 (sealed instrument) ·
  Big Move (unvalidated) · killzone/futures stack (wrong market) · 00-BRAIN ·
  proof-of-life today-only · ict/library_bridge (sanctioned) · live dashboard pages ·
  Library auto-learn core.
- **ATTIC-CANDIDATE: 10** (Colin's ruling list — nothing moved): com.alta.cache.refresh
  Reddit path · dead .env keys (Tiingo, OpenWeather, Firebase×, AV-technical) · Polygon
  (equity batch) · cross_system_bridge.py · .smart-env embeddings · bench telemetry
  (or wire its 5-line alert) · stray_tripwire WatchPaths mode (inert since Jun 7).

## Corrections the audit forced on the morning picture
1. The Alexandrian Library is LIVE in ICT (query every scan; `learn()` live-writes the
   canonical json) — the gap is the memory loop, not existence.
2. GDELT is probably not blocked — the paced path has never been observed failing; the
   June failure was one unpaced burst.
3. The "unidentified OANDA writer" was our own integration test firing on every plain
   suite run (now opt-in gated).
4. The repo had no plist-hash watchdog until today (`scripts/plist_watchdog.py` built,
   baselined, GREEN); `stray_tripwire.py` is a different organ (stray-file quarantine).
