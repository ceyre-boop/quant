# R7 — Schedulers + health retargeting audit (2026-07-03, read-only)

## THE PROBER: SOLVED (Task B)
The unidentified USD_CAD 1-unit OANDA writer is **`tests/test_oanda_set_stop.py`**:
- test_oanda_set_stop.py:28-29 `TEST_PAIR="USD_CAD"`, `TEST_UNITS=1`; :55-58
  `place_trade(..., stop_price=1.00000, tp1_price=2.00000)` — exact sentinel match.
- All 8 fills in data/ledger/oanda_fills.jsonl match pytest runs (bursts Jun 30 13:06-33,
  Jun 30 22:30, Jul 2 18:35-55, Jul 3 01:51 — the Day-1 session's own suite runs).
- It is a "practice account integration test" that fires whenever `pytest tests/` runs
  with OANDA creds in .env (always true locally). No cron/CI trigger (CI lacks secrets).
- **Implication: every full-suite run places a real practice-account order.** Immediate
  handling (this session): env-gate the test (`skipif` unless OANDA_INTEGRATION=1) —
  reported as a baseline change (1039 passed → 1038 passed + 1 skipped). Not a stray
  job; a test-discipline hole. No launchd action needed; nothing killed.

## Scheduler verdicts (Task A)
Fresh + consuming (LEAVE): cache.refresh (but see R1: its Reddit payload is ornamental
— job-level keep/kill is Colin's), forex.scan, oracle.briefing, journal_sync,
forex_exit_manager, oracle.killzone, oracle.session_close, papertrading (Colin's
pending keep/kill), weekly_review, shadow_audit, quant.pulse.

STALE + effectively unmonitored (RETARGET — health wiring, not code):
- oracle_cycle/reflect — last Jun 28 (yfinance fetch errors)
- hypothesis.generator — last Jun 15
- research.factory — last Jun 15
- **health.responder — last Jun 14: THE WATCHDOG ITSELF is the stalest job**
- futures.bias — last Jun 14
- stray_tripwire — WatchPaths trigger inactive since Jun 7 (manual invocation works,
  exit 0 today; the *watch* is what's dead) · ATTIC-CANDIDATE as a watch-job, or re-arm
(Context: generator/session_close/cache.refresh were repaired from .corrupt-20260701
on Jul 1 — repair restored files, evidently not schedules/inputs.)

No failure coverage / logs to /tmp (lost on reboot): bench_throughput (no log at all),
evening_prep, render_keepalive → RETARGET (plist StandardErrorPath redirect = live-organ
config edit → ticket for Colin-reviewed batch, not today).

## Headlines
1. Every full-suite run trades on the practice account (test_oanda_set_stop) — gate it
   today, before the next suite run.
2. The watchdog trifecta (health_responder + the two research nightlies) has been
   silently dead 18+ days; nothing watches the watchdogs. One health-wiring ticket
   covers all: hard staleness deadlines per job, surfaced on the dashboard + morning
   brief.
3. Three jobs log to /tmp or nowhere — reboot erases their history; plist log-path
   redirect batch (Colin-reviewed) fixes visibility permanently.
