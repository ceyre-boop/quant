# FOMC-window logger — self-arming runbook

Built because the 2026-07-28 FOMC window was missed: arming was a manual `python
scripts/fomc_window_logger.py` invocation someone had to remember to run. It's now
self-arming — one-time setup below, then every FOMC meeting arms itself.

## What changed

- `scripts/fomc_window_logger.py` now defaults `--center` to the **next unexpired
  meeting** in `config/fomc_dates.yml`, instead of a hardcoded date. Explicit
  `--center` still overrides it.
- `config/fomc_dates.yml` lists the known 2026 FOMC meeting dates + statement times (ET).
  **⚠️ Colin must verify these against https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm**
  — they were transcribed from `clawd_trading/swing_prediction/fomc_calendar.py`, not
  independently re-checked against the Fed site by this session. The file has a
  `verified_against_federalreserve_gov: false` flag at the top — flip it once checked.
- `scripts/com.alta.fomc_window_logger.plist` fires the logger at 13:55 on each date in
  the config, 5 minutes ahead of the 2:00pm ET statement, so the window is armed before
  the print drops.
- The logger still places **no orders** — pure observation — and still fails loud
  (`FOMC LOGGER FAILURE — window NOT captured: ...` to stderr / plist error log) if the
  MetaTrader5 package, terminal, or demo connection isn't available, instead of quietly
  no-oping.

## One-time setup (on the Windows VM running the MT5 demo terminal)

1. Confirm `config/fomc_dates.yml` dates are correct (see verification note above).
2. Keep the MT5 demo terminal running and logged into the demo account — the logger
   connects through the same `MT5Connector` + demo guard as the execution bridge; if the
   terminal isn't up, it fails loud, it doesn't skip silently.
3. Install the plist once:
   ```
   launchctl load /Users/taboost/quant/scripts/com.alta.fomc_window_logger.plist
   ```
4. Note: `StartCalendarInterval` times are **local system clock time** — the plist
   assumes the VM's system timezone is America/New_York (since 13:55 ET is hardcoded as
   Hour=13/Minute=55). If the VM runs in a different timezone, either set the VM's system
   TZ to America/New_York, or recompute each entry's Hour/Minute in the plist to 13:55 ET
   converted to the VM's local zone.

After that, every FOMC meeting listed in `config/fomc_dates.yml` self-arms — no manual
step required. Output lands at `data/execution/fomc_window_<date>.jsonl`.

## Checking it worked

```
tail logs/fomc_window_logger.log logs/fomc_window_logger.err
```

A clean run prints `FOMC window: ... → ...` then `Done. Wrote N samples → ...`. A failed
run prints `FOMC LOGGER FAILURE — window NOT captured: <reason>` to the `.err` file —
treat any content there as an alarm, not noise.

## Next scheduled arm

As of 2026-07-28, the logger self-arms next for **2026-07-29 14:00 ET** (tomorrow's
FOMC statement) — confirmed via `python scripts/fomc_window_logger.py --dry-run`.
