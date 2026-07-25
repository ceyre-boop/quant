# DISPATCH — Daily Intelligence Pipeline (DIP)
## Alta Investments · Work Order for Claude Code
### Priority: High · Starts: Tonight · Runs: Every weekday, automatically

---

## The Mission

Build a master orchestration script that runs three phases of escalating compute
intensity every weekday — automatically, without Colin present — and diffuses
the day's research into the Obsidian brain and hypothesis ledger as persistent
accumulated knowledge. The goal is compound intelligence, not daily intervention.

One concentrated burst of compute per day. Conducted into Obsidian. Diffused
into every future session. Day after day.

---

## Architecture Overview

```
scripts/daily_intelligence_pipeline.py   — master orchestrator
  Phase 1 (06:00 ET)  — warm-up   : market context, regime, gate scan
  Phase 2 (08:30 ET)  — peak heat : hypothesis batch, ICT pass, oracle
  Phase 3 (16:30 ET)  — diffusion : Obsidian sync, ledger stamp, calibration

~/Library/LaunchAgents/com.alta.dip_warmup.plist    — fires Phase 1
~/Library/LaunchAgents/com.alta.dip_peak.plist      — fires Phase 2
~/Library/LaunchAgents/com.alta.dip_diffuse.plist   — fires Phase 3
```

Each phase is independently callable (`--phase 1|2|3`) so any single phase can
be re-run manually after a failure without re-running the others.

---

## Phase 1 — Warm-Up (06:00 ET, ~10 min)

Lightweight compute. Sets the day's context. No Claude inference.

**Tasks (in order):**
1. Pull macro calendar from Forex Factory for today (cached JSON, EDGAR-style 1s sleep)
2. Read live regime map feature files and write `data/agent/daily_regime.json`
   - Format: `{"GBPUSD": "NARROWING", "EURUSD": "NARROWING", ...}`
3. Check if `data/agent/petrules_gate_scan.json` is fresh (today's date in `scanned_at`)
   - If not: log a warning, do NOT re-run the gate scan here (it has its own plist)
4. Write `data/agent/dip_phase1.json`:
   ```json
   {
     "date": "2026-07-23",
     "phase": 1,
     "completed_at": "...",
     "regime": {...},
     "macro_events_today": [...],
     "gate_scan_fresh": true
   }
   ```

**Failure behavior:** if any step errors, log it and continue. Phase 1 failures do not
block Phase 2. Write `data/agent/dip_phase1_error.json` with the traceback.

---

## Phase 2 — Peak Heat (08:30 ET, ~45-60 min)

The compute-intensive window. Terminates cleanly when done. Fans will spin.
This is not just backtesting — it is recursive model training. The machine
watches the market the way a trader watches it: price, all indicators at once,
and news, aligned on the same timeline. It learns from what actually happened,
not what was predicted.

---

### 2a — Feature Assembly ("the full picture, aligned in time")

Build the training feature matrix for the rolling window. This is the raw
material the XGBoost model trains on. Every row is one bar. Every column is
one thing the market was doing at that moment.

**Price + indicator layer** (yfinance daily bars):
- OHLCV for each instrument in the universe
- Derived: RSI-14, MACD (12/26/9), ATR-14, Bollinger Band position,
  distance from 200SMA, 52-week range percentile, volume z-score
- Carry-specific: differential_trend feature (already computed by regime_map.py),
  rate differential level (in bps), days since last CB decision

**News sentiment layer** (free, local):
- Source: yfinance `ticker.news` (last 72h headlines per symbol)
- Model: VADER sentiment (local, no API cost, no rate limit)
  `from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer`
- Output per bar: `sentiment_compound` (−1.0 to +1.0), `news_volume` (article count),
  `sentiment_momentum` (compound delta vs. prior 72h window)
- This is the "news said right there on the price action map." One float per bar.
  It does not replace macro judgment. It is a feature, not a signal.

**Assembly output:** `data/ml/feature_matrix_<date>.parquet`
Schema: `[date, symbol, open, high, low, close, volume, rsi, macd_hist, atr,
  bb_pct, dist_200sma, range_pct, vol_z, diff_trend, rate_diff_bps, sentiment_compound,
  news_volume, sentiment_momentum, target_return_5d]`

`target_return_5d` = forward 5-day log return (label). This is what we are
trying to predict. It is only available for rows older than 5 days — recent
rows have `target_return_5d = NaN` and are inference-only, not training rows.

---

### 2b — Recursive Walk-Forward XGBoost Training

**The training gate (non-negotiable from CLAUDE.md Art. 6):**
Training runs only when there is at least one hypothesis-ledger entry with
`verdict == CONFIRMED`. On first run, carry_v015 and HYP-093 (Undertow) are
both CONFIRMED — the gate is open. If no CONFIRMED entry exists: log the block
and skip 2b. Do not train on unproven edges.

**The 30-day seed window (start here, tonight):**
- Training set: rows where `date >= today - 30d` AND `target_return_5d is not NaN`
  (i.e., bars older than 5 days within the 30-day window)
- This is ~25 usable training rows per instrument on day 1. Small. That is intentional.
  The model will underfit. That is fine. We are not looking for a perfect model —
  we are building the pipeline that will train a good model in 90 days.

**Recursive expansion protocol:**
- Read `data/ml/training_window.json`: `{"start_date": "2026-06-23", "end_date": "today"}`
- Each week, Dispatch (or Colin) expands `start_date` backward by 30 days.
  Month 1: 30d. Month 2: 60d. Month 3: 90d. After 6 months: 1yr. After 1yr: 2yr.
- This is the recursive accumulation. The window grows as the system proves it
  can handle the data volume and the model proves it is learning something real.
- Never go back more than available data — if yfinance returns 2yr of bars, that
  is the floor. Do not manufacture data.

**XGBoost training:**
```python
import xgboost as xgb
# Already in repo (sovereign/autonomous/research_factory.py uses it)

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,          # shallow — avoids overfit on small N
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror'
)
model.fit(X_train, y_train)
```

- Feature importance: log top-10 features to `data/ml/feature_importance_<date>.json`
  every run. This is how we learn what the market actually responds to.
- Validation: hold out last 5 days of the training window as a mini-val set.
  Log `val_mae`, `val_r2` to `data/ml/training_log.jsonl`. Watch these numbers
  over weeks — if they improve, the model is learning. If they degrade after
  expanding the window, the older data may be regime-incompatible.
- Save model: `data/ml/xgb_model_<date>.json` (XGBoost native format)
  AND symlink `data/ml/xgb_model_latest.json` → today's model.
  The live system picks up `xgb_model_latest.json` next session.

**Inference pass (after training):**
- Run inference on the most recent 5 rows (the NaN-target rows) for each instrument.
- These are the bars where we don't yet know the outcome but the model makes a prediction.
- Write predictions to `data/ml/xgb_inference_<date>.json`:
  ```json
  {"date": "2026-07-23", "symbol": "EURUSD", "predicted_return_5d": 0.0031,
   "confidence": "LOW", "n_training_rows": 18}
  ```
- `confidence` = LOW if `n_training_rows < 50`, MEDIUM if < 200, HIGH if >= 200.
  At 30-day seed window: everything is LOW. That is honest.
- These predictions do NOT trigger trades. They feed the Petrules Gate and the
  hypothesis batch as additional context. Colin decides. Always.

---

### 2c — Hypothesis Batch

Read `data/research/hypothesis_ledger.jsonl`. Find all entries where:
- `status == "PENDING"` (not yet tested)
- OR `status == "ACTIVE"` and `last_batch_run` is more than 7 days ago

Take the top 5 by `priority_score` (field in ledger, default 0.5 if absent).

For each hypothesis, produce a research note combining:
- Pattern search in Obsidian brain (similarity > 0.70 to any historical analog)
- XGBoost feature importance from today's training run (does this hypothesis's
  signal appear in the top-10 features? If yes, the model is independently
  finding the same thing)
- Philosophy gate check (6 tenets — see TRADING_PHILOSOPHY.md)

Write each note to `data/agent/hypothesis_batch/<date>/<hyp_id>.md`.
Append summary to `data/agent/daily_hypothesis_batch.jsonl`.

---

### 2d — Oracle Reflect Cycle

Check `data/agent/decision_log.jsonl` for entries where `outcome_logged == false`
and `entry_time` is older than `hold_bars * bar_duration`. For each:
call `sovereign/oracle/reflect_cycle.py`. The oracle cannot learn without this.
Do not skip it.

---

### 2e — ICT Daily Pass (conditional)

If `scripts/ict_daily_pipeline.py` exists: run it.
If not: log "ICT pipeline not yet built — skip" and continue.

**2026-07-24: this is the intended steady state, not a temporary stub.**
`tickets/DISPATCH_ICT_DAILY_PIPELINE.md` (the spec that would have produced this script)
is SHELVED/RE-SCOPED — ICT's role in this system is a TP/SL/entry reference layer, not a
predictive daily pipeline, so `scripts/ict_daily_pipeline.py` is not expected to exist.
The skip branch above should be treated as the normal, permanent path for 2e, not a
placeholder awaiting a future build.

---

### 2f — Checkpoint

Write `data/agent/dip_phase2.json`:
```json
{
  "date": "2026-07-23",
  "phase": 2,
  "completed_at": "...",
  "feature_matrix_rows": 750,
  "training_window_days": 30,
  "model_val_mae": 0.0041,
  "model_val_r2": 0.12,
  "top_feature": "sentiment_momentum",
  "inference_symbols": 4,
  "hypotheses_run": 5,
  "oracle_cycles": 1,
  "skipped": ["ict_daily_pipeline"]
}
```

---

## Phase 3 — Diffusion (16:30 ET, ~20 min)

The heat bleeds into the conductor. This is where the day's compute becomes
permanent knowledge.

### 3a — Obsidian Brain Sync

For every file written today in `data/agent/hypothesis_batch/<today>/`:
- Convert to Obsidian-compatible markdown with YAML frontmatter:
  ```yaml
  ---
  date: 2026-07-23
  type: hypothesis-research
  status: pending-verdict
  tags: [hypothesis, daily-batch]
  ---
  ```
- Add wikilinks: `[[TRADING_PHILOSOPHY]]`, `[[hypothesis_ledger]]`, any matched
  Library pattern (e.g., `[[ASIAN_CURRENCY_CONTAGION]]`)
- Write to `~/Obsidian/Obsidian/Trading/System/Hypotheses/<hyp_id>_<date>.md`
- If a note for this `hyp_id` already exists: append a dated section, do not overwrite

For the day's regime reading:
- Append to `~/Obsidian/Obsidian/Trading/System/CONTEXT.md`:
  ```
  ## 2026-07-23 Regime
  GBPUSD: NARROWING | EURUSD: NARROWING | AUDUSD: NARROWING | GBPJPY: NARROWING
  Gate scan: {tier3_plus} Tier3+ signals
  ```

### 3b — Ledger Stamp

For each hypothesis that was batch-run today:
- Update `data/research/hypothesis_ledger.jsonl` entry: set `last_batch_run = today`
- If the philosophy gate returned ABORT: set `status = "GATED_OUT"`, log reason
- Append-only: never modify historical entries, only append a new record with
  `{"type": "batch_update", "hyp_id": ..., "date": ..., "result": ...}`

### 3c — Calibration Append

Read `data/agent/petrules_gate_scan.json`. If `tier3_plus > 0` and today's scan
is fresh: the signals are now in the calibration window. Do not log outcomes yet
(outcomes close in the future). Log that they are "open" in `gate_calibration.jsonl`:
```jsonl
{"date": "2026-07-23", "symbol": "NVDA", "tier": 4, "conviction_score": 0.87,
 "outcome_logged": false, "opened_at": "2026-07-23"}
```

### 3d — Final write

Write `data/agent/dip_phase3.json`:
```json
{
  "date": "2026-07-23",
  "phase": 3,
  "completed_at": "...",
  "obsidian_notes_written": 5,
  "ledger_entries_stamped": 5,
  "calibration_rows_opened": 1
}
```

---

## Sleep Safety

The Mac may sleep between phases. All three phases write checkpoint files.
On wakeup, each launchd plist fires once and terminates — no daemons, no loops.
A phase that already ran today (checkpoint exists with today's date) skips
immediately with a log line: "Phase N already completed today at <time>. Skip."
This makes the pipeline idempotent.

---

## Dashboard Integration

The dashboard reads `data/agent/dip_phase2.json` and `data/agent/dip_phase3.json`.
Add a small DIP status row to `dashboard/index.html`:
- Last run timestamp per phase
- Hypotheses batched today
- Obsidian notes written
- "RUNNING" / "COMPLETE" / "FAILED" status per phase

This is a minor dashboard extension — the data files are the primary output.
The dashboard display is optional if time is short; the files are not.

---

## Files (all NEW or additive)

```
scripts/
  daily_intelligence_pipeline.py    — master orchestrator, --phase 1|2|3 flag
  dip_feature_assembly.py           — Phase 2a: price + indicator + sentiment matrix
  dip_xgb_trainer.py                — Phase 2b: walk-forward XGBoost training + inference
  dip_hypothesis_batch.py           — Phase 2c logic (extracted for testability)
  dip_obsidian_sync.py              — Phase 3a logic (extracted for testability)

~/Library/LaunchAgents/
  com.alta.dip_warmup.plist         — 06:00 ET weekdays, --phase 1
  com.alta.dip_peak.plist           — 08:30 ET weekdays, --phase 2
  com.alta.dip_diffuse.plist        — 16:30 ET weekdays, --phase 3

data/ml/
  feature_matrix_<date>.parquet     — assembled feature matrix (price + indicators + sentiment)
  xgb_model_<date>.json             — trained model snapshot (XGBoost native format)
  xgb_model_latest.json             — symlink to most recent model
  xgb_inference_<date>.json         — today's 5-day return predictions (LOW confidence at first)
  feature_importance_<date>.json    — top-10 features from today's training run
  training_log.jsonl                — append-only: date, val_mae, val_r2, n_rows, window_days
  training_window.json              — current window bounds: {"start_date": ..., "end_date": "today"}

data/agent/
  dip_phase1.json                   — today's Phase 1 checkpoint
  dip_phase2.json                   — today's Phase 2 checkpoint (includes model metrics)
  dip_phase3.json                   — today's Phase 3 checkpoint
  hypothesis_batch/<date>/          — per-hypothesis research notes
  daily_hypothesis_batch.jsonl      — append-only batch summary log

config/
  dip.yml                           — training window, feature flags, sentiment toggle,
                                      min training rows before model is trusted (default: 50)
```

**Dependencies to install (in sandbox and VM):**
```bash
pip install vaderSentiment xgboost pyarrow --break-system-packages
# xgboost already likely present (sovereign/autonomous uses it)
# vaderSentiment is new — local, no API key, no rate limit
```

---

## Non-Negotiables (CLAUDE.md)

- **No sovereign/ imports from new scripts unless through orchestrator.py.**
  `dip_hypothesis_batch.py` calls existing scripts as subprocesses or via the
  documented public interface — it does not import ICT pipeline internals.
- **No silent failures.** Every phase writes a checkpoint or an error file. Never exits 0 and leaves nothing behind.
- **No live trades.** The pipeline is a research loop. It reads signals and logs them. It does NOT call `order_send` or any MT5 bridge code.
- **Obsidian writes are additive only.** Append to existing notes; never truncate. The brain is append-only like the calibration ledger.
- **Training gate still applies.** If `dip_hypothesis_batch.py` generates a research note that looks like a CONFIRMED verdict: it still requires a human to formally stamp the ledger. The pipeline surfaces; Colin decides. Always.

---

## Definition of Done

**RECONCILED 2026-07-25 — see NEXT.md "2026-07-25 — DIP honest reconciliation" for the full
spec-to-reality map. Checkboxes below reflect verified current state, not the original ask.**

- [x] `python3 scripts/daily_intelligence_pipeline.py --phase 1` runs end-to-end, writes checkpoint
      (`data/agent/dip_phase1.json`, verified 2026-07-25: 5/5 collectors written)
- [~] `python3 scripts/daily_intelligence_pipeline.py --phase 2` runs end-to-end — RE-SCOPED:
  - [ ] `data/ml/feature_matrix_<date>.parquet` with price + indicator + **sentiment** columns —
        **NOT BUILT.** No VADER/news-sentiment feature layer exists anywhere in this repo. Not faked.
  - [x] Walk-forward XGBoost training — satisfied by the ALREADY-LIVE `continuous_harvester.py` +
        `training/retrain_loop.py` pair (`data/harvest.db`, `models/xgb_veto.json`,
        `models/threshold_history.json`), delegated via `dip_daily.sh` behind `--with-retrain`.
        Different file contract than the spec's `data/ml/xgb_model_<date>.json`; re-scoped rather
        than duplicated.
  - [ ] `data/ml/training_log.jsonl` val_mae/val_r2 row — NOT BUILT (no `data/ml/` layer exists;
        `retrain_loop.py` has its own metrics path in `models/threshold_history.json` instead)
  - [ ] `data/ml/xgb_inference_<date>.json` confidence=LOW — NOT BUILT, same reason
  - [x] hypothesis batch notes written — `sovereign/autonomous/hypothesis_generator.run()`,
        verified live 2026-07-25 (5 candidates generated, L1 briefing context attached)
- [x] `python3 scripts/daily_intelligence_pipeline.py --phase 3` runs end-to-end — BUILT 2026-07-25,
      RE-SCOPED: diffuses regime + hypothesis-batch summary to
      `~/Obsidian/Obsidian/Trading/System/DIP-Daily-Log.md` (append-only) and opens calibration rows
      when the gate scan is fresh. Per-hypothesis notes with Library-pattern-similarity wikilinks are
      NOT built — no similarity matcher exists in this repo. Ledger stamping re-scoped to read the
      existing `data/agent/generator_log.jsonl` batch-append rather than writing a second stamp into
      the adjudicated `hypothesis_ledger.json`.
- [x] All three launchd plists exist and are ready for operator install:
      `scripts/com.alta.dip_{warmup,peak,diffuse}.plist` (built 2026-07-25). **NOT YET LOADED** —
      `launchctl list | grep alta.dip` on Colin's Mac will show zero until Colin runs the install
      commands in each plist's header comment. `com.alta.dip_daily.plist` (harvest+retrain, 02:30 ET)
      is unchanged and separate — `dip_peak` deliberately does not pass `--with-retrain` to avoid
      running that compute twice a day.
- [x] Sleep-safe: verified 2026-07-25 — running `--phase 1` twice same-day prints "already completed
      today. Skip." and does not re-fetch.
- [x] Training gate confirmed: orchestrator checks `hypothesis_ledger.json` for any `CONFIRMED`
      status before delegating to `--with-retrain` (14 CONFIRMED entries present as of 2026-07-25, so
      gate reads open); if none were present it would log `"training gate closed"` and skip.
- [x] `NEXT.md` updated 2026-07-25.

---

## Tonight

Run `python3 scripts/daily_intelligence_pipeline.py --phase 1` manually first.
No launchd yet. Just confirm the warm-up runs clean and writes its checkpoint.
If it does: install the plists. Tomorrow morning Phase 1 fires automatically at 6:00am.
The night begins.

---

*Alta Investments · Dispatch Work Order · 2026-07-23*
*"Heat the conductor. Trust diffusion. Day after day."*
