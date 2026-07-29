# DISPATCH — Petrules Gate Daily Scanner
## Alta Investments · Work Order for Claude Code
### Priority: High · Output: data/agent/petrules_gate_scan.json

---

## The Mission

Build a scanner that wakes up every morning, checks the entire market for
conviction-worthy divergence setups, and writes the result to one JSON file
the dashboard reads. When something clears Tier 3 or Tier 4, Colin stops
what he is doing and looks at it. When nothing clears, the dashboard shows
QUIET and life continues. No manual prompting. No hunting. It finds him.

---

## What "The Whole Market" Means

Universe: S&P 500 + Russell 2000 + 50 major ETFs + the 4 carry forex pairs.
That is approximately 2,550 instruments. Every instrument gets scored every day.
The scanner is fast because the scoring is lightweight — no ML yet, just
feature assembly and a rule-based conviction rubric. ML comes later.

Symbol list: download S&P 500 from Wikipedia (always current), Russell 2000
from iShares IWM holdings CSV, ETF list hardcoded in config. Write the
combined list to `data/agent/gate_universe.json` on first run, refresh weekly.

---

## Data Sources (All Free)

### Form 4 — SEC EDGAR (insider trades)
- URL: `https://efts.sec.gov/LATEST/search-index?q=%22form+4%22&dateRange=custom&startdt={today}&enddt={today}&forms=4`
- Parse: issuer CIK → ticker lookup → transaction type (P=purchase, S=sale) → amount
- Signal: cluster of 3+ insiders buying within 5 days, open-market only (not option exercises)
- Library: `requests` + `xml.etree` — no paid API needed

### 13D/G — Activist stakes
- URL: `https://efts.sec.gov/LATEST/search-index?forms=SC+13D,SC+13G&dateRange=custom&startdt={7_days_ago}&enddt={today}`
- Signal: new 13D (activist) or 13G amendment showing increased stake > 1%
- Parse same as Form 4

### Analyst estimates — Yahoo Finance
- Library: `yfinance` (already in repo)
- `ticker.info` → `targetMeanPrice`, `recommendationKey`, `numberOfAnalystOpinions`
- `ticker.recommendations` → recent upgrade/downgrade history (revision velocity)
- Signal: 5+ upgrades in 10 days with zero downgrades = top-decile revision velocity

### Options flow approximation — Yahoo Finance
- `ticker.option_chain(expiry)` for the nearest 3 expiries
- For each strike: `volume / open_interest` ratio. Ratio > 3.0 on calls = unusual accumulation
- Flag: single strike with volume > 5,000 contracts AND volume/OI > 3.0 AND premium > $500k
- This is an approximation. Not as good as paid flow, but real signal at zero cost.

### Price + momentum
- `yfinance` daily bars: 52-week range position, distance from 200SMA, recent gap
- These are Organ 1 inputs only (consensus context), not divergence signals

---

## The Scoring Rubric (Pre-ML, Rule-Based)

Each instrument gets a conviction score 0.0–1.0 built from weighted factors.
No model. No training data required. This is the Phase 0 version that starts
running immediately while calibration data accumulates.

```python
conviction_score = (
    insider_cluster_score  * 0.30 +   # Form 4: cluster size, recency, amount
    revision_velocity_score * 0.25 +   # analyst upgrades vs. downgrades, 10d
    options_flow_score      * 0.25 +   # unusual volume/OI ratio, premium size
    activist_score          * 0.20     # 13D/G new filing or amendment
)
```

Tier assignment:
- TIER 4 (screamer):  score >= 0.85
- TIER 3 (strong):    score >= 0.70
- TIER 2 (watch):     score >= 0.50  (logged but not surfaced on dashboard)
- TIER 1 (noise):     score < 0.50   (discarded after logging)

### Scoring detail per factor

**insider_cluster_score**
- 0 insiders buying:                          0.0
- 1 insider, < $100k:                         0.1
- 1 insider, > $500k open market:             0.3
- 2-3 insiders, any amount:                   0.5
- 3+ insiders, cluster within 5 days:         0.8
- 4+ insiders including C-suite, > $1M total: 1.0
- Any sells in same window:                   subtract 0.2 (informed selling = negative)

**revision_velocity_score**
- 0 revisions:                                0.0
- 1-2 upgrades, no downgrades, 10d:           0.3
- 3-5 upgrades, 0-1 downgrades:              0.6
- 6+ upgrades, 0 downgrades, top decile:      1.0
- Any downgrades:                             cap at 0.5

**options_flow_score**
- No unusual volume:                          0.0
- One strike with vol/OI > 2.0:              0.3
- One strike with vol/OI > 3.0, > $250k:     0.6
- Single block > $500k premium, vol/OI > 3:  0.8
- Multiple strikes accumulating same side:    1.0

**activist_score**
- No 13D/G activity:                          0.0
- 13G amendment (passive, increased):         0.3
- New 13G filing (first disclosure):          0.5
- 13D amendment (activist, increased stake):  0.7
- New 13D filing (new activist):              1.0

---

## Output Schema — `data/agent/petrules_gate_scan.json`

```json
{
  "scanned_at": "2026-07-23T09:15:00Z",
  "instruments_scanned": 2547,
  "tier3_plus": 2,
  "top_signal": {
    "symbol": "NVDA",
    "tier": 4,
    "conviction_score": 0.87,
    "hypothesis": "Street expects +7.2% move. Insider cluster + options accumulation suggests beat + guide raise → +12-18% actual.",
    "consensus": {
      "eps_est": "$0.89",
      "implied_move": "±7.2%",
      "analyst_buy_pct": 91,
      "narrative": "AI demand priced in"
    },
    "divergence_signals": [
      {"label": "Form 4: CFO bought $4.2M open market Jul 18 — 3 days before blackout. Cluster of 4 insiders.", "direction": "bullish"},
      {"label": "Options: 14,000 calls at $140 strike, vol/OI=4.8, $2.8M premium. Single block.", "direction": "bullish"},
      {"label": "Revisions: 6 upgrades vs 1 cut in 10 days. Top 3% velocity.", "direction": "bullish"}
    ],
    "sizing": {
      "size_multiplier": 1.43,
      "f": 0.057,
      "f_max": 0.08,
      "f_max_hit": false,
      "calibration_status": "BACKTEST ONLY"
    },
    "move_up": ["Insider cluster expands to 6+", "Second options block same strike"],
    "move_down": ["Any analyst downgrade", "FOMC shock tomorrow"]
  },
  "all_signals": [
    {"symbol": "NVDA", "tier": 4, "conviction_score": 0.87},
    {"symbol": "AAPL", "tier": 3, "conviction_score": 0.71}
  ]
}
```

If no instrument clears Tier 2, `top_signal` is null and `tier3_plus` is 0.
The dashboard handles null gracefully (shows QUIET state).

---

## File Structure

```
scripts/
  petrules_gate_scanner.py     — main scanner, called by launchd
  gate_universe.py             — universe builder, called weekly
  gate_edgar_client.py         — Form 4 and 13D/G fetcher
  gate_options_screen.py       — options flow approximation
  gate_scorer.py               — conviction rubric, factor weights

data/agent/
  petrules_gate_scan.json      — today's scan output (dashboard reads this)
  gate_universe.json           — current instrument universe
  gate_scan_history.jsonl      — one line per scan, all tiers 2+, append-only
  gate_calibration.jsonl       — outcome log (ticker, score, actual outcome)
                                  — accumulates until n >= 500 for calibration

config/
  gate_params.yml              — factor weights, tier thresholds, universe config
```

---

## Scheduling

Wire into launchd alongside the paper account writer. New plist:
`~/Library/LaunchAgents/com.alta.petrules_gate.plist`

Run time: 09:00 ET daily (before market open — EDGAR Form 4s for prior day
are available after ~8:30 ET). Do not run on weekends.

If the scanner fails (network error, EDGAR rate-limit), write a failure record
to `petrules_gate_scan.json` with `{"error": "...", "scanned_at": "..."}`.
The dashboard handles this gracefully — shows the error timestamp, not a crash.

---

## Calibration Log (Built Now, Used Later)

Every time the gate surfaces a Tier 3+ signal, log the outcome when it resolves:

```jsonl
{"date": "2026-07-23", "symbol": "NVDA", "tier": 4, "conviction_score": 0.87, "hypothesis": "...", "actual_outcome_pct": 14.2, "outcome_direction": "bullish", "consensus_implied_move": 7.2, "beat_implied": true}
```

This is the data that eventually calibrates the ML model. Start accumulating it
on day one even though the model does not exist yet. The log is append-only.
When n >= 500 closed outcomes, run calibration (separate ticket).

---

## Non-Negotiables (CLAUDE.md)

- No hardcoded thresholds — all weights and tier cutoffs in `config/gate_params.yml`
- No sovereign/ imports — this is a standalone research script, not part of ICT pipeline
- No silent failures — if EDGAR is down, log it explicitly and write the error JSON
- `gate_calibration.jsonl` is append-only — never overwrite historical outcomes
- EDGAR has rate limits: add 1-second sleep between requests, respect 429 responses
- This script is research infrastructure — it does NOT trigger trades automatically.
  It surfaces signals. Colin decides. Always.

---

## Definition of Done

- [ ] `scripts/petrules_gate_scanner.py` runs end-to-end without error
- [ ] `data/agent/petrules_gate_scan.json` is written with valid schema
- [ ] Dashboard Petrules Gate panel renders correctly from the JSON
- [ ] launchd plist installed and confirmed running at 09:00 ET
- [ ] `gate_calibration.jsonl` receives entries for any Tier 3+ signals
- [ ] `NEXT.md` updated: what shipped, push confirmed

---

*Alta Investments · Dispatch Work Order · 2026-07-22*
*"It finds you. You don't hunt for it."*
