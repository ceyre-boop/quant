# Funded-Account Repo — File Manifest
**What to copy from `quant` (the general repo) into the new challenge-only repo.**
*Every path below was verified to exist before being listed. Nothing here is a guess.*

---

## The scoping call I'm making — say so if you disagree

This repo copies **carry only.** Not ICT (permutation p=0.52, NOT PROVEN), not
Undertow/HYP-093 (still INSUFFICIENT_DATA, n_live=4 of 250 needed), not any
research-in-progress line. COLIN_V1.md already made this call last session:
carry is the one edge that's actually confirmed and sized correctly. A repo
whose entire job is "pass the eval" should run the one proven thing, not carry
three unfinished ones alongside it. If you want ICT or Undertow riding along
as a second sleeve, say so — the file list below would grow, not shrink.

---

## 1. The signal engine — what decides a trade exists

| Copy | Why |
|---|---|
| `sovereign/forex/` (whole dir, 27 files) | The entire carry pipeline — rate differentials, entry logic, the 5 gates. This is the edge itself. |
| `sovereign/forex/cb_calendar.py` | Already listed above but calling it out — this is the CB-blackout gate specifically, HYP-061-adjacent logic. |
| `sovereign/features/` (16 files) | The macro feature set every signal is computed from — momentum, volatility, rate diff. Signal engine won't run without it. |

## 2. The risk layer — what stops a trade from happening

| Copy | Why |
|---|---|
| `sovereign/risk/risk_engine.py` | The single gate every trade passes through. |
| `sovereign/risk/layers/` (`base_size.py`, `drawdown.py`, `gates.py`, `kelly.py`, `portfolio.py`, `prop.py`, `regime.py`, `volatility.py`) | Every individual risk check, kept separate on purpose — `prop.py` especially, since that's the funded-account-specific sizing logic. |
| `RISK_CONSTITUTION.md` | The written risk rules this code enforces. Needed as the human-readable spec alongside the code. |

## 3. Execution and logging — what happens when a trade is live

| Copy | Why |
|---|---|
| `execution/harness.py` | Measures whether the edge survives real fills (slippage, spread) — the honesty check between backtest and live. |
| `execution/eod.py` | End-of-day reconciliation. |
| `sovereign/execution/forex_exit_manager.py` | The exit logic — currently SHADOW MODE. **Decision point:** this repo's whole purpose is going live, so you'll need to decide when this comes off shadow. Don't flip it silently — log the unlock like NEXT.md already requires. |
| `sovereign/intelligence/decision_logger.py` | Captures every decision's reasoning before the outcome is known. Mandatory — the Oracle loop (if you keep any of it) can't learn without this. |
| `experience/journal.py` + `experience/journal_sync.py` | One row per decision, abstentions included. This is your real, unfakeable trade record. |

## 4. The funded-account tracking itself

| Copy | Why |
|---|---|
| `sovereign/propfirm/` (whole dir, 5 files, including `deployment_checklist.py`) | `deployment_checklist.py` is the actual script that computes the 5 gates (G1–G5) — not a file that reads the result, the file that produces it. This is the core of "should we go live" for this repo's specific purpose. |
| `data/agent/prop_challenge_state.json` | Current gate state — copy as the starting snapshot, but it'll regenerate. |

## 5. The proof — numbers this repo's decisions rest on

| Copy | Why |
|---|---|
| `data/proof/backtest_trades_v015_2015_2024.csv` | The sealed 411-trade log. Every number in COLIN_V1.md traces back to this file specifically — without it, the claims are unverifiable in the new repo. |
| `data/research/positioning_family/spot_cache/` (`EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD` — both `.parquet` and `_ohlc.parquet`) | The real daily FX price history (2014–2026) everything gets tested against. **Note:** there are two near-duplicate spot_cache directories in the general repo (`positioning_family/` and `modern/`) — pick one as canonical when you copy, don't bring both into the new repo and create a second source-of-truth problem there too. |
| The carry-relevant entries from `data/agent/hypothesis_ledger.json` — specifically **HYP-045** (AUDNZD exclusion, CONFIRMED), **HYP-059** (trailing stop finding, CONFIRMED), **HYP-108** (exit protocol, unsealed) | Don't copy all 88 entries — most are ICT/equities/political-alpha and irrelevant here. Pull just the carry-lineage ones so the new repo's ledger tells an honest, complete story of *this* edge's history without dragging in everything else. |

## 6. The written discipline — the part that isn't code

| Copy | Why |
|---|---|
| `ALTA_METHOD.md` | Already written as the exact live-trading protocol for this edge — 5-step method, entry confirmation, sizing, hold rules, exit logging. This is close to the new repo's actual operating manual already. |
| `COLIN_V1.md` | The sizing/frequency conclusion from last session — 0.5–1.0% risk, no-deadline eval path, 92% pass probability. This IS the strategic thesis for the new repo. |
| `TRADING_PHILOSOPHY.md` | The six tenets. Worth keeping so the new repo doesn't drift from the reasoning that shaped the old one. |
| `advice.md` | The trader-wisdom file directly cited throughout ALTA_METHOD.md's rules — Livermore, PTJ, Kovner, Lipschutz. Keep the two together; ALTA_METHOD.md quotes it. |
| `data/trade_logs/TEMPLATE_PRE_TRADE.md`, `TEMPLATE_DAILY_HOLD.md`, `TEMPLATE_EXIT.md`, `TEMPLATE_META_ANALYSIS.md` | The actual logging templates already built for this exact purpose. |

## 7. The dashboard — the part built last night, specifically for this

| Copy | Why |
|---|---|
| `scripts/build_daily_verdict_page.py` | Generates the plain-English go/no-go page from real gate data. Built last session for exactly this purpose — arguably the single most on-point file for a challenge-only repo. |
| `render.yaml` | Deployment blueprint — trim the `startCommand` and health checks to match whatever subset of `live_signals_server.py` you actually bring over. |
| `requirements-dashboard.txt` (with the `pyarrow` fix already applied) | Don't recopy the broken version — the fix from last session should travel with it. |

**Do not copy `index.html` or `dashboard/dashboard_live.html` wholesale.** Both are built for the general repo's full surface — ICT tab, replay cockpit, chat, TradingView, six systems' worth of panels. Almost none of that belongs in a repo whose only job is "pass this specific eval." Build a new, small front page around `daily_verdict.html` instead of trimming a large one down — trimming a page built for a different purpose tends to leave dead code and false assumptions behind, which is the exact problem the general repo's dashboard already has.

## 8. Config

| Copy | Why |
|---|---|
| `config/parameters.yml` | Live thresholds — but audit every value on the way in. This file holds config for systems you're *not* bringing over too; strip it to carry-only rather than copying the whole thing and hoping the unused keys stay inert. |
| `data/agent/param_change_log.jsonl` | Optional — the audit trail for why thresholds are what they are. Worth it if you want the new repo's config to have provenance from day one instead of starting as unexplained numbers. |

---

## What I'd explicitly leave behind, and why

- **`sovereign/oracle/`, `sovereign/autonomous/`, `sovereign/discovery/`, the research factory, the hypothesis generator, `research/` (all 28+ files across subdirs)** — this is the research engine. The new repo's job is executing one already-proven edge, not generating new ones. If it starts autonomously proposing hypotheses, it's not a challenge-passing repo anymore, it's a smaller copy of this one.
- **`ict/`, `ict-engine/`** — unproven (p=0.52). Carrying it over "just in case" is exactly the infrastructure-as-avoidance pattern from two sessions ago.
- **`research/yield_frontier/` (Undertow)** — real, but not confirmed yet (n=4 of 250 needed). Let it finish proving itself in the general repo; bring it into the challenge repo later, deliberately, once it clears the same bar carry already did.
- **`attic/`, `archive/`, `scratch/`, `lab/`** — already marked RETIRED in the general repo. No reason for dead code to get a second life in a new repo.
- **Most of `scripts/`** — 179 files, only ~22 are actually LIVE-firing (see `TOP_100_WHAT_THE_BACKEND_DOES.md` from last session for the exact list). Copy the ones named above by task, not the directory.
- **The full `tests/` suite (133 test-only files)** — write a new, smaller test suite that matches what actually ships in this repo. Porting tests for systems you didn't copy just leaves failing or meaningless tests on day one.

---

## 9. Data connections and API keys — so you're not starting cold

`.env` is gitignored and has never been committed (checked — `git ls-files | grep .env` returns nothing), so no agent can pull it from git history in either repo. This has to be a manual file copy. Here's exactly what to bring and what to skip.

**Copy `.env` itself, then delete the lines this repo doesn't use.** Of the 29 keys in the general repo's `.env`, a carry-only funded-account repo actually needs:

| Key | Why |
|---|---|
| `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `OANDA_LIVE`, `OANDA_BASE_URL` | The broker connection — real prices, real practice-account fills. This is the one that matters most. |
| `FRED_API_KEY` | Feeds the rate-differential calculation carry is built on. |
| `ANTHROPIC_API_KEY` | Powers the dashboard chat panel and any Oracle-style briefing you keep. |
| `NEWS_API_KEY` (maybe) | Only if you keep any sentiment/calendar-context feature from `sovereign/features/`. |

**Skip these — they're real keys, but nothing in a carry-only repo calls them:** `ALPACA_*`, `POLYGON_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `TIINGO_API_KEY`, `NASDAQ_DATA_LINK_API_KEY`, `DATABENTO_API_KEY`, `IB_*`, `TELEGRAM_*`, `FIREBASE_*`, `OPENWEATHER_API_KEY`.

**On `THETADATA_API_KEY` specifically, since you asked for it by name:** copy it if you want, it costs nothing to have present — but say plainly, same as last session: ThetaData has zero forex coverage, and this repo is carry-only. The key would sit in `.env` unused by anything currently on the copy list. If the real reason you want it there is that you're picturing this repo eventually running an earnings/options sleeve alongside carry, that's a legitimate reason to bring it — just know that's a scope decision, not a data-plumbing one, and it pulls in the Earnings Conviction packet skills too, not just a key.

**On the OANDA practice account — the one thing I'd actually stop and decide before copying, not after:** there's only one `OANDA_ACCOUNT_ID` in the whole repo. Every OANDA-connected system points at the same practice account. If the new repo reuses that same ID and its scheduled jobs ever run at the same time as the general repo's, both codebases are placing orders against the identical account — which corrupts exactly the data this repo depends on (`prop.py`'s sizing, G3's win-rate alignment, G5's non-bust-day count) with trades that came from a different system's logic, not this one's. Two honest options, not a technical requirement either way:
- **Fresh practice account, same key vendor.** Create a second OANDA practice account (free, instant, from OANDA's own site), drop its ID into the new repo's `.env`. Track record starts clean and every trade in it is provably this repo's decision. This is what I'd do.
- **Same account, but only one repo's scheduler is ever live at a time.** Keeps continuity with the paper history you already have, at the cost of having to actively remember which repo is "the one running" whenever both exist on the same machine.

**Don't recopy the broken `requirements-dashboard.txt`.** The `pyarrow` fix from last session needs to travel with it — copy the file as it stands now in this repo, not from an older commit.

## One thing to build fresh, not copy

**A new `CLAUDE.md`.** The general repo's `CLAUDE.md` governs a much bigger system — ICT isolation rules, training gates for research that won't exist here, a test-command block describing a 973-file environment. Write a short one for the new repo instead: this edge, this risk constitution, this sizing, this gate list, nothing else. Keeping the old one and hoping the irrelevant parts don't matter is how repos end up with rules nobody remembers the reason for.

---

*Alta Investments — built to pass the first funded account, not to be the general repo's sibling.*
