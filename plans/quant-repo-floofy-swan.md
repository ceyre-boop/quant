# Repurpose quant as a research terminal

## Context

This repo was built as a strategy-discovery engine. The discovery surfaces have
returned null repeatedly — ICT pattern edge fails permutation testing (p=0.52), the
research queue has executed nothing since 2026-08-16, and the funded-challenge
apparatus is no longer relevant to what the repo is for.

The decision is to stop building discovery and rebuild the front end as a single
**research terminal**: one place that surfaces everything worth looking at, pulled from
sources already wired up. The edge is coverage and speed of access, not a secret signal.

Judgment stays with the user. This is a **pull** surface — you open it and browse. No
alerting, no "the system tells you what to trade."

The new centre of gravity is a **fundamentals and filings layer**, which the stack
currently has essentially nothing of. It is heavy on price and sentiment; the decisive
information — what a company guided, what it printed, how the stock reacted, who is
buying and selling it — is absent.

### Decisions taken

| Question | Answer |
|---|---|
| Data budget | Free sources now, behind a provider interface so a paid key drops in later without a rewrite |
| Universe | Curated watchlist warmed nightly + any ticker on demand |
| Hosting | GitHub Pages first; Vercel or Render later. Build must port cleanly |
| Front end | Vite + React + TypeScript + Tailwind, building to `_site/` |

---

## What exists today (verified)

**Live front end** is `index.html` at the repo root — 3,900 lines, 228 KB, one
hand-written file with inline `<style>` and `<script>`. No bundler, no framework, no
build step. Six tabs: Trade, Signals, Research, Trades, Calendar, Chat.

**Backend** is `scripts/live_signals_server.py` — 1,281 lines, hand-rolled
`BaseHTTPRequestHandler` on port 8765, `do_GET` at :804 as an `elif` chain. Hosted on
Render free tier (sleeps after ~15 min, hence the "waking the live server (~30s)" retry
paths in the UI). Front end picks its base URL at `index.html:1393-1397`.

**Deploy** is `.github/workflows/deploy.yml`, GitHub Pages, **branch `master` only**,
build step is a raw `cp` sequence. Current work branch is `sovereign-v2`, so nothing
committed here reaches Pages until it is pushed to master — the existing
`scripts/build_dashboard.sh` does exactly that.

**QA** is `scripts/qa/dashboard_qa.mjs` (puppeteer-core, headless Chrome against
localhost:8765). Not wired into CI. It asserts no panel is left reading "Loading…"
(`A.noLoading`, L42-45) and that the hypothesis filter buttons work (L171) — both
assertions must be updated as panels change.

---

## Constraint discovered during exploration — read before removing the prop panel

`sovereign/intelligence/allocation_engine.py:170-182` (`_load_challenge_buffer()`) reads
`data/agent/prop_challenge_state.json`, and lines 260-271 use it to **cut equity weight
by up to 50%**. Deleting that file silently changes live allocation, which the
execution-path freeze (CLAUDE.md standing constraints) forbids without a recorded unlock.

**Therefore the funded-challenge removal is UI-only.** The panel, the `/prop-challenge`
endpoint and the Research-tab gates card go. The state file, its generator
(`scripts/build_paper_accounts.py`), the `com.alta.paper_accounts` launchd job and
`sovereign/risk/monte_carlo_prop.py` all stay running untouched. Python scoring code is
left in place and simply stops being surfaced.

Also verified as *not* the feature, do not touch: `execution/risk.py`
(`PROP_EVAL_RISK_CAP`, `FW_PROP_EVAL` breach) is the live risk ladder.

---

## Verified data-source facts

Tested directly against the live endpoints:

| Endpoint | CORS (tested with `Origin:`) | Usable how |
|---|---|---|
| `data.sec.gov/submissions/CIK*.json` | `access-control-allow-origin: *` | **Browser-direct** |
| `www.sec.gov/Archives/...` | none | Server only — **and this is where Form 4 share amounts and 13F holdings live** |
| `efts.sec.gov/LATEST/search-index` | none | Server only |
| `data.sec.gov/api/xbrl/companyfacts` | none | Server or pre-bake |
| `www.sec.gov/files/company_tickers.json` | none | Baked once, committed |
| `api.finra.org` / `api.nasdaq.com` short interest | none | Server or pre-bake |

> **Correction, made during planning.** An earlier read of this table claimed `efts.sec.gov`
> was CORS-open. It is not. The first test omitted an `Origin:` header — `data.sec.gov`
> returns ACAO unconditionally, `efts.sec.gov` returns it never. Re-tested three times with
> `Origin: https://taboost.github.io`. `www.sec.gov/Archives` is likewise closed.

Exactly **one** useful endpoint is browser-reachable: `data.sec.gov/submissions`. It yields
filing forms, dates and accession numbers — so a browser can show *that* a Form 4 was filed
and *how many* in the last 30/90/180 days, but never how many shares at what price, because
that detail lives in `www.sec.gov/Archives`. Everything else is pre-baked or proxied.

**Honest gap:** there is no good free source for forward analyst *consensus* estimates.
`sovereign/sentiment/surprise_feed.py` already documents this and deliberately labels its
output `release_innovation` rather than "surprise". Alpha Vantage `EARNINGS` gives
`reportedEPS` vs `estimatedEPS` per quarter (historical consensus at the time of print),
which covers estimates-vs-actuals **backwards** but not forward guidance. The provider
interface exists so that a paid key (FMP/Finnhub) closes this gap later.

**Build prerequisite:** `.venv313` is missing `lxml`, so `yfinance.get_earnings_dates()`
raises today. Must be installed before the earnings path works.

---

## Panel plan

### Remove
- **Funded-challenge panel** — `index.html:854-895` markup, `:1667-1718` JS, `:110-112`
  and `:595` CSS, `:971-972` control buttons, `:1263` + `resGates` (:3070) research card,
  `:1304`/`:1660` chat context strings. Backend `/prop-challenge` at
  `live_signals_server.py:884-894`. Data generation stays (see constraint above).
- **Research tab analysis sections** — the ledger, queue, velocity, regime, indicator
  consensus, proof-of-life, reflection and version-tracker cards
  (`index.html:1260-1278`, renderers `:3070-3439`).

### Keep, ported verbatim
- **TradingView embed** (`index.html:1077-1087`, `sigTVInit` :1836-1856) and the
  **"↗ Open on TradingView"** link-out (`sigOpenOnTV` :1857-1860). Standard `tv.js`
  widget; the only state is the `data-fut`/`data-proxy` symbol mapping.
- **Signals panel** — live signal status.
- **Calendar + replay cockpit** — markup `:1088-1127` / `:1283-1300`, JS `:1871-2143`.
  Uses LightweightCharts and direct `getElementById` manipulation. Ported as a
  **framework-free island module** that React mounts via `useEffect`, so its behaviour is
  preserved exactly rather than rewritten. `calOpenReplay` (:2117-2128) currently does a
  `.click()` plus nested `setTimeout`; that becomes a real state call.
- **Oracle** — the in-app assistant (Chat tab, `:1301-1320` / `:3474-3573`, `POST /chat`).

### Reduce
- **Research tab → Connections.** Key management and data-connection status only: which
  APIs are wired, which are failing, where to add a key. Absorbs the existing control-token
  widget (`:981-983`, JS :1719-1824) and the `health.json` status rendering (`resHealth`
  :3274).

### Build — the fundamentals + filings layer
Per-ticker, populated by the same symbol selection that drives the chart:
earnings history (guide vs actual), estimates vs actuals with surprise, price reaction
history per print, insider activity (Form 4), institutional positioning (13F), short
interest.

---

## Integration reconciliation

The list in the brief was transcribed from audio and is close to the `components` dict in
`data/agent/health.json`. Reconciled against the repo:

- **Confirmed live:** yfinance (keyless, 209 files), Alpaca, FRED, Alpha Vantage, OANDA
  (the live execution venue — *omitted from the brief's list*, and it matters), Anthropic,
  GDELT, CFTC COT, Databento, Interactive Brokers, Ollama, DuckDB, ICT scanner, backtest
  engine, decision log.
- **Health-ping only, no consumer (dormant):** OpenWeather, Nasdaq Data Link, Tiingo,
  Firebase, Polygon (nominally free-tier "reference data" but live calls fall through to
  yfinance/Alpaca).
- **Degraded:** Reddit — cache 10,264 min stale; its plist exists in `scripts/` but was
  never installed to `~/Library/LaunchAgents`.
- **Conditional:** ThetaData — needs a local ThetaTerminal, skips gracefully when down.
- **In the repo, on neither list:** SEC EDGAR (research-tier), ForexFactory scraper,
  Telegram, Tradovate, TradeLocker, MT5/cTrader, Twitter (dead code).
- **"Oracle Pulse" does not exist as a panel.** Nearest things are `pulse_check` in
  `execution/risk.py` and a "Research Pulse" card in the non-deployed
  `dashboard/index.html`. This is a greenfield slot, not a port.

Two staleness traps to fix while here: `health.json` was last generated 2026-07-27 because
**`sync_dashboard_data.py` has no launchd job**, and `com.alta.dashboard-publish.plist` is
loaded but points at `/Users/taboost/passing-funded-account-1-/daytrade/dashboard_publish.sh`,
outside this repo.

---

## The fundamentals layer — source map (every row verified live)

| # | Category | Free source | Endpoint | Access | Cadence |
|---|---|---|---|---|---|
| 1 | Earnings history, guide vs actual | Alpha Vantage `EARNINGS` + yfinance | `query?function=EARNINGS` | Server, **25 req/day cap** | Nightly, watchlist rotation |
| 2 | Estimates vs actuals + surprise | Alpha Vantage `EARNINGS` (`reportedEPS` vs `estimatedEPS`) | same call as #1 | Server | Same call — free |
| 3 | Reaction history per print | **No new source** — join earnings dates to existing price data | `sovereign/data/adapter.py` | Local | Computed |
| 4 | Insider activity (Form 4) | SEC EDGAR submissions | `data.sec.gov/submissions/CIK*.json` | **Browser-direct (CORS `*`)** | On demand |
| 5 | Institutional positioning (13F) | SEC quarterly **bulk 13F datasets** | `sec.gov/files/structureddata/data/form-13f-data-sets/<range>_form13f.zip` | Server, bulk → DuckDB | Quarterly |
| 6 | Short interest | FINRA / Nasdaq | `api.finra.org`, `api.nasdaq.com` | Server, no CORS | Bi-monthly |
| — | Filings list | SEC EDGAR submissions | same as #4 | **Browser-direct** | On demand |
| — | Ticker → CIK | SEC `company_tickers.json` (795 KB) | `sec.gov/files/company_tickers.json` | Baked once, committed | Monthly |
| — | CUSIP → ticker | **OpenFIGI, keyless** | `api.openfigi.com/v3/mapping` | Server, 25/min ×10 batch | Cached permanently |

Two things this table settles that were not obvious:

- **13F is not in the issuer's feed.** AAPL's submissions returns 589 Form 4s and *zero*
  13F-HR, because 13F is filed by the institution, not the company. Per-ticker
  institutional positioning therefore comes from the quarterly bulk ZIPs aggregated by
  CUSIP — which is why CUSIP→ticker mapping is on the critical path.
- **Reaction history costs nothing.** It is a join between earnings dates and price bars
  the repo already fetches. No new integration.

**The static-hosting floor:** bake the ticker→CIK map into
`data/fundamentals/cik_map.json` and commit it — required, because `company_tickers.json`
is itself not CORS-open, so without the baked map even this tier fails. The browser then
resolves any ticker with zero network and hits `data.sec.gov/submissions` directly. That
gives an **honest but thin** on-demand card for any ticker with no backend: filing calendar,
Form 4 *cadence*, and 13D/G/13F presence flags. Amounts require the server.

### Warm vs on-demand

- **Warm** (nightly → committed JSON under `data/fundamentals/<TICKER>.json`, served
  statically): earnings history, estimates/actuals/surprise, reaction history, short
  interest, 13F positioning. Watchlist is the existing `WATCHLIST_SYMBOLS` in `.env`
  (23 tickers — no new config needed).
- **On demand, any ticker, no backend (Tier B floor)**: filing calendar, Form 4 *counts*
  per window, 13D/G/13F presence flags. Rendered with `partial: true` and a per-gap reason
  string — never an empty chart where a real one would go.
- **On demand, full detail (Tier A)**: needs the server. Insider amounts, estimates,
  reaction, short interest, 13F holdings. On Render's free tier the first call costs
  30–50s, so the UI shows an explicit "waking backend ~40s" state, not a generic spinner.
  Moving to Vercel later removes the cold start and makes Tier A the default.

Because five of six categories are structurally unreachable from a browser, **all six are
pre-baked for the watchlist.** That is a consequence, not a hedge — and it makes the warm
path a single ~20 KB gzipped fetch with no waterfall.

### Reuse, not rebuild

`research/petrules/sources.py` is already an honest free-source layer: real fixtures
copied verbatim from probe responses, explicit `ABSENT` provenance, and a docstring
stating it never fabricates a value. **Promote and generalise it** into
`sovereign/fundamentals/` rather than writing a new client. Its only real limits are a
hardcoded 12-ticker CIK map (replaced by the baked SEC map) and a fourth private
re-implementation of `.env` parsing.

- Reuse `sovereign/data/cache.py` (`DataCache`, parquet, `CacheStats`) for price bars in
  the reaction-history join.
- Follow the `sovereign/sentiment/store.py` DuckDB pattern (`INSERT OR REPLACE` upserts)
  for the new `data/fundamentals.db`. Do **not** import it — the sentiment store is
  isolated by design.
- Do **not** route through `sovereign/data/adapter.py` for filings; it is a bars
  abstraction and fundamentals are not bars.

The provider interface (`sovereign/fundamentals/providers/`) declares per source whether
it is `browser_direct`, so the same declaration drives both the harvester and what the
front end is allowed to fetch itself. A paid provider (FMP/Finnhub) slots in as one more
module and closes the consensus-estimates gap.

**Honest gap, stated plainly:** there is no good free source for *forward* analyst
consensus. Alpha Vantage gives historical consensus-at-the-time-of-print, which covers
estimates-vs-actuals looking backwards but not forward guidance.
`sovereign/sentiment/surprise_feed.py` already documents this and labels its output
`release_innovation` rather than "surprise" for exactly this reason. The panel will do the
same rather than implying a number it does not have.

---

## Build status (updated 2026-09-01)

Work is on `sovereign-v2`, pushed. Commits: `5206259`, `24748b0`, `a92170b`, `174c6d7`, `43d4cfc`.

**Done and verified by running it:**

| Step | State | Evidence |
|---|---|---|
| 0 — prerequisites | DONE | `lxml` installed and pinned; `yfinance.get_earnings_dates('AAPL')` returns 25 quarters |
| 4 — front-end scaffold | DONE | `app/` builds to `_site/`, 112 KB gzipped |
| 5 — port the keepers | DONE | QA confirms the TradingView container + link-out exist and the replay island creates its canvas |
| 7 — Connections panel | DONE | 21 integrations, keyless distinguished, backend probe live |
| 8 — removals (UI only) | DONE | prop panel and 13 research cards gone; generator untouched; isolation test passes |
| 9 — deploy wiring | DONE | `deploy.yml` builds with bun and copies `data/fundamentals/` |
| 10 — QA | DONE | 0 assertion failures across 7 views; HTTP failures gated by URL |
| 2 — fundamentals data layer | PARTIAL | types, errors, store (11 tables), httpcache, all 5 transports verified live; providers landed; registry/panel/reaction in flight |
| 3 — nightly harvester | IN FLIGHT | — |
| 6 — fundamentals panel | DONE (UI) | renders honest empty states; fills once the harvester runs |

**Three real bugs QA caught that code review would not have:**
1. `/api` prefix 404'd every backend call when the Python server served the app.
2. Signals called `.toUpperCase()` on a numeric `signal` field and threw.
3. No error boundary, so (2) blanked every other panel. Fixed; boundaries added per panel.

**Open — needs a real browser, not this environment.**
Headless Chrome here cannot reach any fresh external host, so the browser-direct
SEC fetch could not be verified. `curl` with browser-identical headers *does*
receive `access-control-allow-origin: *`. Since that path is the entire static
on-demand tier, verify in real Chrome (Interceptor) before relying on it. The UI
already degrades to an explained `partial` state if it fails.

**Pre-existing, not caused by this work:** `GET /replay` 500s with
`IB required for MNQ — no data` (`sovereign/futures/bar_feed.py:148`). Recorded in
the QA harness as known-degraded so it stays visible without being absorbed.

---

## Implementation

Work on `sovereign-v2`. The working tree currently has ~70 changed files of data churn —
commit or stash that separately so it does not mix into this work.

**1 — Prerequisites.** `uv pip install --python .venv313/bin/python lxml` (yfinance
earnings is broken without it). Bake `data/fundamentals/cik_map.json` from SEC.
*Verify:* `.venv313/bin/python -c "import yfinance,lxml; print(yfinance.Ticker('AAPL').get_earnings_dates(limit=4))"`

**2 — Fundamentals data layer.** `sovereign/fundamentals/` — provider interface, SEC
client (submissions, bulk 13F, CIK map), Alpha Vantage earnings with a hard call budget,
FINRA/Nasdaq short interest, OpenFIGI CUSIP mapping, DuckDB store at
`data/fundamentals.db`, reaction-history join.
*Verify:* `.venv313/bin/python -m sovereign.fundamentals.probe AAPL` prints all six
categories with a provenance tag on each, `ABSENT` where genuinely unavailable.

**3 — Nightly harvester.** `scripts/harvest_fundamentals.py` → writes
`data/fundamentals/<TICKER>.json` + `index.json`. Budget: Alpha Vantage ≤20 calls/day
(rotating the watchlist so full refresh takes ~2 days), SEC ≤10 req/s, OpenFIGI ≤25/min.
Scheduled as `scripts/com.alta.fundamentals.plist`.
*Verify:* `.venv313/bin/python scripts/harvest_fundamentals.py --dry-run --limit 2`

**4 — Front-end scaffold.** Vite + React + TS + Tailwind at `app/`, building to `_site/`.
`lightweight-charts@4.1.3` becomes an npm dep (it is an unpkg script tag today).
*Verify:* `npm run build && npm run preview`

**5 — Port the keepers.** TradingView embed + link-out as `Chart.tsx` (a standard `tv.js`
widget; the only state is the `data-fut`/`data-proxy` symbol mapping). Replay cockpit and
calendar as a **framework-free island module** — 38 `getElementById` calls across
`index.html:1871-2143`, so it moves as-is into `app/src/islands/replay.ts` and React
mounts it via `useEffect`. `calOpenReplay`'s `.click()` + nested `setTimeout` becomes a
real state call. Then Signals and Oracle/Chat.
*Verify:* replay a known day and compare the order tape against the current dashboard
side by side.

**6 — Fundamentals panel.** Per-ticker, driven by the same symbol selection as the chart.
Six sections; each renders its provenance and an explicit empty state.

**7 — Connections panel.** Research tab reduced to key management and connection status.
Absorbs the control-token widget and `health.json` rendering. Also fixes the two staleness
traps: give `sync_dashboard_data.py` a launchd job so `health.json` stops being five weeks
stale, and either repoint or unload `com.alta.dashboard-publish.plist`, which currently
runs a script outside this repo.

**8 — Removals.** Prop panel markup/JS/CSS/control-buttons/chat-strings and the
`/prop-challenge` endpoint. Research analysis cards. **Data generation stays** — see the
allocation-engine constraint above.
*Verify:* `python3 -m pytest tests/ -k test_pipeline_does_not_import_sovereign` still
passes, and `sovereign/intelligence/allocation_engine.py` still finds its state file.

**9 — Deploy.** `deploy.yml` gains `actions/setup-node` + `npm ci && npm run build`; the
`cp index.html _site/index.html` line is replaced by the Vite output. Data copies stay,
plus `data/fundamentals/`. Note deploy fires on **master only** — publish via the existing
`scripts/build_dashboard.sh` push pattern.
*Verify:* Interceptor against the deployed Pages URL — chart renders, a watchlist ticker
populates all six fundamentals sections, an off-watchlist ticker shows insider and filings
and an honest empty state for the rest.

**10 — QA + docs.** Update `scripts/qa/dashboard_qa.mjs` — its `A.noLoading` assertion
(L42-45) and hypothesis-filter assertion (L171) both break when the Research tab changes.
Update `README.md`, `NEXT.md`, and `.env.example` for any new keys.

---

## Out of scope

`ict/index.html` (the separate ICT Oracle page at `/ict/`) stays exactly as it is — the
brief does not mention it and it deploys independently. The orphaned HTML files
(`dashboard/`, `trading-dashboard.html`, `sovereign_dashboard.html`,
`prediction-framework.html`, `frontend/live_signals.html`) are left alone rather than
deleted; they cost nothing and deleting them is a separate decision.

No new strategy discovery. No automated execution. No rebuilding of charting or indicators.

---

## Top risks

1. **Breaking the replay cockpit in the port.** It is the feature explicitly called out as
   worth keeping. Mitigated by moving it verbatim as an island rather than rewriting it,
   and by side-by-side comparison against the current dashboard before the old file goes.
2. **Silently changing live allocation.** Mitigated by making the prop removal UI-only and
   asserting the allocation engine still reads its state file.
3. **Alpha Vantage's 25-call/day ceiling making the warm set feel stale.** Mitigated by
   rotating the watchlist across days, storing every fetch permanently in DuckDB (earnings
   history is immutable once printed, so it is fetched once per ticker per quarter, not
   daily), and surfacing the fetch timestamp on the panel so staleness is visible rather
   than implied.
