# Petrules Gate — Phase 0 Data Availability Audit

**Date:** 2026-07-21 · **Status:** COMPLETE · **Spec:** `research/PETRULES_GATE_SPEC.md`
**Probe evidence:** `research/petrules_audit/probes/` · **Probe code:** `research/petrules_audit/probe_sources.py`
**Rule enforced throughout:** availability timestamps = filing/publication dates only. Public data only.

---

## Verdict up front: **NARROW**

The Gate is buildable, but not as pre-registered. Two features die cleanly, one is degraded,
and the options history window is 2020-01→present on the current ThetaData tier. The
pre-registration must be narrowed **before** it is filed — none of this requires touching
holdout data, so it is a free correction now and a leak later.

**Single biggest risk (one sentence):** No free source exposes the analyst-estimate *revision
path* as of time T — only the final pre-print consensus snapshot — so any
`consensus_revision_momentum` / `analyst_revision_trend` feature built from today's data
would be a lookahead leak into the organ whose entire product is confidence; these features
must be dropped or replaced, not approximated.

---

## Per-source table

| Source | Access | Years | Point-in-time? | Coverage | Verdict |
|---|---|---|---|---|---|
| Alpha Vantage `EARNINGS` | free key on file (25 req/day) | 1996→present (AAPL: 121 quarters) | **Y — final pre-print consensus only** (`estimatedEPS`, `surprise`, `reportedDate`, `reportTime`); no revision path | full for US listed; spot-checks match historical record | **USABLE** (surprise label + earnings-date spine); **UNUSABLE-lookahead** for revision-path features |
| Alpha Vantage `EARNINGS_ESTIMATES` | same key | current only | N — returned `estimates: []` even for AAPL | n/a | **UNAVAILABLE** |
| Yahoo Finance `earningsTrend` | unofficial | current snapshot only | N | n/a — HTTP 401 "Invalid Crumb" unauthenticated | **UNUSABLE-lookahead** (current-only by design, and access is brittle) |
| Finnhub estimates | no key; free tier | — | — | HTTP 401 | **UNAVAILABLE** (estimates are paid-tier) |
| Nasdaq.com analyst API | free, UA header | current only (`asOf: null`, forward quarters only) | N | n/a | **UNUSABLE-lookahead** |
| Nasdaq Data Link (Zacks) | key on file; Zacks datasets are premium | — | — | not probed (paid) | **UNAVAILABLE** at budget |
| ThetaData options EOD (Value tier) | ThetaTerminal :25503, live | **2020-01→present** (2019-10 and earlier → "requires STANDARD/PROFESSIONAL subscription"; cutoff pinned between 2019-10-14 ✗ and 2020-01-13 ✓) | Y — event-dated chains with bid/ask as of date (AAPL 2023-08-02 chain for 2023-08-04 expiry: 112 contracts, quotes populated) | full chain incl. event-dated weeklies | **USABLE** (implied move + skew), 2020+ window |
| SEC EDGAR Form 4 (submissions JSON) | free, no key | full history | Y — `filingDate` distinct from transaction `reportDate` | n=6,658 filings across 12 tickers | **USABLE** |
| SEC EDGAR 13D/G (full-text search) | free | 2001→present (FTS) | Y — filing date is the event | 2021: 5,445 · 2023: 5,865 · 2025: 5,637 (⚠ 2025 files under new form name `SCHEDULE 13D`, not `SC 13D` — query must cover both) | **USABLE** |
| SEC EDGAR 13F | free | full history | Y — `filingDate` vs `reportDate` explicit | Berkshire/Bridgewater/Renaissance: n=167 filings | **USABLE with structural staleness** (median 44–45d) |
| HouseStockWatcher / SenateStockWatcher S3 | dead | — | — | HTTP 403 on both documented S3 buckets | **UNAVAILABLE** (mirrors defunct) |
| House Clerk official FD archive | free ZIP per year (`disclosures-clerk.house.gov/public_disc/financial-pdfs/{Y}FD.zip`) | ≥2021 verified | index has `FilingDate` (publication) only; **transaction dates are inside per-filing PDFs** | PTR count: 2021: 680 · 2023: 460 · 2025: 515 filings | **USABLE-with-parsing-cost** (lag measurable only after PDF parsing — Phase 1 work if pursued) |
| Senate eFD (`efdsearch.senate.gov`) | reachable (302 → session flow) | — | same PDF problem + CSRF session scraping | not measured | **USABLE-with-parsing-cost** (worse access than House) |
| FINRA consolidated short interest API | free, no key (`api.finra.org`) | ≥2020-04 verified in response | settlement-dated; publication lags settlement (~7 business days — inferred from FINRA docs, not measured) | full market | **USABLE** (must timestamp by publication, not settlement) |
| Daily bars (Alpaca SIP / Tiingo / Polygon) | keys on file | deep, incl. delisted (Alpaca SIP, per prior verified work HYP-092) | Y | full | **USABLE** |

## Q2 measurements — disclosed-flow lead time (measured, not assumed)

- **Form 4 disclosure lag** (filingDate − transaction reportDate), 12 tickers, **n=6,658**:
  median **2 days**, p90 4 days, 96% within 4 days. The 2-business-day statute holds in practice.
- **Form 4 cluster → next earnings lead time** (clusters = ≥3 filings in 30d, dated by *last
  filing date*; earnings spine = AV `reportedDate`; AAPL/DKS/CROX/CAT, **n=791 clusters**):
  median lead **56 days**, p10 = 13 days, **100% arrive ≥1 day before the next earnings event**.
  → Form 4 comfortably precedes the events it should predict. **KEEP.**
  (Caveat: clusters here are direction-blind — buy/sell parsing from the Form 4 XML is Phase 1.)
- **13F staleness** (filingDate − period end), 3 large filers, n=167: median **44–45 days**.
  Structurally the filing still precedes the *next* quarterly earnings of held names (~45 more
  days later), so it is usable — but only as a slow accumulation feature, never as an
  event-timing feature. **KEEP, demoted expectation.**
- **13D/G**: the filing IS the catalyst (lead time is definitionally ≥0). Volume ~5,400–5,900/yr
  supports the sample-size target. **KEEP.**
- **Congressional**: statutory lag up to 45 days vs. transaction; **measured lag: NOT MEASURABLE
  at Phase-0 cost** — both free mirrors are dead (403) and official sources put transaction
  dates inside PDFs. Volume is also thin (~460–680 House PTR filings/yr, further split across
  thousands of tickers → per-ticker-90d hit rate near zero). **DROP from Tier 1.**

## Q1 answer — point-in-time analyst estimates

Split the spec's need in two:

1. **Final pre-print consensus snapshot** (needed for the surprise label and
   `consensus_eps_estimate`): **AVAILABLE FREE.** AV `EARNINGS` carries `estimatedEPS` per
   historical quarter back to 1996 with `reportedDate` and `reportTime`. Spot-checks against
   the known record match (AAPL 2023-08-03: 1.26 reported vs 1.19 estimate; 2024-08-01:
   1.40 vs 1.34; 2020-07-30: 0.65 vs 0.51 split-adjusted) — these are the consensus figures
   as they stood at print, not restated values.
2. **Revision path over T-90..T-1** (needed for `consensus_revision_momentum`,
   `analyst_revision_trend`, `analyst_commentary_trend`, `consensus_price_target` history):
   **NOT AVAILABLE free or cheap.** Every free source (Yahoo, Nasdaq, AV `EARNINGS_ESTIMATES`)
   serves only the current standing estimate. Historical revision vintages are Refinitiv/Zacks
   paid territory. Per the spec's own fallback clause (§Subsystem 1, availability constraint):
   **the consensus baseline falls back to options-implied-move** (+ the pre-print snapshot for
   the label).

## Per-feature verdicts

| Feature (tier) | Buildable clean? | Basis / fallback |
|---|---|---|
| `consensus_revision_momentum` (T1) | **NO — DROP** | No point-in-time revision history at budget. Fallback: none honest. Replace with options-implied-move dynamics (change in implied move over T-30..T-1, ThetaData). |
| `options_skew_direction` (T1) | **YES (2020+)** | ThetaData event-dated chains, bid/ask by date. |
| `disclosed_flow_vs_consensus` (T1) | **PARTIAL** | Form 4 side: yes (lead measured, median 56d pre-event). "vs consensus" side: consensus direction cannot come from revision trend (dead) — redefine vs. options-implied/positioning direction. Deviation note required. |
| `activist_disclosure_recent` (T1) | **YES** | EDGAR 13D/G filing dates; must query both `SC 13D` and `SCHEDULE 13D` form names. |
| `institutional_accumulation` (T1) | **YES, stale by construction** | 13F filingDate only; feature value freezes at filing date, 44–45d after quarter end. |
| `congressional_trade_direction` (T1) | **NO — DROP** | Mirrors dead; official sources = per-filing PDF parsing with ~500 filings/yr total — cost high, per-ticker coverage negligible. |
| `price_vs_52w_high` (T2) | **YES** | Daily bars, incl. delisted (Alpaca SIP). |
| `volume_ratio_20d` (T2) | **YES** | Same. |
| `short_interest_ratio` (T2) | **YES with publication-lag discipline** | FINRA API verified ≥2020-04; timestamp by publication (+~7bd), not settlement. |
| `earnings_surprise_history` (T2) | **YES** | AV `EARNINGS` prior-quarter surprise fields, 1996+. |
| Tier 3 (sentiment) | log-only per spec | Not audited — excluded from model v1 by pre-registration anyway. |

## Sample-size check against the 5,000-event target

Binding constraint is the options window (2020-01→present ≈ 26 quarters). US-listed
≥$500M names ≈ 1,500+ → order 30–40k earnings events with implied-move coverage, plus
~5,500 13D events/yr. **Target is reachable inside the narrowed window.** Upgrading
ThetaData to STANDARD would extend to 2012+ (AAPL expirations verified back to 2012-06)
— an optional paid widening, not a blocker.

## Recommended narrowing (to bake into the pre-registration before hash-lock)

1. Drop `consensus_revision_momentum` and `congressional_trade_direction` from Tier 1.
2. Add `implied_move_change_30d` (options-implied move dynamics) as the replacement
   consensus-expectation feature.
3. Redefine `disclosed_flow_vs_consensus` as disclosed-flow direction vs. options-implied
   direction (not analyst-revision direction).
4. Consensus baseline = options-implied move (both earnings and non-earnings setups);
   AV pre-print snapshot used for the divergence *label* on earnings events.
5. Training window pinned 2020-01→present unless ThetaData STANDARD upgrade is bought.

## Verified-by-query vs. inferred-from-docs

**Verified by live query (evidence in `probes/`):** AV EARNINGS history + field shape + 3
spot-checks; AV EARNINGS_ESTIMATES empty; Yahoo 401; Finnhub 401; Nasdaq current-only; ThetaData
expirations 2012–2028, event-dated 2023 chain, tier cutoff (2019-10 ✗ / 2020-01 ✓); Form 4 lag
(n=6,658); cluster lead times (n=791); 13F staleness (n=167); 13D volumes 2021/2023/2025 incl.
form-name change; stock-watcher 403s; House FD ZIP index shape + PTR counts 2021/2023/2025;
FINRA short-interest API response.
**Inferred from docs (not measured):** FINRA publication lag (~7bd after settlement); Zacks/NDL
pricing; Senate eFD scraping difficulty (reachability verified, session flow not exercised);
AV estimate provenance as at-print snapshot (strongly supported by spot-checks but the
vendor's snapshot methodology itself is taken from documentation).

---

*Phase 0 complete. Per spec: pre-registration (`PETRULES_GATE_prereq.json`) may now be filed —
with the narrowing above — before any Phase 1 code.*
