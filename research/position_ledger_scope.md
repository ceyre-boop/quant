# Unified Position Ledger — Scope Note

**Alta Investments · research/ · 2026-07-22**
**Purpose:** scope the `portfolio` section of the system regime contract
(`specs/system_regime_contract.md`) — the unified open-exposure-by-cluster view
that the portfolio drawdown breaker needs. Today that section is `UNAVAILABLE`
(see `scripts/build_system_regime.py::build_portfolio`). This note says exactly
what exists on disk, what is missing, and what would move it `UNAVAILABLE → OK`.
It does **not** build the ledger — the cluster taxonomy and caps do not exist yet,
so building now would mean fabricating them.

---

## Verdict

**Partial.** Raw per-strategy open-position data exists (ICT directly; forex/carry
reconstructable from fills), and the live forex account already exposes NAV / daily
P&L. But the two pieces the *breaker* actually needs — a **cluster taxonomy**
(USD_MACRO / YEN / EQUITY_SMALLCAP) and **per-cluster caps** — do **not** exist on
disk anywhere. So a clean position-*by-cluster* view cannot be assembled from what
exists today without first authoring those. Scope it; do not build it yet.

---

## What exists on disk (sources)

| Field the contract needs | Source on disk | Status | Notes |
|---|---|---|---|
| ICT open positions | `data/ledger/ict_paper_trades.json` → `open[]` | REAL, structured | `pair, direction, entry, stop, risk_dollars, grade, score, open_time`. **Paper** (each risk_dollars≈100), not live capital. |
| ICT entry/exit event stream | `data/ledger/live_trade_log.jsonl` | REAL | `type: ENTRY \| SESSION_CLOSE`, `source:"ICT"`, `ticker, direction, price, meta{stop,tp1,tp2,risk_dollars,id}`. Open set reconstructable by event replay. |
| Forex/carry fills (live OANDA) | `data/ledger/oanda_fills.json` / `.jsonl` | REAL, but **fills not net positions** | `pair, direction, units, fill_price, stop_price, trade_id, account_id`. Currently-open set requires reconciling fills (or an OANDA account-positions pull). |
| Live account NAV / daily P&L | `data/agent/equity_curve_live.jsonl` | REAL | `nav, balance, unrealized_pl, open_trade_count` per timestamp. **Forex account only** — ICT is paper and separate. |
| Per-account risk limits | `config/parameters.yml::hard_constraints` | REAL | `max_daily_loss_pct: 0.02`, `max_concurrent_positions: 5`. Already read by the writer. |

## What is missing (the gap that keeps it UNAVAILABLE)

1. **Cluster taxonomy** — no mapping of instrument → exposure cluster
   (USD_MACRO / YEN / EQUITY_SMALLCAP) exists anywhere in `config/`. The
   `cluster_*` keys in `config/ict_params.yml` are ICT **setup**-clustering
   (k-means over setups), unrelated to position-exposure clusters. This map must
   be authored.
2. **Per-cluster caps** — no `cluster_caps` key exists; only the per-account
   `max_concurrent_positions` / `max_daily_loss_pct`. New config keys → must be
   logged to `data/agent/param_change_log.jsonl` with rationale before use
   (CLAUDE.md NN#4).
3. **Net-open reconciliation for forex** — no single "currently open forex
   positions" file; must be derived from `oanda_fills.json` (fill replay) or a
   live OANDA positions pull. ICT already has a clean `open[]`.
4. **Cross-strategy P&L attribution** — `equity_curve_live.jsonl` is the forex
   account only; ICT paper P&L lives separately in `ict_paper_trades.json`. A
   single system-wide `daily_pnl_pct` needs both feeds unified (and a decision on
   whether paper ICT counts toward a live-capital breaker — likely not).

## Exact fields for `portfolio` section (drop-in once the above exist)

```
open_exposure_by_cluster: { <cluster>: <sum notional or risk_dollars> }
    ← ict_paper_trades.json open[].risk_dollars  (mapped via cluster taxonomy)
    ← oanda net positions .units × price       (mapped via cluster taxonomy)
cluster_caps:            { <cluster>: <cap> }   ← NEW config key (needs param log)
daily_pnl_pct:           equity_curve_live.jsonl (nav vs session-open balance)
daily_drawdown_limit_pct: config hard_constraints.max_daily_loss_pct (0.02) ✓ have
drawdown_breaker_tripped: (daily_pnl_pct <= -limit)  ← computed once P&L feed unified
```

## Recommended build order (later, not now)

1. Author `config/parameters.yml::clusters` (instrument→cluster) + `cluster_caps`;
   log both to `param_change_log.jsonl` with rationale.
2. Add a small fill→net-open reconciler for OANDA (or pull account positions).
3. Fill `build_portfolio()`: sum ICT `open[]` + forex net into `open_exposure_by_cluster`,
   read caps, compute `daily_pnl_pct` from `equity_curve_live.jsonl`, set the breaker.
4. Flip the portfolio section `UNAVAILABLE → OK`; wire ICT Layer-7 breaker to read
   `portfolio.drawdown_breaker_tripped`.

*Honest status: the plumbing (sources 1–5) is real; the taxonomy + caps (gaps 1–2)
are authored decisions, not data. Until they exist, `UNAVAILABLE` is the correct,
non-fabricated state.*
