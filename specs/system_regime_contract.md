# System Regime Contract — Spec
## Alta Investments · specs/ · 2026-07-22
### The live nervous system that connects every strategy to one regime read

**Status:** SPEC (pre-code) · **Design principle:** [[Connected-Edge-and-Regime]]
**Isolation:** builds as a NEUTRAL layer — imports nothing from `ict/` or `sovereign/`;
communicates by data contract only, so the `ict/` ↔ `sovereign/` wall is never touched.
**Freeze:** touches nothing on the execution path. New files only.

---

## The one idea

Every strategy already produces (or can produce) its own regime read. They are
disconnected — carry reads `forex_proximity.json`, ES/NQ reads `nqes_regime.json`, ICT
reads nothing unified, the gapper reads nothing. **Connected Edge × Regime** says a
strategy is only tradeable when its edge AND its regime are favorable at the same time.
This contract makes that enforceable *live and system-wide*: one canonical file every
strategy consults before it sizes, so no strategy can be long a good edge in a bad regime,
and one portfolio view so risk sees everything at once.

It is the "mirror" pattern from 2026-07-21 (`obsidian_sync`) applied to regime: read the
scattered per-strategy signals, write one honest unified contract, everyone reads it.

---

## Why a data contract, not shared code

`ict/` may never import `sovereign/` (CLAUDE.md NN#1). A shared *import* is therefore
impossible. A shared *file* is not an import — reading JSON crosses no wall. So the
connective layer is a **contract**:

- A neutral writer (lives in `platform/`, a new top-level package that imports neither
  side) reads each strategy's existing regime signal and computes one unified state.
- Each strategy reads its own section of that state before sizing. Reading a file is legal
  from inside `ict/` and from inside `sovereign/`.
- No strategy imports another. The wall stands. The system is connected.

---

## The contract file

`data/agent/system_regime_state.json` — written by the platform writer, read by all.

```json
{
  "generated_at": "2026-07-22T...Z",
  "status": "OK | STALE | DEGRADED",
  "status_reason": "...",
  "strategies": {
    "carry": {
      "verdict": "GO | CAUTION | STAND_ASIDE",
      "favorable": false,
      "reason": "rate differentials NARROWING on 3/4 pairs; none widening",
      "size_multiplier": 0.0,
      "detail": { "per_pair": { "AUDUSD": {"differential_trend": "NARROWING", ...}, ... } },
      "source": "data/agent/forex_proximity.json",
      "source_age_hours": 0.4
    },
    "ict_equities": {
      "verdict": "...", "favorable": ..., "size_multiplier": ...,
      "detail": { "per_symbol": { "META": {"trend_label":"TRENDING","vol_percentile":"HIGH",...} } },
      "source": "computed from PriceStore ADX/ATR", "source_age_hours": ...
    },
    "es_nq":   { "verdict": "...", "detail": {"rotation":"ROTATION_WARN"}, "source": "data/research/nqes_regime.json", ... },
    "undertow_gapper": { "verdict": "...", "detail": {"vol_regime":"CALM","disaster_watch":false}, "source": "...", ... }
  },
  "portfolio": {
    "open_exposure_by_cluster": { "USD_MACRO": 0.0, "YEN": 0.0, "EQUITY_SMALLCAP": 0.0 },
    "cluster_caps": { "...from config/parameters.yml..." },
    "daily_pnl_pct": 0.0,
    "daily_drawdown_limit": "...from RISK_CONSTITUTION.md...",
    "drawdown_breaker_tripped": false
  }
}
```

### Rules the contract enforces
- **`verdict` per strategy** — GO / CAUTION / STAND_ASIDE, derived from that strategy's
  own regime inputs by a small, per-strategy classifier (each documented). This is the
  live form of Connected Edge × Regime.
- **`size_multiplier`** — 0.0 (stand aside) … 1.0 (full) … up to a configured max when the
  regime is not just favorable but *improving*. A strategy multiplies its own sizing by
  this before it ever reaches position sizing.
- **`favorable` requires improving, not merely present** — TRENDING price with NARROWING
  differential is NOT favorable for carry. The classifier must read the *trend of the
  regime driver*, not its level.
- **Portfolio drawdown breaker** — one system-wide flag. If daily loss across ALL
  strategies exceeds the limit, `drawdown_breaker_tripped = true` and every strategy's
  `size_multiplier` is forced to 0 for the rest of the session.
- **Fail loud** — `status` OK/STALE/DEGRADED with a reason. A source older than its
  freshness limit → that strategy's section is STALE and its verdict downgrades to
  STAND_ASIDE (never trade on a stale regime read). Never emit a favorable verdict from
  missing data.

---

## The writer (`platform/regime_contract.py` + `scripts/build_system_regime.py`)

Runs on a slow clock (every 30 min, like the mirror). Steps:

1. **Read each strategy's existing signal** (files, not imports):
   - carry → `data/agent/forex_proximity.json` (`differential_trend` per pair)
   - es_nq → `data/research/nqes_regime.json`
   - macro backdrop → `data/macro/macro_snapshot.json` (VIX, curve, credit)
   - ict_equities → compute ADX/ATR-percentile/correlation from the same PriceStore the
     ICT pipeline uses (or a cached bar source); if unavailable, section = STALE.
   - undertow_gapper → vol/disaster regime from the gapper shadow + VIX.
2. **Classify each into a verdict + size_multiplier** using a per-strategy rule (below).
3. **Assemble the portfolio section** from the unified position ledger + config caps.
4. **Write `system_regime_state.json`** with per-section status and timestamps. Never raise.

### Per-strategy verdict rules (each small, documented, config-driven)
- **carry:** favorable iff ≥1 pair shows `differential_trend == WIDENING` AND VIX not
  EXTREME. STAND_ASIDE if all pairs NARROWING (today's live state) or VIX EXTREME.
  size_multiplier scales with count of widening pairs.
- **ict_equities:** favorable iff `trend_label == TRENDING` AND `vol_percentile in
  {NORMAL, HIGH}` (EXTREME → size down, not out). Per-symbol.
- **es_nq:** research-only today (no execution path) — verdict INFO, never sizes.
- **undertow_gapper:** favorable iff vol CALM and no active disaster/halt watch;
  size_multiplier down in EXTREME vol (the Drawdown Lock lives here).

Thresholds come from `config/parameters.yml` / `config/ict_params.yml`. Any new key is
logged to `data/agent/param_change_log.jsonl` with rationale before use.

---

## The reader (`platform/regime_client.py`)

A tiny helper any strategy can call — importable from BOTH `ict/` and `sovereign/` because
`platform/` imports neither:

```python
from platform.regime_client import get_regime
r = get_regime("carry")          # reads system_regime_state.json
if r.stale or r.verdict == "STAND_ASIDE":
    skip()                       # never trade a stale or adverse regime
size *= r.size_multiplier        # Connected Edge × Regime, enforced at sizing
```

`platform/` sits above both walls, imports only stdlib + pandas/numpy, and is added to the
isolation test's allowlist as a shared-neutral package (it must not import `ict` or
`sovereign`; a test asserts this both directions).

---

## How each strategy plugs in (the follow-on, cheap once the contract exists)

- **carry (frozen):** read-only consumption behind an explicit NEXT.md unlock — a one-line
  `size *= get_regime("carry").size_multiplier` at the sizing boundary. No logic change.
- **ICT pipeline:** its Layer 1 becomes "read the contract" instead of "own a RegimeMap";
  Layer 7's breaker reads `portfolio.drawdown_breaker_tripped`.
- **gapper / Petrules (future):** same one-line read.

---

## Build order

1. `platform/` package skeleton + `regime_client.py` reader + isolation test (both
   directions). Commit.
2. `regime_contract.py` writer: carry + es_nq + macro sections from existing files (these
   exist today — fastest real value). Commit.
3. ict_equities + undertow_gapper sections (need PriceStore / gapper inputs). Commit.
4. Portfolio section (unified exposure + drawdown breaker) from the position ledger +
   config. Commit.
5. `scripts/com.alta.system_regime.plist` — schedule the writer every 30 min (do NOT
   install; hand Colin the load command). Commit.
6. Wire the first *reader* into a non-frozen strategy as proof (ICT), leave carry for the
   post-28th unlock.

Each step: `[PLATFORM]` commit prefix, push, isolation test green, no hardcoded caps.

---

## Definition of done (this spec's scope — the nervous system, not the full ICT pipeline)

- `platform/` exists, imports neither `ict/` nor `sovereign/`, isolation test passes both ways.
- `system_regime_state.json` is written on a 30-min schedule with per-section status.
- Today's live truth reproduces: carry section reads STAND_ASIDE (differentials narrowing).
- `regime_client.get_regime()` works from a scratch script.
- One strategy (ICT) reads the contract and modulates size by it.
- NEXT.md updated; plist load command handed over.

---

*Alta Investments · specs/system_regime_contract.md · v1.0*
*"One regime read, every strategy, live. No good edge in a bad regime — enforced, not remembered."*
