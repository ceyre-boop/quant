# Thesis Ledger Schema (Layer 3)

Machine twin: `sovereign/thesis/schema.py` (`validate`, `canonical_hash`, `SEALED_FIELDS`).
Ledger: `data/thesis/thesis_ledger.jsonl`, one record per line, append-only, written only by
`sovereign/thesis/ledger.py`. Seals: `data/thesis/preregister/THS-NNN.json`.

## Record

| key | sealed | meaning |
|---|---|---|
| `id` | yes | `THS-NNN` (suffix letter for a sibling sleeve, e.g. `THS-001B`). Never reused. |
| `title`, `thesis` | yes | the mechanism in ≤3 sentences: *who is forced to pay, and why* |
| `horizon_months` | yes | when the thesis is judged (scorecards mark 1/3/6/12 m) |
| `basket[{ticker, weight}]` | yes | weights sum to 1, set once at entry, never rebalanced |
| `entry_date`, `entry_prices{}` | yes | first fill date; fill prices if known |
| `invalidation{rule, metric, threshold, manual_check?}` | yes | the only exit. `metric=basket_drawdown` is scored mechanically; `manual_check` is surfaced as a question at each scoring run |
| `benchmark` | yes | `SPY` unless the thesis says otherwise |
| `sizing{account, dollars}` | yes | what was actually put on |
| `source` | yes | `ai-proposed` · `operator` · `retro-registered` (entered before the engine; allowed once, always labelled) |
| `status` | no | `DRAFT → SEALED → LIVE → INVALIDATED | CLOSED` |
| `sealed_at`, `hash_lock` | set by seal | sha256 over the sealed keys, key-sorted, no whitespace |
| `scorecards[]`, `notes[]` | no | appended by `score` and `live`/`set_status` |

## Rules enforced in code

- `create` refuses a duplicate id, a basket whose weights don't sum to 1, and a thesis without
  an invalidation rule.
- `seal` writes the prereg file once and refuses to overwrite it.
- `update` refuses any change to a sealed key (compares the recomputed hash to `hash_lock`).
- `verify` recomputes the hash from both the ledger and the prereg file.
- `live` refuses an unsealed thesis: no fills before the seal, except retro-registration.

## Scoring contract

Buy-and-hold of the sealed weights from `entry_date`, vs the benchmark over the same window;
TWR, excess, max drawdown, marks at 1/3/6/12 months. It measures the *thesis as sealed*, not
what the account did afterwards. What the account did is `reconcile` (TWR + Modified Dietz
from broker fills in `data/thesis/fills/`).
