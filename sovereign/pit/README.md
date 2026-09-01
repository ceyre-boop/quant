# Point-in-time storage

> Every record has two timestamps: **when it happened** and **when it became knowable**.
> Every read states the instant it is pretending to be at.
> A revision adds a vintage; it never destroys the observation it revises.

If you query "what did I know on March 3" and get data published March 8, the
backtest is fiction. This layer makes that structurally impossible rather than
discouraged.

## Use it

```python
from sovereign.pit import as_of, view

v = view(as_of("2026-03-03"))
rows = v.facts("earnings", "AAPL")        # only what was public on March 3
one  = v.latest("earnings", "AAPL")
hist = v.vintages("earnings", "AAPL")     # the revision path
```

There is no default as-of. `as_of(None)` raises. For a deliberate live read use
`as_of_now()` — a separate, deliberately more awkward name, so "read the latest"
is always visible in the diff instead of being an omitted argument.

## The four properties that make leakage impossible

**1. You cannot read without an instant.** `AsOfReader` is only reachable through
`view(as_of(...))`, and `as_of(None)` raises `AsOfRequired`. This mirrors
`backtester/holdout_guard.py`, where an unbounded range is a violation rather
than "everything".

**2. You cannot express a leaking query.** The reader takes no SQL. It takes a
registered fact name and builds the `published < as_of` predicate itself. There
is no parameter to forget, reorder or comment out. Every returned row is then
re-checked in Python (`_verify`), so a regression in SQL generation raises
`LookaheadError` instead of returning a plausible number.

**3. A fact with no publication instant cannot be read at all.** It raises
`NotPointInTime`. Returning rows whose knowability is unknown is the exact silent
leak this exists to prevent. Four facts are currently blocked; each names its fix
in `spec.py` so the gap is a visible TODO, not an invisible assumption.

**4. Writes are append-only.** This is the half that timestamps alone do not buy.

## Why append-only is the load-bearing part

The store previously wrote every table with `INSERT OR REPLACE` on a primary key
that excluded the publication instant. So a revision overwrote the observation it
revised:

- an earnings restatement destroyed the originally-reported figure
- the routine pre-print → post-print transition destroyed the pre-announcement
  consensus view of that quarter
- a 13F-A amendment overwrote the original 13F **in place**, taking the
  original's earlier `filing_date` with it — after which even a correct
  `filing_date < as_of` filter *understates* what was knowable

No filter fixes that. History has to survive first. `scripts/migrate_pit.py`
rebuilt the tables without those primary keys so two vintages can coexist.

Demonstrated on a real restatement (EPS 2.00 reported 2026-01-30, restated to
1.75 on 2026-03-12):

| as_of | answer |
|---|---|
| 2026-01-15 | nothing knowable yet |
| 2026-02-01 | **2.00** (original) |
| 2026-03-01 | **2.00** (original — the restatement had not been filed) |
| 2026-04-01 | **1.75** (restated) |

## The strict rule

```
knowable  <=>  published_ts < as_of      (STRICT)
```

Strict, and ties go to *not knowable* — you cannot read a filing the instant it
appears, and same-instant is where vendor clock skew lives. `published_ts IS
NULL` is never knowable: a row whose publication time we failed to record could
have been published at any moment. Matches
`research/petrules/provenance.py::knowable_at` exactly.

A bare date means **midnight UTC at the start of that day**, so
`as_of("2026-03-03")` excludes everything published on the 3rd. That is the
conservative reading and the one a daily-bar backtest wants.

## Currently blocked facts

| Fact | Why | Fix |
|---|---|---|
| `short_volume` | FINRA posts the daily file *after* the close of the day it describes; no publication column | record the file's posting time in `transports/finra.py` |
| `borrow` | IB locate snapshots carry a bare date, no instant | record the snapshot time in `scripts/ib_shortable_snapshot.py` |
| `institutions_agg` | aggregated with `filing_date` discarded — leaks ~45 days *by construction* | carry `max(filing_date)` of contributors; an aggregate is knowable when its latest input was |
| `price_reaction` | derived, so it has no publication instant of its own | compute inside an as-of view from as-of-filtered inputs |

`short_interest` is registered but every row currently has a NULL publication
date (Nasdaq gives none), so 250 real rows exist and the as-of reader correctly
returns **none** of them. That is the layer working, not a bug.

## Enforcement

- `tests/test_pit_clock.py` — the strict rule, including the equal-instant case
- `tests/test_pit_reader.py` — the 13F period-vs-filing trap, AMC prints,
  restatement vintages, monotonicity, truncation invariance
- `tests/test_pit_enforcement.py` — AST walls; the allowlist of files still
  reading PIT tables directly is **migration debt and must shrink, never grow**
- `scripts/audit_pit.py` — standing DB-level detector, exit code 1 on violation

**Do not fix a failure in these by relaxing the test.**
