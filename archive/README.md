# archive/

Retired code kept for reference. Nothing here is loaded, scheduled, or imported by the
live system. Files land here via the "improve by subtraction" doctrine (DECISIONS D-Q2) —
removed from the working tree but recoverable.

| File | Retired | Why | Replaced by |
|------|---------|-----|-------------|
| `agent_scheduler.py` | 2026-06-23 | Deprecated/unloaded "usage-aware research scheduler". Not in launchd, not imported anywhere (only documentary docstrings reference it). | Individual launchd jobs: `com.alta.oracle.reflect` (reflection trigger), `com.alta.cache.refresh` (Reddit/macro caches), `com.alta.research.factory` (hypothesis dispatch), `com.alta.hypothesis.generator` (queue fill). |

Note: a few stale docstrings ("Called by agent_scheduler.py …") still exist in
`sovereign/oracle/oracle_cycle.py` and `sovereign/intelligence/cross_system_bridge.py`.
They are historical comments only — those functions are now invoked by the launchd jobs
above, not by this scheduler.
