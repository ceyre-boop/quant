# Adversarial Invariant Report — 2026-07-07

spec v1 `3599b77f1b95aaad…` → **FAIL**

- **I1 Oracle contamination:** 2 (allowed 0)
  - USD_CAD src=fills_backfill outcome=LOSS R=-0.0 @ 2026-06-30T13:33:11.455520+00:00 — forbidden pair USD_CAD
  - USD_CAD src=fills_backfill outcome=LOSS R=-0.0 @ 2026-06-30T22:30:04.240520+00:00 — forbidden pair USD_CAD
- **I2 rogue OANDA writes:** 7 (allowed 0)
  - [data/ledger/oanda_fills.jsonl] USD_CAD units=1 stop=1.0 @ 2026-06-30T13:33:11.455520+00:00 — forbidden pair USD_CAD; sentinel/probe fill
  - [data/ledger/oanda_fills.jsonl] USD_CAD units=1 stop=1.0 @ 2026-06-30T22:30:04.240520+00:00 — forbidden pair USD_CAD; sentinel/probe fill
  - [data/ledger/oanda_fills.jsonl] USD_CAD units=1 stop=1.0 @ 2026-07-02T18:35:03.977325+00:00 — forbidden pair USD_CAD; sentinel/probe fill
  - [data/ledger/oanda_fills.jsonl] USD_CAD units=1 stop=1.0 @ 2026-07-02T18:36:04.838104+00:00 — forbidden pair USD_CAD; sentinel/probe fill
  - [data/ledger/oanda_fills.jsonl] USD_CAD units=1 stop=1.0 @ 2026-07-02T18:55:24.056260+00:00 — forbidden pair USD_CAD; sentinel/probe fill
  - [data/ledger/oanda_fills.jsonl] USD_CAD units=1 stop=1.0 @ 2026-07-03T01:51:57.427597+00:00 — forbidden pair USD_CAD; sentinel/probe fill
  - [data/execution/fills.jsonl] USD_JPY units=None stop=None @ 2026-07-01T02:55:52.513384+00:00 — sentinel/probe fill
- **I3 forbidden pairs (broad):** 8 (allowed 0)
  - [decision] USD_CAD @ 2026-06-30T13:33:11.455520+00:00
  - [decision] USD_CAD @ 2026-06-30T22:30:04.240520+00:00
  - [data/ledger/oanda_fills.jsonl] USD_CAD @ 2026-06-30T13:33:11.455520+00:00
  - [data/ledger/oanda_fills.jsonl] USD_CAD @ 2026-06-30T22:30:04.240520+00:00
  - [data/ledger/oanda_fills.jsonl] USD_CAD @ 2026-07-02T18:35:03.977325+00:00
  - [data/ledger/oanda_fills.jsonl] USD_CAD @ 2026-07-02T18:36:04.838104+00:00
  - [data/ledger/oanda_fills.jsonl] USD_CAD @ 2026-07-02T18:55:24.056260+00:00
  - [data/ledger/oanda_fills.jsonl] USD_CAD @ 2026-07-03T01:51:57.427597+00:00

soft: 0 unknown-pair · fills present ['data/ledger/oanda_fills.jsonl', 'data/execution/fills.jsonl'] · stale ['data/ledger/oanda_fills.jsonl', 'data/execution/fills.jsonl']
