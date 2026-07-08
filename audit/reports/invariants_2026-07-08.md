# Adversarial Invariant Report — 2026-07-08

spec v1 `3599b77f1b95aaad…` → **FAIL**

- **I1 Oracle contamination:** 0 (allowed 0)
- **I2 rogue OANDA writes:** 4 (allowed 0)
  - [data/ledger/oanda_fills.jsonl] USD_CAD units=1 stop=1.0 @ 2026-07-02T18:35:03.977325+00:00 — forbidden pair USD_CAD; sentinel/probe fill
  - [data/ledger/oanda_fills.jsonl] USD_CAD units=1 stop=1.0 @ 2026-07-02T18:36:04.838104+00:00 — forbidden pair USD_CAD; sentinel/probe fill
  - [data/ledger/oanda_fills.jsonl] USD_CAD units=1 stop=1.0 @ 2026-07-02T18:55:24.056260+00:00 — forbidden pair USD_CAD; sentinel/probe fill
  - [data/ledger/oanda_fills.jsonl] USD_CAD units=1 stop=1.0 @ 2026-07-03T01:51:57.427597+00:00 — forbidden pair USD_CAD; sentinel/probe fill
- **I3 forbidden pairs (broad):** 4 (allowed 0)
  - [data/ledger/oanda_fills.jsonl] USD_CAD @ 2026-07-02T18:35:03.977325+00:00
  - [data/ledger/oanda_fills.jsonl] USD_CAD @ 2026-07-02T18:36:04.838104+00:00
  - [data/ledger/oanda_fills.jsonl] USD_CAD @ 2026-07-02T18:55:24.056260+00:00
  - [data/ledger/oanda_fills.jsonl] USD_CAD @ 2026-07-03T01:51:57.427597+00:00

soft: 0 unknown-pair · fills present ['data/ledger/oanda_fills.jsonl', 'data/execution/fills.jsonl'] · stale ['data/ledger/oanda_fills.jsonl', 'data/execution/fills.jsonl']
