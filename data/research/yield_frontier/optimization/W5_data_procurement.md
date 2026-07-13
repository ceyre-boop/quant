# W5 — Data Procurement for Scale Trials (specialist agent brief, 2026-07-13)
Stamp: research input to TICK-033; no purchases made.

## Recommended stacks
**(a) Shoestring ~$142/mo + ~$300 one-time:** Polygon/Massive Developer $79/mo (10yr minute aggs,
delisted incl., flat files) + Norgate US Platinum ~$52.50/mo eff. (survivorship spine + delisted
universe + index constituents) + iBorrowDesk Patron $10/mo + FREE daily IBKR FTP borrow snapshotter
(start NOW — forward-fill compounds) + Databento status schema 2018+ for halts (~$100-300 one-time)
+ tape-gap halt reconstruction pre-2018 + EDGAR full-text (FREE) for the 424B/S-1/S-3 dilution table.
**(b) Serious ~$860/mo + ~$3.5k one-time:** Polygon Advanced $199 (2003+, LULD websocket) + Kibot
All-Stocks 1-min $3,000 one-time (1998+, 10,430 delisted — cross-validation + deeper holdouts) +
Ortex API $499/mo (the ONLY clean sub-enterprise 2015-2026 borrow backfill) + Norgate +
DilutionTracker $49/mo (live ops) + Databento halt-window microstructure budget.

## Hard facts
- Databento equities start 2018-05 — cannot be the 10-yr backbone; IS the best halt/SSR status
  source 2018+ (status schema cheap).
- No vendor sells clean 2015→2026 LULD halt records retail; reconstruction plan above.
- Historical per-name borrow fees are the market gap; Ortex is the only clean series below
  enterprise. Shoestring answer: conservative hard-coded HTB haircuts (current prereg approach).
- IBKR FTP is current-day only — every day we don't snapshot is data lost forever.
- EDGAR FTS (free, 2001+, exact timestamps) fully adequate for point-in-time dilution features.
- Polygon rebranded to Massive early 2026; re-confirm pricing at purchase.

## Immediately actionable at $0 (operator-promote convention for jobs)
1. Daily IBKR FTP shortstock snapshotter (borrow fees forward-fill).
2. Daily Nasdaq/NYSE halt-list scraper (halt forward-fill).
