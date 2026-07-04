# Plan — TICK-007: Dashboard parity — positioning-board export + panel

## Step 1 — exporter (SAFE-NOW, this repo, additive)
- NEW `scripts/export_positioning_board.py`: read latest per-pair row from
  `sentiment_board_state` (read_only), emit `data/agent/positioning_board.json`:
  `{as_of, built_at, stale: bool (>3 trading days), pairs: {PAIR: {cot_net_pct_1y,
  cot_net_z, cot_flush_1w, tff_lev_net_pct, vrp_signal, vrp_pct, rr25, bf25,
  atm_term_slope, econ_surprise_z, gdelt_tone, vix_regime}}}` — nulls preserved
  (missing data renders as missing, never fabricated).
- Invocation: appended tail-call in `scripts/update_sentiment.py` main() (additive,
  guarded) so the export refreshes with every board rebuild; also runnable standalone.
- NEW `tests/unit/test_positioning_export.py`: schema, staleness flag, null passthrough
  (fixture db via store.connect(path=tmp)).

## Step 2 — panel (own session; master-branch worktree deploy pattern)
- Root `index.html`: "Positioning" panel reading `data/agent/positioning_board.json`
  (same fetch pattern as existing panels); per-pair mini-table, stale badge.
- Deploy: worktree → master data-only push (see project_dashboard_deploy memory —
  NEVER full-merge sovereign-v2→master); verify with Interceptor (mandatory).

## Constraint
DISPLAY-ONLY. No live gate, readiness score, or decision chain reads this file.
Article 6: board signals reach live decisions only behind a CONFIRMED verdict +
logged param change.

## Verification
Step 1: export exists + regenerates on `update_sentiment.py` run; tests green; suite
baseline holds. Step 2: Interceptor screenshot of the rendered panel on the live URL.
