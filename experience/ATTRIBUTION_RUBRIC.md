# Attribution Rubric v1 — the law under which experience is metabolized

Committed BEFORE the first classification (spec-first). Changes require a new version
section here — old classifications are never rewritten. The classifier
(`experience/attribution.py`) implements this tree mechanically; where inputs are
missing it says AMBIGUOUS — it never guesses.

**Classes:** `thesis_confirmed` · `thesis_invalidated` · `luck_good` · `luck_bad` ·
`AMBIGUOUS`. **Overlays** (orthogonal annotations, never the class):
`execution_variance` · `policy_exit` · `sizing` · `commitment`.

Bad luck and bad thesis are different sins: a stop-out with the thesis still alive is
variance (luck_bad); a loss whose reason died before the exit is a broken thesis
(thesis_invalidated); a win while the reason was already dead is a warning, not a
victory (luck_good).

---

## 1 · Inputs (per closed decision)

engine ∈ {carry, exit_shadow, predictive} · thesis {kind: structural_carry | hypothesis,
id, falsification_predicates} · board rows at entry and exit (rate-diff sign, VIX-gate
state; predicates where they exist) · exit_reason ∈ {TIME, TRAILING, STOP, REVERSAL,
CB_REFRESH, UNKNOWN} · realized R (decision-price) · fill slippage in R.

## 2 · Thesis-alive test

- **hypothesis engines:** thesis is alive iff ALL falsification predicates evaluate TRUE
  at exit (predicates frozen at entry — specs/thesis_exit_spec.md).
- **structural carry (no explicit predicates):** mechanical proxy — the pair's
  rate-differential SIGN at exit equals the sign at entry AND the pair's VIX-gate state
  (gated/ungated) is unchanged. Both from board/decision-log values, never re-derived.
- inputs missing / not evaluable → AMBIGUOUS (class), full stop.

## 3 · The ordered tree (first match wins)

1. `realized_r` missing or `exit_reason == UNKNOWN` → **AMBIGUOUS**.
2. Overlay pass (does not classify): |fill_slippage_r| > **0.15R** → add
   `execution_variance`; classification proceeds on decision-price R.
3. Evaluate thesis-alive (§2). Unevaluable → **AMBIGUOUS**.
4. **WIN (r > 0):** thesis alive → **thesis_confirmed** (TIME-exit-positive with intact
   thesis is the canonical carry case). Thesis dead → **luck_good** — won while the
   reason was gone; flagged for review, never celebrated.
5. **LOSS (r ≤ 0):** thesis dead at/before exit → **thesis_invalidated**.
   STOP / TRAILING / REVERSAL with thesis intact → **luck_bad** (stopped-but-right).
   TIME-negative with thesis intact → **luck_bad** — variance around a live thesis;
   one trade cannot kill a structural thesis.
6. `CB_REFRESH` exits are policy exits: classify by steps 4-5, add overlay `policy_exit`.

## 4 · Mapping to the Oracle harvest labels

THESIS_FAILURE → thesis_invalidated · TIMING_FAILURE / REGIME_FAILURE → luck_bad ·
EXECUTION_FAILURE → overlay execution_variance · SIZING_FAILURE → overlay sizing ·
COMMITMENT_FAILURE → overlay commitment. (Port of
`sovereign/futures/reasoning._post_trade_hypothesis`'s win/loss × confirmed branches,
generalized from CVD-confirmation to predicates.)

## 5 · Committed thresholds

slippage overlay: **0.15R** · carry proxy: rate-diff sign + VIX-gate state (exactly
those two — adding proxy terms is a rubric version bump).

## 6 · Worked examples

- Carry EUR_USD short, TIME exit, r=+0.4, rate-diff sign unchanged → **thesis_confirmed**.
- Carry USD_JPY long, TRAILING exit, r=−0.5, rate-diff sign unchanged → **luck_bad**.
- Predictive HYP-074 fade, exit at target r=+0.8 but `rr25_z` predicate had flipped two
  days prior → **luck_good** (review: the exit engine should have fired thesis_invalidated).
- Carry GBP_USD, STOP exit, r=−1.0, rate differential had flipped sign before exit →
  **thesis_invalidated**.
- Any decision with no realized R (still open / unmatched) → **AMBIGUOUS**.
- Shadow CLOSE decision with 0.2R gap between decision close and OANDA fill →
  class per tree + overlay `execution_variance`.

## 7 · Abstention rows

Abstentions are journaled decisions but are NOT classified by this rubric (no realized
outcome). The weekly review reports them as acted-vs-abstained calibration material.

## 8 · Versioning

v1 — 2026-07-02. sha256 of this file is stamped into every attribution artifact the
classifier writes. Amendments append a v2 section; the classifier refuses to run if the
file hash doesn't match its pinned version list.
