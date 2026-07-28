# Trust Decision Brief — Multiple-Testing Correction Scope (DRAFT, for Colin)

**One decision needed:** does the Benjamini-Hochberg (BH) multiple-testing
correction apply **retroactively** to the whole hypothesis ledger's past
CONFIRMED verdicts, or **forward-only** from now on? 14 entries are currently
`CONFIRMED` in `data/agent/hypothesis_ledger.json` (HYP-045, 046, 046a/b/c,
049, 050, 051, 052, 056, 059, 060, 062, 063) out of 87 total ledger entries
across many hypothesis families tested over time. Not decided here —
Colin's call.

## The two options

- **Retroactive:** re-run BH correction across the full history of tests
  that ever contributed a CONFIRMED verdict, treating the whole ledger as
  one family. More statistically honest — the more hypotheses tested, the
  higher the false-discovery rate on the "confirmed" ones, and the ledger
  has tested 87+ hypotheses over ~2 months. Cost: some existing CONFIRMED
  edges (including load-bearing ones like HYP-045, currently the live v015
  config basis) could get retroactively demoted, which is disruptive if
  live capital or config already reflects them.
- **Forward-only:** apply BH from this point forward — every new
  hypothesis family (like HYP-071-v2) gets corrected against its own
  family size, but past verdicts stand as adjudicated. Cleaner, no
  disruption to live config. Cost: past verdicts were adjudicated under a
  weaker standard, so the ledger is inconsistent — some CONFIRMED entries
  cleared a bar the newer ones won't.

## Tradeoff in plain terms

- Retroactive is **more conservative** (higher bar for what counts as
  proven) but touches settled ground — a demotion isn't just a label
  change, it would mean reopening whether HYP-045 (the live 4-pair
  portfolio basis) or other load-bearing CONFIRMED edges still hold under
  correction.
- Forward-only is **safer operationally** (nothing already live gets
  second-guessed) but means the ledger's evidentiary bar quietly got
  stricter partway through and old entries don't reflect it.
- This directly matters for HYP-071-v2 (`research/HYP-071_v2_prereg.DRAFT.md`,
  criterion 6): that prereg applies BH *within* the HYP-071-v2 family
  regardless of this decision, since it's a fresh test. This brief is only
  about whether the correction also reaches backward into the 14 existing
  CONFIRMED verdicts — HYP-071-v2 is unaffected either way.
- No technical blocker either direction — this is a governance call about
  how much to disturb settled state, not a code change.

## Recommendation

None offered — genuinely Colin's call, flagged per his own instruction not
to decide it here.

**Status: DRAFT_PENDING_OPERATOR_APPROVAL — awaiting Colin's choice.**
