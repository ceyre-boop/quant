# SESSION CLOSING PROTOCOL — Alta Investments
# Every Claude Code session ends by running this list top to bottom.
# You do not go home until every item is GREEN or has a logged reason it was skipped.
# Total expected time: 15–25 minutes.

---

## HOW THIS WORKS

Five levels. Each level has tasks. Each task has:
- A **command** (run it) or **check** (verify it manually)
- A **PASS** definition (what green looks like)
- A **FAIL action** (what to do if it's not green)

Tasks are ordered hardest-to-skip at the top, easiest-to-skip at the bottom.
A FAIL at Levels 1–2 means the session is not done. Fix it or log it loudly in NEXT.md.
A FAIL at Levels 3–5 means ticket it and hand it off — don't block the close.

---

## LEVEL 1 — JANITOR
*No discretion. Always runs. Anyone could do this. ~5 min.*
*These are the things a $12/hr worker checks before clocking out.*

### J-1: No uncommitted changes left behind
```bash
git status
```
**PASS:** Output is `nothing to commit, working tree clean`
**FAIL:** Commit or stash before closing. An uncommitted change is a lost change.

### J-2: On the right branch
```bash
git branch --show-current
```
**PASS:** Returns `sovereign-v2`
**FAIL:** You are on the wrong branch. Do not push. Switch and re-run.

### J-3: Pushed at least once this session
```bash
git log --oneline origin/sovereign-v2..HEAD
```
**PASS:** Returns nothing (everything pushed)
**FAIL:** `git push` right now. An unpushed branch is a single-machine point of failure.

### J-4: No debug print statements left in changed files
```bash
git diff origin/sovereign-v2 --name-only | xargs grep -l "print(" 2>/dev/null || echo "CLEAN"
```
**PASS:** `CLEAN` or only intentional prints (logging, not debugging)
**FAIL:** Remove the print, commit, push.

### J-5: No temp files left in repo root
```bash
ls *.tmp *.bak *.swp 2>/dev/null || echo "CLEAN"
```
**PASS:** `CLEAN`
**FAIL:** Delete them.

### J-6: SHADOW_MODE is still True (frozen path check)
```bash
python3 -c "from sovereign.execution.forex_exit_manager import SHADOW_MODE; print('SHADOW_MODE:', SHADOW_MODE)"
```
**PASS:** Prints `SHADOW_MODE: True`
**FAIL:** STOP. This is a critical safety failure. Do not close the session. Revert whatever changed it and document in NEXT.md.

*(Verified 2026-07-25: `sovereign/execution/forex_exit_manager.py` exports `SHADOW_MODE` at module level, currently `True`. Import path is correct as written.)*

---

## LEVEL 2 — TECHNICIAN
*Verification and data hygiene. Requires knowing what the system does. ~5 min.*
*These are the checks a shift supervisor runs before handing off.*

### T-1: Core isolation test still passes
```bash
python3 -m pytest tests/ -k test_pipeline_does_not_import_sovereign -q
```
**PASS:** 1 passed
**FAIL:** ICT has imported sovereign. This is a critical architecture violation. Do not close. Fix it.

### T-2: ICT pipeline baseline not worse than pre-existing
```bash
python3 -m pytest tests/ -k "ict and pipeline" -q 2>&1 | tail -5
```
**PASS:** 4 failed / 23 passed (or better). The 4 pre-existing failures are tracked in `Plans/restoration-ledger.md` — do NOT absorb new ones.
**FAIL:** New failure appeared. Identify it, ticket it (TICK-XXX), note in NEXT.md.

*(Verified 2026-07-25: baseline is exactly 4 failed / 23 passed / 1 skipped, matching CLAUDE.md's documented baseline.)*

### T-3: Risk constitution drift test passes
```bash
python3 -m pytest tests/ -k "risk_constitution" -q 2>&1 | tail -3
```
**PASS:** All selected tests pass (currently 10 passed, 1 skipped — the skip is expected, not a failure)
**FAIL:** Config and constitution are out of sync. Fix before closing.

*(Reconciled 2026-07-25: original text said "PASS: 1 passed" — the selector actually matches 10 tests + 1 intentional skip, not a single test. Corrected the PASS definition to match reality; the selector itself was fine.)*

### T-4: Shadow log has today's entry (or a logged reason it doesn't)
```bash
python3 -c "
import json, datetime
lines = open('data/exec/exit_manager_shadow.jsonl').readlines()
if not lines: print('EMPTY — INVESTIGATE')
else:
    last = json.loads(lines[-1])
    print('Last entry:', last.get('run_ts','?')[:10], '| mode:', last.get('mode','?'))
"
```
**PASS:** Last entry shows today's date, or shows a SKIP (no open trades = valid)
**FAIL:** Shadow log hasn't run today. Check `com.alta.forex_exit_manager` is loaded.

*(Verified 2026-07-25: `data/exec/exit_manager_shadow.jsonl` exists at this exact path. Path in original text was correct.)*

### T-5: No new URGENT messages waiting for Colin
```bash
python3 -c "
import json
msgs = json.load(open('data/agent/messages_to_colin.json'))
urgent = [m for m in msgs if m.get('level') == 'URGENT']
print(f'{len(urgent)} URGENT items' if urgent else 'CLEAN — no URGENT')
for u in urgent[:3]: print(' -', u.get('text','')[:80])
"
```
**PASS:** `CLEAN — no URGENT`
**FAIL:** Read each item. If it's new and unaddressed, handle it before closing or note explicitly in NEXT.md with a plan.

*(Verified 2026-07-25: `data/agent/messages_to_colin.json` exists at this exact path — confirm the JSON is a flat list of message dicts before relying on this in a hurry; several `.bak-*` snapshots exist alongside it but the live file is unqualified.)*

### T-6: param_change_log has an entry for any config touched this session
```bash
# Run this if you changed config/parameters.yml, config/ict_params.yml, or risk limits
tail -5 data/agent/param_change_log.jsonl
```
**PASS:** Every config change made this session has a logged rationale entry
**FAIL:** Write the entry now. Format: `{"timestamp": "...", "param": "...", "old": ..., "new": ..., "rationale": "..."}`

*(Confirmed 2026-07-25: file exists, actively used this week.)*

### T-7: All new JSON output files are valid JSON
```bash
# Run against any JSON files created or modified this session
git diff origin/sovereign-v2 --name-only | grep "\.json$" | while read f; do
  python3 -c "import json; json.load(open('$f'))" && echo "OK: $f" || echo "INVALID: $f"
done
```
**PASS:** All `OK`
**FAIL:** Malformed JSON in a live file. Fix before closing — bad JSON silently breaks readers.

---

## LEVEL 3 — ANALYST
*Knowledge management and research continuity. ~8 min.*
*These keep the next session from re-deriving context this session already built.*

### A-1: NEXT.md updated with session summary
Open `NEXT.md` and write or verify the entry for today includes:
- What shipped (commits, by hash)
- Push confirmation
- Any new verdicts (hypothesis ledger entries, test results)
- Blockers for next session
- Anything explicitly refused and why

**PASS:** Entry exists, is complete, is committed and pushed
**FAIL:** The next session opens blind. Write it now.

### A-2: Training gate status logged
```bash
python3 scripts/sovereign_train.py --watch 2>&1 | grep -E "GATE|PASS|FAIL|CLOSED|OPEN" | head -10
```
**PASS:** Gate status printed and matches your understanding of current state
**FAIL:** Gate changed unexpectedly. Investigate.

*(Verified 2026-07-25: `scripts/sovereign_train.py --watch` runs and prints gate lines matching this grep. Currently CLOSED, all 4 blockers listed — matches `sovereign/training/gate.py`.)*

### A-3: Backlog tickets updated
Open `tickets/backlog.md`. For every ticket that was:
- **Completed this session** → mark `status: DONE`, add resolution note
- **Partially addressed** → add a progress note
- **Newly created this session** → confirm it exists with acceptance criteria

**PASS:** Backlog reflects reality
**FAIL:** Tickets are stale. Stale tickets create rework.

### A-4: Hypothesis ledger consistent
```bash
python3 scripts/audit_hypothesis_ledger.py 2>&1 | tail -5
```
**PASS:** No schema errors, no orphaned entries
**FAIL:** Note the error. Don't close a session with a corrupted ledger.

*(Verified 2026-07-25: `scripts/audit_hypothesis_ledger.py` exists. Note the ledger itself lives at `data/agent/hypothesis_ledger.json` — NOT under `data/research/` — in case you need to inspect it directly.)*

### A-5: Obsidian brain sync (if substantive session)
```bash
# Only if code architecture, new hypotheses, or major decisions changed
python3 scripts/build_obsidian_graph.py 2>&1 | tail -3
```
**PASS:** Sync completed, no errors
**SKIP:** Valid if session was purely operational (no architectural changes)

*(Verified 2026-07-25: `scripts/build_obsidian_graph.py` exists.)*

### A-6: Plist watchdog baseline current
```bash
python3 -c "
import json, datetime
b = json.load(open('data/system/plist_watchdog_baseline.json'))
print('Baseline:', b.get('timestamp','?'), '| Count:', b.get('loaded_count','?'))
"
```
**PASS:** Baseline timestamp reflects current known state (rebaseline if you added/removed plists)
**FAIL:** Baseline is stale. Run `python3 scripts/plist_watchdog.py --rebaseline "session close"`

*(Verified 2026-07-25: both `data/system/plist_watchdog_baseline.json` and `scripts/plist_watchdog.py` exist at these paths.)*

---

## LEVEL 4 — MANAGER
*Architecture integrity. Verifies the system's safety properties are intact. ~3 min.*
*These are the things a senior engineer checks before merging.*

### M-1: No frozen files were touched without an unlock
```bash
git diff origin/sovereign-v2 --name-only | grep -E "forex_exit_manager|decide_exit|exit_machine" || echo "CLEAN"
```
**PASS:** `CLEAN`
**FAIL:** You touched a frozen file. Check NEXT.md for the explicit unlock entry. If it's not there, revert.

### M-2: Ignition gate still closed for the right reasons
```bash
python3 -c "
from sovereign.training.gate import evaluate_gate
gs = evaluate_gate()
print('Gate:', 'OPEN' if gs.open else 'CLOSED')
for r in gs.reasons: print(' -', r)
"
```
**PASS:** Gate CLOSED, all 4 expected reasons listed
**FAIL A:** Gate unexpectedly OPEN → stop, investigate, do not run training
**FAIL B:** Reasons changed → one of the checks passed without you knowing. Verify it was intentional.

*(Verified 2026-07-25: `evaluate_gate()` returns a `GateStatus` with `.open` and `.reasons` exactly as written. Currently CLOSED with 4 reasons: `tick_024_carry_fix_landed`, `hyp_071_net_confirmed`, `value_board_is_net` (gross_R_caveat), `hyp_071_revival_confirmed`.)*

### M-3: MT5 guard still refuses live accounts
```bash
python3 -c "
from sovereign.execution.mt5.guard import assert_routable
class FakeLive:
    trade_mode = 2  # ACCOUNT_TRADE_MODE_REAL
    login = 12345
    server = 'test'
try:
    assert_routable(FakeLive())
    print('FAIL — live account was NOT rejected')
except Exception as e:
    print('PASS — live rejected:', type(e).__name__)
"
```
**PASS:** Prints `PASS — live rejected: LiveAccountError`
**FAIL:** The live guard is broken. Do not close session. Fix immediately.

*(RECONCILED 2026-07-25 — this command was broken: the original draft set `trade_mode = 0` and commented it as `ACCOUNT_TRADE_MODE_REAL`. In `sovereign/execution/mt5/__init__.py`, 0 is actually `ACCOUNT_TRADE_MODE_DEMO`; `ACCOUNT_TRADE_MODE_REAL` is 2. As written, the fake account would have been accepted as DEMO and never exercised the live-rejection path at all — the check would have silently passed for the wrong reason every time. Corrected to `trade_mode = 2` and added the `login`/`server` attributes `assert_routable` reads via `_attr()`. Module path and `LiveAccountError` name were both already correct.)*

### M-4: divergence_spec.md sha256 unchanged
```bash
python3 -c "
import hashlib
h = hashlib.sha256(open('audit/divergence_spec.md','rb').read()).hexdigest()[:16]
print('spec sha256 prefix:', h)
print('Verify this matches the recorded value in the latest shadow audit report.')
"
```
**PASS:** Hash matches the last audit report
**FAIL:** Spec was modified. If intentional, the change requires a dated §10 entry and spec_version bump (per the spec's own change control). If unintentional, revert.

---

## LEVEL 5 — CEO
*Strategic alignment. Three questions. ~2 min.*
*The session isn't done until you can answer all three.*

### C-1: Did today move the needle?
Write one sentence in NEXT.md answering: "The primary mission is to reach ignition (first honest training cycle). Did today's session bring that closer, and how?"

Valid answers: "Yes — TICK-024 staged" / "Yes — game board defined" / "No — maintenance session, system healthier"
Invalid answer: blank.

### C-2: Is there anything only Colin can decide?
List in NEXT.md any items that require his explicit sign-off before the next session can proceed. Examples: parameter changes, new risk limits, go/no-go on a staged patch, capital decisions.

**PASS:** Either "none" or a clear list with enough context for him to decide in 5 minutes
**FAIL:** A decision is implied by the code but never surfaced. The next session will hit a wall.

### C-3: Is the handoff clean?
Read the first 20 lines of NEXT.md as if you are a fresh Claude Code session with no context. Can you answer: what is the current state, what is the one most important thing to do next, and what must not be touched?

**PASS:** Yes, in under 60 seconds of reading
**FAIL:** The handoff is unclear. Rewrite until it is.

---

## CLOSING STAMP

When all items are GREEN (or explicitly logged as skipped with reason), append this to NEXT.md:

```
SESSION CLOSE [date] [time UTC]
Protocol: SESSION_CLOSING_PROTOCOL.md v1.1
J: 6/6 | T: 7/7 | A: 5/6 (A-5 skipped: no arch changes) | M: 4/4 | C: 3/3
Gate: CLOSED | Shadow: GREEN | Branch: sovereign-v2 | Pushed: YES
```

If any item failed and was not fixed: replace the count with the item ID and a one-line note.

---

*Alta Investments — Sovereign Trading Intelligence*
*SESSION_CLOSING_PROTOCOL.md v1.1 — 2026-07-25 (v1.0 by Colin; v1.1 reconciles commands against the live repo — see inline "Verified/Reconciled" notes on T-3, T-5, M-3)*
*"The machine flags its own problems. So does the person running it."*
