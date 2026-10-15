# Oracle Stage 03 — SDK agent (Layer 2)

**What.** The daily reflection run as a Claude Agent SDK agent with repo tools, instead of one
prompt-and-answer call. Code `sovereign/oracle/sdk_agent.py`, runner `scripts/oracle_sdk.py`,
CLI `alta oracle-agent run [--budget 2.0] [--model …]`. Schedule
`launchd/com.alta.oracle_sdk_agent.plist` (07:00 local, loaded 2026-10-15).

**Inputs (tools, all read-only).** `thesis_list`, `thesis_score`, `loop_health`,
`recent_decisions(days)`, `price(ticker, days)`, plus built-in Read/Grep/Glob/WebSearch over the repo.
The only write is `propose_thesis_draft` → `research/thesis/drafts/<date>_<slug>.json`, a DRAFT
the operator may seal with `alta thesis new`. Write/Edit/Bash/Agent are disallowed by the SDK
tool list, not by the prompt.

**Process.** Mission in `sdk_agent.MISSION`: score every LIVE thesis and answer its manual
invalidation question with sources; state whether the trading system produced closed trades;
propose one lesson or say "no new lesson"; draft a thesis only if evidence warrants.

**Outputs (Layer 4).** `oracle/03-agent/output/<stamp>_<tag>.md` (report) and `.json` (cost,
turns, stop reason, session id); one line per run in `data/agent/oracle_sdk_ledger.jsonl`;
cost appended to `logs/oracle_cost.json` as `oracle_sdk_agent`.

**Budget.** `max_budget_usd` 2.0 per run (measured ≈ $0.40 on claude-opus-5-5, 10 turns).
At one run a day the $100 grant covers ~8 months.

**Gate.** A report is advisory. Nothing here changes config, places orders, seals a thesis, or
writes to the hypothesis ledger. The reflect/test/codify cycle (stages 00–02) is unchanged.
