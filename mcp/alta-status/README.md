# Alta Status MCP

A **read-only** MCP server that exposes the Alta Investments operation's live state — so any MCP
client (Claude Desktop, etc.) can ask "how's Alta doing?" without bespoke scripts.

Every tool is read-only. The only external call is a live OANDA `GET` (account summary / open
positions). **No tool can place, modify, or close a trade**, change config, or write anything.

## Tools

| Tool | What it returns |
|------|-----------------|
| `alta_account_status` | Live OANDA NAV, balance, realized/unrealized P&L, margin, open positions (practice/LIVE mode) |
| `alta_morning_brief` | Latest briefing: regime, bias, confidence, key level, FRED macro block, Big-Move headline, scorecard, narrative |
| `alta_signals` | Forex proximity scan — per pair: conviction, direction, %-to-trigger, regime, rate differential |
| `alta_oracle_reflection` | Oracle's daily candidate lesson(s): text, mechanism, testable rule, target component, health note (`count` for last N days) |
| `alta_loop_health` | Scheduled-loop health: ALIVE/DOWN, heartbeats, frozen/kill-switch state, market hours |
| `alta_research_panel` | The daily multi-source research panel by domain (`date`, `source` filters). RAW recorded data — not findings |

## Build

```bash
cd mcp/alta-status
bun install
bun run build          # tsc -> dist/
bun run smoke          # exercise every tool against real data + live OANDA
bun run inspect        # optional: MCP Inspector
```

Credentials and data come from the quant repo. The server resolves the repo root automatically
(two levels up); override with `ALTA_ROOT=/path/to/quant` if running from elsewhere. OANDA
credentials are read directly from `quant/.env` (`OANDA_API_KEY`, `OANDA_ACCOUNT_ID`,
`OANDA_BASE_URL`, `OANDA_LIVE`) — never written to the environment or logged.

## Wire into Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "alta-status": {
      "command": "node",
      "args": ["/Users/taboost/quant/mcp/alta-status/dist/index.js"]
    }
  }
}
```

(Run `bun run build` first so `dist/index.js` exists. Restart the client to pick it up.)

## Notes

- **Read-only by design.** All tools carry `readOnlyHint: true`. The account tool hits the live
  OANDA practice endpoint; it reads, never trades.
- Data sources are the operation's own files (`data/oracle/`, `data/agent/`, `data/research/panel/`)
  plus the live OANDA pull — the server is a thin, typed window, not a second source of truth.
- The research panel is **raw recorded variables, not findings** — relationship testing goes through
  the research factory's gauntlet, never inferred here.
