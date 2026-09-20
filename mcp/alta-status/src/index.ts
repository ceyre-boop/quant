#!/usr/bin/env node
/**
 * Alta Status MCP server — read-only window into the Alta Investments operation.
 *
 * Eight tools expose the live state Colin keeps asking for by hand: account/positions (live OANDA),
 * morning brief, forex signals, oracle reflections, loop health, the daily research panel, proof of
 * life, and the pre-trade context packet (Alexandrian Library + the whole written research record).
 * Everything is read-only; the only external call is a live OANDA GET (account summary/positions).
 * No tool can place, modify, or close a trade.
 *
 * Transport: stdio (local server). Wire it into an MCP client (e.g. Claude Desktop) with:
 *   { "command": "node", "args": ["<repo>/mcp/alta-status/dist/index.js"] }
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import {
  accountStatus,
  morningBrief,
  signals,
  oracleReflection,
  loopHealth,
  researchPanel,
  proofOfLife,
  tradeContext,
} from "./alta.js";

const READONLY = {
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: true,
} as const;

function ok(data: unknown) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }],
    structuredContent: data as Record<string, unknown>,
  };
}

const server = new McpServer({ name: "alta-status", version: "0.1.0" });

server.registerTool(
  "alta_account_status",
  {
    title: "Alta account status (live)",
    description:
      "Live OANDA account snapshot: NAV, balance, realized/unrealized P&L, margin, and every open " +
      "position (instrument, side, units, unrealized P&L). Read-only — performs a live GET against " +
      "the configured OANDA account; cannot place or modify trades. Reports mode (practice/LIVE).",
    inputSchema: {},
    annotations: READONLY,
  },
  async () => ok(await accountStatus()),
);

server.registerTool(
  "alta_morning_brief",
  {
    title: "Alta morning brief",
    description:
      "Latest morning briefing: regime call, directional bias + confidence, key level, the daily " +
      "FRED macro backdrop block, the Big-Move headline (display-only), the scorecard line, and the " +
      "narrative. Qualitative context — never a verified trading signal.",
    inputSchema: {
      narrative_chars: z
        .number()
        .int()
        .min(0)
        .max(20000)
        .optional()
        .describe("How many chars of the narrative to include (default 2000; 0 omits it)."),
    },
    annotations: READONLY,
  },
  async ({ narrative_chars }) => ok(morningBrief(narrative_chars ?? 2000)),
);

server.registerTool(
  "alta_signals",
  {
    title: "Alta forex signals",
    description:
      "Current forex proximity scan — per pair: conviction score, direction (often NO_TRADE), " +
      "percent-to-trigger, regime, and rate differential. Shows whether the macro carry edge is " +
      "near firing. Read-only.",
    inputSchema: {},
    annotations: READONLY,
  },
  async () => ok(signals()),
);

server.registerTool(
  "alta_oracle_reflection",
  {
    title: "Alta oracle reflection(s)",
    description:
      "The Oracle's daily candidate lesson(s): lesson text, mechanism, testable rule, the reasoning " +
      "component it targets, and a system-health note. Pass count to retrieve the last N days " +
      "(default 1, max 14).",
    inputSchema: {
      count: z.number().int().min(1).max(14).optional().describe("How many recent days (default 1)."),
    },
    annotations: READONLY,
  },
  async ({ count }) => ok(oracleReflection(count ?? 1)),
);

server.registerTool(
  "alta_loop_health",
  {
    title: "Alta loop health",
    description:
      "Scheduled-loop health: which loops are ALIVE vs DOWN, last heartbeat + silence hours per loop, " +
      "whether the system is frozen (kill switch), and market-hours state. Use to spot a stalled " +
      "scanner or briefing. Read-only.",
    inputSchema: {},
    annotations: READONLY,
  },
  async () => ok(loopHealth()),
);

server.registerTool(
  "alta_research_panel",
  {
    title: "Alta research panel",
    description:
      "The daily multi-source research panel: which sources harvested OK, and each domain's recorded " +
      "variables (macro, markets, sentiment, FX, equities, vol-premium proxy, positioning). RAW " +
      "RECORDED DATA — not findings; relationships are tested separately through the research factory. " +
      "Optional date (YYYY-MM-DD, default latest) and source filter.",
    inputSchema: {
      date: z
        .string()
        .regex(/^\d{4}-\d{2}-\d{2}$/)
        .optional()
        .describe("Panel day YYYY-MM-DD (default latest)."),
      source: z
        .string()
        .optional()
        .describe("Filter to one domain: macro_fred, markets, sentiment_reddit, news, fx_macro, equities, vrp_proxy, positioning."),
    },
    annotations: READONLY,
  },
  async ({ date, source }) => ok(researchPanel({ date, source })),
);

server.registerTool(
  "alta_proof_of_life",
  {
    title: "Alta proof of life",
    description:
      "The honest 'is the system alive and producing signal?' read: whether a trade fired today, " +
      "would-be signals (pairs currently signalling vs NO_TRADE), how close each pair is to firing, " +
      "the last actual fill + its age, and loop health. The answer to 'is it working?' WITHOUT " +
      "real-money risk — no forced trades, read-only.",
    inputSchema: {},
    annotations: READONLY,
  },
  async () => ok(proofOfLife()),
);

server.registerTool(
  "alta_trade_context",
  {
    title: "Alta trade context packet",
    description:
      "EVERYTHING this desk already knows about one instrument, assembled for the moment a trade is " +
      "being considered. Six sources in one packet: (1) the Alexandrian Library — the nearest " +
      "historical analogues to today's tape across 63 sealed episodes in 10 volumes, with similarity, " +
      "threat level, size modifier and what actually followed each precedent; (2) the edge ledger — " +
      "what is CONFIRMED vs FRAGILE vs null, filtered to this instrument; (3) CLOSED DOORS — every " +
      "hypothesis already refuted here, so nothing re-proposes an idea the desk has already paid for; " +
      "(4) the one-line lesson from each relevant hypothesis; (5) the ratified risk caps, quoted from " +
      "RISK_CONSTITUTION.md at call time rather than hardcoded; (6) this instrument's own logged " +
      "decisions. READ THE CLOSED DOORS FIRST. Read-only: writes nothing, places nothing, recommends " +
      "nothing, and never returns a position size. A degraded packet always says so in `warnings` and " +
      "`completeness`; an unavailable source states its reason. Absence of a closed door is not " +
      "evidence of an edge.",
    inputSchema: {
      instrument: z
        .string()
        .describe("EURUSD, GBP_JPY, USDJPY, SPY, QQQ — FX pairs in any separator style."),
      offline: z
        .boolean()
        .optional()
        .describe(
          "Skip the live price fetch and use the local cache for the Library query. Much faster, " +
            "but the cache is stale — the packet labels every series it serves as STALE.",
        ),
      include_library: z
        .boolean()
        .optional()
        .describe(
          "Set false to skip the Alexandrian Library entirely and return only the written record " +
            "(instant). Default true.",
        ),
    },
    annotations: READONLY,
  },
  async ({ instrument, offline, include_library }) =>
    ok(await tradeContext({ instrument, offline, include_library })),
);

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("alta-status MCP server running (stdio)");
