#!/usr/bin/env node
/**
 * Alta Status MCP server — read-only window into the Alta Investments operation.
 *
 * Six tools expose the live state Colin keeps asking for by hand: account/positions (live OANDA),
 * morning brief, forex signals, oracle reflections, loop health, and the daily research panel.
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

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("alta-status MCP server running (stdio)");
