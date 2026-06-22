/**
 * Resolve the Alta repo root and load OANDA credentials from quant/.env.
 *
 * ALTA_ROOT defaults to the quant repo (two levels up from this MCP package). Override with the
 * ALTA_ROOT env var if the server is run from elsewhere. Credentials are read from the repo's
 * .env file directly (never echoed, never written to process.env).
 */
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
// src/ -> alta-status/ -> mcp/ -> quant/  (compiled dist/ is one deeper but resolve() handles ../..)
export const ALTA_ROOT = process.env.ALTA_ROOT
  ? resolve(process.env.ALTA_ROOT)
  : resolve(here, "..", "..", "..");

export const PATHS = {
  morningBrief: join(ALTA_ROOT, "data", "oracle", "market_briefings", "latest.json"),
  forexProximity: join(ALTA_ROOT, "data", "agent", "forex_proximity.json"),
  loopHealth: join(ALTA_ROOT, "data", "oracle", "loop_health_status.json"),
  reflectionsDir: join(ALTA_ROOT, "data", "oracle", "reflections"),
  panelDir: join(ALTA_ROOT, "data", "research", "panel"),
  fredEconomic: join(ALTA_ROOT, "data", "macro", "fred_economic_latest.json"),
  env: join(ALTA_ROOT, ".env"),
};

export interface OandaCreds {
  apiKey: string;
  accountId: string;
  baseUrl: string;
  live: boolean;
}

/** Parse quant/.env (manual; no dotenv dep) for the OANDA credentials only. */
export function loadOandaCreds(): OandaCreds | null {
  let raw: string;
  try {
    raw = readFileSync(PATHS.env, "utf-8");
  } catch {
    return null;
  }
  const env: Record<string, string> = {};
  for (const line of raw.split("\n")) {
    const t = line.trim();
    if (!t || t.startsWith("#") || !t.includes("=")) continue;
    const i = t.indexOf("=");
    env[t.slice(0, i).trim()] = t.slice(i + 1).trim();
  }
  const apiKey = env.OANDA_API_KEY ?? "";
  const accountId = env.OANDA_ACCOUNT_ID ?? "";
  if (!apiKey || !accountId) return null;
  return {
    apiKey,
    accountId,
    baseUrl: (env.OANDA_BASE_URL || "https://api-fxpractice.oanda.com").replace(/\/+$/, ""),
    live: env.OANDA_LIVE === "1",
  };
}
