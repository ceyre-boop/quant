/**
 * Alta data access — read-only readers over the operation's state files + a live OANDA pull.
 * Every reader degrades gracefully (returns a `{ available: false, note }` shape) so a missing
 * file or a dead source never throws across the MCP boundary.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { PATHS, loadOandaCreds } from "./config.js";

function readJson(path: string): any | null {
  try {
    return JSON.parse(readFileSync(path, "utf-8"));
  } catch {
    return null;
  }
}

// ── account (live OANDA) ──────────────────────────────────────────────────────
export async function accountStatus(): Promise<any> {
  const creds = loadOandaCreds();
  if (!creds) return { available: false, note: "OANDA credentials not found in quant/.env" };
  const headers = { Authorization: `Bearer ${creds.apiKey}` };
  const acctUrl = `${creds.baseUrl}/v3/accounts/${creds.accountId}`;
  try {
    const [sumRes, posRes] = await Promise.all([
      fetch(`${acctUrl}/summary`, { headers }),
      fetch(`${acctUrl}/openPositions`, { headers }),
    ]);
    if (!sumRes.ok) {
      return { available: false, note: `OANDA HTTP ${sumRes.status} ${sumRes.statusText}` };
    }
    const s = (await sumRes.json()).account;
    const positions = posRes.ok ? ((await posRes.json()).positions ?? []) : [];
    return {
      available: true,
      mode: creds.live ? "LIVE" : "practice",
      nav: Number(s.NAV),
      balance: Number(s.balance),
      unrealized_pl: Number(s.unrealizedPL),
      realized_pl_lifetime: Number(s.pl),
      margin_used: Number(s.marginUsed),
      margin_available: Number(s.marginAvailable),
      open_trade_count: Number(s.openTradeCount),
      open_position_count: Number(s.openPositionCount),
      pending_order_count: Number(s.pendingOrderCount),
      last_transaction_id: s.lastTransactionID,
      open_positions: positions.map((p: any) => {
        const long = Number(p.long?.units ?? 0);
        const short = Number(p.short?.units ?? 0);
        const side = long > 0 ? "LONG" : short < 0 ? "SHORT" : "FLAT";
        return {
          instrument: p.instrument,
          side,
          units: side === "LONG" ? long : short,
          unrealized_pl: Number(p.long?.unrealizedPL ?? 0) + Number(p.short?.unrealizedPL ?? 0),
        };
      }),
    };
  } catch (e: any) {
    return { available: false, note: `OANDA request failed: ${e?.name ?? "error"}` };
  }
}

// ── morning brief ─────────────────────────────────────────────────────────────
export function morningBrief(narrativeChars = 2000): any {
  const b = readJson(PATHS.morningBrief);
  if (!b) return { available: false, note: "No morning briefing found (latest.json)." };
  return {
    available: true,
    date: b.date,
    generated_at: b.generated_at,
    synthesis_source: b.synthesis_source,
    regime: b.meta_regime ?? b.regime_call,
    directional_bias: b.directional_bias,
    confidence: b.confidence,
    key_level: b.key_level,
    big_move_headline: b.big_move_headline?.headline ?? null,
    macro_economic: b.macro_economic?.text ?? null,
    scorecard_summary: b.scorecard_summary,
    narrative: typeof b.narrative === "string" ? b.narrative.slice(0, narrativeChars) : null,
    provenance_note: b.provenance?.note,
  };
}

// ── forex signals (proximity) ───────────────────────────────────────────────────
export function signals(): any {
  const d = readJson(PATHS.forexProximity);
  if (!d) return { available: false, note: "No forex proximity scan found." };
  return {
    available: true,
    last_scan: d.last_scan,
    pairs: (d.pairs ?? []).map((p: any) => ({
      pair: p.pair,
      conviction: p.conviction,
      direction: p.direction,
      pct_to_trigger: p.pct_to_trigger,
      regime: p.regime,
      rate_differential: p.rate_differential,
    })),
  };
}

// ── oracle reflection (latest, or last N) ───────────────────────────────────────
export function oracleReflection(count = 1): any {
  let files: string[];
  try {
    // Sort by mtime (newest first) — the dir mixes naming conventions (YYYY-MM-DD.json and
    // older reflection_<ts>.json with a different schema), so alphabetical sort is unreliable.
    files = readdirSync(PATHS.reflectionsDir)
      .filter((f) => f.endsWith(".json"))
      .map((f) => ({ f, m: statSync(join(PATHS.reflectionsDir, f)).mtimeMs }))
      .sort((a, b) => b.m - a.m)
      .slice(0, Math.max(1, Math.min(count, 14)))
      .map((x) => x.f);
  } catch {
    return { available: false, note: "No reflections directory." };
  }
  if (files.length === 0) return { available: false, note: "No reflections found." };
  const out = files.map((f) => {
    const r = readJson(join(PATHS.reflectionsDir, f));
    const ref = r?.reflection ?? {};
    const cl = ref.candidate_lesson ?? {};
    return {
      date: f.replace(".json", ""),
      lesson_text: typeof cl === "object" ? cl.lesson_text : cl,
      mechanism: typeof cl === "object" ? cl.mechanism : null,
      testable_rule: typeof cl === "object" ? cl.testable_rule : null,
      reasoning_component_targeted: typeof cl === "object" ? cl.reasoning_component_targeted : null,
      system_health_note: ref.system_health_note ?? null,
    };
  });
  return { available: true, reflections: out };
}

// ── loop health ─────────────────────────────────────────────────────────────────
export function loopHealth(): any {
  const d = readJson(PATHS.loopHealth);
  if (!d) return { available: false, note: "No loop health status found." };
  const loops = d.loops ?? {};
  return {
    available: true,
    checked_at: d.checked_at,
    market_hours: d.market_hours,
    frozen: d.frozen,
    down: d.down ?? [],
    loops: Object.entries(loops).map(([name, v]: [string, any]) => ({
      name,
      status: v.status,
      last: v.last,
      silence_hours: v.silence_hours,
      threshold: v.threshold,
    })),
  };
}

// ── proof of life (is the system alive & producing signal) ──────────────────────
export function proofOfLife(): any {
  const d = readJson(PATHS.proofOfLife);
  if (!d) return { available: false, note: "No proof-of-life snapshot yet (run scripts/proof_of_life.py)." };
  return {
    available: true,
    generated_at: d.generated_at,
    alive: d.alive,
    fired_today: d.fired_today,
    summary: d.summary_line,
    n_would_fire_today: d.n_would_fire_today,
    strongest_signal: d.strongest_signal,
    pairs: d.pairs,
    last_fill: d.last_fill,
    loops_alive: d.loops_alive,
    loops_down: d.loops_down,
    scan_age_hours: d.scan_age_hours,
  };
}

// ── research panel (a given day, default latest) ────────────────────────────────
export function researchPanel(opts: { date?: string; source?: string } = {}): any {
  let file: string;
  if (opts.date) {
    file = join(PATHS.panelDir, `${opts.date}.json`);
  } else {
    let days: string[];
    try {
      days = readdirSync(PATHS.panelDir)
        .filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f))
        .sort()
        .reverse();
    } catch {
      return { available: false, note: "No panel directory." };
    }
    if (days.length === 0) return { available: false, note: "No panel snapshots yet." };
    file = join(PATHS.panelDir, days[0]);
  }
  const d = readJson(file);
  if (!d) return { available: false, note: `No panel snapshot for ${opts.date ?? "latest"}.` };
  let sources = d.sources ?? {};
  if (opts.source) {
    sources = sources[opts.source]
      ? { [opts.source]: sources[opts.source] }
      : {};
  }
  return {
    available: true,
    date: d.date,
    generated_at: d.generated_at,
    sources_ok: d.sources_ok,
    sources_null: d.sources_null,
    sources,
    provenance_note: d.provenance?.note,
  };
}
