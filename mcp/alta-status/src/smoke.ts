/**
 * Smoke test — exercises every data-access function against the real Alta files (and the live
 * OANDA pull) without the MCP transport. Confirms the tools return real data, not stubs.
 * Run: bun run src/smoke.ts
 */
import { accountStatus, morningBrief, signals, oracleReflection, loopHealth, researchPanel } from "./alta.js";
import { ALTA_ROOT } from "./config.js";

const line = (s: string) => console.log(s);

line(`ALTA_ROOT = ${ALTA_ROOT}\n`);

const acct = await accountStatus();
line(`account_status        : ${acct.available ? `OK  NAV=${acct.nav} mode=${acct.mode} openPos=${acct.open_position_count}` : `unavailable (${acct.note})`}`);

const mb = morningBrief(0);
line(`morning_brief         : ${mb.available ? `OK  ${mb.date} regime=${mb.regime} bias=${mb.directional_bias} conf=${mb.confidence}` : `unavailable (${mb.note})`}`);

const sg = signals();
line(`signals               : ${sg.available ? `OK  ${sg.pairs.length} pairs, scan=${sg.last_scan}` : `unavailable (${sg.note})`}`);

const orc = oracleReflection(1);
line(`oracle_reflection     : ${orc.available ? `OK  ${orc.reflections[0].date}: "${String(orc.reflections[0].lesson_text).slice(0, 70)}..."` : `unavailable (${orc.note})`}`);

const lh = loopHealth();
line(`loop_health           : ${lh.available ? `OK  ${lh.loops.length} loops, down=[${lh.down}], market_hours=${lh.market_hours}` : `unavailable (${lh.note})`}`);

const rp = researchPanel({});
line(`research_panel        : ${rp.available ? `OK  ${rp.date}, sources_ok=${rp.sources_ok?.length}, null=[${rp.sources_null}]` : `unavailable (${rp.note})`}`);

const fail = [acct, mb, sg, orc, lh, rp].filter((r) => !r.available && !String(r.note).includes("OANDA"));
line(`\n${fail.length === 0 ? "✅ all file-backed tools returned data" : `⚠ ${fail.length} unexpected failures`}`);
