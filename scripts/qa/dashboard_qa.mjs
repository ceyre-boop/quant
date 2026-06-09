// Real-Chrome QA harness for the SOVEREIGN dashboard.
// Drives the installed Google Chrome (headless) against a running dashboard, captures every
// console error / uncaught throw / failed request, asserts the rendered DOM, and screenshots
// desktop + 375px mobile for every tab + the ICT page.
//
// Usage: node scripts/qa/dashboard_qa.mjs [baseUrl]   (default http://localhost:8765)
import puppeteer from 'puppeteer-core';
import { mkdirSync, writeFileSync } from 'fs';

const BASE = process.argv[2] || 'http://localhost:8765';
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const SHOTS = new URL('./shots/', import.meta.url).pathname;
mkdirSync(SHOTS, { recursive: true });

const BENIGN = [/favicon/i, /tradingview/i, /tv\.js/i, /s3\.tradingview/i, /telemetry/i]; // harmless 3rd-party (TV embed) noise
const sleep = ms => new Promise(r => setTimeout(r, ms));
const report = { base: BASE, views: {}, ok: true };

function record(view, errors, failures) {
  const realErrors = errors.filter(e => !BENIGN.some(rx => rx.test(e)));
  report.views[view] = { consoleErrors: realErrors, assertionFailures: failures };
  if (realErrors.length || failures.length) report.ok = false;
}

// attach error collectors to a page; returns the live arrays
const http4xx = []; // global list of 4xx/5xx URLs seen across the run
function wire(page) {
  const errs = [], reqfail = [];
  page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
  page.on('pageerror', e => errs.push('pageerror: ' + (e?.message || e)));
  page.on('requestfailed', r => {
    const u = r.url();
    if (!/favicon/i.test(u)) reqfail.push('reqfail: ' + u + ' (' + (r.failure()?.errorText) + ')');
  });
  page.on('response', r => { if (r.status() >= 400) http4xx.push(r.status() + ' ' + r.url()); });
  return { errs, reqfail };
}

async function shot(page, name) { await page.screenshot({ path: SHOTS + name + '.png', fullPage: true }); }

// ---- assertion helpers run in the browser ----
const A = {
  noLoading: () => [...document.querySelectorAll('.rp-empty,.tab-panel.active *')]
      .some(el => /Loading…|Loading\.\.\./.test(el.textContent) && el.offsetParent !== null)
      ? 'still shows Loading…' : null,
};

async function evalAsserts(page, fn) {
  try { return await page.evaluate(fn); } catch (e) { return ['evaluate threw: ' + e.message]; }
}

async function run() {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage'] });

  // ============ DESKTOP ============
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  const collect = wire(page);

  await page.goto(BASE + '/', { waitUntil: 'networkidle2', timeout: 30000 }).catch(() => {});
  await sleep(1500);

  // helper to click a tab and settle
  const tab = async name => { await page.click(`.hdr-tab[data-tab="${name}"]`).catch(()=>{}); await sleep(2500); };

  // TRADE (default active) — pass-probability cockpit + de-stale checks
  await sleep(2500); // let /prop-challenge + /next-move land
  await shot(page, 'trade');
  let f = await evalAsserts(page, () => {
    const out = [];
    if (!document.querySelector('#tab-trade.active')) out.push('TRADE tab not active by default');
    const cards = document.querySelectorAll('#forex-grid .forex-card').length;
    if (cards !== 4) out.push('forex grid should have 4 cards (AUDNZD excluded), found ' + cards);
    const bodyTxt = document.getElementById('tab-trade').textContent;
    if (/AUD\/NZD/.test(bodyTxt)) out.push('AUDNZD card still present (HYP-045 excluded it)');
    // stale FRAMING (not honest legacy labels): v013-as-current / the uncosted achievement banner
    if (/v013 current/.test(bodyTxt)) out.push('"v013 current" still shown');
    if (/INSTITUTIONAL GRADE ACHIEVED/.test(bodyTxt)) out.push('false "INSTITUTIONAL GRADE ACHIEVED" banner still shown');
    if (/1\.8552/.test(bodyTxt)) out.push('uncosted 1.8552 headline still shown');
    if (!/v015/.test(bodyTxt)) out.push('v015 not shown');
    // pass probability must be a real number, not the "—" placeholder or "Loading"
    const pass = (document.getElementById('ph-prob-pass') || {}).textContent || '';
    if (!/\d+\.\d%|\d+%/.test(pass)) out.push('P(pass) not populated: "' + pass + '"');
    if (/Loading/.test((document.getElementById('ph-caveat') || {}).textContent || '')) out.push('pass-prob caveat stuck on Loading');
    // control panel present
    if (!document.querySelector('.ctl-section')) out.push('control panel missing');
    if (!document.getElementById('nbm-body')) out.push('Next Best Move card missing');
    if (!document.querySelector('[data-ctl="run-queue"]')) out.push('Run Queue button missing');
    return out;
  });
  record('trade', collect.errs.splice(0), f);

  // CONTROL: token gating — without a token the status should read "locked"; the Next Best Move
  // card should render the queued Oracle prompt (read-only, no token needed).
  const ctlCheck = await evalAsserts(page, async () => {
    const out = [];
    localStorage.removeItem('sq_ctl_token');
    const nbm = (document.getElementById('nbm-body') || {}).textContent || '';
    if (/Loading/.test(nbm)) out.push('Next Best Move stuck on Loading');
    // try an action without a token → must NOT spawn a job; status flips to an error/lock hint
    const runBtn = document.querySelector('[data-ctl="checklist"]');
    if (runBtn) runBtn.click();
    await new Promise(r => setTimeout(r, 400));
    const st = (document.getElementById('ctl-status') || {}).textContent || '';
    if (!/token/i.test(st)) out.push('no-token action did not warn about token: "' + st + '"');
    return out;
  });
  if (ctlCheck.length) { report.views['trade'].assertionFailures.push(...ctlCheck); report.ok = false; }

  // SIGNALS — default TradingView embed mode
  await tab('signals'); await sleep(5000); await shot(page, 'signals');
  f = await evalAsserts(page, () => {
    const out = [];
    if (!document.querySelector('#tab-signals')) out.push('no signals panel');
    if (!document.getElementById('sigmode-tv')) out.push('mode toggle missing');
    if (document.getElementById('sig-tv').classList.contains('hidden')) out.push('TV mode not active by default');
    if (!document.querySelector('#tv_chart iframe')) out.push('TradingView embed iframe did not load');
    return out;
  });
  record('signals', collect.errs.splice(0), f);
  // click the free QQQ proxy (futures are gated in the free widget) and screenshot a real chart
  await page.click('#sig-tv-symbols .tvsym[data-sym="NASDAQ:QQQ"]').catch(()=>{}); await sleep(6000); await shot(page, 'signals-qqq');
  // switch to Replay Cockpit, load a day, screenshot, then start playback
  await page.click('#sigmode-replay').catch(()=>{}); await sleep(6000); await shot(page, 'signals-replay');
  let rf = await evalAsserts(page, () => {
    const out = [];
    if (!document.getElementById('replay-cockpit')) out.push('replay cockpit missing');
    if (!document.getElementById('rp-orders')) out.push('LIVE ORDERS panel missing');
    if (document.getElementById('sig-bridge') && document.getElementById('sig-bridge').offsetParent) out.push('Bridge State still visible');
    const st = (document.getElementById('rp-status')||{}).textContent || '';
    if (!/ready|trades|bars/.test(st)) out.push('replay did not load: ' + st);
    return out;
  });
  record('replay', collect.errs.splice(0), rf);
  // press PLAY, let a few bars animate, screenshot a mid-play frame
  await page.click('#rp-play').catch(()=>{}); await sleep(8000); await shot(page, 'signals-replay-playing');

  // RESEARCH — the card that kept breaking
  await tab('research'); await sleep(2500); await shot(page, 'research');
  f = await evalAsserts(page, () => {
    const out = [];
    const cards = ['rp-health','rp-messages','rp-usage','rp-hyp','rp-queue','rp-gates','rp-reflection','rp-fills','rp-versions','rp-regime','rp-indicators'];
    for (const id of cards) {
      const el = document.getElementById(id);
      if (!el) { out.push(id + ' MISSING'); continue; }
      if (/Loading…|Loading\.\.\./.test(el.textContent)) out.push(id + ' stuck on Loading');
    }
    const rows = document.querySelectorAll('#rp-hyp tbody tr').length;
    if (rows < 1) out.push('ledger has no rows');
    if (/[>\s]\?\s*</.test(document.getElementById('rp-hyp')?.innerHTML || '')) out.push('ledger has ? status icon');
    return out;
  });
  record('research', collect.errs.splice(0), f);

  // RESEARCH filter interaction: click REJECTED, count must change & be >0
  const filterCheck = await evalAsserts(page, async () => {
    const out = [];
    const before = document.querySelectorAll('#rp-hyp tbody tr').length;
    const btn = document.querySelector('#hyp-filter .sig-tb-btn[data-status="REJECTED"]');
    if (!btn) return ['REJECTED filter button missing'];
    btn.click();
    await new Promise(r => setTimeout(r, 300));
    const after = document.querySelectorAll('#rp-hyp tbody tr').length;
    if (after < 1) out.push('REJECTED filter shows 0 rows');
    if (after === before && before > after) out.push('filter did not change rows');
    // reset to ALL
    document.querySelector('#hyp-filter .sig-tb-btn[data-status="ALL"]')?.click();
    return out;
  });
  if (filterCheck.length) { report.views['research'].assertionFailures.push(...filterCheck); report.ok = false; }

  // TRADES
  await tab('trades'); await shot(page, 'trades');
  f = await evalAsserts(page, () => {
    const out = [];
    if (!document.querySelector('#tab-trades')) out.push('no trades panel');
    return out;
  });
  record('trades', collect.errs.splice(0), f);

  // CALENDAR
  await tab('calendar'); await sleep(6000); await shot(page, 'calendar');
  f = await evalAsserts(page, () => {
    const out = [];
    if (!document.getElementById('tab-calendar')) out.push('calendar tab missing');
    if (!document.querySelector('#cal-grid table')) out.push('calendar grid did not render');
    const title = (document.getElementById('cal-title')||{}).textContent || '';
    if (!/\d{4}/.test(title)) out.push('calendar title missing: ' + title);
    return out;
  });
  record('calendar', collect.errs.splice(0), f);

  // CHAT
  await tab('chat'); await shot(page, 'chat');
  f = await evalAsserts(page, () => {
    const out = [];
    if (!document.getElementById('chat-input')) out.push('chat input missing');
    const chips = document.querySelectorAll('#chat-chips button').length;
    if (chips < 1) out.push('chat chips missing');
    return out;
  });
  record('chat', collect.errs.splice(0), f);

  // ============ ICT PAGE ============
  await page.goto(BASE + '/ict/', { waitUntil: 'networkidle2', timeout: 30000 }).catch(() => {});
  await sleep(2500); await shot(page, 'ict');
  f = await evalAsserts(page, () => {
    const out = [];
    const sb = document.getElementById('sc-bridge');
    if (!sb) out.push('sc-bridge missing');
    const card = sb?.closest('.state-card');
    // live cross-state is NORMAL/0.00 -> cards should NOT be in alert(red)
    if (card && card.classList.contains('alert')) out.push('Bridge card still RED on NORMAL state');
    const tsub = document.getElementById('sc-threat-sub')?.textContent || '';
    if (/CRITICAL/.test(tsub)) out.push('threat subtext stuck on CRITICAL: ' + tsub);
    if (/Loading…/.test(document.getElementById('oracle-text')?.textContent || '')) out.push('oracle text stuck Loading');
    return out;
  });
  record('ict', collect.errs.splice(0), f);

  // ============ MOBILE 375 ============
  await page.setViewport({ width: 375, height: 800, isMobile: true });
  for (const v of [['/', 'home'], ['/ict/', 'ict']]) {
    await page.goto(BASE + v[0], { waitUntil: 'networkidle2', timeout: 30000 }).catch(() => {});
    await sleep(2000);
    await shot(page, v[1] + '-mobile');
    const ov = await evalAsserts(page, () => {
      const sw = document.documentElement.scrollWidth, cw = document.documentElement.clientWidth;
      return sw > cw + 2 ? ['horizontal overflow: scrollWidth ' + sw + ' > ' + cw] : [];
    });
    record('mobile' + (v[1] === 'home' ? '' : '-ict'), collect.errs.splice(0), ov);
  }

  report.http4xx = [...new Set(http4xx)];
  await browser.close();
  writeFileSync(new URL('./report.json', import.meta.url), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  process.exit(report.ok ? 0 : 1);
}
run().catch(e => { console.error('HARNESS CRASH:', e); process.exit(2); });
