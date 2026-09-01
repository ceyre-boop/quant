/**
 * Front-end QA for the research terminal.
 *
 * Rewritten for the Vite app (app/). The previous version drove the old
 * single-file dashboard and asserted on panels that no longer exist — its
 * hypothesis-filter and prop-challenge checks were the first things to break.
 *
 * Run against a served build:
 *   cd app && bun run build
 *   python3 scripts/live_signals_server.py      # serves _site + the API
 *   node scripts/qa/dashboard_qa.mjs
 *
 * The backend being asleep or absent is NOT a failure — the terminal is designed
 * to degrade to committed data and to the browser-direct SEC path. What IS a
 * failure is a panel that sits on "loading" forever, or a JS error.
 */
import puppeteer from 'puppeteer-core'
import { mkdirSync, writeFileSync } from 'node:fs'

const BASE = process.env.QA_BASE || 'http://localhost:8765'
const CHROME = process.env.CHROME_PATH ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const SHOTS = new URL('./shots/', import.meta.url).pathname
mkdirSync(SHOTS, { recursive: true })

const sleep = ms => new Promise(r => setTimeout(r, ms))
const report = { ok: true, base: BASE, at: new Date().toISOString(), views: {}, http4xx: [] }

/**
 * Known-degraded conditions, each with the reason it is not a regression.
 * These are REPORTED under `degraded`, not silently dropped -- a blanket filter
 * on 500s would hide exactly the bugs this harness exists to catch.
 */
const KNOWN = [
  [/favicon/i, 'no favicon yet'],
  [/net::ERR_CONNECTION_REFUSED/, 'backend not running'],
  [/data\.sec\.gov.*blocked by CORS/i,
   'browser-direct SEC lookup blocked in this environment; curl with identical headers does receive access-control-allow-origin: *, so verify in a real browser'],
  [/net::ERR_FAILED/, 'follows the blocked SEC fetch above'],
  [/status of 404/, 'ticker not in the warm set, or artifact not harvested yet'],
  // Generic 5xx console lines carry no URL, so they are duplicative of the
  // http4xx list below, which IS classified by URL and does gate the result.
  [/status of 5\d\d/, 'see http4xx below -- classified by URL there'],
]

/**
 * HTTP failures, classified by URL. Anything not listed here fails the run.
 * This is the real gate: it is specific enough that a new broken route cannot
 * hide behind a generic filter.
 */
const KNOWN_HTTP = [
  [/\/favicon\.ico/, 'no favicon yet'],
  [/\/data\/fundamentals\/tickers\//, 'ticker not in the warm set -- run scripts/harvest_fundamentals.py'],
  [/\/replay\?/, 'PRE-EXISTING: sovereign/futures/bar_feed.py raises "IB required" -- IB is not connected. Not a regression from the terminal rebuild.'],
]

const degraded = []
function classify(msg) {
  for (const [rx, why] of KNOWN) {
    if (rx.test(msg)) { degraded.push({ msg: msg.slice(0, 140), why }); return true }
  }
  return false
}

const http4xx = []
function wire(page) {
  const errs = []
  page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()) })
  page.on('pageerror', e => errs.push('pageerror: ' + (e?.message || e)))
  page.on('response', r => { if (r.status() >= 400) http4xx.push(r.status() + ' ' + r.url()) })
  return errs
}

function record(view, errs, failures) {
  const real = errs.filter(e => !classify(e))
  report.views[view] = { consoleErrors: real, assertionFailures: failures }
  if (real.length || failures.length) report.ok = false
  errs.length = 0
}

const shot = (page, name) => page.screenshot({ path: SHOTS + name + '.png', fullPage: true })

/** The one assertion that matters everywhere: nothing stuck on "loading". */
const notStuck = () => {
  const out = []
  const main = document.querySelector('main')
  if (!main) return ['no <main>']
  const txt = main.innerText || ''
  if (/^\s*loading\b/im.test(txt) && txt.trim().split('\n').length < 3) {
    out.push('panel appears stuck on loading')
  }
  return out
}

async function evalAsserts(page, fn) {
  try { return await page.evaluate(fn) } catch (e) { return ['evaluate threw: ' + e.message] }
}

async function run() {
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  })
  const page = await browser.newPage()
  await page.setViewport({ width: 1440, height: 900 })
  const errs = wire(page)

  await page.goto(BASE + '/', { waitUntil: 'networkidle2', timeout: 30000 }).catch(() => {})
  await sleep(2000)

  const go = async name => {
    await page.click(`[data-tab="${name}"]`).catch(() => {})
    await sleep(2200)
  }

  // ---- Terminal: chart + fundamentals, same ticker ----
  await sleep(2500)
  await shot(page, 'terminal')
  record('terminal', errs, await evalAsserts(page, () => {
    const out = []
    if (!document.querySelector('[data-panel="terminal"]')) out.push('terminal not the default panel')
    if (!document.querySelector('#tv_chart')) out.push('TradingView container missing')
    const btns = [...document.querySelectorAll('button')]
    if (!btns.some(b => /Open on TradingView/i.test(b.textContent)))
      out.push('TradingView link-out button missing')
    const txt = document.querySelector('main')?.innerText || ''
    // Fundamentals must render SOMETHING honest, warm or partial.
    if (!/Earnings|Insider|Institutional|Short interest|Could not load/i.test(txt))
      out.push('fundamentals sections did not render')
    return out
  }))

  // ---- Signals + replay cockpit ----
  await go('signals')
  await shot(page, 'signals')
  record('signals', errs, await evalAsserts(page, () => {
    const out = []
    const btns = [...document.querySelectorAll('button')]
    if (!btns.some(b => /Replay cockpit/i.test(b.textContent))) out.push('replay mode toggle missing')
    return out
  }))

  // Switch into the cockpit and confirm the island actually mounted a chart.
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('button')].find(x => /Replay cockpit/i.test(x.textContent))
    b?.click()
  })
  await sleep(3000)
  await shot(page, 'replay')
  record('replay', errs, await evalAsserts(page, () => {
    const out = []
    const btns = [...document.querySelectorAll('button')]
    if (!btns.some(b => /Play day/i.test(b.textContent))) out.push('replay Play control missing')
    // lightweight-charts renders into a canvas; no canvas means the island failed.
    if (!document.querySelector('main canvas')) out.push('replay chart canvas not created')
    return out
  }))

  // ---- Calendar ----
  await go('calendar')
  await shot(page, 'calendar')
  record('calendar', errs, await evalAsserts(page, () => {
    const out = []
    const txt = document.querySelector('main')?.innerText || ''
    if (!/Month|unavailable|offline|snapshot/i.test(txt)) out.push('calendar rendered neither grid nor an honest empty state')
    return out
  }))

  // ---- Oracle ----
  await go('oracle')
  await shot(page, 'oracle')
  record('oracle', errs, await evalAsserts(page, () => {
    const out = []
    if (!document.querySelector('main input')) out.push('oracle input missing')
    return out
  }))

  // ---- Connections: the reduced Research tab ----
  await go('connections')
  await shot(page, 'connections')
  record('connections', errs, [
    ...(await evalAsserts(page, () => {
      const out = []
      const txt = document.querySelector('main')?.innerText || ''
      for (const k of ['oanda', 'sec_edgar', 'alpha_vantage', 'finra'])
        if (!txt.includes(k)) out.push('connections missing integration: ' + k)
      if (!/keyless/i.test(txt)) out.push('keyless integrations not distinguished')
      return out
    })),
    ...(await evalAsserts(page, notStuck)),
  ])

  // ---- Mobile ----
  await page.setViewport({ width: 390, height: 844 })
  await page.click('[data-tab="terminal"]').catch(() => {})
  await sleep(2000)
  await shot(page, 'terminal-mobile')
  record('mobile', errs, await evalAsserts(page, () => {
    const out = []
    if (document.documentElement.scrollWidth > window.innerWidth + 2)
      out.push('horizontal overflow at 390px')
    return out
  }))

  await browser.close()
  const seen = [...new Set(http4xx)]
  const unexpected = []
  for (const line of seen) {
    const hit = KNOWN_HTTP.find(([rx]) => rx.test(line))
    if (hit) degraded.push({ msg: line, why: hit[1] })
    else { unexpected.push(line); report.ok = false }
  }
  report.http4xx = { unexpected, total: seen.length }
  report.degraded = [...new Map(degraded.map(d => [d.why, d])).values()]
  writeFileSync(new URL('./report.json', import.meta.url), JSON.stringify(report, null, 2))

  console.log(JSON.stringify(report, null, 2))
  const asserts = Object.values(report.views).reduce((n, v) => n + v.assertionFailures.length, 0)
  console.log(`\nassertions failed: ${asserts} | unexpected HTTP: ${report.http4xx.unexpected.length} | degraded (expected): ${report.degraded.length}`)
  if (report.http4xx.unexpected.length) console.log('UNEXPECTED:', report.http4xx.unexpected)
  console.log(report.ok ? 'QA PASS' : 'QA FAIL')
  process.exit(report.ok ? 0 : 1)
}

run().catch(e => { console.error(e); process.exit(1) })
