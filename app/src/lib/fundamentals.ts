/**
 * Shape of data/fundamentals/tickers/{TICKER}.json, mirroring the Python
 * dataclasses in sovereign/fundamentals/types.py.
 *
 * Every section carries as_of / staleness_days / sources / gaps. That is not
 * decoration: three of the six categories are structurally lagged (13F ~45 days,
 * official short interest ~8 days, Form 4 T+2), so a panel that renders them next
 * to a live price without saying when they are from is actively misleading.
 */
import { fetchStatic, fetchJSON, apiPath, backendAwake } from './api'
import { lookup, loadSymbols, KIND_LABEL } from './symbols'

export type Section<T> = {
  as_of: string | null
  staleness_days: number | null
  sources: string[]
  gaps: string[]
  rows: T[]
}

export type EarningsRow = {
  fiscal_end: string | null
  report_date: string | null
  report_time: 'bmo' | 'amc' | 'unknown'
  eps_estimate: number | null
  eps_actual: number | null
  eps_surprise: number | null
  eps_surprise_pct: number | null
  rev_estimate: number | null
  rev_actual: number | null
  /** Free tier cannot source forward guidance. Null by design, rendered greyed. */
  guide_eps_low: number | null
  guide_eps_high: number | null
  reaction?: {
    gap_pct: number | null
    d0_pct: number | null
    d1_pct: number | null
    d5_pct: number | null
    d0_excess_spy: number | null
    gap_over_atr: number | null
  } | null
  source: string
}

export type InsiderRow = {
  owner_name: string
  owner_title: string
  txn_date: string | null
  filing_date: string | null
  code: string
  shares: number | null
  price: number | null
  value_usd: number | null
  is_open_market: boolean
}

export type InsiderSection = Section<InsiderRow> & {
  summary: {
    buys_180d: number
    sells_180d: number
    net_shares_180d: number | null
    net_value_usd_180d: number | null
    /** True when counts come from filing cadence only (the browser-direct path,
     *  which cannot see share amounts because those live on www.sec.gov/Archives). */
    counts_only?: boolean
  } | null
}

export type HolderRow = { filer_name: string; shares: number | null; d_shares: number | null }

export type InstitutionsSection = Section<never> & {
  period_end: string | null
  filing_date_max: string | null
  n_holders: number | null
  d_holders_qoq: number | null
  total_shares: number | null
  d_shares_qoq: number | null
  top_buyers: HolderRow[]
  top_sellers: HolderRow[]
}

export type ShortSection = {
  as_of: string | null
  staleness_days: number | null
  sources: string[]
  gaps: string[]
  /** Official bimonthly short INTEREST. */
  interest: { settlement_date: string; shares_short: number | null; days_to_cover: number | null }[]
  /** Daily short VOLUME. A different measurement — never merge into interest. */
  short_volume: { date: string; short_pct: number | null }[]
  borrow: { date: string; tier: string; fee_rate: number | null } | null
}

export type Panel = {
  schema_version: number
  ticker: string
  cik: number | null
  name: string | null
  generated_at: string | null
  /** Which sections the active provider can actually serve. The UI greys the
   *  rest rather than hiding them, so the gap stays visible. */
  capabilities: string[]
  partial?: boolean
  sections: {
    earnings?: Section<EarningsRow>
    insider?: InsiderSection
    institutions?: InstitutionsSection
    short?: ShortSection
  }
}

export type Load =
  | { state: 'loading' }
  | { state: 'warm';    panel: Panel }
  | { state: 'live';    panel: Panel }
  | { state: 'partial'; panel: Panel; reason: string }
  | { state: 'waking' }
  | { state: 'error';   reason: string }

/** SEC submissions is the ONE endpoint a browser can reach cross-origin. */
const SEC_SUBMISSIONS = (cik: number) =>
  `https://data.sec.gov/submissions/CIK${String(cik).padStart(10, '0')}.json`

type SecSubmissions = {
  name?: string
  filings?: { recent?: { form?: string[]; filingDate?: string[]; accessionNumber?: string[] } }
}

/**
 * Tier B — the static floor. Works with no backend at all, on a bare Pages host.
 * Gives filing cadence and presence flags, and is explicit that it cannot give
 * amounts, because those live behind a non-CORS origin.
 */
export async function loadFromSEC(ticker: string, cik: number): Promise<Panel> {
  const j = await fetchJSON<SecSubmissions>(SEC_SUBMISSIONS(cik), 12_000)
  const r = j.filings?.recent
  const forms = r?.form ?? []
  const dates = r?.filingDate ?? []

  const now = Date.now()
  const withinDays = (iso: string, d: number) =>
    (now - Date.parse(iso)) / 86_400_000 <= d

  let f4_30 = 0, f4_90 = 0, f4_180 = 0
  const rows: InsiderRow[] = []
  let has13F = false, has13DG = false

  for (let i = 0; i < forms.length; i++) {
    const form = forms[i], d = dates[i]
    if (!d) continue
    if (form === '4') {
      if (withinDays(d, 30)) f4_30++
      if (withinDays(d, 90)) f4_90++
      if (withinDays(d, 180)) {
        f4_180++
        rows.push({
          owner_name: '—', owner_title: '', txn_date: null, filing_date: d,
          code: '', shares: null, price: null, value_usd: null, is_open_market: false,
        })
      }
    }
    if (form?.startsWith('13F')) has13F = true
    if (form?.startsWith('SC 13')) has13DG = true
  }

  const gaps = [
    'insider amounts: www.sec.gov/Archives is not CORS-open — needs the backend',
    'estimates: no browser-reachable source',
    'price reaction: needs bars from the backend',
    'short interest: FINRA and Nasdaq are not CORS-open',
  ]
  if (!has13F) gaps.push('institutions: 13F is filed by the institution, not the issuer')

  return {
    schema_version: 1, ticker, cik, name: j.name ?? null,
    generated_at: new Date().toISOString(),
    capabilities: ['insider_cadence', 'filings'],
    partial: true,
    sections: {
      insider: {
        as_of: dates[0] ?? null, staleness_days: null,
        sources: ['sec_submissions'],
        gaps: ['share amounts unavailable in-browser'],
        summary: {
          buys_180d: 0, sells_180d: 0,
          net_shares_180d: null, net_value_usd_180d: null,
          counts_only: true,
        },
        rows: rows.slice(0, 60),
      },
      institutions: {
        as_of: null, staleness_days: null, sources: ['sec_submissions'],
        gaps: [has13DG ? '13D/G present — detail needs the backend' : 'no 13D/G on file'],
        period_end: null, filing_date_max: null,
        n_holders: null, d_holders_qoq: null,
        total_shares: null, d_shares_qoq: null,
        top_buyers: [], top_sellers: [], rows: [] as never[],
      },
    },
  }
}

/** Warm -> live -> SEC floor, in that order, saying honestly which one answered. */
export async function loadPanel(ticker: string, cikMap: Record<string, number>): Promise<Load> {
  try {
    const p = await fetchStatic<Panel>(`fundamentals/tickers/${ticker}.json`)
    return { state: 'warm', panel: p }
  } catch { /* not in the warm set — expected for most tickers */ }

  const cik = cikMap[ticker]
  const awake = await backendAwake()
  if (awake) {
    try {
      const p = await fetchJSON<Panel>(apiPath(`fundamentals?ticker=${encodeURIComponent(ticker)}`))
      return { state: 'live', panel: p }
    } catch { /* fall through to the static floor */ }
  }

  if (cik == null) {
    // Absent from the SEC map is NOT an error. FX pairs, futures and indices have
    // no issuer and never file, so there is nothing to look up — that is a fact
    // about the instrument, not a failed lookup. Saying "check the symbol" here
    // made a working tool look broken.
    //
    // Await the index rather than reading a possibly-cold cache: getting this
    // wrong would tell the user a symbol "was not found" purely because the
    // lookup table had not finished loading yet.
    await loadSymbols()
    const sym = lookup(ticker)
    const reason = sym && sym.kind !== 'equity'
      ? `${ticker} is ${KIND_LABEL[sym.kind]} — no issuer files with the SEC, so there are no `
        + 'earnings, insider or 13F records. The chart and price data still work.'
      : `${ticker} was not found. Search by ticker (NVDA) or company name (NVIDIA).`

    return {
      state: 'partial',
      panel: {
        schema_version: 1, ticker, cik: null, name: sym?.name ?? null,
        generated_at: new Date().toISOString(),
        capabilities: [], partial: true, sections: {},
      },
      reason,
    }
  }
  try {
    const p = await loadFromSEC(ticker, cik)
    return {
      state: 'partial', panel: p,
      reason: awake
        ? 'Backend reachable but returned no panel — showing SEC filings only.'
        : 'Backend asleep — showing the SEC filings that a browser can reach directly.',
    }
  } catch (e) {
    // The browser-direct SEC path can fail for reasons outside our control:
    // an SEC rate-limit response carries no CORS header, and some networks
    // block the host outright. That is a degraded state with a real
    // explanation, not an unexpected error — say which, and say what fixes it.
    const msg = e instanceof Error ? e.message : String(e)
    return {
      state: 'partial',
      panel: {
        schema_version: 1, ticker, cik, name: null,
        generated_at: new Date().toISOString(),
        capabilities: [], partial: true, sections: {},
      },
      reason: `${ticker} is not in the warm set, and the direct SEC lookup did not complete (${msg}). `
        + 'Start the local backend, or add this ticker to the warm watchlist and re-run the harvester.',
    }
  }
}
