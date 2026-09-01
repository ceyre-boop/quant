/**
 * Symbol resolution for the terminal.
 *
 * Two universes, deliberately merged into one search box:
 *
 *   SEC equities  10,391 US-listed tickers from company_tickers.json, WITH the
 *                 company name. An earlier build baked only ticker->CIK and threw
 *                 the name away, so typing "NVIDIA" failed while "NVDA" worked --
 *                 which reads as missing coverage when the data was there all along.
 *
 *   Non-SEC       FX pairs and futures. No issuer, no filings, so they can never
 *                 appear in an SEC file. They are traded here daily and must be
 *                 first-class rather than falling through to "not in the ticker map".
 *
 * The index is loaded lazily on first keystroke (~455 KB, ~1 fetch, then browser
 * cached) so it never costs anything on initial page load.
 */
import { fetchStatic } from './api'

export type Kind = 'equity' | 'fx' | 'future' | 'index' | 'crypto'

export type Symbol = {
  ticker: string
  name: string
  kind: Kind
  cik?: number
  /** TradingView symbol; equities use the bare ticker. */
  tv?: string
}

type RawIndex = {
  schema_version: number
  t: string[]
  c: number[]
  n: string[]
  non_sec?: { t: string; n: string; kind: Kind; tv: string }[]
}

let cache: Symbol[] | null = null
let inflight: Promise<Symbol[]> | null = null

export function loadSymbols(): Promise<Symbol[]> {
  if (cache) return Promise.resolve(cache)
  if (inflight) return inflight

  inflight = fetchStatic<RawIndex>('fundamentals/symbol_index.json')
    .then(raw => {
      const out: Symbol[] = []
      // Non-SEC first so an exact FX/futures match outranks a same-named equity.
      for (const s of raw.non_sec ?? []) {
        out.push({ ticker: s.t, name: s.n, kind: s.kind, tv: s.tv })
      }
      for (let i = 0; i < raw.t.length; i++) {
        out.push({ ticker: raw.t[i], name: raw.n[i] ?? '', kind: 'equity', cik: raw.c[i] })
      }
      cache = out
      return out
    })
    .catch(() => {
      cache = []
      return cache
    })
    .finally(() => { inflight = null })

  return inflight
}

/** Already-loaded symbols, or null. Lets callers resolve without awaiting. */
export const symbolsIfLoaded = () => cache

export function lookup(ticker: string): Symbol | null {
  const up = ticker.trim().toUpperCase()
  if (!cache || !up) return null
  return cache.find(s => s.ticker === up) ?? null
}

/**
 * Rank matters more than recall here: typing "AAPL" must put Apple first, not
 * "Apple iSports Group". Exact ticker wins outright, then ticker prefix, then a
 * word-start in the name, then anything containing the query.
 */
export function search(query: string, limit = 10): Symbol[] {
  const q = query.trim().toUpperCase()
  if (!q || !cache) return []

  const scored: { s: Symbol; rank: number }[] = []

  for (const s of cache) {
    const t = s.ticker
    const n = s.name.toUpperCase()
    let rank = -1

    if (t === q) rank = 0
    else if (t.startsWith(q)) rank = 1
    else if (n.startsWith(q)) rank = 2
    // Word-start inside the name: "MOTORS" should find "GENERAL MOTORS".
    else if (n.includes(' ' + q)) rank = 3
    else if (q.length >= 3 && n.includes(q)) rank = 4
    else if (q.length >= 3 && t.includes(q)) rank = 5

    if (rank >= 0) {
      // Nudge non-equities up a half-step within their band: a desk that trades
      // FX wants EURUSD before a micro-cap whose name happens to contain "EUR".
      if (s.kind !== 'equity') rank -= 0.5
      scored.push({ s, rank })
    }
  }

  scored.sort((a, b) =>
    a.rank - b.rank ||
    a.s.ticker.length - b.s.ticker.length ||
    a.s.ticker.localeCompare(b.s.ticker),
  )
  return scored.slice(0, limit).map(x => x.s)
}

/** What TradingView should load. Equities use the bare ticker. */
export const tvSymbol = (s: Symbol | null, fallback: string) =>
  s?.tv ?? (s ? s.ticker : fallback)

export const KIND_LABEL: Record<Kind, string> = {
  equity: 'stock',
  fx: 'fx',
  future: 'futures',
  index: 'index',
  crypto: 'crypto',
}
