/**
 * Where data comes from, and how honestly we say so when it doesn't.
 *
 * Two planes:
 *   STATIC  - committed JSON under data/, served by GitHub Pages or by
 *             live_signals_server.py's /data/ passthrough. Always available.
 *   LIVE    - the Python backend. On Render's free tier it sleeps after ~15min,
 *             so a cold call costs 30-50s. Callers must surface that as a real
 *             state, never as a spinner that looks broken.
 */

const isRemote =
  location.protocol === 'https:' ||
  !['localhost', '127.0.0.1', ''].includes(location.hostname)

export const API_BASE = isRemote
  ? 'https://sovereign-quant-dashboard.onrender.com'
  : ''                       // dev: vite proxies /api and /data to :8765

export const IS_REMOTE = isRemote
/** Remote needs a long ceiling purely because of Render cold starts. */
export const NET_TIMEOUT = isRemote ? 30_000 : 5_000

export class Timeout extends Error {}

export async function fetchJSON<T>(url: string, ms = NET_TIMEOUT): Promise<T> {
  const ctrl = new AbortController()
  const t = setTimeout(() => ctrl.abort(), ms)
  try {
    const r = await fetch(url, { signal: ctrl.signal })
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
    return (await r.json()) as T
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw new Timeout(url)
    throw e
  } finally {
    clearTimeout(t)
  }
}

/** Committed artifacts. Relative so Pages project-subpaths work unchanged. */
export const staticPath = (p: string) => `data/${p.replace(/^\/+/, '')}`
export const fetchStatic = <T,>(p: string) => fetchJSON<T>(staticPath(p))

/**
 * Backend routes.
 *
 * The Python server exposes them at the ROOT (/health, /data, /replay, ...), and
 * it also serves the built app, so same-origin calls must not be prefixed. Only
 * the Vite dev server on :5173 needs the /api prefix, because that is where the
 * proxy rule lives. Getting this wrong 404s every backend call — it did.
 */
const VITE_DEV = location.port === '5173'

export const apiPath = (p: string) =>
  `${API_BASE}${VITE_DEV ? '/api' : ''}/${p.replace(/^\/+/, '')}`

/**
 * Is the backend awake? /health is the existing no-compute liveness route.
 * Short timeout on purpose: we want a fast "asleep" answer so the UI can say so,
 * not a 30s hang before we can render anything.
 */
export async function backendAwake(ms = 3000): Promise<boolean> {
  try {
    await fetchJSON(apiPath('health'), ms)
    return true
  } catch {
    return false
  }
}
