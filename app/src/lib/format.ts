export const pct = (v: number | null | undefined, d = 2) =>
  v == null || !isFinite(v) ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(d)}%`

export const num = (v: number | null | undefined, d = 2) =>
  v == null || !isFinite(v) ? '—' : v.toFixed(d)

export function compact(v: number | null | undefined): string {
  if (v == null || !isFinite(v)) return '—'
  const a = Math.abs(v)
  const s = v < 0 ? '-' : ''
  if (a >= 1e12) return `${s}${(a / 1e12).toFixed(2)}T`
  if (a >= 1e9) return `${s}${(a / 1e9).toFixed(2)}B`
  if (a >= 1e6) return `${s}${(a / 1e6).toFixed(2)}M`
  if (a >= 1e3) return `${s}${(a / 1e3).toFixed(1)}K`
  return `${s}${a.toFixed(0)}`
}

export const usd = (v: number | null | undefined) =>
  v == null || !isFinite(v) ? '—' : `${v < 0 ? '-' : ''}$${compact(Math.abs(v))}`

export const day = (d: string | null | undefined) => (d ? d.slice(0, 10) : '—')

/** Staleness in days, for the "as of" chips. */
export function staleDays(iso: string | null | undefined): number | null {
  if (!iso) return null
  const t = Date.parse(iso)
  if (isNaN(t)) return null
  return Math.floor((Date.now() - t) / 86_400_000)
}
