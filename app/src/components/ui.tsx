import type { ReactNode } from 'react'

export function Panel({
  title, right, children, className = '', dense = false,
}: {
  title?: ReactNode; right?: ReactNode; children: ReactNode
  className?: string; dense?: boolean
}) {
  return (
    <section className={`bg-surface border border-line rounded-md flex flex-col min-h-0 ${className}`}>
      {(title || right) && (
        <header className="flex items-center justify-between gap-3 px-3 py-2 border-b border-line-soft shrink-0">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-muted">{title}</h2>
          {right}
        </header>
      )}
      <div className={`min-h-0 flex-1 max-w-full overflow-x-auto ${dense ? '' : 'p-3'}`}>{children}</div>
    </section>
  )
}

/** A source + freshness chip. Used on every fundamentals section — staleness is
 *  first-class UI here because three of six categories are structurally lagged
 *  (13F ~45d, official short interest ~8d, Form 4 T+2). */
export function AsOf({ date, days, sources }: {
  date?: string | null; days?: number | null; sources?: string[]
}) {
  if (!date && !sources?.length) return null
  const stale = days != null && days > 45
  return (
    <div className="flex items-center gap-2 text-[10px] text-faint">
      {sources?.length ? <span className="uppercase tracking-wider">{sources.join(' · ')}</span> : null}
      {date ? (
        <span className={stale ? 'text-warn' : ''}>
          as of {date.slice(0, 10)}{days != null ? ` · ${days}d` : ''}
        </span>
      ) : null}
    </div>
  )
}

/** Never render an empty chart where a real one belongs. Say why it's empty. */
export function Empty({ reason, hint }: { reason: string; hint?: string }) {
  return (
    <div className="h-full min-h-[80px] flex flex-col items-center justify-center text-center gap-1 px-4 py-6">
      <p className="text-[12px] text-muted">{reason}</p>
      {hint && <p className="text-[11px] text-faint max-w-md">{hint}</p>}
    </div>
  )
}

export function Loading({ label = 'loading' }: { label?: string }) {
  return (
    <div className="h-full min-h-[80px] flex items-center justify-center gap-2 text-[11px] text-faint">
      <span className="w-1.5 h-1.5 rounded-full bg-accent pulse" />
      {label}
    </div>
  )
}

export function Delta({ v, digits = 2, suffix = '%' }: {
  v: number | null | undefined; digits?: number; suffix?: string
}) {
  if (v == null || !isFinite(v)) return <span className="num text-faint">—</span>
  const c = v > 0 ? 'text-up' : v < 0 ? 'text-down' : 'text-muted'
  return <span className={`num ${c}`}>{v > 0 ? '+' : ''}{v.toFixed(digits)}{suffix}</span>
}

export function Pill({ tone = 'muted', children }: {
  tone?: 'muted' | 'up' | 'down' | 'warn' | 'accent'; children: ReactNode
}) {
  const tones = {
    muted:  'bg-raised text-muted border-line',
    up:     'bg-up/10 text-up border-up/25',
    down:   'bg-down/10 text-down border-down/25',
    warn:   'bg-warn/10 text-warn border-warn/25',
    accent: 'bg-accent/10 text-accent border-accent/25',
  }[tone]
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded border text-[10px] uppercase tracking-wider ${tones}`}>
      {children}
    </span>
  )
}
