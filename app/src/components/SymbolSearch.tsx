import { useEffect, useRef, useState } from 'react'
import { loadSymbols, search, KIND_LABEL, type Symbol } from '../lib/symbols'

/**
 * One box for everything the desk looks at: 10,391 US equities searchable by
 * ticker OR company name, plus the FX pairs and futures that have no issuer and
 * therefore never appear in an SEC file.
 *
 * The index loads on first keystroke, not on mount, so it costs nothing until
 * someone actually searches.
 */
export default function SymbolSearch({
  value, onChange,
}: { value: string; onChange: (t: string) => void }) {
  const [q, setQ] = useState('')
  const [hits, setHits] = useState<Symbol[]>([])
  const [open, setOpen] = useState(false)
  const [ready, setReady] = useState(false)
  const [cursor, setCursor] = useState(0)
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const away = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', away)
    return () => document.removeEventListener('mousedown', away)
  }, [])

  useEffect(() => {
    if (!q) { setHits([]); return }
    let dead = false
    loadSymbols().then(() => {
      if (dead) return
      setReady(true)
      setHits(search(q, 10))
      setCursor(0)
    })
    return () => { dead = true }
  }, [q])

  const commit = (s?: Symbol) => {
    const t = (s?.ticker ?? q).trim().toUpperCase()
    if (!t) return
    onChange(t)
    setQ(''); setHits([]); setOpen(false)
  }

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setCursor(c => Math.min(c + 1, hits.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setCursor(c => Math.max(c - 1, 0)) }
    else if (e.key === 'Enter') { e.preventDefault(); commit(hits[cursor]) }
    else if (e.key === 'Escape') { setQ(''); setOpen(false) }
  }

  return (
    <div ref={box} className="relative">
      <div className="flex items-center gap-2">
        <span className="hidden sm:inline text-[11px] text-faint uppercase tracking-wider">Symbol</span>
        <input
          value={q}
          placeholder={value}
          onChange={e => { setQ(e.target.value); setOpen(true) }}
          onFocus={() => { loadSymbols().then(() => setReady(true)); setOpen(true) }}
          onKeyDown={onKey}
          aria-label="Search ticker or company name"
          className="num w-40 sm:w-52 bg-raised border border-line rounded px-2 py-1 text-[12px]
                     text-ink placeholder:text-faint
                     focus:outline-none focus:border-accent/60"
        />
      </div>

      {open && q.length > 0 && (
        <ul className="absolute right-0 top-full mt-1 w-[22rem] max-w-[85vw] z-50 bg-raised
                       border border-line rounded-md shadow-xl overflow-hidden max-h-80 overflow-y-auto">
          {hits.length === 0 ? (
            <li className="px-3 py-2 text-[11px] text-faint">
              {ready
                ? <>No match for “{q}”. Try a ticker (NVDA) or a company name (NVIDIA).</>
                : 'loading symbols…'}
            </li>
          ) : hits.map((s, i) => (
            <li key={s.kind + s.ticker}>
              <button
                onMouseDown={() => commit(s)}
                onMouseEnter={() => setCursor(i)}
                className={`w-full flex items-center gap-2 px-3 py-1.5 text-left
                  ${i === cursor ? 'bg-surface' : ''} hover:bg-surface`}
              >
                <span className="num text-[12px] text-ink w-16 shrink-0">{s.ticker}</span>
                <span className="flex-1 truncate text-[11px] text-muted">{s.name}</span>
                {s.kind !== 'equity' && (
                  <span className="shrink-0 text-[9px] uppercase tracking-wider text-accent/80
                                   border border-accent/25 rounded px-1 py-0.5">
                    {KIND_LABEL[s.kind]}
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
