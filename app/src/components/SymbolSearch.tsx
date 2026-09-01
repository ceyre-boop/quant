import { useEffect, useRef, useState } from 'react'
import { fetchStatic } from '../lib/api'

/**
 * Resolves a ticker with zero network, from the committed SEC ticker->CIK map.
 *
 * This matters more than it looks: www.sec.gov/files/company_tickers.json is NOT
 * CORS-open, so a browser cannot fetch it live. Baking it into data/fundamentals/
 * is what makes on-demand lookup of an arbitrary ticker possible at all on a
 * static host.
 */
type CikMap = Record<string, number>

export default function SymbolSearch({
  value, onChange,
}: { value: string; onChange: (t: string) => void }) {
  const [map, setMap] = useState<CikMap | null>(null)
  const [q, setQ] = useState('')
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchStatic<CikMap>('fundamentals/cik_map.json')
      .then(setMap)
      .catch(() => setMap({}))   // absent map degrades to free text, not a crash
  }, [])

  useEffect(() => {
    const away = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', away)
    return () => document.removeEventListener('mousedown', away)
  }, [])

  const hits = (() => {
    if (!map || q.length < 1) return []
    const up = q.toUpperCase()
    const out: string[] = []
    for (const t of Object.keys(map)) {
      if (t.startsWith(up)) { out.push(t); if (out.length >= 8) break }
    }
    return out
  })()

  const commit = (t: string) => {
    const up = t.trim().toUpperCase()
    if (!up) return
    onChange(up); setQ(''); setOpen(false)
  }

  return (
    <div ref={box} className="relative">
      <div className="flex items-center gap-2">
        <span className="text-[11px] text-faint uppercase tracking-wider">Ticker</span>
        <input
          value={q}
          placeholder={value}
          onChange={e => { setQ(e.target.value); setOpen(true) }}
          onKeyDown={e => {
            if (e.key === 'Enter') commit(hits[0] ?? q)
            if (e.key === 'Escape') { setQ(''); setOpen(false) }
          }}
          className="num w-28 bg-raised border border-line rounded px-2 py-1 text-[12px]
                     text-ink placeholder:text-faint uppercase
                     focus:outline-none focus:border-accent/60"
        />
      </div>

      {open && hits.length > 0 && (
        <ul className="absolute right-0 top-full mt-1 w-56 z-50 bg-raised border border-line
                       rounded-md shadow-xl overflow-hidden">
          {hits.map(t => (
            <li key={t}>
              <button
                onMouseDown={() => commit(t)}
                className="w-full text-left px-3 py-1.5 text-[12px] num text-muted
                           hover:bg-surface hover:text-ink"
              >{t}</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
