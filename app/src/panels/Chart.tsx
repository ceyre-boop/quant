import { useEffect, useRef, useState } from 'react'

/**
 * TradingView embed — ported verbatim from index.html:1836-1860.
 *
 * Deliberately NOT rebuilt. The indicators live on the user's real TradingView
 * account; this widget is the read-only view and the link-out hands off to the
 * logged-in chart where those indicators actually are.
 *
 * `proxy` is what the free widget can render; `fut` is the real contract the
 * link-out opens. That two-symbol mapping is the only state this ever had.
 */
declare global { interface Window { TradingView?: any } }

type Sym = { label: string; proxy: string; fut: string }

const SYMBOLS: Sym[] = [
  { label: 'NQ',  proxy: 'NASDAQ:QQQ',   fut: 'CME_MINI:NQ1!' },
  { label: 'ES',  proxy: 'AMEX:SPY',     fut: 'CME_MINI:ES1!' },
  { label: 'YM',  proxy: 'AMEX:DIA',     fut: 'CBOT_MINI:YM1!' },
  { label: 'RTY', proxy: 'AMEX:IWM',     fut: 'CME_MINI:RTY1!' },
  { label: 'CL',  proxy: 'AMEX:USO',     fut: 'NYMEX:CL1!' },
  { label: 'GC',  proxy: 'AMEX:GLD',     fut: 'COMEX:GC1!' },
]

const LS_KEY = 'sq_tv_symbol'

export default function Chart({ symbol }: { symbol?: string }) {
  const box = useRef<HTMLDivElement>(null)
  const [active, setActive] = useState<Sym>(() => {
    const saved = localStorage.getItem(LS_KEY)
    return SYMBOLS.find(s => s.label === saved) ?? SYMBOLS[0]
  })

  // An explicitly selected ticker (from the fundamentals search) overrides the
  // futures buttons — same symbol drives chart and fundamentals.
  const shown = symbol ? symbol.toUpperCase() : active.proxy

  useEffect(() => {
    let cancelled = false
    const mount = () => {
      if (cancelled) return
      if (!window.TradingView) { setTimeout(mount, 300); return }   // tv.js still loading
      const el = box.current
      if (!el) return
      el.innerHTML = ''
      new window.TradingView.widget({
        container_id: el.id, autosize: true, symbol: shown, interval: '5',
        timezone: 'America/New_York', theme: 'dark', style: '1', locale: 'en',
        toolbar_bg: '#11141a', enable_publishing: false, allow_symbol_change: true,
        withdateranges: true, details: true, hide_side_toolbar: false,
      })
    }
    mount()
    return () => { cancelled = true }
  }, [shown])

  const openReal = () => {
    const real = symbol ? symbol.toUpperCase() : active.fut
    window.open(`https://www.tradingview.com/chart/?symbol=${encodeURIComponent(real)}`, '_blank')
  }

  const pick = (s: Sym) => { setActive(s); localStorage.setItem(LS_KEY, s.label) }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-line-soft shrink-0">
        {SYMBOLS.map(s => (
          <button
            key={s.label}
            onClick={() => pick(s)}
            disabled={!!symbol}
            className={`px-2 py-1 rounded text-[11px] num transition-colors disabled:opacity-30
              ${!symbol && s.label === active.label
                ? 'bg-accent/15 text-accent'
                : 'text-muted hover:text-ink hover:bg-raised'}`}
          >{s.label}</button>
        ))}
        {symbol && <span className="px-2 py-1 rounded text-[11px] num bg-accent/15 text-accent">{symbol.toUpperCase()}</span>}
        <div className="flex-1" />
        <button
          onClick={openReal}
          title="Open in your real TradingView account, with your indicators"
          className="px-2 py-1 rounded text-[11px] text-muted hover:text-accent hover:bg-raised transition-colors"
        >↗ Open on TradingView</button>
      </div>
      <div id="tv_chart" ref={box} className="flex-1 min-h-0" />
    </div>
  )
}
