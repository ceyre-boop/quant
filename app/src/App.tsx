import { useEffect, useState } from 'react'
import Chart from './panels/Chart'
import Fundamentals from './panels/Fundamentals'
import Signals from './panels/Signals'
import CalendarPanel from './panels/Calendar'
import Oracle from './panels/Oracle'
import Connections from './panels/Connections'
import SymbolSearch from './components/SymbolSearch'

/**
 * A research terminal, not an alerting system. Everything here is pull: you open
 * it and look. Nothing pushes, nothing recommends a trade.
 *
 * Layout is deliberately two-plane. The chart and the fundamentals for the SAME
 * ticker sit side by side, because the whole premise is that the decision signal
 * lives in earnings and filings rather than in price alone — so they have to be
 * readable together, not in separate tabs.
 */
type Tab = 'terminal' | 'signals' | 'calendar' | 'oracle' | 'connections'

const TABS: { id: Tab; label: string }[] = [
  { id: 'terminal',    label: 'Terminal' },
  { id: 'signals',     label: 'Signals' },
  { id: 'calendar',    label: 'Calendar' },
  { id: 'oracle',      label: 'Oracle' },
  { id: 'connections', label: 'Connections' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>(
    () => (localStorage.getItem('sq_tab') as Tab) || 'terminal',
  )
  const [ticker, setTicker] = useState<string>(
    () => localStorage.getItem('sq_ticker') || 'AAPL',
  )
  // Calendar day-click jumps to the replay cockpit. In the old dashboard this
  // was a .click() plus two nested setTimeouts; here it is just state.
  const [replayDay, setReplayDay] = useState<string | null>(null)

  useEffect(() => { localStorage.setItem('sq_tab', tab) }, [tab])
  useEffect(() => { localStorage.setItem('sq_ticker', ticker) }, [ticker])

  const openReplay = (day: string) => { setReplayDay(day); setTab('signals') }

  return (
    <div className="h-full flex flex-col bg-bg">
      <nav className="flex items-center gap-4 px-3 h-11 border-b border-line shrink-0">
        <div className="flex items-center gap-2 pr-2">
          <span className="w-2 h-2 rounded-sm bg-accent" />
          <span className="text-[12px] font-semibold tracking-[0.18em] uppercase">
            Sovereign <span className="text-faint">//</span> Quant
          </span>
        </div>

        <div className="flex items-center gap-0.5">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-3 py-1.5 rounded text-[12px] transition-colors ${
                tab === t.id
                  ? 'bg-raised text-ink'
                  : 'text-muted hover:text-ink hover:bg-raised/60'
              }`}
            >{t.label}</button>
          ))}
        </div>

        <div className="flex-1" />
        <SymbolSearch value={ticker} onChange={setTicker} />
      </nav>

      <main className="flex-1 min-h-0 p-2">
        {tab === 'terminal' && (
          <div className="h-full grid grid-cols-1 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] gap-2">
            <div className="bg-surface border border-line rounded-md overflow-hidden min-h-[360px]">
              <Chart symbol={ticker} />
            </div>
            <div className="min-h-0 overflow-y-auto">
              <Fundamentals ticker={ticker} />
            </div>
          </div>
        )}
        {tab === 'signals'     && <Signals replayDay={replayDay} />}
        {tab === 'calendar'    && <CalendarPanel onOpenReplay={openReplay} />}
        {tab === 'oracle'      && <Oracle />}
        {tab === 'connections' && <Connections />}
      </main>
    </div>
  )
}
