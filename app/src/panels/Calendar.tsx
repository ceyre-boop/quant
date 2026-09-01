import { useCallback, useEffect, useState } from 'react'
import { apiPath, staticPath, fetchJSON, IS_REMOTE } from '../lib/api'
import { Panel as Card, Empty, Loading } from '../components/ui'

/**
 * Day-P&L calendar. Ported from index.html:2027-2116, keeping all three data
 * paths: live /calendar, the ICARUS shadow merge, and the committed snapshot
 * fallback for when the backend is asleep.
 *
 * Clicking a day opens the replay cockpit on that date. In the old dashboard
 * that was a .click() on the nav plus two nested setTimeouts racing the DOM;
 * here the parent just gets the date.
 */
type DayInfo = { closed?: number; pnl: number; n: number; wins?: number; src?: string }
type CalData = { month: string; days: Record<string, DayInfo>; month_total?: { pnl: number; n: number }; error?: string }

const ym = (d: Date) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
const key = (d: Date) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`

/** Fold ICARUS shadow sim results in on a $100k reference basis. */
async function mergeIcarus(d: CalData): Promise<CalData> {
  try {
    const ic = await fetchJSON<any>(`${staticPath('icarus_status.json')}?t=${Date.now()}`, 6000)
    d.days ||= {}
    const mt = (d.month_total ||= { pnl: 0, n: 0 })
    for (const e of ic?.shadow?.daily ?? []) {
      if (!String(e.date).startsWith(d.month)) continue
      const pnl = (e.ret || 0) * 100_000
      const n = e.n || 0
      const wins = (e.trades ?? []).filter((t: any) => t.ret > 0).length
      const cur = d.days[e.date]
      if (cur) {
        cur.pnl += pnl; cur.n += n
        cur.wins = (cur.wins || 0) + wins
        cur.closed = (cur.closed || 0) + n
      } else {
        d.days[e.date] = { closed: n, pnl, n, wins, src: 'icarus' }
      }
      mt.pnl += pnl; mt.n += n
    }
  } catch { /* shadow data is optional enrichment, never a hard failure */ }
  return d
}

export default function CalendarPanel({ onOpenReplay }: { onOpenReplay: (day: string) => void }) {
  const [month, setMonth] = useState(() => localStorage.getItem('sq_calmonth') || ym(new Date()))
  const [data, setData] = useState<CalData | null>(null)
  const [status, setStatus] = useState<'loading' | 'live' | 'snapshot' | 'waking' | 'down'>('loading')

  const load = useCallback(async (m: string, retried = false) => {
    setStatus('loading')
    try {
      const d = await fetchJSON<CalData>(apiPath(`calendar?month=${m}`))
      if (d.error) throw new Error(d.error)
      setData(await mergeIcarus(d)); setStatus('live'); return
    } catch {
      if (IS_REMOTE && !retried) {
        setStatus('waking')
        setTimeout(() => load(m, true), 9000)
        return
      }
      try {
        const snap = await fetchJSON<any>(`${staticPath('agent/calendar_snapshot.json')}?t=${Date.now()}`)
        const sd: CalData = { month: m, days: snap.days ?? {}, month_total: snap.month_total ?? { pnl: 0, n: 0 } }
        setData(await mergeIcarus(sd)); setStatus('snapshot'); return
      } catch { /* fall through */ }
      setData(null); setStatus('down')
    }
  }, [])

  useEffect(() => { localStorage.setItem('sq_calmonth', month); load(month) }, [month, load])

  const nav = (delta: number) => {
    const [y, m] = month.split('-').map(Number)
    setMonth(ym(new Date(y, m - 1 + delta, 1)))
  }

  const [y, m] = month.split('-').map(Number)
  const title = new Date(y, m - 1, 1).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
  const mt = data?.month_total ?? { pnl: 0, n: 0 }

  // Build the 6x7 grid starting from the Sunday on or before the 1st.
  const weeks: (Date | null)[][] = []
  const last = new Date(y, m, 0)
  const cur = new Date(y, m - 1, 1)
  cur.setDate(cur.getDate() - cur.getDay())
  for (let w = 0; w < 6 && cur <= last; w++) {
    const row: (Date | null)[] = []
    for (let i = 0; i < 7; i++) {
      row.push(cur.getMonth() === m - 1 ? new Date(cur) : null)
      cur.setDate(cur.getDate() + 1)
    }
    weeks.push(row)
  }

  return (
    <Card
      className="h-full"
      title={
        <div className="flex items-center gap-2">
          <button onClick={() => nav(-1)} className="px-1.5 text-muted hover:text-ink">‹</button>
          <span className="text-ink normal-case tracking-normal text-[12px]">{title}</span>
          <button onClick={() => nav(1)} className="px-1.5 text-muted hover:text-ink">›</button>
        </div>
      }
      right={
        <div className="flex items-center gap-3 text-[11px]">
          {status === 'snapshot' && <span className="text-warn text-[10px]">snapshot — no live calendar</span>}
          <span className="text-muted">
            Month <span className={`num ${mt.pnl >= 0 ? 'text-up' : 'text-down'}`}>
              {mt.pnl >= 0 ? '+' : ''}${Math.round(mt.pnl || 0).toLocaleString()}
            </span> · {mt.n || 0} trades
          </span>
        </div>
      }
    >
      {status === 'loading' && <Loading />}
      {status === 'waking' && <Loading label="waking the live server (~30s)" />}
      {status === 'down' && (
        <Empty
          reason={IS_REMOTE ? 'Live server unavailable.' : 'Backend offline.'}
          hint={IS_REMOTE ? 'Try again in a moment.' : 'The calendar needs: python3 scripts/live_signals_server.py'}
        />
      )}
      {data && (status === 'live' || status === 'snapshot') && (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[11px]">
            <thead>
              <tr className="text-faint uppercase tracking-wider text-[10px]">
                {['Sun','Mon','Tue','Wed','Thu','Fri','Sat','Total'].map(d => (
                  <th key={d} className="font-medium py-1 px-1 text-left">{d}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {weeks.map((row, wi) => {
                let wkPnl = 0, wkN = 0
                const cells = row.map((d, i) => {
                  if (!d) return <td key={i} className="border border-line-soft bg-bg/40 h-16" />
                  const k = key(d)
                  const info = data.days?.[k]
                  const traded = info && (info.closed ?? 0) > 0
                  if (traded) { wkPnl += info!.pnl; wkN += info!.n }
                  const wr = traded && info!.closed
                    ? `${Math.round((info!.wins ?? 0) / info!.closed! * 100)}%` : '—'
                  return (
                    <td
                      key={i}
                      onClick={() => onOpenReplay(k)}
                      title={`Replay ${k}`}
                      className={`relative border border-line-soft h-16 align-top p-1 cursor-pointer
                        transition-colors hover:border-accent/50
                        ${traded ? (info!.pnl > 0 ? 'bg-up/8' : info!.pnl < 0 ? 'bg-down/8' : '') : ''}`}
                    >
                      {info?.src === 'icarus' && (
                        <span className="absolute top-1 right-1 w-1 h-1 rounded-full bg-accent" title="icarus" />
                      )}
                      <div className="num text-faint text-[10px]">{d.getDate()}</div>
                      {traded ? (
                        <>
                          <div className={`num text-[11px] ${info!.pnl >= 0 ? 'text-up' : 'text-down'}`}>
                            {info!.pnl >= 0 ? '+' : ''}${Math.round(info!.pnl).toLocaleString()}
                          </div>
                          <div className="text-[9px] text-faint num">{info!.n} tr · {wr}</div>
                        </>
                      ) : <div className="text-faint text-[10px] mt-2">·</div>}
                    </td>
                  )
                })
                return (
                  <tr key={wi}>
                    {cells}
                    <td className="border border-line-soft h-16 align-top p-1 bg-raised/40">
                      <div className="text-[9px] uppercase tracking-wider text-faint">Week {wi + 1}</div>
                      <div className={`num text-[11px] ${wkPnl >= 0 ? 'text-up' : 'text-down'}`}>
                        {wkN ? `${wkPnl >= 0 ? '+' : ''}$${Math.round(wkPnl).toLocaleString()}` : '—'}
                      </div>
                      <div className="text-[9px] text-faint num">{wkN} tr</div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}
