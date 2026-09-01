import { useEffect, useRef, useState } from 'react'
import { apiPath, fetchJSON, IS_REMOTE } from '../lib/api'
import { mountReplay, type ReplayHandle, type ReplayData, type Trade } from '../islands/replay'
import { Panel as Card, Empty, Loading, Pill } from '../components/ui'
import { usd, num } from '../lib/format'
import Chart from './Chart'

const SYMBOLS = ['MNQ', 'MES', 'NQ', 'ES']

/** Live signal status, plus the replay cockpit. The cockpit is an imperative
 *  island (src/islands/replay.ts) — React mounts it and feeds it callbacks. */
export default function Signals({ replayDay }: { replayDay: string | null }) {
  const [mode, setMode] = useState<'live' | 'replay'>(replayDay ? 'replay' : 'live')
  useEffect(() => { if (replayDay) setMode('replay') }, [replayDay])

  return (
    <div className="h-full flex flex-col gap-2 min-h-0">
      <div className="flex items-center gap-1 shrink-0">
        {(['live', 'replay'] as const).map(m => (
          <button key={m} onClick={() => setMode(m)}
            className={`px-3 py-1 rounded text-[11px] uppercase tracking-wider transition-colors ${
              mode === m ? 'bg-accent/15 text-accent' : 'text-muted hover:text-ink hover:bg-raised'}`}>
            {m === 'live' ? 'Live signals' : 'Replay cockpit'}
          </button>
        ))}
      </div>
      {mode === 'live' ? <LiveSignals /> : <ReplayCockpit initialDay={replayDay} />}
    </div>
  )
}

/* ── Live signal status ────────────────────────────────────────────────────── */
/**
 * Shape confirmed against the live GET /data payload, not assumed:
 *   { ticker: "EURUSD=X", label: "EUR/USD", price: 1.16,
 *     signal: 0, conviction: 0.0, size_mult: 1.0, price_history: [...] }
 * `signal` is NUMERIC (-1 short / 0 flat / +1 long). An earlier version treated
 * it as a string and called .toUpperCase() on it, which threw and — with no
 * boundary — unmounted the whole terminal.
 */
type SignalRow = {
  ticker?: string; label?: string
  price?: number; signal?: number | string
  conviction?: number; size_mult?: number
}

function direction(sig: number | string | undefined): 'LONG' | 'SHORT' | 'FLAT' {
  if (typeof sig === 'number') return sig > 0 ? 'LONG' : sig < 0 ? 'SHORT' : 'FLAT'
  const s = String(sig ?? '').toUpperCase()
  if (s.includes('LONG') || s.includes('BUY')) return 'LONG'
  if (s.includes('SHORT') || s.includes('SELL')) return 'SHORT'
  return 'FLAT'
}

function LiveSignals() {
  const [rows, setRows] = useState<SignalRow[] | null>(null)
  const [state, setState] = useState<'loading' | 'ok' | 'down'>('loading')

  useEffect(() => {
    let dead = false
    fetchJSON<any>(apiPath('data'))
      .then(d => {
        if (dead) return
        const list = d.forex_signals ?? d.signals ?? d.equity_signals ?? []
        setRows(Array.isArray(list) ? list : Object.values(list))
        setState('ok')
      })
      .catch(() => { if (!dead) setState('down') })
    return () => { dead = true }
  }, [])

  if (state === 'loading') return <Card className="flex-1"><Loading /></Card>
  if (state === 'down') {
    return (
      <Card className="flex-1">
        <Empty
          reason={IS_REMOTE ? 'Live server unavailable.' : 'Backend offline.'}
          hint={IS_REMOTE
            ? 'Free tier sleeps after ~15 minutes idle; the first request wakes it (30-50s).'
            : 'Start it with: python3 scripts/live_signals_server.py'}
        />
      </Card>
    )
  }

  return (
    <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px] gap-2">
      <div className="bg-surface border border-line rounded-md overflow-hidden min-h-[300px]">
        <Chart />
      </div>
      <Card title="Signals" right={<Pill tone="up">live</Pill>} className="min-h-0">
        {!rows?.length ? <Empty reason="No active signals." /> : (
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-faint uppercase tracking-wider text-[10px]">
                <th className="text-left font-medium py-1">Pair</th>
                <th className="text-left font-medium">Signal</th>
                <th className="text-right font-medium">Conv</th>
                <th className="text-right font-medium">Size</th>
                <th className="text-right font-medium">Price</th>
              </tr>
            </thead>
            <tbody className="num">
              {rows.map((r, i) => {
                const dir = direction(r.signal)
                const tone = dir === 'LONG' ? 'text-up' : dir === 'SHORT' ? 'text-down' : 'text-faint'
                return (
                  <tr key={r.ticker ?? i} className="border-t border-line-soft">
                    <td className="py-1">{r.label ?? r.ticker ?? '—'}</td>
                    <td className={tone}>{dir}</td>
                    <td className="text-right text-muted">{num(r.conviction)}</td>
                    <td className="text-right text-muted">{num(r.size_mult, 1)}x</td>
                    <td className="text-right">{num(r.price, 4)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}

/* ── Replay cockpit ────────────────────────────────────────────────────────── */
function ReplayCockpit({ initialDay }: { initialDay: string | null }) {
  const chartBox = useRef<HTMLDivElement>(null)
  const spark = useRef<HTMLCanvasElement>(null)
  const handle = useRef<ReplayHandle | null>(null)

  const [symbol, setSymbol] = useState(() => localStorage.getItem('sq_rpsym') || 'MNQ')
  const [days, setDays] = useState<string[]>([])
  const [day, setDay] = useState<string>(initialDay ?? '')
  const [dur, setDur] = useState(() => Number(localStorage.getItem('sq_rpdur')) || 3)
  const [muted, setMuted] = useState(() => localStorage.getItem('sq_rpmute') === '1')

  const [status, setStatus] = useState('initialising…')
  const [clock, setClock] = useState('')
  const [pnl, setPnl] = useState(0)
  const [orders, setOrders] = useState<{ t: Trade; kind: 'ENTRY' | 'EXIT' }[]>([])
  const [summary, setSummary] = useState<ReplayData | null>(null)
  const [bias, setBias] = useState('')

  // Mount the island once. It owns its DOM subtree from here on.
  useEffect(() => {
    if (!chartBox.current) return
    handle.current = mountReplay(chartBox.current, spark.current, {
      onStatus: setStatus,
      onClock: setClock,
      onRunningPnl: setPnl,
      onOrder: (t, kind) => setOrders(o => [...o, { t, kind }]),
      onOrdersReset: () => { setOrders([]); setSummary(null) },
      onComplete: d => setSummary(d),
      // The replay route answers 500 with a JSON traceback body when the bar
      // feed has nothing (e.g. IB not connected). fetchJSON would throw away
      // that explanation, so read the body directly and hand the island a
      // structured error it can display honestly.
      fetchReplay: async (sym, d) => {
        const r = await fetch(apiPath(`replay?symbol=${sym}${d ? `&date=${d}` : ''}`))
        const body = await r.json().catch(() => null)
        if (!r.ok) return { day: '', bars: [], trades: [], error: body?.error ?? `HTTP ${r.status}` }
        return body as ReplayData
      },
    })
    return () => { handle.current?.destroy(); handle.current = null }
  }, [])

  useEffect(() => { handle.current?.setMuted(muted); localStorage.setItem('sq_rpmute', muted ? '1' : '0') }, [muted])
  useEffect(() => { localStorage.setItem('sq_rpsym', symbol) }, [symbol])
  useEffect(() => { localStorage.setItem('sq_rpdur', String(dur)) }, [dur])

  // Load whenever symbol or day changes — including the day the calendar hands us.
  useEffect(() => {
    let dead = false
    handle.current?.load(symbol, day || undefined).then(d => {
      if (dead || !d) return
      setBias(d.bias ?? '')
      if (d.available_days?.length) {
        setDays(d.available_days.slice().reverse())
        if (!day) setDay(d.day)
      }
    })
    return () => { dead = true }
  }, [symbol, day])

  const net = summary?.summary?.net_usd ?? 0

  return (
    <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px] gap-2">
      <Card
        className="min-h-[340px]"
        dense
        title={
          <div className="flex items-center gap-1">
            {SYMBOLS.map(s => (
              <button key={s} onClick={() => setSymbol(s)}
                className={`px-2 py-0.5 rounded text-[11px] num ${
                  s === symbol ? 'bg-accent/15 text-accent' : 'text-muted hover:text-ink'}`}>{s}</button>
            ))}
          </div>
        }
        right={
          <div className="flex items-center gap-2">
            <select value={day} onChange={e => setDay(e.target.value)}
              className="num bg-raised border border-line rounded px-1.5 py-0.5 text-[11px] text-muted">
              {days.length === 0 && <option value="">—</option>}
              {days.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
            {bias && <Pill tone="muted">bias {bias}</Pill>}
          </div>
        }
      >
        <div className="flex flex-col h-full min-h-0">
          <div ref={chartBox} className="flex-1 min-h-0" />
          <div className="flex items-center gap-2 px-2 py-1.5 border-t border-line-soft shrink-0">
            <button onClick={() => handle.current?.play(dur)}
              className="px-2.5 py-1 rounded bg-up/15 text-up text-[11px]">▶ Play day</button>
            <button onClick={() => handle.current?.pause()}
              className="px-2.5 py-1 rounded bg-raised text-muted text-[11px] hover:text-ink">⏸</button>
            <button onClick={() => handle.current?.restart(dur)}
              className="px-2.5 py-1 rounded bg-raised text-muted text-[11px] hover:text-ink">↻</button>
            <button onClick={() => setMuted(m => !m)}
              className="px-2 py-1 rounded bg-raised text-[11px]">{muted ? '🔇' : '🔊'}</button>
            <label className="flex items-center gap-1.5 text-[10px] text-faint ml-1">
              over
              <input type="range" min={3} max={20} value={dur}
                onChange={e => setDur(Number(e.target.value))} className="w-20 accent-[#4dd4e8]" />
              <span className="num text-muted w-7">{dur}m</span>
            </label>
            <div className="flex-1" />
            <span className="num text-[11px] text-muted">{clock}</span>
          </div>
          <p className="px-2 pb-1.5 text-[10px] text-faint shrink-0">{status}</p>
        </div>
      </Card>

      <div className="flex flex-col gap-2 min-h-0">
        <Card title="Running P&L" dense>
          <div className="px-3 py-2">
            <p className={`num text-[20px] ${pnl >= 0 ? 'text-up' : 'text-down'}`}>
              {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
            </p>
            <canvas ref={spark} width={280} height={40} className="w-full mt-1" />
          </div>
        </Card>

        <Card title="Order tape" dense className="flex-1 min-h-0">
          <div className="overflow-y-auto h-full text-[11px]">
            {orders.length === 0 ? (
              <p className="px-3 py-3 text-faint">Press ▶ Play day to watch it unfold.</p>
            ) : orders.map((o, i) => (
              <div key={i}
                className={`flex items-center gap-2 px-3 py-1 border-l-2 border-b border-line-soft
                  ${o.t.direction === 'LONG' ? 'border-l-up' : 'border-l-down'}`}>
                <span className={`num text-[10px] ${o.t.direction === 'LONG' ? 'text-up' : 'text-down'}`}>
                  {o.t.direction}
                </span>
                <span className="text-muted truncate">
                  {o.kind} {o.t.setup ?? ''} @ {o.kind === 'ENTRY' ? o.t.entry : o.t.exit}
                </span>
                <div className="flex-1" />
                {o.kind === 'EXIT'
                  ? <span className={`num ${o.t.net_usd >= 0 ? 'text-up' : 'text-down'}`}>
                      {o.t.net_usd >= 0 ? '+' : ''}${(o.t.net_usd || 0).toFixed(2)}
                    </span>
                  : <span className="text-[10px] text-faint">working…</span>}
              </div>
            ))}
          </div>
        </Card>

        {summary && (
          <Card title="Day complete" dense>
            <div className="px-3 py-2">
              <p className={`num text-[22px] ${net >= 0 ? 'text-up' : 'text-down'}`}>
                {net >= 0 ? '+' : ''}${net.toFixed(2)}
              </p>
              <p className="text-[10px] text-faint num mt-0.5">
                {summary.summary?.n_trades ?? 0} trades · max DD {usd(summary.summary?.max_drawdown_usd)} · {summary.day}
              </p>
              <p className="text-[10px] text-faint mt-1">
                Exactly what the live engine would have traded.
              </p>
            </div>
          </Card>
        )}
      </div>
    </div>
  )
}
