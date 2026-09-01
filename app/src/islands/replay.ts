/**
 * Replay cockpit — ported from index.html:1871-2026.
 *
 * Deliberately framework-free. The original is timer-driven, mutates trade
 * objects as it walks bars, and accumulates chart markers; React would add
 * nothing and the risk of changing its behaviour while "modernising" it is
 * real. So it stays imperative, owns its own DOM subtree, and React just mounts
 * and unmounts it. The stepping algorithm, the 60s entry/exit match window, the
 * equity spark and the audio blip are all unchanged.
 */
import {
  createChart, CrosshairMode,
  type IChartApi, type ISeriesApi, type SeriesMarker, type Time,
} from 'lightweight-charts'

export type Bar = { t: number; o: number; h: number; l: number; c: number }
export type Trade = {
  direction: 'LONG' | 'SHORT'
  entry: number; exit: number
  entry_ts: string; exit_ts: string
  net_usd: number; setup?: string
  _entered?: boolean; _exited?: boolean
}
export type ReplayData = {
  day: string; bars: Bar[]; trades: Trade[]
  bias?: string; available_days?: string[]
  summary?: { net_usd?: number; n_trades?: number; max_drawdown_usd?: number }
  error?: string
}

const toSec = (ms: number) => Math.floor(ms / 1000) as unknown as Time

/** Backend errors arrive as full tracebacks; show the part that means something. */
const firstLine = (s: string) => {
  const lines = String(s).trim().split('\n').filter(Boolean)
  return (lines[lines.length - 1] || String(s)).slice(0, 160)
}

export type ReplayHandle = {
  load: (symbol: string, day?: string) => Promise<ReplayData | null>
  play: (durationMin: number) => void
  pause: () => void
  restart: (durationMin: number) => void
  setMuted: (m: boolean) => void
  destroy: () => void
}

export type ReplayCallbacks = {
  onStatus: (s: string) => void
  onClock: (s: string) => void
  onRunningPnl: (v: number) => void
  onOrder: (t: Trade, kind: 'ENTRY' | 'EXIT') => void
  onOrdersReset: () => void
  onComplete: (d: ReplayData) => void
  fetchReplay: (symbol: string, day?: string) => Promise<ReplayData>
}

export function mountReplay(
  chartEl: HTMLElement,
  sparkEl: HTMLCanvasElement | null,
  cb: ReplayCallbacks,
): ReplayHandle {
  const chart: IChartApi = createChart(chartEl, {
    layout: { background: { color: '#0a0c10' }, textColor: '#8b97ad' },
    grid: { vertLines: { color: '#171b23' }, horzLines: { color: '#171b23' } },
    rightPriceScale: { borderColor: '#232833' },
    timeScale: { borderColor: '#232833', timeVisible: true, secondsVisible: false },
    crosshair: { mode: CrosshairMode.Normal },
  })
  const candle: ISeriesApi<'Candlestick'> = chart.addCandlestickSeries({
    upColor: '#2ecc9a', downColor: '#ff5f6b',
    wickUpColor: '#2ecc9a', wickDownColor: '#ff5f6b',
    borderVisible: false,
  })

  const ro = new ResizeObserver(() =>
    chart.applyOptions({ width: chartEl.offsetWidth, height: chartEl.offsetHeight }))
  ro.observe(chartEl)

  let data: ReplayData | null = null
  let timer: number | null = null
  let muted = false
  let run = 0
  let equity: number[] = []
  let audio: AudioContext | null = null

  /** A short blip on each exit — pitch carries win/loss. */
  function blip(win: boolean) {
    if (muted) return
    try {
      audio ||= new AudioContext()
      if (audio.state === 'suspended') void audio.resume()
      const o = audio.createOscillator(), g = audio.createGain()
      o.type = 'sine'
      o.frequency.value = win ? 680 : 430
      g.gain.setValueAtTime(0.07, audio.currentTime)
      g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.11)
      o.connect(g); g.connect(audio.destination)
      o.start(); o.stop(audio.currentTime + 0.12)
    } catch { /* autoplay policy; silence is fine */ }
  }

  function drawSpark() {
    if (!sparkEl?.getContext) return
    const ctx = sparkEl.getContext('2d')!
    ctx.clearRect(0, 0, sparkEl.width, sparkEl.height)
    if (!equity.length) return
    const vals = [0, ...equity]
    const mn = Math.min(...vals, 0), mx = Math.max(...vals, 0)
    const rng = (mx - mn) || 1
    const zy = sparkEl.height - ((0 - mn) / rng) * (sparkEl.height - 4) - 2
    ctx.strokeStyle = 'rgba(255,255,255,.10)'; ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(0, zy); ctx.lineTo(sparkEl.width, zy); ctx.stroke()
    const last = equity[equity.length - 1]
    ctx.strokeStyle = last >= 0 ? '#2ecc9a' : '#ff5f6b'; ctx.lineWidth = 1.5
    ctx.beginPath()
    vals.forEach((v, i) => {
      const x = (i / (vals.length - 1 || 1)) * sparkEl.width
      const y = sparkEl.height - ((v - mn) / rng) * (sparkEl.height - 4) - 2
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
    })
    ctx.stroke()
  }

  function resetRun() { run = 0; equity = []; cb.onRunningPnl(0); drawSpark() }
  function bookExit(t: Trade) {
    run += t.net_usd || 0
    equity.push(run)
    cb.onRunningPnl(run)
    drawSpark()
    blip((t.net_usd || 0) >= 0)
  }

  function renderFull() {
    if (!data?.bars) return
    candle.setData(data.bars.map(b => ({ time: toSec(b.t), open: b.o, high: b.h, low: b.l, close: b.c })))
    candle.setMarkers([])
    chart.timeScale().fitContent()
  }

  function pause() { if (timer != null) { clearInterval(timer); timer = null } }

  async function load(symbol: string, day?: string): Promise<ReplayData | null> {
    pause()
    cb.onStatus(`loading ${symbol}…`)
    try {
      const d = await cb.fetchReplay(symbol, day)
      data = d
      if (d.error) {
        // The backend answered; it just has no session to replay (commonly the
        // futures bar feed needs IB and IB is not connected). Say that, rather
        // than blaming the connection.
        cb.onStatus(`no data for ${symbol}: ${firstLine(d.error)}`)
        return d
      }
      renderFull()
      cb.onStatus(`${d.day} · ${d.bars.length} bars · ${d.trades.length} trades · ready`)
      cb.onOrdersReset()
      return d
    } catch (e) {
      // Distinguish "cannot reach the server" from "the server said no". The
      // first is a connection problem; the second is a data problem, and
      // telling the user to start a server that is already running is worse
      // than saying nothing.
      const msg = e instanceof Error ? e.message : String(e)
      cb.onStatus(
        /Failed to fetch|NetworkError|Timeout|abort/i.test(msg)
          ? 'cannot reach the backend — start it with: python3 scripts/live_signals_server.py'
          : `replay unavailable: ${firstLine(msg)}`,
      )
      return null
    }
  }

  function play(durationMin: number) {
    if (!data?.bars?.length) return
    pause()
    const bars = data.bars
    const trades = data.trades ?? []
    // Fresh run: clear the per-trade fired flags the walk sets as it goes.
    trades.forEach(t => { delete t._entered; delete t._exited })

    const durMin = Math.max(3, durationMin || 3)
    const stepMs = Math.max(40, (durMin * 60 * 1000) / bars.length)
    cb.onOrdersReset()
    resetRun()
    candle.setData([]); candle.setMarkers([])

    const markers: SeriesMarker<Time>[] = []
    let i = 0
    timer = window.setInterval(() => {
      if (i >= bars.length) {
        pause()
        cb.onComplete(data!)
        cb.onStatus('✓ day complete')
        return
      }
      const b = bars[i]
      const tSec = toSec(b.t)
      candle.update({ time: tSec, open: b.o, high: b.h, low: b.l, close: b.c })
      cb.onClock(
        new Date(b.t).toLocaleTimeString('en-US', { hour12: false, timeZone: 'America/New_York' }) + ' ET',
      )

      for (const t of trades) {
        const eMs = Date.parse(t.entry_ts), xMs = Date.parse(t.exit_ts)
        // 60s window: bars are 1-minute, so a fill lands on the bar it belongs to.
        if (!t._entered && Math.abs(eMs - b.t) < 60_000) {
          t._entered = true
          markers.push({
            time: tSec,
            position: t.direction === 'LONG' ? 'belowBar' : 'aboveBar',
            color: t.direction === 'LONG' ? '#2ecc9a' : '#ff5f6b',
            shape: t.direction === 'LONG' ? 'arrowUp' : 'arrowDown',
            text: t.direction[0],
          })
          candle.setMarkers(markers.slice())
          cb.onOrder(t, 'ENTRY')
          blip(true)
        }
        if (!t._exited && Math.abs(xMs - b.t) < 60_000) {
          t._exited = true
          markers.push({
            time: tSec, position: 'inBar',
            color: t.net_usd >= 0 ? '#2ecc9a' : '#ff5f6b',
            shape: 'circle', text: '×',
          })
          candle.setMarkers(markers.slice())
          cb.onOrder(t, 'EXIT')
          bookExit(t)
        }
      }
      i++
    }, stepMs)

    cb.onStatus(`▶ replaying ${data.day} over ${durMin}m…`)
  }

  return {
    load, play, pause,
    restart: (d: number) => play(d),
    setMuted: (m: boolean) => { muted = m },
    destroy: () => { pause(); ro.disconnect(); chart.remove() },
  }
}
