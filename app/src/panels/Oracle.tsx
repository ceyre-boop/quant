import { useEffect, useRef, useState } from 'react'
import { apiPath, IS_REMOTE } from '../lib/api'
import { Panel as Card, Loading } from '../components/ui'

/** The in-app assistant. Ported from index.html:3474-3573 — POST /chat, which is
 *  backed by ANTHROPIC_API_KEY as a Render env secret, never a browser key. */
type Msg = { role: 'user' | 'assistant'; text: string }

const CHIPS = [
  'What changed in the last session?',
  'Summarise the current forex signals',
  'Why is this ticker moving?',
]

export default function Oracle() {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const end = useRef<HTMLDivElement>(null)

  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs, busy])

  async function send(text: string) {
    const t = text.trim()
    if (!t || busy) return
    setMsgs(m => [...m, { role: 'user', text: t }])
    setInput(''); setBusy(true)
    try {
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), IS_REMOTE ? 45_000 : 20_000)
      const r = await fetch(apiPath('chat'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: t, context: '' }),
        signal: ctrl.signal,
      })
      clearTimeout(timer)
      const j = await r.json()
      setMsgs(m => [...m, { role: 'assistant', text: j.reply ?? j.response ?? JSON.stringify(j) }])
    } catch (e) {
      setMsgs(m => [...m, {
        role: 'assistant',
        text: IS_REMOTE
          ? 'The backend did not answer in time. On the free tier it sleeps after ~15 minutes; try again and it should be awake.'
          : 'No local backend. Start it with: python3 scripts/live_signals_server.py',
      }])
    } finally { setBusy(false) }
  }

  return (
    <Card title="Oracle" className="h-full">
      <div className="flex flex-col h-full min-h-0">
        <div className="flex-1 min-h-0 overflow-y-auto space-y-2 pr-1">
          {msgs.length === 0 && (
            <div className="flex flex-wrap gap-1.5 pt-1">
              {CHIPS.map(c => (
                <button key={c} onClick={() => send(c)}
                  className="px-2 py-1 rounded border border-line text-[11px] text-muted
                             hover:text-ink hover:border-accent/40 transition-colors">
                  {c}
                </button>
              ))}
            </div>
          )}
          {msgs.map((m, i) => (
            <div key={i} className={m.role === 'user' ? 'text-right' : ''}>
              <div className={`inline-block max-w-[85%] text-left px-3 py-2 rounded-md text-[12px] leading-relaxed whitespace-pre-wrap
                ${m.role === 'user' ? 'bg-accent/10 text-ink' : 'bg-raised text-muted'}`}>
                {m.text}
              </div>
            </div>
          ))}
          {busy && <Loading label="thinking" />}
          <div ref={end} />
        </div>

        <form
          onSubmit={e => { e.preventDefault(); send(input) }}
          className="flex gap-2 pt-2 border-t border-line-soft shrink-0"
        >
          <input
            value={input} onChange={e => setInput(e.target.value)}
            placeholder="Ask the Oracle…"
            className="flex-1 bg-raised border border-line rounded px-3 py-1.5 text-[12px]
                       placeholder:text-faint focus:outline-none focus:border-accent/60"
          />
          <button type="submit" disabled={busy || !input.trim()}
            className="px-3 py-1.5 rounded bg-accent/15 text-accent text-[12px]
                       disabled:opacity-30 hover:bg-accent/25 transition-colors">
            Send
          </button>
        </form>
      </div>
    </Card>
  )
}
