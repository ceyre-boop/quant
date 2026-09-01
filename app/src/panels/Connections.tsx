import { useEffect, useState } from 'react'
import { fetchStatic, apiPath, IS_REMOTE } from '../lib/api'
import { Panel as Card, Loading, Pill } from '../components/ui'

/**
 * What the Research tab became: key management and data-connection status only.
 *
 * The old tab carried thirteen analysis cards fed by discovery surfaces that
 * returned null. What is genuinely useful is knowing which integrations are
 * wired, which are failing, and where to put a key.
 */
type Health = {
  generated_at?: string
  components?: Record<string, { status?: string; detail?: string; note?: string }>
}

/** Ground truth from the repo, so the panel is honest about dormant vs live
 *  rather than only echoing whatever health.json last managed to write. */
const REGISTRY: { key: string; env: string[]; role: string; tier: string }[] = [
  { key: 'oanda',        env: ['OANDA_API_KEY', 'OANDA_ACCOUNT_ID'], role: 'Live FX execution + fills ledger', tier: 'live' },
  { key: 'alpaca',       env: ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY'], role: 'Equity bars + paper account', tier: 'live' },
  { key: 'yfinance',     env: [], role: 'Bars, VIX, earnings dates (keyless)', tier: 'live' },
  { key: 'fred',         env: ['FRED_API_KEY'], role: 'Macro series + first-print surprises', tier: 'live' },
  { key: 'alpha_vantage',env: ['ALPHA_VANTAGE_API_KEY'], role: 'FX news sentiment; earnings backfill', tier: 'capped' },
  { key: 'anthropic',    env: ['ANTHROPIC_API_KEY'], role: 'Oracle reflection + chat', tier: 'live' },
  { key: 'sec_edgar',    env: [], role: 'Filings, Form 4, 13F (keyless)', tier: 'live' },
  { key: 'finra',        env: [], role: 'Daily short volume (keyless)', tier: 'live' },
  { key: 'openfigi',     env: [], role: 'CUSIP → ticker mapping (keyless)', tier: 'live' },
  { key: 'databento',    env: ['DATABENTO_API_KEY'], role: 'CME ES/NQ 1-min bars', tier: 'live' },
  { key: 'ib',           env: ['IB_HOST', 'IB_ACCOUNT'], role: 'Borrow / shortable snapshots', tier: 'live' },
  { key: 'gdelt',        env: [], role: 'Article tone + volume (keyless)', tier: 'live' },
  { key: 'cftc_cot',     env: [], role: 'Weekly positioning (keyless)', tier: 'live' },
  { key: 'thetadata',    env: ['THETADATA_API_KEY'], role: 'Option chains — needs local ThetaTerminal', tier: 'conditional' },
  { key: 'news_api',     env: ['NEWS_API_KEY'], role: 'Headlines — largely superseded by GDELT', tier: 'degraded' },
  { key: 'reddit',       env: [], role: 'Subreddit sentiment — plist never installed', tier: 'degraded' },
  { key: 'polygon',      env: ['POLYGON_API_KEY'], role: 'Free tier; live calls fall through to yfinance', tier: 'dormant' },
  { key: 'tiingo',       env: ['TIINGO_API_KEY'], role: 'Health ping only, no consumer', tier: 'dormant' },
  { key: 'nasdaq_data_link', env: ['NASDAQ_DATA_LINK_API_KEY'], role: 'Health ping only; WIKI deprecated upstream', tier: 'dormant' },
  { key: 'openweather',  env: ['OPENWEATHER_API_KEY'], role: 'No trading consumer — vestigial', tier: 'dormant' },
  { key: 'firebase',     env: ['FIREBASE_API_KEY'], role: 'RTDB publish — dormant since April', tier: 'dormant' },
]

const TIER_TONE = {
  live: 'up', capped: 'warn', conditional: 'warn',
  degraded: 'warn', dormant: 'muted',
} as const

export default function Connections() {
  const [health, setHealth] = useState<Health | null>(null)
  const [loading, setLoading] = useState(true)
  const [backend, setBackend] = useState<'checking' | 'up' | 'down'>('checking')

  useEffect(() => {
    fetchStatic<Health>('agent/health.json')
      .then(setHealth).catch(() => setHealth(null)).finally(() => setLoading(false))
    fetch(apiPath('health')).then(r => setBackend(r.ok ? 'up' : 'down')).catch(() => setBackend('down'))
  }, [])

  const stale = health?.generated_at
    ? Math.floor((Date.now() - Date.parse(health.generated_at)) / 86_400_000)
    : null

  return (
    <div className="h-full overflow-y-auto grid grid-cols-1 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)] gap-2">
      <Card
        title="Data connections"
        right={
          <div className="flex items-center gap-2 text-[10px] text-faint">
            {stale != null && (
              <span className={stale > 7 ? 'text-warn' : ''}>health.json · {stale}d old</span>
            )}
          </div>
        }
      >
        {loading ? <Loading /> : (
          <>
            {stale != null && stale > 7 && (
              <p className="mb-2 text-[11px] text-warn/90 bg-warn/5 border border-warn/20 rounded px-3 py-2">
                health.json is {stale} days old. Its API components only refresh on a manual
                <code className="mx-1 text-ink">sync_dashboard_data.py</code> run.
                The status column below reflects the repo, not a live probe.
              </p>
            )}
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="text-faint uppercase tracking-wider text-[10px]">
                    <th className="text-left font-medium py-1">Integration</th>
                    <th className="text-left font-medium">Role</th>
                    <th className="text-left font-medium">Keys</th>
                    <th className="text-right font-medium">Tier</th>
                  </tr>
                </thead>
                <tbody>
                  {REGISTRY.map(r => {
                    const h = health?.components?.[r.key]
                    return (
                      <tr key={r.key} className="border-t border-line-soft align-top">
                        <td className="py-1.5 num text-ink">{r.key}</td>
                        <td className="text-muted pr-2">{r.role}</td>
                        <td className="num text-[10px] text-faint">
                          {r.env.length ? r.env.join(', ') : <span className="text-up/70">keyless</span>}
                        </td>
                        <td className="text-right">
                          <Pill tone={TIER_TONE[r.tier as keyof typeof TIER_TONE]}>{r.tier}</Pill>
                          {h?.status && <span className="ml-1 text-[9px] text-faint">{h.status}</span>}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Card>

      <div className="flex flex-col gap-2">
        <Card title="Backend">
          <div className="space-y-2 text-[11px]">
            <div className="flex items-center justify-between">
              <span className="text-muted">{IS_REMOTE ? 'Render (free tier)' : 'localhost:8765'}</span>
              <Pill tone={backend === 'up' ? 'up' : backend === 'down' ? 'down' : 'muted'}>
                {backend === 'checking' ? '…' : backend}
              </Pill>
            </div>
            {backend === 'down' && (
              <p className="text-faint leading-relaxed">
                {IS_REMOTE
                  ? 'Free tier sleeps after ~15 minutes idle. The first request wakes it and takes 30–50s.'
                  : 'Start it with:'}
              </p>
            )}
            {!IS_REMOTE && backend === 'down' && (
              <code className="block bg-raised border border-line rounded px-2 py-1 text-[10px] text-accent">
                python3 scripts/live_signals_server.py
              </code>
            )}
          </div>
        </Card>

        <Card title="Adding a key">
          <div className="space-y-2 text-[11px] text-muted leading-relaxed">
            <p>
              Keys live in <code className="text-ink">.env</code> at the repo root, never in the
              browser bundle. Add the variable, then restart the backend.
            </p>
            <p className="text-faint">
              To close the forward-guidance and consensus-estimate gap in the fundamentals
              panel, add an FMP or Finnhub key and set
              <code className="mx-1 text-ink">FUNDAMENTALS_PRIMARY</code>.
              Everything downstream is unchanged.
            </p>
          </div>
        </Card>
      </div>
    </div>
  )
}
