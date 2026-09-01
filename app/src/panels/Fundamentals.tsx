import { useEffect, useState } from 'react'
import { fetchStatic } from '../lib/api'
import { loadPanel, type Load, type EarningsRow } from '../lib/fundamentals'
import { Panel as Card, AsOf, Empty, Loading, Delta, Pill } from '../components/ui'
import { compact, usd, num, day } from '../lib/format'

/** The centre of gravity: what the company guided, what it printed, how the
 *  stock reacted, and who is buying or selling it. */
export default function Fundamentals({ ticker }: { ticker: string }) {
  const [load, setLoad] = useState<Load>({ state: 'loading' })
  const [cikMap, setCikMap] = useState<Record<string, number>>({})

  useEffect(() => {
    fetchStatic<Record<string, number>>('fundamentals/cik_map.json')
      .then(setCikMap).catch(() => setCikMap({}))
  }, [])

  useEffect(() => {
    let dead = false
    setLoad({ state: 'loading' })
    loadPanel(ticker, cikMap).then(r => { if (!dead) setLoad(r) })
    return () => { dead = true }
  }, [ticker, cikMap])

  if (load.state === 'loading') return <Loading label={`loading ${ticker}`} />
  if (load.state === 'waking')  return <Loading label="waking backend — up to 40s" />
  if (load.state === 'error') {
    return (
      <Card title={`${ticker} — fundamentals`}>
        <Empty reason="Could not load fundamentals." hint={load.reason} />
      </Card>
    )
  }

  const p = load.panel
  const s = p.sections

  return (
    <div className="flex flex-col gap-2">
      <header className="flex items-baseline gap-2 px-1">
        <h1 className="text-[15px] font-semibold num">{p.ticker}</h1>
        <span className="text-[12px] text-muted truncate">{p.name ?? ''}</span>
        <div className="flex-1" />
        {load.state === 'warm'    && <Pill tone="accent">warm</Pill>}
        {load.state === 'live'    && <Pill tone="up">live</Pill>}
        {load.state === 'partial' && <Pill tone="warn">partial</Pill>}
      </header>

      {load.state === 'partial' && (
        <p className="text-[11px] text-warn/90 bg-warn/5 border border-warn/20 rounded px-3 py-2">
          {load.reason}
        </p>
      )}

      <EarningsCard rows={s.earnings?.rows ?? []} meta={s.earnings} />
      <ReactionCard rows={s.earnings?.rows ?? []} />
      <InsiderCard section={s.insider} />
      <InstitutionsCard section={s.institutions} />
      <ShortCard section={s.short} />
    </div>
  )
}

/* ── 1 + 2: earnings history, estimate vs actual, surprise ─────────────────── */
function EarningsCard({ rows, meta }: { rows: EarningsRow[]; meta?: any }) {
  const past = rows.filter(r => r.eps_actual != null).slice(0, 12)
  const next = rows.find(r => r.eps_actual == null && r.eps_estimate != null)

  return (
    <Card
      title="Earnings — estimate vs actual"
      right={<AsOf date={meta?.as_of} days={meta?.staleness_days} sources={meta?.sources} />}
    >
      {next && (
        <div className="mb-2 flex items-center gap-2 text-[11px] text-muted">
          <Pill tone="accent">next</Pill>
          <span className="num">{day(next.report_date)}</span>
          <span className="uppercase text-faint">{next.report_time}</span>
          <span className="text-faint">est</span>
          <span className="num text-ink">{num(next.eps_estimate)}</span>
        </div>
      )}

      {past.length === 0 ? (
        <Empty reason="No earnings history available." hint={meta?.gaps?.join(' · ')} />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-faint uppercase tracking-wider text-[10px]">
                <th className="text-left font-medium py-1">Report</th>
                <th className="text-right font-medium">Est</th>
                <th className="text-right font-medium">Actual</th>
                <th className="text-right font-medium">Surprise</th>
                <th className="text-right font-medium" title="Forward guidance has no free source — see the gap note below">Guide</th>
                <th className="text-right font-medium">Revenue</th>
              </tr>
            </thead>
            <tbody className="num">
              {past.map(r => (
                <tr key={r.fiscal_end ?? r.report_date} className="border-t border-line-soft">
                  <td className="py-1 text-muted">
                    {day(r.report_date)}
                    <span className="ml-1 text-[9px] text-faint uppercase">{r.report_time}</span>
                  </td>
                  <td className="text-right text-muted">{num(r.eps_estimate)}</td>
                  <td className="text-right">{num(r.eps_actual)}</td>
                  <td className="text-right"><Delta v={r.eps_surprise_pct} /></td>
                  <td className="text-right text-faint">
                    {r.guide_eps_low != null ? `${num(r.guide_eps_low)}–${num(r.guide_eps_high)}` : '—'}
                  </td>
                  <td className="text-right text-muted">{r.rev_actual != null ? usd(r.rev_actual) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {meta?.gaps?.length ? (
        <p className="mt-2 text-[10px] text-faint leading-relaxed">
          {meta.gaps.join(' · ')}
        </p>
      ) : null}
    </Card>
  )
}

/* ── 3: how the stock actually moved on each print ─────────────────────────── */
function ReactionCard({ rows }: { rows: EarningsRow[] }) {
  const withReaction = rows.filter(r => r.reaction && r.reaction.d0_pct != null).slice(0, 12)

  return (
    <Card title="Reaction history">
      {withReaction.length === 0 ? (
        <Empty
          reason="No reaction history."
          hint="Computed by joining earnings dates to price bars — needs the backend or a warm ticker."
        />
      ) : (
        <>
          <div className="flex items-end gap-1 h-16 mb-2">
            {withReaction.slice().reverse().map(r => {
              const v = r.reaction!.d0_pct ?? 0
              const mag = Math.min(Math.abs(v) / 12, 1)
              return (
                <div key={r.report_date} className="flex-1 flex flex-col justify-end h-full" title={`${day(r.report_date)} ${v.toFixed(2)}%`}>
                  <div
                    className={v >= 0 ? 'bg-up/70' : 'bg-down/70'}
                    style={{ height: `${Math.max(mag * 100, 3)}%` }}
                  />
                </div>
              )
            })}
          </div>
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-faint uppercase tracking-wider text-[10px]">
                <th className="text-left font-medium py-1">Print</th>
                <th className="text-right font-medium">Gap</th>
                <th className="text-right font-medium">D0</th>
                <th className="text-right font-medium">D1</th>
                <th className="text-right font-medium">D5</th>
                <th className="text-right font-medium" title="D0 move minus SPY over the same session">vs SPY</th>
              </tr>
            </thead>
            <tbody className="num">
              {withReaction.map(r => (
                <tr key={r.report_date} className="border-t border-line-soft">
                  <td className="py-1 text-muted">{day(r.report_date)}</td>
                  <td className="text-right"><Delta v={r.reaction!.gap_pct} /></td>
                  <td className="text-right"><Delta v={r.reaction!.d0_pct} /></td>
                  <td className="text-right"><Delta v={r.reaction!.d1_pct} /></td>
                  <td className="text-right"><Delta v={r.reaction!.d5_pct} /></td>
                  <td className="text-right"><Delta v={r.reaction!.d0_excess_spy} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Card>
  )
}

/* ── 4: insider activity ───────────────────────────────────────────────────── */
function InsiderCard({ section }: { section?: any }) {
  const sum = section?.summary
  const rows = (section?.rows ?? []) as any[]
  const countsOnly = sum?.counts_only

  return (
    <Card
      title="Insider activity"
      right={<AsOf date={section?.as_of} days={section?.staleness_days} sources={section?.sources} />}
    >
      {!section ? (
        <Empty reason="No insider data." />
      ) : countsOnly ? (
        <>
          <div className="flex items-center gap-3 mb-2">
            <Pill tone="warn">cadence only</Pill>
            <span className="num text-[13px]">{rows.length}</span>
            <span className="text-[11px] text-muted">Form 4 filings in 180d</span>
          </div>
          <Empty
            reason="Share amounts are not available in the browser."
            hint="Form 4 detail lives on www.sec.gov/Archives, which does not send CORS headers. Start the local backend, or add this ticker to the warm watchlist."
          />
        </>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-2 mb-2">
            <Stat label="Buys 180d"  value={String(sum?.buys_180d ?? '—')} tone="up" />
            <Stat label="Sells 180d" value={String(sum?.sells_180d ?? '—')} tone="down" />
            <Stat label="Net value"  value={usd(sum?.net_value_usd_180d)}
                  tone={(sum?.net_value_usd_180d ?? 0) >= 0 ? 'up' : 'down'} />
          </div>
          {rows.length === 0 ? (
            <Empty reason="No open-market insider transactions in the window." />
          ) : (
            <table className="w-full text-[11px]">
              <thead>
                <tr className="text-faint uppercase tracking-wider text-[10px]">
                  <th className="text-left font-medium py-1">Insider</th>
                  <th className="text-left font-medium">Filed</th>
                  <th className="text-center font-medium">Code</th>
                  <th className="text-right font-medium">Shares</th>
                  <th className="text-right font-medium">Value</th>
                </tr>
              </thead>
              <tbody className="num">
                {rows.filter(r => r.is_open_market).slice(0, 15).map((r, i) => (
                  <tr key={i} className="border-t border-line-soft">
                    <td className="py-1 truncate max-w-[140px]">
                      <span className="text-ink">{r.owner_name}</span>
                      {r.owner_title && <span className="ml-1 text-[9px] text-faint">{r.owner_title}</span>}
                    </td>
                    <td className="text-muted">{day(r.filing_date)}</td>
                    <td className="text-center">
                      <span className={r.code === 'P' ? 'text-up' : 'text-down'}>{r.code}</span>
                    </td>
                    <td className="text-right">{compact(r.shares)}</td>
                    <td className="text-right text-muted">{usd(r.value_usd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="mt-2 text-[10px] text-faint">
            Open-market only. Grants (A) and tax withholding (F) are excluded — they are not decisions to buy or sell.
          </p>
        </>
      )}
    </Card>
  )
}

/* ── 5: institutional positioning ──────────────────────────────────────────── */
function InstitutionsCard({ section }: { section?: any }) {
  const has = section?.n_holders != null
  return (
    <Card
      title="Institutional positioning"
      right={section?.period_end
        ? <span className="text-[10px] text-warn">as of {day(section.period_end)} · filed {day(section.filing_date_max)}</span>
        : null}
    >
      {!has ? (
        <Empty
          reason="No 13F positioning available."
          hint={section?.gaps?.join(' · ') ?? 'Aggregated from SEC quarterly 13F bulk datasets — needs the backend.'}
        />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <Stat label="Holders" value={compact(section.n_holders)}
                  sub={section.d_holders_qoq != null ? `${section.d_holders_qoq > 0 ? '+' : ''}${section.d_holders_qoq} QoQ` : undefined}
                  tone={(section.d_holders_qoq ?? 0) >= 0 ? 'up' : 'down'} />
            <Stat label="Shares held" value={compact(section.total_shares)}
                  sub={section.d_shares_qoq != null ? `${compact(section.d_shares_qoq)} QoQ` : undefined}
                  tone={(section.d_shares_qoq ?? 0) >= 0 ? 'up' : 'down'} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <HolderList title="Top buyers"  rows={section.top_buyers}  tone="up" />
            <HolderList title="Top sellers" rows={section.top_sellers} tone="down" />
          </div>
          <p className="mt-2 text-[10px] text-faint">
            13F is long-only US equity, filed up to 45 days after quarter end. Stale by construction, not by failure.
          </p>
        </>
      )}
    </Card>
  )
}

function HolderList({ title, rows, tone }: { title: string; rows: any[]; tone: 'up' | 'down' }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-faint mb-1">{title}</p>
      {!rows?.length ? <p className="text-[11px] text-faint">—</p> : (
        <ul className="space-y-0.5">
          {rows.slice(0, 5).map((h, i) => (
            <li key={i} className="flex justify-between gap-2 text-[11px]">
              <span className="truncate text-muted">{h.filer_name}</span>
              <span className={`num shrink-0 ${tone === 'up' ? 'text-up' : 'text-down'}`}>{compact(h.d_shares)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/* ── 6: short interest — three distinct measurements, never blended ────────── */
function ShortCard({ section }: { section?: any }) {
  const si = section?.interest?.[0]
  const sv = section?.short_volume ?? []
  const bo = section?.borrow

  return (
    <Card title="Short interest" right={<AsOf sources={section?.sources} />}>
      {!section ? (
        <Empty reason="No short-interest data." hint="FINRA and Nasdaq are not CORS-open — needs the backend." />
      ) : (
        <>
          <div className="grid grid-cols-3 gap-2">
            <Stat label="Shares short" value={compact(si?.shares_short)}
                  sub={si ? `settled ${day(si.settlement_date)}` : undefined} />
            <Stat label="Days to cover" value={num(si?.days_to_cover, 1)} />
            <Stat label="Borrow" value={bo?.tier || '—'}
                  sub={bo?.fee_rate != null ? `${num(bo.fee_rate, 2)}% fee` : undefined} />
          </div>
          {sv.length > 0 && (
            <div className="mt-3">
              <p className="text-[10px] uppercase tracking-wider text-faint mb-1">
                Daily short volume — a different measurement, not short interest
              </p>
              <div className="flex items-end gap-px h-10">
                {sv.slice(-40).map((d: any, i: number) => (
                  <div key={i} className="flex-1 bg-accent/40" title={`${d.date} ${num(d.short_pct ? d.short_pct * 100 : null, 1)}%`}
                       style={{ height: `${Math.min((d.short_pct ?? 0) * 100 * 1.6, 100)}%` }} />
                ))}
              </div>
            </div>
          )}
          <p className="mt-2 text-[10px] text-faint">
            Official short interest is bimonthly and ~8 days stale at publication — a regulatory fact, not a sourcing failure.
          </p>
        </>
      )}
    </Card>
  )
}

function Stat({ label, value, sub, tone }: {
  label: string; value: string; sub?: string; tone?: 'up' | 'down'
}) {
  const c = tone === 'up' ? 'text-up' : tone === 'down' ? 'text-down' : 'text-ink'
  return (
    <div className="bg-raised/60 border border-line-soft rounded px-2 py-1.5">
      <p className="text-[9px] uppercase tracking-wider text-faint">{label}</p>
      <p className={`num text-[14px] leading-tight ${c}`}>{value}</p>
      {sub && <p className="text-[10px] text-faint num">{sub}</p>}
    </div>
  )
}
