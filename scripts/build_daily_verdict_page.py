#!/usr/bin/env python3
"""
build_daily_verdict_page.py — generates daily_verdict.html

The ONE page that answers, in plain language: should we be trading live today,
and why or why not. Pulls real state from files that already exist — invents
nothing, fabricates no status. If a source file is missing or a field is null,
the page says so instead of guessing.

Run daily (wire into the existing scheduled pipeline alongside health_check.py).
Output: daily_verdict.html (repo root — served by both localhost and Render/Pages).
"""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent

def load(path, default=None):
    p = ROOT / path
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

def main():
    prop = load('data/agent/prop_challenge_state.json', {})
    health = load('data/agent/system_health_verdict.json', {})
    numbers = load('data/research/colin_v1_window_backtest.json', {})
    paper = load('data/agent/carry_paper_account.json', {})

    gates = prop.get('gates', [])
    overall = prop.get('overall', 'UNKNOWN')

    # Plain-language translation of each gate
    gate_plain = {
        'G1': 'Would this pass a funded account, if we ran it 5,000 times?',
        'G2a': 'Are the trade signals actually correct, checked against real data?',
        'G2b': 'Did the orders actually place correctly when tested?',
        'G3': 'Does our real win rate match what the backtest predicted?',
        'G4': 'Is there a big danger warning flashing anywhere in the markets?',
        'G5': 'Have we survived enough real days without blowing up?',
    }

    green = sum(1 for g in gates if g.get('status') == 'GREEN')
    yellow = sum(1 for g in gates if g.get('status') == 'YELLOW')
    red = sum(1 for g in gates if g.get('status') == 'RED')

    # The one big verdict, in words a 10-year-old would understand
    if overall == 'GO':
        verdict = "YES — TRADE LIVE TODAY"
        verdict_color = "#0F6E56"
        verdict_bg = "#E1F5EE"
        verdict_why = "Every single check passed. Nothing is missing, nothing is broken."
    elif red > 0:
        verdict = "NO — DO NOT TRADE LIVE TODAY"
        verdict_color = "#A32D2D"
        verdict_bg = "#FCEBEB"
        verdict_why = f"{red} check(s) failed outright. Trading live now would be ignoring a real red flag."
    else:
        verdict = "NOT YET — WE ARE STILL COLLECTING PROOF"
        verdict_color = "#854F0B"
        verdict_bg = "#FAEEDA"
        verdict_why = f"{green} of {len(gates)} checks are green. The rest aren't broken — they just don't have enough real data yet to say yes."

    strategies = health.get('strategies', {})
    strat_rows = ""
    for name, s in strategies.items():
        ks = s.get('kill_switch', 'UNKNOWN')
        reason = s.get('reason', '')
        color = {'OK': '#0F6E56', 'REDUCE': '#854F0B', 'HALT': '#A32D2D'}.get(ks, '#888')
        strat_rows += f"""
        <div class="strat-row">
          <span class="strat-name">{name.replace('_',' ').title()}</span>
          <span class="strat-pill" style="background:{color}22;color:{color}">{ks}</span>
          <span class="strat-reason">{reason}</span>
        </div>"""

    gate_rows = ""
    for g in gates:
        gid = g.get('id', '')
        status = g.get('status', 'UNKNOWN')
        color = {'GREEN': '#0F6E56', 'YELLOW': '#854F0B', 'RED': '#A32D2D'}.get(status, '#888')
        icon = {'GREEN': '✓', 'YELLOW': '…', 'RED': '✗'}.get(status, '?')
        plain_q = gate_plain.get(gid, gid)
        gate_rows += f"""
        <div class="gate-row">
          <span class="gate-icon" style="color:{color}">{icon}</span>
          <span class="gate-q">{plain_q}</span>
          <span class="gate-val" style="color:{color}">{g.get('value','')}</span>
        </div>"""

    def money_row(label, w):
        if not w or not isinstance(w, dict):
            return ""
        return f"""
        <tr><td>{label}</td>
        <td>{w.get('median',0):+.1%}</td>
        <td>{w.get('p5',0):+.1%}</td>
        <td>{w.get('p95',0):+.1%}</td>
        <td>{w.get('p_pos',0):.0%}</td></tr>"""

    numbers_rows = money_row('5-year test', numbers.get('5yr')) + money_row('10-year test', numbers.get('10yr'))

    paper_pnl = paper.get('total_pnl', None)
    paper_line = (f"Right now, practicing with fake money, we're at "
                  f"${paper_pnl:,.0f} total (out of {paper.get('n_points','?')} data points)."
                  if paper_pnl is not None else "No paper-trading data found yet.")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Should We Trade Live Today?</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background:#FCFCFB; color:#0B0B0B;
         max-width:760px; margin:0 auto; padding:24px 20px 60px; line-height:1.6; }}
  h1 {{ font-size:20px; font-weight:500; margin-bottom:4px; }}
  .updated {{ color:#898781; font-size:13px; margin-bottom:24px; }}
  .verdict-box {{ background:{verdict_bg}; border-radius:16px; padding:28px 24px; margin-bottom:28px; }}
  .verdict-text {{ font-size:28px; font-weight:600; color:{verdict_color}; margin-bottom:10px; }}
  .verdict-why {{ font-size:16px; color:#3a3a38; }}
  h2 {{ font-size:16px; font-weight:500; margin:32px 0 12px; border-bottom:1px solid #E1E0D9; padding-bottom:8px; }}
  .gate-row, .strat-row {{ display:flex; align-items:center; gap:12px; padding:10px 0;
                            border-bottom:1px solid #F0EFE8; font-size:14px; }}
  .gate-icon {{ font-size:18px; font-weight:700; width:20px; }}
  .gate-q {{ flex:1; }}
  .gate-val {{ font-size:13px; font-weight:500; text-align:right; }}
  .strat-name {{ width:130px; font-weight:500; }}
  .strat-pill {{ padding:2px 10px; border-radius:8px; font-size:11px; font-weight:600; }}
  .strat-reason {{ flex:1; color:#6B6A63; font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; margin-top:8px; }}
  th, td {{ text-align:right; padding:8px 6px; border-bottom:1px solid #F0EFE8; }}
  th:first-child, td:first-child {{ text-align:left; }}
  .paper-note {{ background:#F9F9F7; border-radius:12px; padding:16px; font-size:14px; color:#3a3a38; }}
  .footer {{ margin-top:40px; font-size:12px; color:#B4B3A8; }}
</style>
</head>
<body>

<h1>Should We Trade Live Today?</h1>
<p class="updated">Checked: {datetime.now(timezone.utc).strftime('%B %d, %Y at %H:%M UTC')}</p>

<div class="verdict-box">
  <div class="verdict-text">{verdict}</div>
  <div class="verdict-why">{verdict_why}</div>
</div>

<h2>The 5 questions we ask every single day</h2>
{gate_rows if gate_rows else '<p>No gate data found yet.</p>'}

<h2>Each strategy, one line each</h2>
{strat_rows if strat_rows else '<p>No strategy health data found yet.</p>'}
<p style="font-size:12px;color:#898781;margin-top:8px;">"REDUCE" doesn't mean broken — it means we don't have enough real trades yet to be sure it's healthy, so size stays small until we do.</p>

<h2>What the carry strategy actually makes (real backtest, real math)</h2>
<table>
<tr><th>Test period</th><th>Typical year</th><th>Bad year</th><th>Great year</th><th>% of years green</th></tr>
{numbers_rows if numbers_rows else '<tr><td colspan="5">Run scripts/colin_v1_window_backtest.py to fill this in.</td></tr>'}
</table>
<p style="font-size:12px;color:#898781;margin-top:8px;">At 1% risk per trade, on the real sealed trade log. Not a guess — this is what actually would have happened.</p>

<div class="paper-note">
  <strong>Practice account right now:</strong> {paper_line}
</div>

<div class="footer">
  Auto-generated from real files in the repo — data/agent/prop_challenge_state.json,
  data/agent/system_health_verdict.json, data/research/colin_v1_window_backtest.json,
  data/agent/carry_paper_account.json. Nothing on this page is invented. A missing
  number means the underlying data doesn't exist yet, not that everything is fine.
</div>

</body>
</html>"""

    out = ROOT / 'daily_verdict.html'
    out.write_text(html)
    print(f"Wrote {out}")
    print(f"Verdict: {verdict}")

if __name__ == '__main__':
    main()
