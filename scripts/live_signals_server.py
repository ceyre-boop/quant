#!/usr/bin/env python3
"""
Live Signals Server — serves JSON data for the live_signals.html dashboard.
Run: python3 scripts/live_signals_server.py
Port: 8765
"""
import json
import os
import sys
import traceback
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, '.')

try:
    from sovereign.ledger.live_trade_log import LiveTradeLog as _LTL
    _live_log = _LTL()
except Exception:
    _live_log = None

PAIRS = [
    ('EURUSD=X', 'US', 'EU', 'EUR/USD'),
    ('GBPUSD=X', 'US', 'GB', 'GBP/USD'),
    ('AUDUSD=X', 'AU', 'US', 'AUD/USD'),
    ('AUDNZD=X', 'AU', 'NZ', 'AUD/NZD'),
    ('USDJPY=X', 'US', 'JP', 'USD/JPY'),
]
EQUITY_PAIRS = ['SPY', 'QQQ']


TV_REGIME_PATH = 'data/agent/tv_regime_signals.json'
_TV_RETAIN_HOURS = 48


# ── Trader control layer: token-gated mutating actions + async background jobs ──── #
import threading
import subprocess

_JOBS_DIR = 'data/agent/_jobs'
_JOBS: dict = {}  # id -> status dict (in-memory mirror of the on-disk status file)


def _control_token() -> str:
    """Shared secret for mutating control endpoints. Unset → controls disabled (safe default)."""
    return os.environ.get('SOVEREIGN_CONTROL_TOKEN', '').strip()


def _job_path(job_id: str) -> str:
    return os.path.join(_JOBS_DIR, job_id + '.json')


def _write_job(job: dict) -> None:
    _JOBS[job['id']] = job
    try:
        os.makedirs(_JOBS_DIR, exist_ok=True)
        with open(_job_path(job['id']), 'w') as f:
            json.dump(job, f, default=str)
    except Exception:
        pass


def _read_job(job_id: str):
    if job_id in _JOBS:
        return _JOBS[job_id]
    try:
        with open(_job_path(job_id)) as f:
            return json.load(f)
    except Exception:
        return None


def _spawn_job(kind: str, argv: list, label: str = None) -> str:
    """Run argv in a background daemon thread, streaming combined stdout/stderr into a capped
    tail. Returns the job id immediately; poll /control/job?id=<id> for status. Status persists
    to data/agent/_jobs/<id>.json so it survives across requests."""
    job_id = kind + '-' + datetime.now().strftime('%Y%m%d-%H%M%S')
    job = {'id': job_id, 'kind': kind, 'label': label or kind, 'cmd': ' '.join(argv),
           'status': 'running', 'started': datetime.now(timezone.utc).isoformat(),
           'finished': None, 'returncode': None, 'tail': ''}
    _write_job(job)

    def _run():
        tail: list = []
        try:
            proc = subprocess.Popen(argv, cwd='.', stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    env={**os.environ})
            for line in proc.stdout:
                tail.append(line.rstrip('\n'))
                del tail[:-80]  # keep last 80 lines
                job['tail'] = '\n'.join(tail)
                _write_job(job)
            proc.wait()
            job['returncode'] = proc.returncode
            job['status'] = 'done' if proc.returncode == 0 else 'failed'
        except Exception:
            job['status'] = 'failed'
            job['tail'] = (job.get('tail', '') + '\n' + traceback.format_exc())[-4000:]
        job['finished'] = datetime.now(timezone.utc).isoformat()
        _write_job(job)

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def _ingest_tv_regime(payload: dict) -> None:
    """Append a regime_update alert to tv_regime_signals.json, pruning entries >48h old."""
    from pathlib import Path
    path = Path(TV_REGIME_PATH)
    try:
        signals = json.loads(path.read_text()) if path.exists() else []
        if not isinstance(signals, list):
            signals = []
    except Exception:
        signals = []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=_TV_RETAIN_HOURS)
    signals = [
        s for s in signals
        if _parse_ts(s.get("timestamp", "")) >= cutoff
    ]

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker":    str(payload.get("ticker", payload.get("symbol", "UNKNOWN"))).upper(),
        "regime":    str(payload.get("regime", "UNKNOWN")).upper(),
        "strength":  float(payload.get("strength", 0.0)),
        "indicator": str(payload.get("indicator", "UNKNOWN")),
        "timeframe": str(payload.get("timeframe", "UNKNOWN")),
        "raw":       payload,
    }
    signals.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signals, indent=2))
    print(f"[TV-REGIME] {entry['regime']} {entry['ticker']} str={entry['strength']:.2f} [{entry['indicator']}]")


def _parse_ts(ts_str: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _load_cross_system_state():
    try:
        with open('data/forensics/cross_system_state.json') as f:
            return json.load(f)
    except Exception:
        return {}


def _load_big_move():
    """Big-Move-of-the-Day estimates (written by the 2h pulse). DISPLAY-ONLY until
    scripts/validate_big_move.py PASSes — carries its own validation_status flag."""
    try:
        with open('data/agent/big_move.json') as f:
            return json.load(f)
    except Exception:
        return {}


def _load_ict_trades():
    try:
        with open('data/ledger/ict_paper_trades.json') as f:
            d = json.load(f)
        return d if isinstance(d, list) else d.get('trades', [])
    except Exception:
        return []


def _fetch_forex_signals():
    import yfinance as yf
    import pandas as pd
    from sovereign.forex.signal_engine import build_signal_frame

    from concurrent.futures import ThreadPoolExecutor

    def _one(_pair):
        ticker, base, quote, label = _pair
        try:
            prices = yf.Ticker(ticker).history(period='2y', interval='1d')
            if prices.empty:
                return None
            prices.index = prices.index.tz_convert(None)
            df = build_signal_frame(ticker, prices, base, quote)
            last = df.iloc[-1]
            close = float(prices['Close'].iloc[-1])
            signal = int(last['signal'])
            size_mult = float(last.get('size_mult', 1.0))

            # conviction = signal strength * size_mult
            conviction = round(abs(signal) * size_mult, 3) if signal != 0 else 0.0

            # recent price history (last 90 bars) for the chart
            n = min(504, len(prices))  # up to 2 years of daily bars
            px = prices.iloc[-n:]
            price_history = [
                {'t': int(ts.timestamp() * 1000), 'v': round(float(v), 5)}
                for ts, v in zip(px.index, px['Close'])
            ]
            # OHLCV for candlestick chart
            ohlcv = [
                {
                    't': int(ts.timestamp() * 1000),
                    'o': round(float(row['Open']), 5),
                    'h': round(float(row['High']), 5),
                    'l': round(float(row['Low']), 5),
                    'c': round(float(row['Close']), 5),
                    'v': int(row.get('Volume', 0)),
                }
                for ts, row in px.iterrows()
            ]

            # recent signals
            signal_marks = []
            for ts, row in df.iterrows():
                if row['signal'] != 0:
                    signal_marks.append({
                        't': int(pd.Timestamp(ts).timestamp() * 1000),
                        'dir': int(row['signal']),
                        'hold': int(row.get('hold_days', row.get('hold', 60))),
                    })

            # conviction history aligned to price bars
            conv_history = []
            for ts, row in df.iloc[-n:].iterrows():
                sig_val = int(row.get('signal', 0))
                sm = float(row.get('size_mult', 1.0))
                conv_history.append({
                    't': int(pd.Timestamp(ts).timestamp() * 1000),
                    'v': round(abs(sig_val) * sm, 3) if sig_val != 0 else 0.0,
                    's': sig_val,
                })

            return {
                'ticker': ticker,
                'label': label,
                'price': close,
                'signal': signal,
                'conviction': conviction,
                'size_mult': size_mult,
                'price_history': price_history,
                'ohlcv': ohlcv,
                'conv_history': conv_history,
                'signal_marks': signal_marks[-20:],
                'error': None,
            }
        except Exception as e:
            return {'ticker': ticker, 'label': label, 'price': None,
                    'signal': 0, 'conviction': 0.0, 'error': str(e)}

    # Fetch all pairs in parallel — turns ~5 sequential yfinance calls (~22s) into ~5s.
    with ThreadPoolExecutor(max_workers=len(PAIRS)) as _ex:
        return [r for r in _ex.map(_one, PAIRS) if r is not None]


def _fetch_equity_signals():
    import yfinance as yf

    results = []
    for ticker in EQUITY_PAIRS:
        try:
            prices = yf.Ticker(ticker).history(period='10d', interval='5m')
            if prices.empty:
                continue
            close = float(prices['Close'].iloc[-1])
            change_pct = float((prices['Close'].iloc[-1] / prices['Close'].iloc[0] - 1) * 100)
            price_history = [
                {'t': int(ts.timestamp() * 1000), 'v': round(float(v), 2)}
                for ts, v in zip(prices.index[-100:], prices['Close'].iloc[-100:])
            ]
            results.append({
                'ticker': ticker,
                'price': round(close, 2),
                'change_pct': round(change_pct, 2),
                'price_history': price_history,
            })
        except Exception as e:
            results.append({'ticker': ticker, 'price': None, 'change_pct': 0,
                            'error': str(e)})
    return results


def _best_trade_today(forex_signals, bridge_state):
    """Score every live setup and return rank-ordered list."""
    candidates = []

    ict_mode = bridge_state.get('ict_mode', 'NORMAL')
    threat = bridge_state.get('library_threat_score', 0.0)

    for s in forex_signals:
        if s.get('error') or s['signal'] == 0:
            continue

        conviction = s['conviction']
        direction = 'LONG' if s['signal'] > 0 else 'SHORT'

        # Regime penalty
        regime_penalty = 0.0
        if ict_mode == 'HALT_NEW':
            regime_penalty = 0.5
        elif ict_mode == 'TIGHTEN':
            regime_penalty = 0.25

        threat_penalty = threat * 0.3
        final_score = max(0.0, conviction - regime_penalty - threat_penalty)

        gate_reasons = []
        if ict_mode == 'HALT_NEW':
            gate_reasons.append(f'Bridge: HALT_NEW (threat={threat:.2f})')
        if s['size_mult'] < 1.0:
            gate_reasons.append(f'Size reduced: {s["size_mult"]:.2f}x (VIX/regime)')

        candidates.append({
            'ticker': s['ticker'],
            'label': s['label'],
            'direction': direction,
            'raw_conviction': s['conviction'],
            'regime_penalty': round(regime_penalty + threat_penalty, 3),
            'final_score': round(final_score, 3),
            'gate_notes': gate_reasons,
            'price': s['price'],
        })

    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return candidates


def _build_oracle_board(forex_signals, bridge_state):
    """Compute Oracle's board assessment — recommended play, market structure, conviction."""
    best = None
    best_score = -1.0
    for s in forex_signals:
        if s.get('error') or s['signal'] == 0:
            continue
        score = s['conviction'] * s.get('size_mult', 1.0)
        if score > best_score:
            best_score = score
            best = s

    threat = bridge_state.get('library_threat_score', 0.0)
    ict_mode = bridge_state.get('ict_mode', 'NORMAL')
    commitment = bridge_state.get('commitment_score_avg', 0.5)

    # Recommended play from best setup
    play = None
    if best:
        direction = 'LONG' if best['signal'] > 0 else 'SHORT'
        price = best['price'] or 0
        # Estimate SL/TP from ATR proxy (1% of price per unit conviction)
        atr_est = price * 0.008
        sl = round(price - atr_est if direction == 'LONG' else price + atr_est, 5)
        tp1 = round(price + atr_est * 1.5 if direction == 'LONG' else price - atr_est * 1.5, 5)
        tp2 = round(price + atr_est * 3.0 if direction == 'LONG' else price - atr_est * 3.0, 5)

        # Oracle reasoning — plain English
        regime_note = ''
        if ict_mode == 'HALT_NEW':
            regime_note = f'CAUTION: bridge HALT_NEW (threat={threat:.2f}). '
        elif ict_mode == 'TIGHTEN':
            regime_note = f'Size reduced: bridge TIGHTEN (threat={threat:.2f}). '

        commitment_label = 'COMMITTED' if commitment > 0.6 else ('DEVELOPING' if commitment > 0.3 else 'UNCOMMITTED')

        reasoning = (
            f"{regime_note}"
            f"{'Strong' if best['conviction'] > 1.5 else 'Moderate'} {direction} signal on {best['label']} "
            f"(conviction={best['conviction']:.2f}, size={best.get('size_mult', 1.0):.2f}x). "
            f"Market commitment: {commitment_label}."
        )

        play = {
            'pair': best['label'],
            'ticker': best['ticker'],
            'direction': direction,
            'entry': best['price'],
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'conviction': best['conviction'],
            'size_mult': best.get('size_mult', 1.0),
            'reasoning': reasoning,
        }

    # Last Oracle suggestions (read-only)
    suggestions = []
    try:
        with open('data/agent/suggestions.json') as f:
            d = json.load(f)
        items = d if isinstance(d, list) else d.get('suggestions', [])
        suggestions = [s for s in items[-10:] if s.get('status') not in ('VETOED',)][-5:]
    except Exception:
        pass

    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'board': {
            'threat': threat,
            'ict_mode': ict_mode,
            'commitment': round(commitment, 3),
            'regime': 'RISK_OFF' if threat > 0.85 else ('CAUTION' if threat > 0.5 else 'NORMAL'),
        },
        'recommended_play': play,
        'active_setups': [
            {
                'pair': s['label'],
                'ticker': s['ticker'],
                'direction': 'LONG' if s['signal'] > 0 else 'SHORT',
                'conviction': s['conviction'],
                'size_mult': s.get('size_mult', 1.0),
            }
            for s in forex_signals if s.get('signal', 0) != 0 and not s.get('error')
        ],
        'recent_suggestions': suggestions,
    }


_PAYLOAD_CACHE = {'ts': 0.0, 'data': None}

def build_payload(ttl=120):
    # Serve a cached payload for `ttl` seconds — the forex/equity fetches hit yfinance, so this
    # keeps repeat dashboard loads instant instead of recomputing (~5s) every time.
    import time as _t
    now = _t.time()
    if _PAYLOAD_CACHE['data'] is not None and (now - _PAYLOAD_CACHE['ts']) < ttl:
        return _PAYLOAD_CACHE['data']
    bridge = _load_cross_system_state()
    ict_trades = _load_ict_trades()
    forex = _fetch_forex_signals()
    equity = _fetch_equity_signals()
    best = _best_trade_today(forex, bridge)

    payload = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'bridge': bridge,
        'forex_signals': forex,
        'equity': equity,
        'best_trades': best,
        'ict_trades_count': len(ict_trades),
        'ict_recent': ict_trades[-5:] if ict_trades else [],
        'big_move': _load_big_move(),
    }
    _PAYLOAD_CACHE['data'] = payload
    _PAYLOAD_CACHE['ts'] = now
    return payload


# ── Replay engine: replay a futures session as a live trading day (Replay Cockpit) ──
# Reuses the EXACT live scalp+ORB engine (scripts/futures_replay.py: simulate_session) over
# historical 1-min bars, so the orders shown are the real ones the strategy would have taken.
_REPLAY_CACHE = {}
_FR_MOD = None
def _futures_replay_mod():
    global _FR_MOD
    if _FR_MOD is None:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location('futures_replay',
                                            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'futures_replay.py'))
        m = _ilu.module_from_spec(spec); spec.loader.exec_module(m); _FR_MOD = m
    return _FR_MOD

def _run_replay(symbol=None, day=None):
    import time as _t
    # ES/NQ map to the micro engine (same price action; the contract spec sets $/point).
    inst = {'NQ': 'MNQ', 'ES': 'MES'}.get((symbol or 'MNQ').upper(), (symbol or 'MNQ').upper())
    if inst not in ('MES', 'MNQ'):
        inst = 'MNQ'
    ck = (inst, day or 'latest')
    c = _REPLAY_CACHE.get(ck)
    if c and (_t.time() - c[0] < 600):
        return c[1]
    fr = _futures_replay_mod()
    from sovereign.futures import bar_feed as bf
    df = bf.load_history(inst, source='yf', day=None, lookback='7d')
    if df is None or len(df) == 0:
        return {'error': 'no_data', 'instrument': inst, 'available_days': []}
    all_days = bf.session_days(df)
    sel = day if (day and day in all_days) else (all_days[-1] if all_days else None)
    if not sel:
        return {'error': 'no_session', 'instrument': inst, 'available_days': all_days}
    prior_close = None; day_df = None
    for d in all_days:
        ddf = df[df.index.tz_convert(bf.ET).strftime('%Y-%m-%d') == d]
        if d == sel:
            day_df = ddf; break
        if len(ddf):
            prior_close = float(ddf['Close'].iloc[-1])
    if day_df is None or len(day_df) < 3:
        return {'error': 'thin_session', 'instrument': inst, 'day': sel, 'available_days': all_days}
    bias_dir, key_levels = fr._day_bias(day_df, sel, prior_close, inst, 'auto')
    session = fr.simulate_session(day_df, sel, bias_dir, key_levels, inst, 'safe')
    bars = [{'t': int(ts.timestamp() * 1000),
             'o': round(float(r['Open']), 2), 'h': round(float(r['High']), 2),
             'l': round(float(r['Low']), 2), 'c': round(float(r['Close']), 2),
             'v': int(r.get('Volume', 0) or 0)} for ts, r in day_df.iterrows()]
    out = {'instrument': inst, 'day': sel, 'tf': '1m', 'bias': bias_dir,
           'available_days': all_days, 'bars': bars, 'trades': session.get('trades', []),
           'summary': {'net_usd': session.get('net_usd', 0), 'n_trades': session.get('n_trades', 0),
                       'max_drawdown_usd': session.get('max_drawdown_usd', 0)}}
    _REPLAY_CACHE[ck] = (_t.time(), out)
    return out


# ── Calendar: Warrior-style monthly P&L from paper trades (+ engine fill-in for unlogged days) ──
_CAL_POINT_VALUE = {'MES': 5.0, 'MNQ': 2.0}
_CAL_CACHE = {}
def _calendar_data(month=None):
    import time as _t
    if not month:
        month = datetime.now(timezone.utc).strftime('%Y-%m')
    c = _CAL_CACHE.get(month)
    if c and (_t.time() - c[0] < 300):   # 5-min cache — keeps the free tier responsive
        return c[1]
    days = {}
    # 1) Real logged futures paper trades (the source that fills at market close).
    log = 'data/futures/trade_log.jsonl'
    if os.path.exists(log):
        for line in open(log):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('size_contracts', 0) == 0:
                continue
            day = str(r.get('ts', ''))[:10]
            if not day.startswith(month):
                continue
            d = days.setdefault(day, {'pnl': 0.0, 'n': 0, 'wins': 0, 'closed': 0, 'src': 'logged'})
            d['n'] += 1
            entry, exit_, size = r.get('entry'), r.get('exit'), r.get('size_contracts') or 0
            if entry is not None and exit_ is not None and size:
                pv = _CAL_POINT_VALUE.get(r.get('instrument'), 5.0)
                mult = 1 if r.get('direction') == 'LONG' else -1
                pnl = (float(exit_) - float(entry)) * mult * pv * float(size)
                d['pnl'] = round(d['pnl'] + pnl, 2); d['closed'] += 1
                if pnl > 0:
                    d['wins'] += 1
    # 2) Engine fill-in: for recent available sessions this month with no logged trades, show what
    #    the live engine actually produced. Load the 1-min history ONCE and simulate every day from it
    #    (instead of one yfinance fetch per day) — keeps the free-tier request well under the timeout.
    try:
        fr = _futures_replay_mod()
        from sovereign.futures import bar_feed as bf
        df = bf.load_history('MNQ', source='yf', day=None, lookback='7d')
        if df is not None and len(df):
            prior_close = None
            for day in bf.session_days(df):
                day_df = df[df.index.tz_convert(bf.ET).strftime('%Y-%m-%d') == day]
                if day.startswith(month) and day not in days and len(day_df) >= 3:
                    bias_dir, key_levels = fr._day_bias(day_df, day, prior_close, 'MNQ', 'auto')
                    s = fr.simulate_session(day_df, day, bias_dir, key_levels, 'MNQ', 'safe')
                    days[day] = {'pnl': round(s.get('net_usd', 0), 2), 'n': s.get('n_trades', 0),
                                 'wins': sum(1 for t in s.get('trades', []) if t.get('net_usd', 0) > 0),
                                 'closed': s.get('n_trades', 0), 'src': 'engine'}
                if len(day_df):
                    prior_close = float(day_df['Close'].iloc[-1])
    except Exception:
        pass
    total = {'pnl': round(sum(d['pnl'] for d in days.values()), 2),
             'n': sum(d['n'] for d in days.values()),
             'wins': sum(d['wins'] for d in days.values()),
             'closed': sum(d['closed'] for d in days.values())}
    out = {'month': month, 'days': days, 'month_total': total}
    _CAL_CACHE[month] = (_t.time(), out)
    return out


def _build_chat_system() -> str:
    bridge  = _load_cross_system_state()
    mode    = bridge.get('ict_mode', 'UNKNOWN')
    threat  = bridge.get('library_threat_score', 0.0)
    regime  = bridge.get('library_primary_regime', 'UNKNOWN')
    updated = bridge.get('last_updated', '')[:19]
    return (
        f"You are SOVEREIGN, an AI trading research assistant for a quant trading system.\n"
        f"Current state: Forex system v014 (fully-costed OOS Sharpe 0.76, 95% CI [0.55, 0.96]; IS 0.67; "
        f"edge proven vs random p<0.001 but regime-fragile; ICT patterns NOT proven p=0.52). "
        f"ICT system: London+GradeA, WR=41%, avgR=+0.840. "
        f"Bridge: {mode} ({regime}, threat={threat:.2f}). "
        f"State as of {updated} UTC. "
        f"Answer concisely using system data when relevant."
    )

_chat_history: list[dict] = []


def _handle_chat(message: str, context: str) -> str:
    """Send message to Claude via PAI Inference tool, return reply text."""
    import subprocess, shutil
    if not message.strip():
        return 'No message received.'

    _chat_history.append({'role': 'user', 'content': message})

    # Build messages with rolling 10-turn window
    messages = _chat_history[-20:]

    payload = json.dumps({
        'system': _build_chat_system(),
        'messages': messages,
        'max_tokens': 512,
    })

    # Try PAI Inference tool first
    inference_ts = os.path.expanduser('~/.claude/PAI/TOOLS/Inference.ts')
    bun = shutil.which('bun')
    if bun and os.path.exists(inference_ts):
        try:
            result = subprocess.run(
                [bun, inference_ts, 'fast', '--stdin'],
                input=payload, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                reply = data.get('content', [{}])[0].get('text', result.stdout.strip())
                _chat_history.append({'role': 'assistant', 'content': reply})
                return reply
        except Exception:
            pass

    # Fallback: direct anthropic SDK
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=512,
            system=_build_chat_system(),
            messages=messages,
        )
        reply = resp.content[0].text
        _chat_history.append({'role': 'assistant', 'content': reply})
        return reply
    except Exception as e:
        return f'Chat unavailable: {e}. Ensure ANTHROPIC_API_KEY is set.'


def _scalp_journal_payload() -> dict:
    """Aggregate data/futures/scalp_log.jsonl for the discretionary scalping panel +
    calendar. Empty-safe (zeros / empty lists before any trade is logged). This file is
    isolated from the systematic decision logs and is never read by the Oracle."""
    import pathlib
    from datetime import datetime, timezone, timedelta

    log = pathlib.Path('data/futures/scalp_log.jsonl')
    rows = []
    if log.exists():
        for line in log.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def _dt(r):
        try:
            return datetime.fromisoformat(str(r.get('timestamp_entry')).replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None

    now = datetime.now(timezone.utc)
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week0 = now - timedelta(days=7)
    month0 = now - timedelta(days=30)

    def _agg(sel):
        closed = [r for r in sel if r.get('outcome') != 'OPEN']
        net_r = round(sum(r.get('R_realized') or 0 for r in closed), 2)
        pnl = round(sum(r.get('pnl_dollars') or 0 for r in closed), 2)
        wins = sum(1 for r in closed if (r.get('R_realized') or 0) > 0)
        wr = round(100 * wins / len(closed)) if closed else 0
        return {'trades': len(sel), 'closed': len(closed), 'net_R': net_r,
                'pnl': pnl, 'win_rate': wr}

    dated = [(r, _dt(r)) for r in rows]
    today = _agg([r for r, d in dated if d and d >= day0])
    week = _agg([r for r, d in dated if d and d >= week0])
    month_rows = [r for r, d in dated if d and d >= month0]
    month = _agg(month_rows)

    open_positions = [{
        'trade_id': r.get('trade_id'), 'instrument': r.get('instrument'),
        'direction': r.get('direction'), 'entry_price': r.get('entry_price'),
        'stop_price': r.get('stop_price'), 'target_price': r.get('target_price'),
        'size_contracts': r.get('size_contracts'), 'thesis': r.get('thesis'),
        'setup_type': r.get('setup_type'), 'session': r.get('session'),
    } for r in rows if r.get('outcome') == 'OPEN']

    recent = sorted([r for r in rows if r.get('outcome') != 'OPEN'],
                    key=lambda x: x.get('timestamp_exit') or x.get('timestamp_entry') or '',
                    reverse=True)[:12]

    # Setup breakdown over the trailing month (closed trades only).
    setup_breakdown = {}
    for r in month_rows:
        if r.get('outcome') == 'OPEN':
            continue
        s = r.get('setup_type') or 'other'
        b = setup_breakdown.setdefault(s, {'trades': 0, 'wins': 0, 'net_R': 0.0})
        b['trades'] += 1
        b['net_R'] = round(b['net_R'] + (r.get('R_realized') or 0), 2)
        if (r.get('R_realized') or 0) > 0:
            b['wins'] += 1
    for s, b in setup_breakdown.items():
        b['win_rate'] = round(100 * b['wins'] / b['trades']) if b['trades'] else 0

    return {
        'today': today, 'week': week, 'month': month,
        'open_positions': open_positions, 'recent': recent,
        'setup_breakdown': setup_breakdown, 'raw': rows,
        'discretionary': True,
        'note': 'Discretionary paper scalping — not a systematic edge.',
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence request logs

    def _send_json(self, code, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    # Serve dashboard HTML/assets from the repo root so http://localhost:8765/ is the full
    # dashboard on ONE origin — avoids the HTTPS->localhost mixed-content block that makes the
    # public GitHub Pages site show "server offline".
    _CTYPES = {'.html': 'text/html; charset=utf-8', '.css': 'text/css',
               '.js': 'application/javascript', '.json': 'application/json',
               '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
               '.png': 'image/png', '.map': 'application/json'}

    # Secret-shaped paths that must never be served, even though they live inside
    # the repo root. The traversal guard below only stops escapes UP and out; it
    # does nothing about .env / keys sitting right next to index.html. Without
    # this, running the server from ~/quant serves .env over HTTP verbatim.
    _STATIC_DENY_EXT = {'.env', '.key', '.pem', '.p12', '.pfx', '.crt'}
    _STATIC_DENY_SUBSTR = ('secret', 'credential', 'password', 'token', 'holdout')

    def _is_denied_static(self, rel_path: str) -> bool:
        parts = [p for p in rel_path.replace('\\', '/').split('/') if p]
        for p in parts:
            low = p.lower()
            # Any dotfile/dotdir (.env, .git, .ssh, .env.local, ...).
            if low.startswith('.'):
                return True
            if os.path.splitext(low)[1] in self._STATIC_DENY_EXT:
                return True
            if any(s in low for s in self._STATIC_DENY_SUBSTR):
                return True
        return False

    def _send_static(self, rel_path):
        root = os.path.realpath('.')
        target = os.path.realpath(os.path.join(root, rel_path))
        # Path-traversal guard: refuse anything resolving outside the repo root.
        # Secret guard: refuse secret-shaped paths INSIDE the root (see above).
        if self._is_denied_static(rel_path) or \
                not (target == root or target.startswith(root + os.sep)) or not os.path.isfile(target):
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            return
        ext = os.path.splitext(target)[1].lower()
        with open(target, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', self._CTYPES.get(ext, 'application/octet-stream'))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Control-Token')
        self.end_headers()

    def _control_ok(self) -> bool:
        """True only if a control token is configured AND the request carries the matching
        X-Control-Token header. Sends a 403 + hint and returns False otherwise."""
        token = _control_token()
        if not token:
            self._send_json(403, {'error': 'controls_disabled',
                                  'hint': 'set SOVEREIGN_CONTROL_TOKEN in the server env to enable controls'})
            return False
        if self.headers.get('X-Control-Token', '') != token:
            self._send_json(403, {'error': 'bad_token',
                                  'hint': 'set your control token (🔒) in the dashboard — it must match the server'})
            return False
        return True

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/health':
            # Instant liveness probe — no yfinance/forex compute, so cloud health checks never flap.
            self._send_json(200, {'ok': True, 'service': 'sovereign-dashboard'})
        elif path == '/data':
            try:
                self._send_json(200, build_payload())
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})
        elif path == '/replay':
            try:
                import urllib.parse as _up
                qs = dict(_up.parse_qsl(self.path.split('?')[1] if '?' in self.path else ''))
                self._send_json(200, _run_replay(qs.get('symbol', 'MNQ'), qs.get('date')))
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})
        elif path == '/calendar':
            try:
                import urllib.parse as _up
                qs = dict(_up.parse_qsl(self.path.split('?')[1] if '?' in self.path else ''))
                self._send_json(200, _calendar_data(qs.get('month')))
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})
        elif path == '/oracle':
            try:
                bridge = _load_cross_system_state()
                forex = _fetch_forex_signals()
                self._send_json(200, _build_oracle_board(forex, bridge))
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})
        elif path == '/trades':
            try:
                from sovereign.ledger.live_trade_log import LiveTradeLog
                import urllib.parse as _up
                qs = dict(_up.parse_qsl(self.path.split('?')[1] if '?' in self.path else ''))
                n = int(qs.get('n', 200))
                events = LiveTradeLog.read(n)

                # Build equity curve (running R-multiples from closed trades)
                equity_curve = []
                equity = 0.0
                entries = {}
                for ev in sorted(events, key=lambda x: x.get('ts', '')):
                    etype = ev.get('type', '')
                    ticker = ev.get('ticker', '')
                    price = float(ev.get('price') or 0)
                    if etype == 'ENTRY':
                        entries[ticker] = {'price': price, 'dir': ev.get('direction', 'LONG'), 'ts': ev.get('ts', '')}
                    elif etype in ('TP1', 'TP2', 'STOP', 'SESSION_CLOSE') and ticker in entries:
                        entry = entries[ticker]
                        ep = entry['price']
                        if ep and ep != 0:
                            raw_r = (price - ep) / ep if entry['dir'] == 'LONG' else (ep - price) / ep
                            # Normalize to R (assume 1% stop = 1R)
                            r_val = raw_r * 100
                            if etype == 'STOP': r_val = max(r_val, -1.0)
                            equity += r_val
                            equity_curve.append({'ts': ev.get('ts', ''), 'equity': round(equity, 4), 'r': round(r_val, 4), 'type': etype, 'ticker': ticker})
                        if etype in ('TP2', 'STOP', 'SESSION_CLOSE'):
                            entries.pop(ticker, None)

                self._send_json(200, {
                    'events': events,
                    'open_positions': LiveTradeLog.open_positions(),
                    'count': len(events),
                    'equity_curve': equity_curve,
                })
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        elif path == '/scanner_state':
            try:
                with open('logs/scanner_state.json') as f:
                    self._send_json(200, json.load(f))
            except FileNotFoundError:
                self._send_json(200, {'pairs': {}, 'error': 'not_ready'})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        elif path == '/prop-challenge':
            # Monte Carlo prop-challenge risk (bootstrap of the real v015 edge).
            # Regenerate with: python3 -m sovereign.risk.monte_carlo_prop
            try:
                with open('data/risk/prop_monte_carlo.json') as f:
                    self._send_json(200, json.load(f))
            except FileNotFoundError:
                self._send_json(200, {'error': 'no_data',
                                      'hint': 'run: python3 -m sovereign.risk.monte_carlo_prop'})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        elif path == '/activity':
            try:
                now_s = datetime.now(timezone.utc).strftime('%Y_%m')
                ledger = f'data/ledger/ict_veto_ledger_{now_s}.jsonl'
                veto_log = []
                try:
                    with open(ledger) as f:
                        lines = f.readlines()
                    for line in lines[-30:]:
                        try: veto_log.append(json.loads(line.strip()))
                        except Exception: pass
                except FileNotFoundError:
                    pass
                paper = {'open': [], 'closed': []}
                try:
                    with open('data/ledger/ict_paper_trades.json') as f:
                        paper = json.load(f)
                except Exception:
                    pass
                self._send_json(200, {'veto_log': veto_log[-20:], 'paper_trades': paper})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        elif path == '/session_levels':
            try:
                import urllib.parse as _up
                _qs = dict(_up.parse_qsl(self.path.split('?')[1] if '?' in self.path else ''))
                pair = (_qs.get('pair', 'GBPUSD') or 'GBPUSD').upper()
                _YTICKER = {
                    'GBPUSD': 'GBPUSD=X', 'EURUSD': 'EURUSD=X',
                    'AUDUSD': 'AUDUSD=X', 'AUDNZD': 'AUDNZD=X',
                    'USDJPY': 'USDJPY=X',
                }
                yticker = _YTICKER.get(pair, pair + '=X')
                import yfinance as yf
                df = yf.download(yticker, period='5d', interval='1h', progress=False, auto_adjust=True)
                if df is None or df.empty:
                    self._send_json(200, {'error': 'no_data'})
                    return
                if hasattr(df.columns, 'levels'):
                    df.columns = df.columns.get_level_values(0)
                df.index = df.index.tz_convert('UTC')
                today = datetime.now(timezone.utc).date()
                yesterday = today - timedelta(days=1)
                yday = df[df.index.date == yesterday]
                pdh = float(yday['High'].max()) if not yday.empty else None
                pdl = float(yday['Low'].min()) if not yday.empty else None
                todays = df[df.index.date == today]
                asian = todays[(todays.index.hour >= 0) & (todays.index.hour < 8)]
                london = todays[(todays.index.hour >= 8) & (todays.index.hour < 16)]
                levels = {
                    'pair': pair,
                    'date': str(today),
                    'pdh': pdh,
                    'pdl': pdl,
                    'asian_high': float(asian['High'].max()) if not asian.empty else None,
                    'asian_low': float(asian['Low'].min()) if not asian.empty else None,
                    'london_high': float(london['High'].max()) if not london.empty else None,
                    'london_low': float(london['Low'].min()) if not london.empty else None,
                }
                import pathlib, json as _json
                pathlib.Path('logs').mkdir(parents=True, exist_ok=True)
                pathlib.Path('logs/session_levels.json').write_text(_json.dumps(levels, indent=2))
                self._send_json(200, levels)
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        elif path == '/ohlcv':
            try:
                import urllib.parse as _up
                import yfinance as yf
                _qs = dict(_up.parse_qsl(self.path.split('?')[1] if '?' in self.path else ''))
                pair = (_qs.get('pair', 'GBPUSD') or 'GBPUSD').upper()
                _YTICKER = {
                    'GBPUSD': 'GBPUSD=X', 'EURUSD': 'EURUSD=X',
                    'AUDUSD': 'AUDUSD=X', 'AUDNZD': 'AUDNZD=X',
                    'USDJPY': 'USDJPY=X',
                }
                yticker = _YTICKER.get(pair, pair + '=X')
                prices = yf.Ticker(yticker).history(period='2y', interval='1d')
                if prices.empty:
                    self._send_json(200, {'error': 'no_data', 'pair': pair})
                    return
                prices.index = prices.index.tz_convert(None)
                n = min(504, len(prices))
                px = prices.iloc[-n:]
                ohlcv = [
                    {
                        't': int(ts.timestamp() * 1000),
                        'o': round(float(row['Open']), 5),
                        'h': round(float(row['High']), 5),
                        'l': round(float(row['Low']), 5),
                        'c': round(float(row['Close']), 5),
                        'v': int(row.get('Volume', 0)) or round((float(row['High']) - float(row['Low'])) * 100000),
                    }
                    for ts, row in px.iterrows()
                ]
                price_history = [
                    {'t': int(ts.timestamp() * 1000), 'v': round(float(v), 5)}
                    for ts, v in zip(px.index, px['Close'])
                ]
                self._send_json(200, {
                    'pair': pair,
                    'ticker': yticker,
                    'ohlcv': ohlcv,
                    'price_history': price_history,
                    'current': round(float(prices['Close'].iloc[-1]), 5),
                })
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        elif path == '/macro':
            try:
                import subprocess, time as _time
                snap_path = 'data/macro/macro_snapshot.json'
                stale = True
                try:
                    mtime = os.path.getmtime(snap_path)
                    stale = (_time.time() - mtime) > 4 * 3600  # 4h
                except FileNotFoundError:
                    pass
                if stale:
                    subprocess.Popen(
                        [sys.executable, 'scripts/fetch_macro_cache.py'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                try:
                    with open(snap_path) as f:
                        self._send_json(200, json.load(f))
                except FileNotFoundError:
                    self._send_json(200, {'series': {}, 'summary': {}, 'fetched_at': None})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        elif path == '/next-move':
            # The Oracle's top queued "what to tell Claude Code next" prompt (read-only, safe everywhere).
            try:
                order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
                data = {}
                try:
                    with open('data/agent/prompt_queue.json') as f:
                        data = json.load(f)
                except FileNotFoundError:
                    pass
                prompts = [p for p in data.get('prompts', []) if p.get('status') == 'QUEUED']
                prompts.sort(key=lambda p: (order.get(p.get('priority', 'MEDIUM'), 1),
                                            p.get('queued_at', '')))
                self._send_json(200, {'next': prompts[0] if prompts else None,
                                      'pending': len(prompts), 'stats': data.get('stats', {})})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        elif path == '/control/status':
            # Is the control layer enabled? Is trading frozen? Recent background jobs. (read-only)
            try:
                frozen = None
                try:
                    from sovereign.utils import kill_switch as _ks
                    frozen = _ks.state()
                except Exception:
                    frozen = None
                jobs = {}
                try:
                    for fn in sorted(os.listdir(_JOBS_DIR))[-12:]:
                        if fn.endswith('.json'):
                            j = _read_job(fn[:-5])
                            if j:
                                jobs[j['id']] = {k: j.get(k) for k in
                                                 ('id', 'kind', 'label', 'status', 'started',
                                                  'finished', 'returncode')}
                except FileNotFoundError:
                    pass
                self._send_json(200, {'enabled': bool(_control_token()), 'frozen': frozen, 'jobs': jobs})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        elif path == '/control/job':
            try:
                import urllib.parse as _up
                qs = dict(_up.parse_qsl(self.path.split('?')[1] if '?' in self.path else ''))
                j = _read_job(qs.get('id', ''))
                if j is None:
                    self._send_json(404, {'error': 'no_such_job'})
                else:
                    self._send_json(200, j)
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        elif path == '/escalations':
            # Intelligence feed for the dashboard — recent triaged events from the
            # autonomous layer's Escalation Router (Component 4). JSONL → JSON array,
            # newest first, capped. Empty list if the feed doesn't exist yet.
            try:
                import pathlib
                feed = pathlib.Path('data/agent/escalation_log.jsonl')
                rows = []
                if feed.exists():
                    for line in feed.read_text().splitlines():
                        line = line.strip()
                        if line:
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                rows.reverse()  # newest first
                self._send_json(200, {'escalations': rows[:100]})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})
        elif path == '/scalp_journal':
            # Discretionary ES/NQ scalping journal (Colin's manual paper trades).
            # NOT a systematic signal — separate file, never read by the Oracle.
            try:
                self._send_json(200, _scalp_journal_payload())
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})
        elif path.startswith('/data/'):
            # NOTE: this route used to open(path) directly, with no guards — so
            # /data/../.env escaped the repo and was read off disk. Route it
            # through the same traversal + secret checks the static handler uses.
            rel = path.lstrip('/')
            root = os.path.realpath('.')
            target = os.path.realpath(os.path.join(root, rel))
            if self._is_denied_static(rel) or not target.startswith(root + os.sep):
                self._send_json(404, {'error': 'not_found', 'path': path})
                return
            try:
                with open(target) as f:
                    if target.endswith('.jsonl'):
                        # Line-delimited: json.load() would choke on line 2.
                        rows = [json.loads(ln) for ln in f if ln.strip()]
                        self._send_json(200, rows)
                    else:
                        self._send_json(200, json.load(f))
            except FileNotFoundError:
                self._send_json(404, {'error': 'not_found', 'path': path})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        else:
            # Static dashboard files (HTML/assets) served from repo root.
            if path == '/':
                self._send_static('index.html')
            elif path in ('/ict', '/ict/'):
                self._send_static('ict/index.html')
            elif path in ('/scalp_calendar', '/scalp_calendar/'):
                self._send_static('templates/scalp_calendar.html')
            elif path.endswith('/'):
                # Directory request (e.g. /dashboard/) — resolve to its index.html.
                # The dashboard HTML sets BASE='..', so it must be reached at its
                # own subdir path, not the repo root.
                self._send_static(path.lstrip('/') + 'index.html')
            else:
                self._send_static(path.lstrip('/'))

    def do_POST(self):
        path = self.path.split('?')[0]
        if path == '/chat':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length))
                reply = _handle_chat(body.get('message', ''), body.get('context', ''))
                self._send_json(200, {'reply': reply})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})
        elif path == '/webhook/tradingview':
            # TradingView alert webhook receiver.
            # Alert message JSON: {"action":"buy","ticker":"GBPUSD","price":1.2500,"strategy":"MySystem","comment":"entry"}
            # or the TradingView default format with just a text body like "buy GBPUSD 1.2500"
            try:
                length = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(length)
                # Try JSON first, fall back to plain-text parsing
                try:
                    body = json.loads(raw)
                except Exception:
                    text = raw.decode('utf-8', errors='replace').strip()
                    parts = text.split()
                    body = {
                        'action': parts[0] if parts else 'alert',
                        'ticker': parts[1] if len(parts) > 1 else 'UNKNOWN',
                        'price': float(parts[2]) if len(parts) > 2 else 0.0,
                        'strategy': 'pine_script',
                        '_raw': text,
                    }

                # Regime-update alerts are stored separately and don't go to trade log
                if body.get('action', '').lower() == 'regime_update':
                    _ingest_tv_regime(body)
                    self._send_json(200, {'ok': True, 'type': 'regime_update'})
                    return

                action    = body.get('action', 'alert').lower()
                ticker    = body.get('ticker', body.get('symbol', 'UNKNOWN')).upper()
                price     = float(body.get('price', body.get('close', 0.0)))
                strategy  = body.get('strategy', body.get('comment', 'tradingview'))

                direction_map = {
                    'buy': 'LONG', 'long': 'LONG', 'entry_long': 'LONG',
                    'sell': 'SHORT', 'short': 'SHORT', 'entry_short': 'SHORT',
                    'close': 'FLAT', 'exit': 'FLAT', 'close_long': 'FLAT', 'close_short': 'FLAT',
                }
                direction = direction_map.get(action, 'FLAT')
                event_type = 'TV_ALERT' if direction == 'FLAT' else 'ENTRY'

                if _live_log:
                    event = _live_log.log(
                        event_type, 'TRADINGVIEW', ticker, direction, price,
                        meta={'strategy': strategy, 'action': action, 'raw': body}
                    )
                    print(f'[TV] {action} {ticker} @ {price} [{strategy}]')
                    self._send_json(200, {'ok': True, 'logged': event})
                else:
                    self._send_json(503, {'ok': False, 'error': 'live_trade_log unavailable'})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        # ── Trader control layer (token-gated) ──────────────────────────────────
        elif path == '/control/run-queue':
            if not self._control_ok():
                return
            jid = _spawn_job('run-queue',
                             [sys.executable, 'scripts/run_research_queue.py'],
                             'Run research queue')
            self._send_json(200, {'ok': True, 'job': jid})

        elif path == '/control/nudge-oracle':
            if not self._control_ok():
                return
            # Fresh Oracle reflect cycle → a new "next best move" prompt lands in the queue.
            # Heavy: needs ANTHROPIC_API_KEY + sovereign.* deps (realistically a LOCAL action).
            jid = _spawn_job('nudge-oracle',
                             [sys.executable, '-m', 'sovereign.oracle.reflect_cycle'],
                             'Ask Oracle for a fresh take')
            self._send_json(200, {'ok': True, 'job': jid})

        elif path == '/control/checklist':
            if not self._control_ok():
                return
            jid = _spawn_job('checklist',
                             [sys.executable, 'scripts/agent_scheduler.py', '--checklist'],
                             'Run prop-firm checklist')
            self._send_json(200, {'ok': True, 'job': jid})

        elif path == '/control/refresh-prop':
            if not self._control_ok():
                return
            jid = _spawn_job('refresh-prop',
                             [sys.executable, '-m', 'sovereign.risk.monte_carlo_prop'],
                             'Refresh pass-probability')
            self._send_json(200, {'ok': True, 'job': jid})

        elif path == '/control/freeze':
            if not self._control_ok():
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                from sovereign.utils import kill_switch as _ks
                st = _ks.freeze(body.get('reason', 'manual freeze from dashboard'),
                                hard=bool(body.get('hard')), by='dashboard')
                self._send_json(200, {'ok': True, 'frozen': st})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        elif path == '/control/thaw':
            if not self._control_ok():
                return
            try:
                from sovereign.utils import kill_switch as _ks
                prior = _ks.thaw(by='dashboard')
                self._send_json(200, {'ok': True, 'was': prior})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    # PORT from env for cloud hosts (Render/Fly); defaults to 8765 locally.
    port = int(os.environ.get('PORT', 8765))
    print(f'Live Signals Server → http://0.0.0.0:{port}')
    print('Open the dashboard at http://localhost:%d/ (serves index.html + live data).' % port)
    print('Ctrl+C to stop.\n')
    HTTPServer(('0.0.0.0', port), Handler).serve_forever()
