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
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, '.')

PAIRS = [
    ('EURUSD=X', 'US', 'EU', 'EUR/USD'),
    ('GBPUSD=X', 'US', 'GB', 'GBP/USD'),
    ('AUDUSD=X', 'AU', 'US', 'AUD/USD'),
    ('AUDNZD=X', 'AU', 'NZ', 'AUD/NZD'),
    ('USDJPY=X', 'US', 'JP', 'USD/JPY'),
]
EQUITY_PAIRS = ['SPY', 'QQQ']


def _load_cross_system_state():
    try:
        with open('data/forensics/cross_system_state.json') as f:
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

    results = []
    for ticker, base, quote, label in PAIRS:
        try:
            prices = yf.Ticker(ticker).history(period='90d', interval='1d')
            if prices.empty:
                continue
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            df = build_signal_frame(ticker, prices, base, quote)
            last = df.iloc[-1]
            close = float(prices['Close'].iloc[-1])
            signal = int(last['signal'])
            size_mult = float(last.get('size_mult', 1.0))

            # conviction = signal strength * size_mult
            conviction = round(abs(signal) * size_mult, 3) if signal != 0 else 0.0

            # recent price history (last 60 bars) for the sparkline
            price_history = [
                {'t': int(ts.timestamp() * 1000), 'v': round(float(v), 5)}
                for ts, v in zip(prices.index[-60:], prices['Close'].iloc[-60:])
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

            results.append({
                'ticker': ticker,
                'label': label,
                'price': close,
                'signal': signal,
                'conviction': conviction,
                'size_mult': size_mult,
                'price_history': price_history,
                'signal_marks': signal_marks[-20:],
                'error': None,
            })
        except Exception as e:
            results.append({'ticker': ticker, 'label': label, 'price': None,
                            'signal': 0, 'conviction': 0.0, 'error': str(e)})
    return results


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


def build_payload():
    bridge = _load_cross_system_state()
    ict_trades = _load_ict_trades()
    forex = _fetch_forex_signals()
    equity = _fetch_equity_signals()
    best = _best_trade_today(forex, bridge)

    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'bridge': bridge,
        'forex_signals': forex,
        'equity': equity,
        'best_trades': best,
        'ict_trades_count': len(ict_trades),
        'ict_recent': ict_trades[-5:] if ict_trades else [],
    }


_CHAT_SYSTEM = """You are SOVEREIGN, an AI trading research assistant for a quant trading system.
Current state: Forex system v013 (Sharpe 1.8552, institutional grade). ICT system: London+GradeA,
WR=41%, avgR=+0.840. Bridge: HALT_NEW (ASIAN_CURRENCY_CONTAGION, threat=1.00).
Prop challenge: ACTIVE Day 0, $100k balance. Answer concisely using system data when relevant."""

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
        'system': _CHAT_SYSTEM,
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
            system=_CHAT_SYSTEM,
            messages=messages,
        )
        reply = resp.content[0].text
        _chat_history.append({'role': 'assistant', 'content': reply})
        return reply
    except Exception as e:
        return f'Chat unavailable: {e}. Ensure ANTHROPIC_API_KEY is set.'


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

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path.split('?')[0] in ('/', '/data'):
            try:
                self._send_json(200, build_payload())
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/chat':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length))
                reply = _handle_chat(body.get('message', ''), body.get('context', ''))
                self._send_json(200, {'reply': reply})
            except Exception:
                self._send_json(500, {'error': traceback.format_exc()})
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    port = 8765
    print(f'Live Signals Server → http://localhost:{port}')
    print('Open frontend/live_signals.html in your browser.')
    print('Ctrl+C to stop.\n')
    HTTPServer(('', port), Handler).serve_forever()
