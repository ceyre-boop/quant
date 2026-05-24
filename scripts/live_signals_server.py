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
            prices = yf.Ticker(ticker).history(period='2y', interval='1d')
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

            results.append({
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
        path = self.path.split('?')[0]
        if path in ('/', '/data'):
            try:
                self._send_json(200, build_payload())
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
        else:
            self.send_response(404)
            self.end_headers()

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
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    port = 8765
    print(f'Live Signals Server → http://localhost:{port}')
    print('Open frontend/live_signals.html in your browser.')
    print('Ctrl+C to stop.\n')
    HTTPServer(('', port), Handler).serve_forever()
