"""
Phase 7 — Trade Ledger (V1.0)
Logs EVERY executed trade for performance audit.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

_CSV_HEADER = [
    'symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price',
    'direction', 'confidence', 'game_score', 'reason', 'pnl',
    'equity', 'win', 'mae', 'mfe',
]


class TradeLedger:
    def __init__(self):
        self.path = Path(__file__).parent.parent.parent / 'data' / 'ledger'
        self.path.mkdir(parents=True, exist_ok=True)

    def log_entry(self, trade_id: str, symbol: str, direction: str, entry_price: float,
                  size: float, sl: float, tp: float, confidence: float,
                  strategy: str = 'momentum'):
        """Records a new open trade to both JSONL and CSV."""
        now = datetime.utcnow()
        entry = {
            'trade_id':   trade_id,
            'symbol':     symbol,
            'direction':  direction,
            'entry_price': entry_price,
            'size':       size,
            'stop_loss':  sl,
            'take_profit': tp,
            'confidence': confidence,
            'strategy':   strategy,
            'entry_time': now.isoformat(),
            'status':     'OPEN',
        }
        self._write_jsonl(entry, now)
        self._write_csv(symbol, direction, entry_price, confidence, now)

    def log_close(self, trade_id: str, symbol: str, direction: str,
                  entry_price: float, exit_price: float, size: float,
                  sl: float, tp: float, confidence: float, pnl: float,
                  strategy: str = 'momentum', exit_reason: str = 'CLOSE',
                  entry_time: datetime | None = None, exit_time: datetime | None = None):
        """Records a closed trade to both JSONL and CSV."""
        now = exit_time or datetime.utcnow()
        entry_time = entry_time or now
        entry = {
            'trade_id': trade_id,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'stop_loss': sl,
            'take_profit': tp,
            'confidence': confidence,
            'strategy': strategy,
            'entry_time': entry_time.isoformat(),
            'exit_time': now.isoformat(),
            'exit_reason': exit_reason,
            'pnl': pnl,
            'status': 'closed',
        }
        self._write_jsonl(entry, now)
        self._write_csv_close(symbol, direction, entry_price, exit_price, confidence, pnl, now)

    def _write_jsonl(self, entry: dict, now: datetime):
        month = now.strftime('%Y_%m')
        with open(self.path / f'trade_ledger_{month}.jsonl', 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def _write_csv(self, symbol: str, direction: str, entry_price: float,
                   confidence: float, now: datetime):
        month = now.strftime('%Y_%m')
        csv_path = self.path / f'trade_ledger_{month}.csv'
        write_header = not csv_path.exists()
        row = {
            'symbol':      symbol,
            'entry_date':  now.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
            'exit_date':   '',
            'entry_price': entry_price,
            'exit_price':  '',
            'direction':   direction,
            'confidence':  round(confidence, 6),
            'game_score':  0.5,
            'reason':      'OPEN',
            'pnl':         '',
            'equity':      '',
            'win':         '',
            'mae':         '',
            'mfe':         '',
        }
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _write_csv_close(self, symbol: str, direction: str, entry_price: float,
                         exit_price: float, confidence: float, pnl: float,
                         now: datetime):
        month = now.strftime('%Y_%m')
        csv_path = self.path / f'trade_ledger_{month}.csv'
        write_header = not csv_path.exists()
        row = {
            'symbol': symbol,
            'entry_date': now.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
            'exit_date': now.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'confidence': round(confidence, 6),
            'game_score': 0.5,
            'reason': 'CLOSED',
            'pnl': round(pnl, 6),
            'equity': '',
            'win': pnl > 0,
            'mae': '',
            'mfe': '',
        }
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
