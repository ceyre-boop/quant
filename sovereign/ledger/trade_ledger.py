"""
Phase 7 — Trade Ledger (V1.0)
Logs EVERY executed trade for performance audit.
"""

import json
from datetime import datetime
from pathlib import Path

class TradeLedger:
    def __init__(self):
        self.path = Path('data/ledger')
        self.path.mkdir(parents=True, exist_ok=True)

    def log_entry(self, trade_id: str, symbol: str, direction: str, entry_price: float, 
                  size: float, sl: float, tp: float, confidence: float):
        """Records a new open trade."""
        entry = {
            'trade_id': trade_id,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'size': size,
            'stop_loss': sl,
            'take_profit': tp,
            'confidence': confidence,
            'entry_time': datetime.utcnow().isoformat(),
            'status': 'OPEN'
        }
        self._write_entry(entry)

    def _write_entry(self, entry: dict):
        month = datetime.utcnow().strftime('%Y_%m')
        with open(self.path / f'trade_ledger_{month}.jsonl', 'a') as f:
            f.write(json.dumps(entry) + '\n')
