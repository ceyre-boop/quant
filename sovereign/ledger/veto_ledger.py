"""
Phase 7 — Veto Ledger (V1.0)
Logs EVERY signal rejection to monitor system filter health.
"""

import json
from datetime import datetime
from pathlib import Path
from collections import Counter
from contracts.types import VetoRecord

class VetoLedger:
    """
    Records why signals were rejected to ensure the system doesn't over-filter.
    Stores logs in data/ledger/veto_ledger_YYYY_MM.jsonl
    """

    def __init__(self):
        self.path = Path('data/ledger')
        self.path.mkdir(parents=True, exist_ok=True)

    def log_veto(self, record: VetoRecord):
        """Archives a signal rejection event."""
        entry = {
            'timestamp': record.timestamp if hasattr(record, 'timestamp') else datetime.utcnow().isoformat(),
            'symbol': record.symbol,
            'stage': record.veto_stage,
            'reason': record.veto_reason,
            'logged_at': datetime.utcnow().isoformat()
        }
        
        # Monthly shard for high-frequency archival
        month = datetime.utcnow().strftime('%Y_%m')
        log_file = self.path / f'veto_ledger_{month}.jsonl'
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_veto_rate(self, days: int = 30) -> dict:
        """
        Processes logs to count rejections by stage.
        Used to identify 'Rule Bloat' or over-filtering.
        """
        counts = Counter()
        month = datetime.utcnow().strftime('%Y_%m')
        log_file = self.path / f'veto_ledger_{month}.jsonl'
        
        if not log_file.exists():
            return {}

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    counts[entry['stage']] += 1
        except Exception as e:
            print(f"Error reading veto ledger: {e}")
            
        return dict(counts)

    def print_health_report(self):
        """Prints a summary of the current filter stack health."""
        rates = self.get_veto_rate()
        print("\n--- SOVEREIGN FILTER HEALTH REPORT ---")
        
        # Healthy bounds per spec
        bounds = {
            'PETROULAS': 5,
            'ROUTER/FLAT': 40,
            'SPECIALIST': 10,
            'RISK/EV': 20,
            'GAME': 5
        }
        
        for stage, count in rates.items():
            limit = bounds.get(stage, 100)
            status = "✅" if count <= limit else "⚠️ OVER-FILTERING"
            print(f"{stage:15}: {count:4d} entries {status}")
        print("--------------------------------------\n")
