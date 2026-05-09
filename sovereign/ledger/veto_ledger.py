"""
Phase 7 — Veto Ledger (V1.0)
Logs EVERY signal rejection to monitor system filter health.
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from contracts.types import VetoRecord

class VetoLedger:
    """
    Records why signals were rejected to ensure the system doesn't over-filter.
    Stores logs in data/ledger/veto_ledger_YYYY_MM.jsonl
    """

    def __init__(self):
        self.path = Path(__file__).parent.parent.parent / 'data' / 'ledger'
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

    def gate_cooccurrence_report(self) -> Dict[str, object]:
        """
        Reads all veto records (JSONL shards + CSV legacy files) and reports:
          - How often each gate fires
          - How often each gate fires WITHOUT any other gate also firing that day
          - Co-occurrence pairs (gates that always fire together = one is redundant)

        A gate that never fires alone carries no independent information — it is a
        candidate for removal per the problem statement's gate-redundancy analysis.

        Returns a dict with:
          'gate_counts'      : {gate: total_fires}
          'solo_fires'       : {gate: fires_where_no_other_gate_fired_same_day}
          'solo_pct'         : {gate: solo_fires / total_fires}
          'cooccurrence'     : {(gA, gB): count_both_fired_same_day}
          'likely_redundant' : [gate, ...]  — solo_pct < 0.05
        """
        # ── 1. Collect all veto events as (date, gate) ─────────────────── #
        events: List[tuple[str, str]] = []

        # JSONL shards (sovereign/ledger/veto_ledger_*.jsonl)
        for f in sorted(self.path.glob('veto_ledger_*.jsonl')):
            try:
                with open(f) as fh:
                    for line in fh:
                        rec = json.loads(line)
                        date_str = (rec.get('timestamp') or rec.get('logged_at') or '')[:10]
                        gate = rec.get('stage') or rec.get('reason') or 'UNKNOWN'
                        events.append((date_str, gate))
            except Exception:
                pass

        # CSV legacy files (data/paper_trading/veto_ledger_*.csv)
        legacy_paths = list(
            (self.path.parent / 'paper_trading').glob('veto_ledger*.csv')
        )
        for f in legacy_paths:
            try:
                with open(f, newline='') as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        date_str = (row.get('date') or '')[:10]
                        gate = row.get('veto_reason') or row.get('stage') or 'UNKNOWN'
                        # Normalise: trim trailing detail (e.g. "PROB: 0.25 < 0.45" → "PROB")
                        gate_key = gate.split(':')[0].strip()
                        events.append((date_str, gate_key))
            except Exception:
                pass

        if not events:
            return {
                'gate_counts': {},
                'solo_fires': {},
                'solo_pct': {},
                'cooccurrence': {},
                'likely_redundant': [],
                'note': 'No veto data found.',
            }

        # ── 2. Group by date ─────────────────────────────────────────────── #
        by_date: Dict[str, set] = defaultdict(set)
        for date_str, gate in events:
            by_date[date_str].add(gate)

        gate_counts: Counter = Counter()
        solo_fires: Counter = Counter()
        cooccurrence: Counter = Counter()

        for date_str, gates in by_date.items():
            gates_list = sorted(gates)
            for g in gates_list:
                gate_counts[g] += 1
                if len(gates_list) == 1:
                    solo_fires[g] += 1
            # Record pairwise co-occurrences
            for i in range(len(gates_list)):
                for j in range(i + 1, len(gates_list)):
                    cooccurrence[(gates_list[i], gates_list[j])] += 1

        solo_pct = {
            g: round(solo_fires[g] / gate_counts[g], 3) if gate_counts[g] else 0.0
            for g in gate_counts
        }
        likely_redundant = [g for g, pct in solo_pct.items() if pct < 0.05]

        return {
            'gate_counts': dict(gate_counts.most_common()),
            'solo_fires': dict(solo_fires),
            'solo_pct': solo_pct,
            'cooccurrence': {f'{a}+{b}': cnt for (a, b), cnt in cooccurrence.most_common()},
            'likely_redundant': likely_redundant,
        }

    def print_cooccurrence_report(self) -> None:
        """Human-readable gate redundancy report printed to stdout."""
        report = self.gate_cooccurrence_report()
        if report.get('note'):
            print(f"\n[VetoLedger] {report['note']}")
            return

        print('\n─── GATE CO-OCCURRENCE REPORT ───────────────────────────────────')
        print(f"{'Gate':<25} {'Total':>7} {'Solo':>7} {'Solo%':>7}")
        print('─' * 50)
        for gate, total in report['gate_counts'].items():
            solo = report['solo_fires'].get(gate, 0)
            pct = report['solo_pct'].get(gate, 0.0)
            flag = ' ⚠ REDUNDANT?' if gate in report['likely_redundant'] else ''
            print(f'{gate:<25} {total:>7} {solo:>7} {pct:>6.1%}{flag}')

        if report['cooccurrence']:
            print('\n─── MOST COMMON CO-FIRING PAIRS ─────────────────────────────────')
            for pair, cnt in list(report['cooccurrence'].items())[:10]:
                print(f'  {pair:<40} {cnt:>5} days')

        if report['likely_redundant']:
            print(f'\n⚠  Likely redundant (solo% < 5%): {report["likely_redundant"]}')
        print('─────────────────────────────────────────────────────────────────\n')

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
