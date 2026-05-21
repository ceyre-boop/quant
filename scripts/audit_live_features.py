#!/usr/bin/env python3
"""
Feature Audit — print every LIVE feature with its last measured IC and
flag anything not re-validated in 90 days.

Usage:
    python scripts/audit_live_features.py
    python scripts/audit_live_features.py --days 60
    python scripts/audit_live_features.py --all
    python scripts/audit_live_features.py --seed   # seed pd_alignment graveyard entry
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lab.feature_registry import FeatureRegistry, Verdict, STALE_DAYS


def _fmt(v, fmt=".4f"):
    if v is None:
        return "n/a"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _date_age(date_str: str) -> str:
    if not date_str:
        return "never"
    try:
        d = datetime.fromisoformat(date_str).date()
        today = datetime.now(timezone.utc).date()
        age = (today - d).days
        return f"{age}d ago  ({d})"
    except ValueError:
        return date_str


def audit(days: int = STALE_DAYS, show_all: bool = False) -> int:
    """
    Print feature audit report.  Returns count of stale LIVE features.
    """
    reg = FeatureRegistry()
    live = reg.get_live()
    stale = reg.get_stale(days=days)
    graveyard = reg.get_graveyard()
    testing = reg.get_testing()

    width = 80
    print("=" * width)
    print(f"FEATURE REGISTRY AUDIT  —  {datetime.now(timezone.utc).date()}")
    print("=" * width)

    # ── LIVE features ─────────────────────────────────────────────── #
    print(f"\n{'LIVE FEATURES':─<{width}}")
    if not live:
        print("  (none)")
    else:
        for name, rec in sorted(live.items()):
            ic_str = _fmt(rec.get("ic_oos"), ".3f")
            mc_str = _fmt(rec.get("marginal_contribution"), "+.3f")
            n_str  = str(rec.get("sample_size", "?"))
            hyp    = rec.get("hypothesis_id", "")
            last   = _date_age(rec.get("last_validated_date", ""))
            is_stale = name in stale
            stale_flag = "  ⚠ STALE" if is_stale else ""

            print(f"\n  {name}  [{hyp}]{stale_flag}")
            print(f"    IC_OOS         : {ic_str}")
            print(f"    marginal_contrib: {mc_str}")
            print(f"    sample_size    : {n_str}")
            print(f"    last_validated : {last}")
            note = rec.get("note", "")
            if note:
                # wrap at 70 chars
                words = note.split()
                line, lines = [], []
                for w in words:
                    if len(" ".join(line + [w])) > 68:
                        lines.append("    " + " ".join(line))
                        line = [w]
                    else:
                        line.append(w)
                if line:
                    lines.append("    " + " ".join(line))
                print(f"    note           :")
                for l in lines:
                    print(f"  {l}")

    # ── Stale summary ─────────────────────────────────────────────── #
    if stale:
        print(f"\n{'STALE LIVE FEATURES  (>{days} days since last validation)':─<{width}}")
        for name in sorted(stale):
            print(f"  ⚠  {name}  — last validated: "
                  f"{_date_age(stale[name].get('last_validated_date', ''))}")
        print(f"\n  Action required: re-validate or graveyard these features.")

    # ── GRAVEYARD ─────────────────────────────────────────────────── #
    if show_all and graveyard:
        print(f"\n{'GRAVEYARD':─<{width}}")
        for name, rec in sorted(graveyard.items()):
            hyp = rec.get("hypothesis_id", "")
            reason = rec.get("graveyard_reason", "")
            n_str  = str(rec.get("sample_size", "?"))
            print(f"\n  {name}  [{hyp}]")
            print(f"    sample_size: {n_str}")
            print(f"    reason     : {reason[:120]}")

    # ── TESTING ───────────────────────────────────────────────────── #
    if show_all and testing:
        print(f"\n{'TESTING  (in evaluation, not yet promoted)':─<{width}}")
        for name, rec in sorted(testing.items()):
            hyp = rec.get("hypothesis_id", "")
            n_str = str(rec.get("sample_size", "?"))
            print(f"  {name}  [{hyp}]  n={n_str}")

    # ── Summary ───────────────────────────────────────────────────── #
    print(f"\n{'─' * width}")
    print(
        f"  LIVE: {len(live)}   STALE: {len(stale)}   "
        f"GRAVEYARD: {len(graveyard)}   TESTING: {len(testing)}"
    )
    if stale:
        print(f"\n  ⚠  {len(stale)} LIVE feature(s) need re-validation within {days} days.")
    else:
        print(f"\n  ✓  All LIVE features re-validated within {days} days.")
    print("=" * width)

    return len(stale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature registry audit")
    parser.add_argument(
        "--days", type=int, default=STALE_DAYS,
        help=f"Stale threshold in days (default: {STALE_DAYS})",
    )
    parser.add_argument(
        "--all", dest="show_all", action="store_true",
        help="Also show GRAVEYARD and TESTING entries",
    )
    parser.add_argument(
        "--seed", action="store_true",
        help="Seed the registry with pd_alignment graveyard entry",
    )
    args = parser.parse_args()

    if args.seed:
        reg = FeatureRegistry()
        reg.seed_pd_alignment()
        print("Seeded pd_alignment → GRAVEYARD")

    stale_count = audit(days=args.days, show_all=args.show_all)
    sys.exit(1 if stale_count > 0 else 0)


if __name__ == "__main__":
    main()
