#!/usr/bin/env python3
"""
Cursus Honorum — Human Game Session Runner
games/cursus_honorum/session.py

Runs an interactive quiz session for Colin, tracks Elo, and broadcasts
results to Firebase RTDB (/mesolimbic/).

Usage:
    python3 games/cursus_honorum/session.py               # full 100-question run
    python3 games/cursus_honorum/session.py --n 20        # 20 random questions
    python3 games/cursus_honorum/session.py --cats EXITS CARRY  # specific categories
    python3 games/cursus_honorum/session.py --dry-run     # don't write to Firebase

Elo system:
    K = 32 (standard for active players)
    Expected score = 1 / (1 + 10^((opponent_elo - player_elo)/400))
    New Elo = old_elo + K * (actual - expected)
    "Opponent" for each question is the question's difficulty_elo (default 1200).

Firebase path: /mesolimbic/elo/colin and /mesolimbic/category_scores/colin
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUESTION_BANK = Path(__file__).parent / "question_bank.json"
CATEGORIES_FILE = Path(__file__).parent / "categories.json"

# Elo config
K_FACTOR = 32
DEFAULT_QUESTION_ELO = 1200   # treat each question as a 1200-rated opponent
STARTING_ELO = 1200

# 8 canonical mesolimbic categories (maps from the 15-bucket question tags)
CANONICAL_CATEGORIES = [
    "EXPECTANCY", "SIZING", "CARRY", "KELLY",
    "SHARPE", "EXITS", "RECOVERY", "TRUST",
]

# Map question category prefix → canonical bucket
CATEGORY_MAP = {
    "CAL": "EXPECTANCY",    # Calibration / general expectancy
    "SIZ": "SIZING",
    "CAR": "CARRY",
    "KEL": "KELLY",
    "SHA": "SHARPE",
    "EXI": "EXITS",
    "REC": "RECOVERY",
    "TRU": "TRUST",
    "DIAG": "EXPECTANCY",   # Diagnostics → expectancy bucket
    "PORT": "SIZING",
    "PHIL": "TRUST",
    "ARCH": "RECOVERY",
    "RISK": "SIZING",
}


def _load_questions() -> list[dict]:
    with open(QUESTION_BANK) as f:
        data = json.load(f)
    # Support both list and dict-of-dicts formats
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    # dict keyed by question_id
    return [{"id": k, **v} for k, v in data.items()]


def _get_bucket(q: dict) -> str:
    """Map a question to one of the 8 canonical categories."""
    cat = q.get("category", q.get("id", "CAL"))
    prefix = cat.split("-")[0].upper()
    return CATEGORY_MAP.get(prefix, "EXPECTANCY")


def _elo_expected(player: int, opponent: int) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent - player) / 400.0))


def _elo_update(player: int, score: float, expected: float) -> int:
    return round(player + K_FACTOR * (score - expected))


def _get_current_elo(player: str = "colin") -> int:
    """Read current Elo from Firebase, fall back to file cache, then default."""
    try:
        from integration.firebase_state_writer import _rtdb_ref
        ref = _rtdb_ref(f"mesolimbic/elo/{player}")
        if ref:
            data = ref.get()
            if data and "current" in data:
                return int(data["current"])
    except Exception:
        pass

    # File cache fallback
    cache = ROOT / "data" / "mesolimbic" / f"elo_{player}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text()).get("elo", STARTING_ELO)
        except Exception:
            pass

    return STARTING_ELO


def _save_elo_local(player: str, elo: int) -> None:
    """Save Elo to local file as backup."""
    cache = ROOT / "data" / "mesolimbic"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"elo_{player}.json").write_text(json.dumps({
        "player": player,
        "elo": elo,
        "updated": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


def run_session(
    questions: list[dict],
    player: str = "colin",
    dry_run: bool = False,
) -> dict:
    """
    Run an interactive quiz session and return results.
    Writes to Firebase and local cache unless dry_run=True.
    """
    if not questions:
        print("No questions to run.")
        return {}

    elo_before = _get_current_elo(player)
    elo = elo_before
    correct = 0
    total = len(questions)
    category_results: dict[str, dict] = {c: {"correct": 0, "total": 0} for c in CANONICAL_CATEGORIES}

    print(f"\n{'='*60}")
    print(f"  Cursus Honorum — {player.title()} | Elo {elo_before}")
    print(f"  {total} questions | K={K_FACTOR}")
    print(f"{'='*60}\n")

    for i, q in enumerate(questions, 1):
        bucket = _get_bucket(q)
        category_results[bucket]["total"] += 1

        print(f"Q{i}/{total}  [{bucket}]  {q.get('id', '')}")
        print(f"  {q.get('question', q.get('text', ''))}\n")

        choices = q.get("choices", q.get("options", {}))
        if isinstance(choices, dict):
            for letter, text in sorted(choices.items()):
                print(f"    {letter}) {text}")
        elif isinstance(choices, list):
            for j, text in enumerate(choices):
                letter = chr(65 + j)
                print(f"    {letter}) {text}")

        answer = input("\n  Your answer (A/B/C/D or ? to reveal): ").strip().upper()

        correct_answer = q.get("correct", q.get("answer", "")).strip().upper()
        revealed = answer == "?"

        if revealed:
            print(f"  → Correct answer: {correct_answer}")
            print(f"  → {q.get('explanation', '')}")
            # Treat reveal as wrong
            result = 0.0
        elif answer == correct_answer:
            print("  ✓ Correct!")
            if q.get("explanation"):
                print(f"  {q['explanation']}")
            result = 1.0
            correct += 1
            category_results[bucket]["correct"] += 1
        else:
            print(f"  ✗ Wrong. Correct: {correct_answer}")
            print(f"  → {q.get('explanation', '')}")
            result = 0.0

        # Elo update per question
        expected = _elo_expected(elo, DEFAULT_QUESTION_ELO)
        elo = _elo_update(elo, result, expected)
        print(f"  Elo: {elo_before} → {elo}  ({'+' if elo-elo_before >= 0 else ''}{elo - elo_before})\n")

    # Session summary
    pct = correct / total * 100 if total else 0
    elo_delta = elo - elo_before

    print(f"\n{'='*60}")
    print(f"  Session Complete")
    print(f"  Score:   {correct}/{total}  ({pct:.0f}%)")
    print(f"  Elo:     {elo_before} → {elo}  ({'+' if elo_delta >= 0 else ''}{elo_delta})")
    print(f"\n  Category breakdown:")
    for cat, res in category_results.items():
        if res["total"] > 0:
            cat_pct = res["correct"] / res["total"] * 100
            bar = "█" * int(cat_pct / 10) + "░" * (10 - int(cat_pct / 10))
            print(f"    {cat:<12} {bar}  {res['correct']}/{res['total']}  ({cat_pct:.0f}%)")
    print(f"{'='*60}\n")

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results = {
        "player": player,
        "score": correct,
        "total": total,
        "pct": round(pct, 1),
        "elo_before": elo_before,
        "elo_after": elo,
        "elo_delta": elo_delta,
        "category_results": category_results,
        "session_id": session_id,
    }

    if dry_run:
        print("  [dry-run] Firebase write skipped.")
        return results

    # Write to Firebase
    try:
        from integration.firebase_state_writer import broadcast_mesolimbic_session
        broadcast_mesolimbic_session(
            player=player,
            score=correct,
            total=total,
            elo_before=elo_before,
            elo_after=elo,
            category_results=category_results,
            session_id=session_id,
        )
        print("  Firebase updated ✓")
    except Exception as exc:
        print(f"  Firebase write failed (non-fatal): {exc}")

    # Local backup
    _save_elo_local(player, elo)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Cursus Honorum — human game session")
    parser.add_argument("--n", type=int, default=None, help="Number of random questions (default: all)")
    parser.add_argument("--cats", nargs="+", choices=CANONICAL_CATEGORIES, help="Filter by categories")
    parser.add_argument("--player", default="colin", help="Player name (default: colin)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Firebase")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle questions (default: True)")
    args = parser.parse_args()

    all_questions = _load_questions()

    # Filter by category
    if args.cats:
        all_questions = [q for q in all_questions if _get_bucket(q) in args.cats]
        print(f"Filtered to {len(all_questions)} questions in {args.cats}")

    # Shuffle and limit
    if args.shuffle:
        random.shuffle(all_questions)
    if args.n:
        all_questions = all_questions[:args.n]

    run_session(all_questions, player=args.player, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
