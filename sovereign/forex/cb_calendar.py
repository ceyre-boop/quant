"""
Central bank meeting calendar for major forex central banks.

Uses the policy decision / announcement date for each meeting. For two-day
meetings, that is the second (final) day.

Verified against official central bank calendars and policy schedule pages on
2026-04-24.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List

CB_MEETINGS: Dict[str, List[date]] = {
    'FED': [
        date(2025, 1, 29),
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 10, 29),
        date(2025, 12, 10),
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 10, 28),
        date(2026, 12, 9),
    ],
    'ECB': [
        date(2025, 1, 30),
        date(2025, 3, 6),
        date(2025, 4, 17),
        date(2025, 6, 5),
        date(2025, 7, 24),
        date(2025, 9, 11),
        date(2025, 10, 30),
        date(2025, 12, 18),
        date(2026, 2, 5),
        date(2026, 3, 19),
        date(2026, 4, 30),
        date(2026, 6, 11),
        date(2026, 7, 23),
        date(2026, 9, 10),
        date(2026, 10, 29),
        date(2026, 12, 17),
    ],
    'BOJ': [
        date(2025, 1, 24),
        date(2025, 3, 19),
        date(2025, 5, 1),
        date(2025, 6, 17),
        date(2025, 7, 31),
        date(2025, 9, 19),
        date(2025, 10, 30),
        date(2025, 12, 19),
        date(2026, 1, 23),
        date(2026, 3, 19),
        date(2026, 4, 28),
        date(2026, 6, 16),
        date(2026, 7, 31),
        date(2026, 9, 18),
        date(2026, 10, 30),
        date(2026, 12, 18),
    ],
    'BOE': [
        date(2025, 2, 6),
        date(2025, 3, 20),
        date(2025, 5, 8),
        date(2025, 6, 19),
        date(2025, 8, 7),
        date(2025, 9, 18),
        date(2025, 11, 6),
        date(2025, 12, 18),
        date(2026, 2, 5),
        date(2026, 3, 19),
        date(2026, 4, 30),
        date(2026, 6, 18),
        date(2026, 7, 30),
        date(2026, 9, 17),
        date(2026, 11, 5),
        date(2026, 12, 17),
    ],
    'SNB': [
        date(2025, 3, 20),
        date(2025, 6, 19),
        date(2025, 9, 25),
        date(2025, 12, 11),
        date(2026, 3, 19),
        date(2026, 6, 18),
        date(2026, 9, 24),
        date(2026, 12, 10),
    ],
    'RBA': [
        date(2025, 2, 18),
        date(2025, 4, 1),
        date(2025, 5, 20),
        date(2025, 7, 8),
        date(2025, 8, 12),
        date(2025, 9, 30),
        date(2025, 11, 4),
        date(2025, 12, 9),
        date(2026, 2, 3),
        date(2026, 3, 17),
        date(2026, 5, 5),
        date(2026, 6, 16),
        date(2026, 8, 11),
        date(2026, 9, 29),
        date(2026, 11, 3),
        date(2026, 12, 8),
    ],
    'BOC': [
        date(2025, 1, 29),
        date(2025, 3, 12),
        date(2025, 4, 16),
        date(2025, 6, 4),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 10, 29),
        date(2025, 12, 10),
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 10),
        date(2026, 7, 15),
        date(2026, 9, 2),
        date(2026, 10, 28),
        date(2026, 12, 9),
    ],
    'RBNZ': [
        date(2025, 2, 19),
        date(2025, 4, 9),
        date(2025, 5, 28),
        date(2025, 7, 9),
        date(2025, 8, 20),
        date(2025, 10, 8),
        date(2025, 11, 26),
        date(2026, 2, 18),
        date(2026, 4, 8),
        date(2026, 5, 27),
        date(2026, 7, 8),
        date(2026, 9, 2),
        date(2026, 10, 28),
        date(2026, 12, 9),
    ],
}

__all__ = [
    'CB_MEETINGS',
    'get_days_to_next_meeting',
    'get_days_since_last_meeting',
    'is_in_blackout_period',
    'get_next_meeting',
]


def _normalize_bank(bank: str) -> str:
    bank_name = bank.strip().upper()
    if bank_name not in CB_MEETINGS:
        raise ValueError(f'Unknown central bank: {bank}')
    return bank_name


def _as_of_date(as_of: date | None) -> date:
    return as_of or date.today()


def get_next_meeting(bank: str, as_of: date | None = None) -> date | None:
    bank_name = _normalize_bank(bank)
    current_date = _as_of_date(as_of)
    for meeting_date in CB_MEETINGS[bank_name]:
        if meeting_date >= current_date:
            return meeting_date
    return None


def get_days_to_next_meeting(bank: str, as_of: date | None = None) -> int:
    current_date = _as_of_date(as_of)
    next_meeting = get_next_meeting(bank, current_date)
    if next_meeting is None:
        return 999
    return (next_meeting - current_date).days


def get_days_since_last_meeting(bank: str, as_of: date | None = None) -> int:
    bank_name = _normalize_bank(bank)
    current_date = _as_of_date(as_of)
    for meeting_date in reversed(CB_MEETINGS[bank_name]):
        if meeting_date < current_date:
            return (current_date - meeting_date).days
    return 999


def is_in_blackout_period(
    bank: str,
    as_of: date | None = None,
    blackout_days: int = 10,
) -> bool:
    days_to_next = get_days_to_next_meeting(bank, as_of)
    return 0 <= days_to_next <= blackout_days


if __name__ == '__main__':
    today = date.today()
    for bank_name in CB_MEETINGS:
        next_meeting = get_next_meeting(bank_name, today)
        print(f'{bank_name}: {next_meeting}')
