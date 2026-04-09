#!/usr/bin/env python3
"""
Fix hardcoded $100k equity values across all files.
Replaces hardcoded 100000.0 with config-driven values.

ARCHIVED: One-off migration script for Windows development environment.
Not intended to be run in production. Preserved for historical reference only.
"""

import os
import re

BASE = r"C:\Users\Admin\clawd\quant"


def patch_file(filepath, replacements):
    """Apply replacements to a file."""
    full_path = os.path.join(BASE, filepath)

    if not os.path.exists(full_path):
        print(f"  [SKIP] {filepath} - not found")
        return

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(full_path, "r", encoding="latin-1") as f:
            content = f.read()

    original = content

    for old, new in replacements:
        content = content.replace(old, new)

    if content != original:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [FIXED] {filepath}")
    else:
        print(f"  [OK] {filepath} - no changes needed")


# Files to patch and their replacements
patches = {
    "dashboard_api.py": [
        (
            "from execution.paper_trading import PaperTradingEngine\n",
            "from execution.paper_trading import PaperTradingEngine\nfrom config.settings import get_starting_equity\n",
        ),
        (
            "paper_trading = PaperTradingEngine(starting_equity=100000.0)",
            "paper_trading = PaperTradingEngine(starting_equity=get_starting_equity())",
        ),
    ],
    "realtime_publisher.py": [
        (
            "from execution.paper_trading import PaperTradingEngine\n",
            "from execution.paper_trading import PaperTradingEngine\nfrom config.settings import get_starting_equity\n",
        ),
        (
            "self.paper = PaperTradingEngine(starting_equity=100000.0)",
            "self.paper = PaperTradingEngine(starting_equity=get_starting_equity())",
        ),
    ],
    "run_engine_live.py": [
        (
            "from execution.paper_trading import PaperTradingEngine\n",
            "from execution.paper_trading import PaperTradingEngine\nfrom config.settings import get_starting_equity\n",
        ),
        (
            "self.paper_trading = PaperTradingEngine(starting_equity=100000.0)",
            "self.paper_trading = PaperTradingEngine(starting_equity=get_starting_equity())",
        ),
    ],
    "run_engine_production.py": [
        (
            "from execution.paper_trading import PaperTradingEngine\n",
            "from execution.paper_trading import PaperTradingEngine\nfrom config.settings import get_starting_equity\n",
        ),
        (
            'equity=account_data.get("equity", 100000)',
            'equity=account_data.get("equity", get_starting_equity())',
        ),
        ("equity=100000,", "equity=get_starting_equity(),"),
    ],
    "run_engine_v2_real_data.py": [
        (
            "from execution.paper_trading import PaperTradingEngine\n",
            "from execution.paper_trading import PaperTradingEngine\nfrom config.settings import get_starting_equity\n",
        ),
        (
            "self.paper_trading = PaperTradingEngine(starting_equity=100000.0)",
            "self.paper_trading = PaperTradingEngine(starting_equity=get_starting_equity())",
        ),
    ],
    "simple_publisher.py": [
        (
            "from execution.paper_trading import PaperTradingEngine\n",
            "from execution.paper_trading import PaperTradingEngine\nfrom config.settings import get_starting_equity\n",
        ),
        (
            "self.paper = PaperTradingEngine(starting_equity=100000.0)",
            "self.paper = PaperTradingEngine(starting_equity=get_starting_equity())",
        ),
    ],
    "verify_data_flow.py": [
        (
            "from contracts.types import AccountState\n",
            "from contracts.types import AccountState\nfrom config.settings import get_starting_equity\n",
        ),
        ("equity=100000,", "equity=get_starting_equity(),"),
        ("balance=100000,", "balance=get_starting_equity(),"),
        ("margin_available=100000,", "margin_available=get_starting_equity(),"),
    ],
}

print("Fixing hardcoded $100k equity values...")
print("=" * 60)

for filepath, replacements in patches.items():
    patch_file(filepath, replacements)

print("=" * 60)
print("Done! Equity is now config-driven from .env STARTING_EQUITY")
print("Change the value in .env to update across all files.")
