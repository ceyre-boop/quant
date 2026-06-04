"""Risk-engine config loader (mirrors config/loader.py: @lru_cache + yaml.safe_load).

ALL tunable numbers live in risk_config.yaml. Code reads them through here — zero magic numbers.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "risk_config.yaml"


@lru_cache(maxsize=1)
def load_risk_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"risk_config.yaml missing at {_CONFIG_PATH} — refusing to size on defaults.")
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("risk_config.yaml is empty — refusing to size on an empty config.")
    return cfg


def config_path() -> Path:
    return _CONFIG_PATH
