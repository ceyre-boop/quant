"""
config/loader.py — Single cached loader for parameters.yml
All modules import `params` from here. Never read the YAML directly.
"""

import yaml
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def load_params() -> dict:
    path = Path(__file__).parent / "parameters.yml"
    with open(path) as f:
        return yaml.safe_load(f)


# Module-level singleton — import this everywhere
params = load_params()
