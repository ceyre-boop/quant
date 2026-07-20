"""
sovereign.brain — the Obsidian intelligence layer.

Bidirectional knowledge bridge between the trading agents and the Obsidian
vault at ~/Obsidian/Obsidian/. The READ side (obsidian_reader) loads relevant
long-term memory before an agent acts; the WRITE side (obsidian_writer) posts
structured knowledge back after it acts.

Isolation note: this package lives under sovereign/ and imports nothing from
ict/ or the live execution path. It is pure-stdlib and side-effect-free on
import. Every reader function degrades gracefully (empty structure, never a
crash) when the vault or a source file is missing.
"""

from . import obsidian_reader, obsidian_writer  # noqa: F401

__all__ = ["obsidian_reader", "obsidian_writer"]
