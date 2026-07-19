"""Obsidian vault writer — the sync that was specified and never built.

`~/Obsidian/Obsidian/Trading/Research/Oracle-Sync-Spec.md:5` records:

    status: spec (code not yet implemented — Phase E, staged for a reviewed pass)

and names the missing artifact as `sovereign/oracle/obsidian_sync.py`. It was
never written, which is why `Trading/Oracle-Log/` contains a README and a
template and **zero dated entries**, and why `Oracle-Context.md:32,35` still
carry the literal placeholders "_(seed — wrapper appends here each cycle)_".

This is that writer, built where the execution layer can use it rather than
inside the Oracle, so both the bias log (Layer 2) and the EOD reconciliation
(Layer 6) share one implementation.

CONVENTIONS FOLLOWED (not invented)
-----------------------------------
- Vault root constant matches `scripts/build_obsidian_graph.py:35`.
- Frontmatter shape matches `Trading/Oracle-Log/_TEMPLATE.md`: `date`, `title`,
  `type`, then a `> Links:` breadcrumb line.
- Append-per-run blocks (`## [HH:MM ET] LABEL`) rather than rewrite, matching the
  template's own instruction: "One block is appended per run. Do not edit."

SAFETY
------
- `dry_run=True` renders and returns the text without touching the vault. Every
  caller defaults to it in tests.
- Writes are confined to `Trading/` — `_safe_path()` raises on any attempt to
  escape, so a bad slug cannot write into the user's personal notes.
- Appends never truncate: if the file exists, new blocks go on the end.
- A missing vault is reported, never created. If the vault is not mounted (the
  scheduled-agent sandbox failure that blocked 8 consecutive runs), this fails
  loudly rather than silently writing nowhere.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VAULT = Path("/Users/taboost/Obsidian/Obsidian")
TRADING = VAULT / "Trading"
ET = ZoneInfo("America/New_York")


class VaultUnavailable(RuntimeError):
    """The Obsidian vault is not reachable. Raised, never swallowed."""


def vault_available() -> bool:
    return VAULT.exists() and TRADING.exists()


def _require_vault() -> None:
    if not vault_available():
        raise VaultUnavailable(
            f"Obsidian vault not reachable at {VAULT}. This is the failure mode "
            f"that blocked the 7 AM scheduled agent for 8 consecutive runs — the "
            f"sandbox mounted only part of the filesystem. Not writing anywhere."
        )


def _safe_path(relative: str) -> Path:
    """Resolve a path inside Trading/, refusing anything that escapes it."""
    p = (TRADING / relative).resolve()
    if not str(p).startswith(str(TRADING.resolve())):
        raise ValueError(f"refusing to write outside {TRADING}: {relative}")
    return p


def frontmatter(day: date | str, title: str, doc_type: str,
                links: list[str] | None = None) -> str:
    links = links or ["Alta Investments"]
    link_line = " · ".join(f"[[{l}]]" for l in links)
    return (
        "---\n"
        f"date: {day}\n"
        f"title: {title}\n"
        f"type: {doc_type}\n"
        "generated_by: execution/obsidian.py\n"
        "---\n\n"
        f"# {title}\n\n"
        f"> Auto-generated. Links: {link_line}\n"
    )


def write_note(relative: str, body: str, *, day: date | str, title: str,
               doc_type: str = "trading-log", links: list[str] | None = None,
               dry_run: bool = False) -> tuple[Path, str]:
    """Create or overwrite a note. Returns (path, rendered_text)."""
    text = frontmatter(day, title, doc_type, links) + "\n" + body.rstrip() + "\n"
    if dry_run:
        return _safe_path(relative), text
    _require_vault()
    p = _safe_path(relative)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p, text


def append_block(relative: str, block: str, *, day: date | str, title: str,
                 doc_type: str = "trading-log", links: list[str] | None = None,
                 dry_run: bool = False) -> tuple[Path, str]:
    """Append a block, creating the note with frontmatter if absent.

    Never truncates an existing note — matches the template's "one block appended
    per run, do not edit" convention.
    """
    if dry_run:
        return _safe_path(relative), block.rstrip() + "\n"
    _require_vault()
    p = _safe_path(relative)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(frontmatter(day, title, doc_type, links))
    with open(p, "a") as fh:
        fh.write("\n" + block.rstrip() + "\n")
    return p, block


def et_stamp() -> str:
    return datetime.now(ET).strftime("%H:%M")


# ── Callers ───────────────────────────────────────────────────────────────────

def write_bias_block(b, *, dry_run: bool = False) -> tuple[Path, str]:
    """Append one bias block to Trading/Oracle-Log/{date}.md.

    This is the file the Oracle-Sync-Spec promised and never delivered — the
    directory has held only a README and a template since it was created.
    """
    lines = [
        f"## [{et_stamp()} ET] DAILY BIAS",
        f"- **Direction:** {b.direction}",
        f"- **Confidence:** {b.confidence:.3f} "
        f"(scaled by context freshness {b.context_fraction_fresh:.0%})",
        f"- **Inputs used:** {', '.join(b.inputs_used) or 'none'}",
        f"- **Inputs missing:** {', '.join(b.inputs_missing) or 'none'}",
        "- **Reasoning:**",
    ]
    lines += [f"  - {r}" for r in b.reasoning]
    lines.append("- **Gating:** none — this bias is recorded and scored, it does "
                 "not filter signals (ARCHITECTURE.md L1/L2 wall).")
    if b.realised:
        lines.append(f"- **Realised:** {b.realised.get('direction')} "
                     f"(correct={b.realised.get('correct')})")
    return append_block(
        f"Oracle-Log/{b.date}.md", "\n".join(lines),
        day=b.date, title=f"Oracle Log — {b.date}",
        links=["Oracle-Context", "Alta Investments", "Oracle"], dry_run=dry_run)


def write_eod_note(day: date | str, body: str, *,
                   dry_run: bool = False) -> tuple[Path, str]:
    """Write the end-of-day system reconciliation to Trading/Ops/."""
    return write_note(
        f"Ops/System-EOD-{day}.md", body, day=day,
        title=f"System EOD — {day}", doc_type="trading-log",
        links=["Alta Investments", "Oracle-Context"], dry_run=dry_run)
