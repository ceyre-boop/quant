#!/usr/bin/env python3
"""
build_standalone_dashboard.py — bake the prop-challenge dashboard into ONE
self-contained HTML file that opens with a double-click (file://), no server.

WHY THIS EXISTS
---------------
serve_dashboard.sh only works while a python http.server is running in a
terminal. When it isn't, the localhost URL is dead. This script instead reads
the SAME allowlisted JSON files, inlines their contents directly into the HTML,
and writes a portable file that is never "down" — the data is baked in.

SECURITY BOUNDARY (do NOT weaken)
---------------------------------
The repo root contains .env, API keys, and sealed holdouts. This script copies
NOTHING by directory glob. It reads ONLY the explicit allowlist below — the
exact same set serve_dashboard.sh stages — and hard-asserts that no secret-
looking content ever lands in the output before writing it.

Because the dashboard fetches `../data/...` and fetch() is blocked under
file://, we inject a tiny fetch shim BEFORE the app script. The shim resolves
those exact paths from an inlined data map. The dashboard's own fj/ft/fjl
helpers are left completely untouched — they just call the shimmed fetch.

Usage:  python3 scripts/build_standalone_dashboard.py
Output: dashboard/dashboard_live.html   (double-click to open)
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_HTML = REPO / "dashboard" / "index.html"
OUT_HTML = REPO / "dashboard" / "dashboard_live.html"

# ── Allowlist — IDENTICAL to serve_dashboard.sh. This is the security boundary.
#    Nothing not named here is ever read. Do not replace with a directory glob.
FILES = [
    "data/agent/oracle_briefing_morning.json",
    "data/agent/prop_account_balance.json",
    "data/agent/carry_paper_account.json",
    "data/agent/system_regime_state.json",
    "data/agent/petrules_gate_scan.json",
    "data/agent/prop_challenge_state.json",
    "data/agent/training_gate_status.json",
    "data/agent/daily_briefing.json",
    "data/briefing/scorecard_status.json",
    "data/proof/v015_manifest.json",
    "logs/training_log.jsonl",
    "data/execution/fills.json",
    "data/execution/fills.jsonl",
    "data/execution/heartbeat.json",
    "data/oracle/daily_digest.json",
    "data/oracle/loop_health_status.json",
    "data/oracle/market_briefings/latest.json",
]
# Small non-sensitive dirs with date-stamped filenames (bias_${date}.json).
DIRS = ["data/bias"]

# ── Secret guardrails. Belt-and-suspenders on top of the allowlist.
SECRET_NAME_RE = re.compile(r"(\.env|\.key$|secret|password|token|credential)", re.I)
# Content sniff: anything that looks like a real secret assignment/value.
SECRET_CONTENT_RE = re.compile(
    r"(api[_-]?key|secret|password|bearer|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})",
    re.I,
)


def _assert_not_secret(rel: str, text: str) -> None:
    """Refuse to inline anything whose NAME or CONTENT looks like a secret."""
    if SECRET_NAME_RE.search(Path(rel).name):
        sys.exit(f"ABORT: allowlisted path looks secret-named: {rel}")
    if SECRET_CONTENT_RE.search(text):
        sys.exit(f"ABORT: content of {rel} matches a secret pattern — not inlining.")


def collect() -> dict[str, str]:
    """Read the allowlist into a {served_path: raw_text} map. Raw text so the
    dashboard's JSON.parse / line-splitting behaves exactly as over HTTP."""
    data: dict[str, str] = {}
    for rel in FILES:
        src = REPO / rel
        if src.is_file():
            text = src.read_text(encoding="utf-8")
            _assert_not_secret(rel, text)
            data[rel] = text
            print(f"  + {rel}")
        else:
            print(f"  ~ {rel} (missing — dashboard shows 'no data' for it)")

    for d in DIRS:
        src = REPO / d
        if not src.is_dir():
            print(f"  ~ {d} (missing)")
            continue
        for f in sorted(src.glob("*.json")):  # ONLY *.json, never anything else
            rel = f"{d}/{f.name}"
            text = f.read_text(encoding="utf-8")
            _assert_not_secret(rel, text)
            data[rel] = text
            print(f"  + {rel}")
    return data


def build_shim(data: dict[str, str]) -> str:
    """A fetch() shim that resolves `../data/...` from the inlined map. Falls
    back to the newest bias file when today's date-stamped one is absent (the
    view date can differ from the build date)."""
    payload = json.dumps(data, ensure_ascii=False)
    bias_keys = sorted(k for k in data if re.match(r"data/bias/bias_.*\.json$", k))
    newest_bias = bias_keys[-1] if bias_keys else ""
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    return f"""<script>
/* ===== STANDALONE DATA (baked {built}) — injected by build_standalone_dashboard.py =====
   All dashboard data is inlined below so the page works offline via file://
   with NO server. The shim below intercepts the dashboard's fetch('../data/..')
   calls and serves this map. Nothing else is touched. */
(function () {{
  const DATA = {payload};
  const NEWEST_BIAS = {json.dumps(newest_bias)};
  const norm = (u) => String(u).replace(/^(\\.\\.\\/)+/, "").replace(/^\\/+/, "");
  function lookup(path) {{
    if (path in DATA) return DATA[path];
    // Date-stamped bias file requested but not baked for that day → newest baked.
    if (/^data\\/bias\\/bias_.*\\.json$/.test(path) && NEWEST_BIAS) return DATA[NEWEST_BIAS];
    return null;
  }}
  window.__STANDALONE__ = true;
  window.fetch = function (url) {{
    const body = lookup(norm(url));
    if (body == null) return Promise.resolve({{ ok: false, status: 404,
      json: () => Promise.reject(new Error("404")), text: () => Promise.resolve("") }});
    return Promise.resolve({{
      ok: true, status: 200,
      json: () => Promise.resolve(JSON.parse(body)),
      text: () => Promise.resolve(body),
    }});
  }};
}})();
</script>
"""


def main() -> None:
    if not SRC_HTML.is_file():
        sys.exit(f"ABORT: source dashboard not found: {SRC_HTML}")
    print(f"Reading allowlist ({len(FILES)} files + {DIRS}) ...")
    data = collect()

    html = SRC_HTML.read_text(encoding="utf-8")
    shim = build_shim(data)

    # Inject the shim immediately before the app's <script> block.
    marker = "<script>"
    idx = html.find(marker)
    if idx == -1:
        sys.exit("ABORT: could not find <script> block to inject before.")
    html = html[:idx] + shim + "\n" + html[idx:]

    # Neutralize the file:// fetch-warning banner — fetch works now via the shim.
    html = html.replace(
        "if (location.protocol === 'file:')\n"
        "  document.getElementById('file-warn').style.display = 'block';",
        "/* file:// warning suppressed — standalone build inlines all data */",
    )

    # ── FINAL hard assertion: no secret pattern survives into the output.
    hits = SECRET_CONTENT_RE.findall(html)
    if hits:
        sys.exit(f"ABORT: secret-looking content in generated HTML: {set(hits)!r}. "
                 "Not writing output.")
    if re.search(r"\.env\b|BEGIN [A-Z ]*PRIVATE KEY", html):
        sys.exit("ABORT: .env or private-key marker in output. Not writing.")

    OUT_HTML.write_text(html, encoding="utf-8")
    kb = OUT_HTML.stat().st_size / 1024
    print(f"\nWrote {OUT_HTML}  ({kb:.0f} KB, {len(data)} data files inlined)")
    print(f"Open (double-click or):  file://{OUT_HTML}")


if __name__ == "__main__":
    main()
