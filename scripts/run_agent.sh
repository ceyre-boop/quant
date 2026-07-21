#!/bin/bash
#
# run_agent.sh — single entry point for the three Claude-driven agent jobs
# (morning, EOD, research). Replaces three plists that each hardcoded their own
# path to the `claude` binary and their own .env handling.
#
# WHY THIS EXISTS
# ---------------
# 1. PATH DRIFT. The three agent plists hardcoded a `claude` path each, and they
#    drifted apart. As of 2026-07-21 the installed morning_agent plist pointed at
#    /usr/local/bin/claude, which does not exist — line 1 of that day's log is
#    "/bin/sh: /usr/local/bin/claude: No such file or directory" — while eod_agent
#    and research_agent invoked claude successfully from somewhere else. A whole
#    week of triage was spent on "the claude binary is missing" when the binary
#    was present and one plist was pointing at the wrong place. Resolving at
#    runtime, in one file, makes that class of bug impossible to repeat.
#
# 2. AUTH LEAK. All three plists did `set -a; . .env; set +a`, which exports
#    ANTHROPIC_API_KEY. The Claude CLI then prefers the API key over the
#    subscription login — every agent run billed the API, and the CLI logged
#    "claude.ai connectors are disabled because ANTHROPIC_API_KEY ... takes
#    precedence over your claude.ai login". This script loads .env for the
#    trading credentials the routines need (OANDA, FRED, etc.) and then
#    explicitly unsets the Anthropic auth vars so the CLI uses the subscription.
#    Override with ALTA_AGENT_USE_API_KEY=1 if you ever genuinely want API auth.
#
# 3. FAIL LOUD. If claude cannot be found, this exits non-zero with the list of
#    paths tried, rather than emitting a shell error into a log nobody reads and
#    leaving downstream jobs to infer a "zero-signal day" from the silence.
#
# Usage:  scripts/run_agent.sh <morning|eod|research>
#         scripts/run_agent.sh --which     # print resolved claude path and exit

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

log() { printf '[run_agent %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# --------------------------------------------------------------------------
# 1. Resolve the claude binary. Ordered: explicit override, then PATH, then the
#    known install locations. First hit that is executable wins.
# --------------------------------------------------------------------------
resolve_claude() {
  if [[ -n "${CLAUDE_BIN:-}" ]]; then
    [[ -x "$CLAUDE_BIN" ]] && { echo "$CLAUDE_BIN"; return 0; }
    log "CLAUDE_BIN is set to '$CLAUDE_BIN' but that is not executable — ignoring it"
  fi

  local candidates=(
    "$(command -v claude 2>/dev/null || true)"
    "$HOME/.local/bin/claude"
    "$HOME/.claude/local/claude"
    "/opt/homebrew/bin/claude"
    "/usr/local/bin/claude"
    "$HOME/.bun/bin/claude"
    "$HOME/.volta/bin/claude"
    "/opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js"
  )
  local c
  for c in "${candidates[@]}"; do
    [[ -n "$c" && -x "$c" ]] && { echo "$c"; return 0; }
  done

  # Last resort: npm global bin, which is slow to query so it is checked last.
  local npmbin
  npmbin="$(npm bin -g 2>/dev/null || true)"
  if [[ -n "$npmbin" && -x "$npmbin/claude" ]]; then echo "$npmbin/claude"; return 0; fi

  return 1
}

CLAUDE="$(resolve_claude)" || {
  log "FATAL: no executable 'claude' found. Tried PATH plus:"
  log "  \$CLAUDE_BIN, ~/.local/bin, ~/.claude/local, /opt/homebrew/bin,"
  log "  /usr/local/bin, ~/.bun/bin, ~/.volta/bin, npm global bin"
  log "Set CLAUDE_BIN in the environment or install the CLI. NOT running the routine."
  exit 127
}

if [[ "${1:-}" == "--which" ]]; then
  echo "$CLAUDE"
  exit 0
fi

# --------------------------------------------------------------------------
# 2. Load .env for trading credentials, then strip Anthropic auth so the CLI
#    uses the subscription login rather than billing the API.
# --------------------------------------------------------------------------
if [[ -f "$REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO/.env"
  set +a
else
  log "WARNING: no .env at $REPO/.env — routines needing OANDA/FRED credentials will fail loudly"
fi

if [[ "${ALTA_AGENT_USE_API_KEY:-0}" != "1" ]]; then
  unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
fi

# --------------------------------------------------------------------------
# 3. Dispatch.
# --------------------------------------------------------------------------
ROUTINE="${1:-}"
case "$ROUTINE" in
  morning)  PROMPT="Read ~/quant/AGENT_DIRECTIVE.md and execute the 08:00 morning routine" ;;
  eod)      PROMPT="Read ~/quant/AGENT_DIRECTIVE.md and execute the 16:05 EOD routine" ;;
  research) PROMPT="Read ~/quant/AGENT_DIRECTIVE.md and execute the 21:00 research routine" ;;
  *)
    log "FATAL: unknown routine '${ROUTINE}'. Expected one of: morning, eod, research"
    exit 2
    ;;
esac

log "routine=$ROUTINE claude=$CLAUDE api_key_auth=${ALTA_AGENT_USE_API_KEY:-0}"

"$CLAUDE" --print "$PROMPT" \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep" 2>&1
rc=$?

log "routine=$ROUTINE exited rc=$rc"
exit $rc
