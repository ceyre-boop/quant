#!/usr/bin/env bash
#
# serve_dashboard.sh — serve the prop-challenge dashboard SAFELY.
#
# WHY THIS EXISTS: the repo root contains .env, API keys, and sealed holdouts
# (data/research/gapper/holdout/, data/research/yield_frontier/gauntlet/).
# Running `python3 -m http.server` from ~/quant would expose ALL of it over HTTP.
# Instead we copy ONLY the dashboard HTML and the specific JSON files it fetches
# into a throwaway staging dir, and serve that — bound to loopback only.
#
# The dashboard (dashboard/index.html) sets `const BASE = '..'`, i.e. it fetches
# `../data/...`. So the HTML must live in a SUBDIR of the served root, with a
# sibling data/ tree. We reproduce exactly that layout under STAGE.
#
# Usage:  scripts/serve_dashboard.sh
# Then open:  http://127.0.0.1:8080/dashboard/
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="/tmp/quant-dashboard"
PORT="8080"

# Explicit allowlist. NOTHING is copied that is not named here — this is the
# security boundary. Do not replace with a blanket `cp -r data/`.
FILES=(
  "dashboard/index.html"
  "data/agent/oracle_briefing_morning.json"
  "data/agent/prop_account_balance.json"
  "data/agent/carry_paper_account.json"
  "data/agent/system_regime_state.json"
  "data/agent/petrules_gate_scan.json"
  "data/agent/prop_challenge_state.json"
  "data/agent/training_gate_status.json"
  "data/proof/v015_manifest.json"
  "logs/training_log.jsonl"
  "data/execution/fills.json"
  "data/execution/fills.jsonl"
  "data/execution/heartbeat.json"
  "data/oracle/daily_digest.json"
  "data/oracle/loop_health_status.json"
  "data/oracle/market_briefings/latest.json"
)
# Small, non-sensitive dirs with date-stamped filenames the dashboard builds at
# runtime (bias_${today}.json etc.). Copied whole; they hold only bias records.
DIRS=(
  "data/bias"
)

echo "Staging dashboard into ${STAGE} (allowlist copy) ..."
rm -rf "${STAGE}"
mkdir -p "${STAGE}"

for rel in "${FILES[@]}"; do
  src="${REPO}/${rel}"
  dst="${STAGE}/${rel}"
  mkdir -p "$(dirname "${dst}")"
  if [[ -f "${src}" ]]; then
    cp "${src}" "${dst}"
    echo "  + ${rel}"
  else
    echo "  ~ ${rel} (missing at source — dashboard will show 'no data' for it)"
  fi
done

for rel in "${DIRS[@]}"; do
  src="${REPO}/${rel}"
  dst="${STAGE}/${rel}"
  if [[ -d "${src}" ]]; then
    mkdir -p "${dst}"
    # Copy only *.json — never anything else that might land in the dir.
    find "${src}" -maxdepth 1 -type f -name '*.json' -exec cp {} "${dst}/" \;
    echo "  + ${rel}/*.json"
  else
    echo "  ~ ${rel} (missing at source)"
  fi
done

# Hard safety assertion: no secrets may exist under the served root.
if find "${STAGE}" -iname '.env' -o -iname '*.key' -o -iname '*secret*' | grep -q .; then
  echo "ABORT: secret-looking file found under ${STAGE}. Not serving." >&2
  exit 1
fi

echo
echo "Served root: ${STAGE}  (bound to 127.0.0.1 only)"
echo "Open:        http://127.0.0.1:${PORT}/dashboard/"
echo "Contents:"
find "${STAGE}" -type f | sed "s#${STAGE}/#  #"
echo
exec python3 -m http.server "${PORT}" --bind 127.0.0.1 --directory "${STAGE}"
