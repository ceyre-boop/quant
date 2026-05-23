#!/bin/bash
# SOVEREIGN TradingView Tunnel
# Exposes localhost:8765 to the internet so TradingView webhooks can reach it.
#
# Requires: cloudflared (installed below if missing)
# Run: bash scripts/start_tunnel.sh
#
# The public URL printed here is what you paste into TradingView alert webhooks.
# TradingView alert message format (paste into "Message" box):
#   {"action":"buy","ticker":"{{ticker}}","price":{{close}},"strategy":"{{strategy_title}}"}
#
# Webhook URL for TradingView: <public-url>/webhook/tradingview

set -e

PORT=8765

# ── Install cloudflared if missing ─────────────────────────────────────────
if ! command -v cloudflared &>/dev/null; then
  echo "Installing cloudflared..."
  if command -v brew &>/dev/null; then
    brew install cloudflared
  else
    echo "ERROR: brew not found. Install cloudflared manually:"
    echo "  https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    exit 1
  fi
fi

# ── Check live_signals_server is running ──────────────────────────────────
if ! curl -s "http://localhost:${PORT}/trades" | python3 -c "import sys,json; json.load(sys.stdin)" &>/dev/null; then
  echo ""
  echo "⚠  live_signals_server.py is not running on port ${PORT}."
  echo "   Start it first:"
  echo "     python3 scripts/live_signals_server.py"
  echo ""
fi

# ── Start tunnel ──────────────────────────────────────────────────────────
echo ""
echo "Starting cloudflared tunnel → localhost:${PORT}"
echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│  Copy the https://... URL below and paste it into TradingView   │"
echo "│  Alerts → Webhook URL field as:                                 │"
echo "│                                                                  │"
echo "│    https://<random>.trycloudflare.com/webhook/tradingview       │"
echo "│                                                                  │"
echo "│  TradingView alert Message (JSON):                               │"
echo '│    {"action":"{{strategy.order.action}}",                        │'
echo '│     "ticker":"{{ticker}}",                                       │'
echo '│     "price":{{close}},                                           │'
echo '│     "strategy":"{{strategy.title}}",                             │'
echo '│     "comment":"{{strategy.order.comment}}"}                      │'
echo "│                                                                  │"
echo "│  Test it from terminal:                                          │"
echo "│    curl -X POST <url>/webhook/tradingview \\                      │"
echo "│      -H 'Content-Type: application/json' \\                      │"
echo '│      -d '"'"'{"action":"buy","ticker":"EURUSD","price":1.08}'"'"'          │"
echo "│                                                                  │"
echo "│  Then check: python3 scripts/trades.py                          │"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""

cloudflared tunnel --url "http://localhost:${PORT}"
