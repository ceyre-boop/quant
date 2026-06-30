#!/usr/bin/env python3
"""Render free-tier keep-alive ping.

The Sovereign dashboard (ceyre-boop.github.io/quant/) reads from a Render
free-tier web service (sovereign-quant-dashboard) that spins down after ~15min
of inactivity. The next request then pays a 30-60s cold start, during which
every backend-fed panel shows "Loading…".

This script pings the backend's /health endpoint every 10 minutes so the
service never idles long enough to sleep, eliminating the cold-start stall.

/health is an instant liveness probe (no yfinance/forex compute), so these
pings are cheap and won't flap.

Run as a long-lived launchd daemon (see com.alta.render_keepalive.plist) or
directly:  python3 scripts/keep_alive_render.py
"""

import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

API_BASE = "https://sovereign-quant-dashboard.onrender.com"
HEALTH_URL = f"{API_BASE}/health"
INTERVAL_SECONDS = 600  # 10 minutes — well under Render's ~15min idle timeout
REQUEST_TIMEOUT = 90    # generous: a cold start can take 30-60s to respond


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ping() -> bool:
    """Hit /health once. Return True on HTTP 200, False otherwise."""
    started = time.monotonic()
    try:
        req = urllib.request.Request(
            HEALTH_URL,
            headers={"User-Agent": "alta-render-keepalive/1.0"},
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            elapsed = time.monotonic() - started
            ok = resp.status == 200
            print(
                f"[{_now()}] {'OK ' if ok else 'WARN'} "
                f"{resp.status} {HEALTH_URL} ({elapsed:.1f}s)",
                flush=True,
            )
            return ok
    except urllib.error.HTTPError as e:
        elapsed = time.monotonic() - started
        print(
            f"[{_now()}] FAIL HTTP {e.code} {HEALTH_URL} ({elapsed:.1f}s)",
            flush=True,
        )
        return False
    except Exception as e:  # URLError, timeout, DNS, etc.
        elapsed = time.monotonic() - started
        print(
            f"[{_now()}] FAIL {type(e).__name__}: {e} {HEALTH_URL} ({elapsed:.1f}s)",
            flush=True,
        )
        return False


def main() -> int:
    print(
        f"[{_now()}] keep_alive_render started — pinging {HEALTH_URL} "
        f"every {INTERVAL_SECONDS}s",
        flush=True,
    )
    try:
        while True:
            ping()
            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print(f"[{_now()}] keep_alive_render stopped (KeyboardInterrupt)", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
