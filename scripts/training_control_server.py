#!/usr/bin/env python3
"""
training_control_server.py — localhost-only control server for the self-play
training dashboard buttons (Run --watch / Kill / Undo last cycle / Restore).

SECURITY MODEL (read this before touching anything):
  - Binds 127.0.0.1 ONLY. Never 0.0.0.0. Never configurable via request input.
  - Exposes a FIXED allowlist of exactly four mutating actions
    (start_watch, kill, undo_last_cycle, restore_last_cycle) plus one
    read-only status endpoint. There is no generic "run command" route, no
    shell passthrough, and no query parameter is ever interpolated into a
    shell string or file path.
  - The training subprocess is launched with a HARD-CODED argv list
    (`[sys.executable, "scripts/sovereign_train.py", "--watch"]`),
    `shell=False`. No request data reaches the subprocess invocation.
  - This server NEVER imports or calls anything from sovereign/training/gate.py
    that could write config/training.yml, NEVER writes that file itself, and
    NEVER imports any execution-path module (forex_exit_manager, decide_exit,
    exit_machine, ict.pipeline, carry_engine, harness). grep the repo for
    "training_control_server" if you need to confirm nothing else calls in.
  - Even if the ignition gate were OPEN, this server does not change what the
    training runner does — sovereign_train.py's own gate + placebo + director
    human-approval checks are the only thing that can ever commit a cycle.
    This server can start/stop that process and move between snapshots it
    already wrote; it cannot make the process do anything it wouldn't already
    do when run from a terminal.
  - Serves no repo files. There is no static file handler and no directory
    listing. GET /api/status embeds the tail of the training run's own
    stdout log (a file this server wrote) as a JSON string list — that is
    reading the server's own output, not exposing the filesystem.

Usage:
    python3 scripts/training_control_server.py            # binds 127.0.0.1:8787
    python3 scripts/training_control_server.py --port 0    # ephemeral port (tests)

Endpoints (all on 127.0.0.1):
    POST /api/start_watch       -> launch sovereign_train.py --watch
    POST /api/kill               -> terminate the running training subprocess
    POST /api/undo_last_cycle    -> sovereign.training.snapshots.undo_last_cycle()
    POST /api/restore_last_cycle -> sovereign.training.snapshots.restore_last_cycle()
    GET  /api/status             -> current run status + snapshot history (read-only)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.training import snapshots as snapshots_mod  # noqa: E402

STATUS_PATH = ROOT / "data" / "agent" / "training_run_status.json"
LOG_PATH = ROOT / "logs" / "training_run_output.log"

# The one and only command this server is ever allowed to launch. Hard-coded,
# no substitution, no user-controllable arguments.
TRAIN_CMD = [sys.executable, "scripts/sovereign_train.py", "--watch"]

_lock = threading.Lock()
_proc: subprocess.Popen | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(**fields) -> dict:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if STATUS_PATH.exists():
        try:
            payload = json.loads(STATUS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            payload = {}
    payload.update(fields)
    payload["updated_at"] = _now()
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _read_status() -> dict:
    if not STATUS_PATH.exists():
        return {"running": False, "pid": None, "note": "no run yet"}
    try:
        return json.loads(STATUS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"running": False, "pid": None, "note": "status file unreadable"}


def _watch_process(proc: subprocess.Popen) -> None:
    """Background thread: block on the subprocess, then flip running=false."""
    exit_code = proc.wait()
    with _lock:
        global _proc
        if _proc is proc:
            _proc = None
    _write_status(running=False, exit_code=exit_code, finished_at=_now())


def action_start_watch() -> tuple[int, dict]:
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return 409, {"ok": False, "message": "a training run is already active"}

        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(LOG_PATH, "w")  # noqa: SIM115 — lives for subprocess lifetime
        proc = subprocess.Popen(
            TRAIN_CMD,
            cwd=str(ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            shell=False,
        )
        _proc = proc

    _write_status(
        running=True, pid=proc.pid, started_at=_now(),
        exit_code=None, finished_at=None, killed=False,
        log_path=str(LOG_PATH.relative_to(ROOT)), mode="DRY-SCAFFOLD (ignition gate gates real commits)",
    )
    threading.Thread(target=_watch_process, args=(proc,), daemon=True).start()
    return 200, {"ok": True, "message": f"started (pid={proc.pid})", "pid": proc.pid}


def action_kill() -> tuple[int, dict]:
    global _proc
    with _lock:
        proc = _proc
    if proc is None or proc.poll() is not None:
        _write_status(running=False)
        return 200, {"ok": True, "message": "no active run (already stopped)"}

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    with _lock:
        if _proc is proc:
            _proc = None
    _write_status(running=False, killed=True, finished_at=_now())
    return 200, {"ok": True, "message": "killed"}


def action_undo_last_cycle() -> tuple[int, dict]:
    result = snapshots_mod.undo_last_cycle()
    return (200 if result["ok"] else 404), result


def action_restore_last_cycle() -> tuple[int, dict]:
    result = snapshots_mod.restore_last_cycle()
    return (200 if result["ok"] else 404), result


def _tail_log(n: int = 40) -> list[str]:
    if not LOG_PATH.exists():
        return []
    try:
        lines = LOG_PATH.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


def action_status() -> tuple[int, dict]:
    run = _read_status()
    run["log_tail"] = _tail_log()
    return 200, {
        "run": run,
        "snapshots": snapshots_mod.status(),
    }


# Fixed allowlist. This dict IS the security boundary — nothing outside these
# five keys is ever dispatched to.
ROUTES = {
    ("POST", "/api/start_watch"): action_start_watch,
    ("POST", "/api/kill"): action_kill,
    ("POST", "/api/undo_last_cycle"): action_undo_last_cycle,
    ("POST", "/api/restore_last_cycle"): action_restore_last_cycle,
    ("GET", "/api/status"): action_status,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:  # quieter test output
        pass

    def _respond(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8080")
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, method: str) -> None:
        route = ROUTES.get((method, self.path))
        if route is None:
            self._respond(404, {"ok": False, "message": f"unknown route: {method} {self.path}"})
            return
        try:
            code, payload = route()
        except Exception as exc:  # noqa: BLE001 — never let a bug crash the loopback server
            self._respond(500, {"ok": False, "message": f"internal error: {exc}"})
            return
        self._respond(code, payload)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_OPTIONS(self) -> None:  # noqa: N802 — CORS preflight for the dashboard fetch()
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8080")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def create_server(port: int = 8787) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), Handler)


def main() -> int:
    ap = argparse.ArgumentParser(description="Localhost-only training control server")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    server = create_server(args.port)
    bound_port = server.server_address[1]
    print(f"training_control_server listening on http://127.0.0.1:{bound_port}")
    print("Allowlisted actions: start_watch, kill, undo_last_cycle, restore_last_cycle (+ GET status)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
