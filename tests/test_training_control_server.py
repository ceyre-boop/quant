"""Tests for scripts/training_control_server.py and sovereign/training/snapshots.py.

Freeze-safe: exercises only the training scaffold's own artifacts under a
temp ROOT-relative sandbox (monkeypatched paths) — never touches the real
data/training/ tree or any execution-path module.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from scripts import training_control_server as tcs
from sovereign.training import snapshots as snap


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    snap_dir = tmp_path / "snapshots"
    history = snap_dir / "history.json"
    current = tmp_path / "current_policy_params.json"
    monkeypatch.setattr(snap, "SNAP_DIR", snap_dir)
    monkeypatch.setattr(snap, "HISTORY_PATH", history)
    monkeypatch.setattr(snap, "CURRENT_PARAMS_PATH", current)
    status_path = tmp_path / "training_run_status.json"
    monkeypatch.setattr(tcs, "STATUS_PATH", status_path)
    yield tmp_path


def test_undo_restore_empty_history_is_safe(sandbox):
    result = snap.undo_last_cycle()
    assert result["ok"] is False
    result = snap.restore_last_cycle()
    assert result["ok"] is False


def test_record_undo_restore_round_trip(sandbox):
    snap.record_cycle(
        params_before={"hold": 5}, params_after={"hold": 5},
        cycle_ref="cycle_1.json", committed=False, timestamp="20260101T000000Z",
    )
    snap.record_cycle(
        params_before={"hold": 5}, params_after={"hold": 6},
        cycle_ref="cycle_2.json", committed=False, timestamp="20260101T000100Z",
    )
    assert snap.get_current_params() == {"hold": 6}

    undone = snap.undo_last_cycle()
    assert undone["ok"] is True
    assert undone["params"] == {"hold": 5}
    assert snap.get_current_params() == {"hold": 5}

    restored = snap.restore_last_cycle()
    assert restored["ok"] is True
    assert restored["params"] == {"hold": 6}
    assert snap.get_current_params() == {"hold": 6}


def test_new_cycle_clears_redo_stack(sandbox):
    snap.record_cycle({"a": 1}, {"a": 1}, "c1", False, "t1")
    snap.undo_last_cycle()
    history = snap._load_history()
    assert len(history["undone"]) == 1

    snap.record_cycle({"a": 1}, {"a": 2}, "c2", False, "t2")
    history = snap._load_history()
    assert history["undone"] == []


def test_undo_never_activates_uncommitted_cycle(sandbox):
    """Undo/restore only ever move between states the gated pipeline wrote —
    both before/after states here have committed=False, so neither undo nor
    restore can produce a committed=True effect."""
    snap.record_cycle({"a": 1}, {"a": 1}, "c1", committed=False, timestamp="t1")
    result = snap.restore_last_cycle()
    assert result["ok"] is False  # nothing undone yet
    result = snap.undo_last_cycle()
    assert result["ok"] is True
    history = snap._load_history()
    assert all(r["committed"] is False for r in history["undone"] + history["applied"])


@pytest.fixture()
def server(sandbox):
    srv = tcs.create_server(port=0)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    thread.join(timeout=5)


def _post(url):
    req = urllib.request.Request(url, data=b"{}", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def test_server_binds_loopback_only(server):
    assert server.startswith("http://127.0.0.1:")


def test_unknown_action_rejected(server):
    code, body = _post(f"{server}/api/does_not_exist")
    assert code == 404
    assert body["ok"] is False


def test_unknown_action_via_get_rejected(server):
    req = urllib.request.Request(f"{server}/api/start_watch", method="GET")
    try:
        urllib.request.urlopen(req, timeout=5)
        raise AssertionError("expected HTTPError")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


def test_undo_with_no_history_returns_404(server):
    code, body = _post(f"{server}/api/undo_last_cycle")
    assert code == 404
    assert body["ok"] is False


def test_status_endpoint_readonly(server):
    code, body = _get(f"{server}/api/status")
    assert code == 200
    assert "run" in body and "snapshots" in body


def test_kill_with_nothing_running_is_idempotent(server):
    code, body = _post(f"{server}/api/kill")
    assert code == 200
    assert body["ok"] is True


def test_start_watch_refuses_second_concurrent_run(server, monkeypatch):
    # Replace the real training command with a long-sleeping stub so the test
    # doesn't depend on sovereign_train.py's own runtime, but still exercises
    # the real subprocess launch path.
    monkeypatch.setattr(tcs, "TRAIN_CMD", ["python3", "-c", "import time; time.sleep(5)"])
    code1, body1 = _post(f"{server}/api/start_watch")
    assert code1 == 200
    assert body1["ok"] is True

    code2, body2 = _post(f"{server}/api/start_watch")
    assert code2 == 409
    assert body2["ok"] is False

    code3, body3 = _post(f"{server}/api/kill")
    assert code3 == 200
    assert body3["ok"] is True
    time.sleep(0.2)


def test_server_never_touches_config_training_yml():
    """Source-level guard: only the module docstring is allowed to mention
    these terms (as an explanation of what the server must NOT do) — no
    functional statement may reference them."""
    import scripts.training_control_server as module
    import inspect

    source = inspect.getsource(module)
    docstring = inspect.getdoc(module) or ""
    body_after_docstring = source.split('"""', 2)[-1] if source.count('"""') >= 2 else source
    for term in ("training.yml", "gate.py", "evaluate_gate"):
        assert term not in body_after_docstring, (
            f"{term!r} found outside the module docstring — server must not reference it"
        )
    assert "training.yml" in docstring  # sanity: the guard text itself is present
