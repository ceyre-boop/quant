"""Error handling contract for the consolidated Alpaca client.

The bug being prevented: both shadow scripts wrapped every failure in
`except Exception: time.sleep(5)` and then raised `RuntimeError(url[:110])`,
destroying the original exception. A retry loop that discards the exception it
is retrying cannot be debugged — that is why the gapper_shadow_scan exit-1 was
misdiagnosed.
"""
import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from execution.alpaca import (AlpacaEntitlementError, AlpacaError, get)

URL = "https://data.alpaca.markets/v2/stocks/TEST/quotes"


def _http_error(code, body=b'{"message":"nope"}'):
    return urllib.error.HTTPError(URL, code, "err", {}, io.BytesIO(body))


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")


def test_403_raises_immediately_without_retrying():
    """403 is an entitlement fact, not a transient error. The old code slept 65s
    and retried, turning a permanent subscription failure into a silent stall."""
    body = b'{"message":"subscription does not permit querying recent SIP data"}'
    with patch("urllib.request.urlopen", side_effect=_http_error(403, body)) as m:
        with pytest.raises(AlpacaEntitlementError) as ei:
            get(URL, max_attempts=5)
    assert m.call_count == 1, "403 must not be retried"
    assert ei.value.status == 403
    assert "subscription does not permit" in ei.value.body


def test_404_raises_without_retrying():
    with patch("urllib.request.urlopen", side_effect=_http_error(404)) as m:
        with pytest.raises(AlpacaError):
            get(URL, max_attempts=5)
    assert m.call_count == 1


def test_500_retries_then_raises_with_context():
    with patch("urllib.request.urlopen", side_effect=_http_error(500)), \
         patch("time.sleep"):
        with pytest.raises(AlpacaError) as ei:
            get(URL, max_attempts=3)
    assert ei.value.attempts == 3
    assert ei.value.status == 500
    assert ei.value.__cause__ is not None, "original exception must be preserved"


def test_timeout_retries_then_raises_preserving_cause():
    with patch("urllib.request.urlopen", side_effect=TimeoutError("slow")), \
         patch("time.sleep"):
        with pytest.raises(AlpacaError) as ei:
            get(URL, max_attempts=2)
    assert isinstance(ei.value.__cause__, TimeoutError)


def test_malformed_json_is_not_retried():
    """Retrying a parse error hides the real problem."""
    class _R:
        def read(self): return b"not json at all"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch("urllib.request.urlopen", return_value=_R()) as m:
        with pytest.raises(AlpacaError) as ei:
            get(URL, max_attempts=5)
    assert m.call_count == 1
    assert "invalid JSON" in ei.value.body


def test_unexpected_exception_is_reraised_not_swallowed():
    """The precise failure mode of the old code: a bare `except Exception`."""
    with patch("urllib.request.urlopen", side_effect=ValueError("boom")):
        with pytest.raises(ValueError):
            get(URL, max_attempts=3)


def test_success_returns_parsed_json():
    class _R:
        def read(self): return json.dumps({"quotes": [{"bp": 1.0}]}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch("urllib.request.urlopen", return_value=_R()):
        assert get(URL)["quotes"][0]["bp"] == 1.0


def test_error_message_carries_url_and_body():
    with patch("urllib.request.urlopen", side_effect=_http_error(500)), \
         patch("time.sleep"):
        with pytest.raises(AlpacaError) as ei:
            get(URL, max_attempts=1)
    msg = str(ei.value)
    assert URL in msg and "500" in msg


def test_missing_keys_fail_loud(monkeypatch):
    from execution import alpaca
    monkeypatch.setenv("ALPACA_API_KEY", "")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "")
    monkeypatch.setattr(alpaca, "load_env", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
        alpaca.keys()
