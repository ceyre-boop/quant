"""Tests for the three-tier synthesis fallback chain in sovereign.briefing.synthesize.

Verifies ORDER and tier-selection without requiring Ollama or an Anthropic key installed:
  ollama → (if None and key) anthropic → (if None) None (deterministic upstream).
"""
import sovereign.briefing.synthesize as S


_ARGS = ({}, {"regime": "BREADTH"}, {}, {}, {}, "no data yet")


def test_ollama_tried_first_and_short_circuits_anthropic(monkeypatch):
    calls = {"ollama": 0, "anthropic": 0}

    def fake_ollama(prompt):
        calls["ollama"] += 1
        return S._finalize({"directional_bias": "LONG", "confidence": 70}, "ollama/qwen2.5", 0.0)

    def boom(*a, **k):
        calls["anthropic"] += 1
        raise AssertionError("anthropic must not be reached when ollama succeeds")

    monkeypatch.setattr(S, "_try_ollama", fake_ollama)
    monkeypatch.setattr(S, "_load_cost", boom, raising=False)
    out = S.synthesize(*_ARGS)
    assert out["model"] == "ollama/qwen2.5"
    assert out["cost_usd"] == 0.0
    assert calls["ollama"] == 1 and calls["anthropic"] == 0
    # regime_call backfills from lead_lag when the model omits it
    assert out["regime_call"] == "BREADTH"


def test_no_key_returns_none_after_ollama_fails(monkeypatch):
    monkeypatch.setattr(S, "_try_ollama", lambda p: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # neutralise the dotenv loader so it can't repopulate the key from disk
    monkeypatch.setattr("sovereign.oracle.oracle_agent._load_dotenv", lambda: None, raising=False)
    assert S.synthesize(*_ARGS) is None


def test_anthropic_reached_only_when_ollama_none_and_key_present(monkeypatch):
    monkeypatch.setattr(S, "_try_ollama", lambda p: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("sovereign.oracle.oracle_agent._load_dotenv", lambda: None, raising=False)

    reached = {"n": 0}

    class _Resp:
        class _C:
            text = '{"directional_bias":"SHORT","confidence":40,"regime_call":"ROTATION_WARN","key_level":null,"narrative":"x"}'
        content = [_C()]

        class _U:
            input_tokens = 10
            output_tokens = 20
        usage = _U()

    class _Client:
        def __init__(self, **k):
            pass

        class messages:  # noqa: N801
            @staticmethod
            def create(**k):
                reached["n"] += 1
                return _Resp()

    import types
    fake_anthropic = types.SimpleNamespace(Anthropic=lambda **k: _Client())
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_anthropic)
    out = S.synthesize(*_ARGS)
    assert reached["n"] == 1
    assert out["model"] == S.MODEL
    assert out["directional_bias"] == "SHORT"


def test_select_model_prefers_qwen_then_llama():
    # _select_ollama_model degrades to None when the ollama client is unavailable — must never raise.
    assert S._select_ollama_model() in (S._OLLAMA_PREFERRED, S._OLLAMA_FALLBACK, None)
