from __future__ import annotations

import pytest


def test_registry_and_transport_error() -> None:
    from cure_system_one.errors import SystemOneConfigError, SystemOneError
    from cure_system_one.providers import (
        ProviderTransportError,
        create_provider,
        provider_names,
    )

    assert set(provider_names()) == {"openrouter", "openrouter-decisions", "local"}

    with pytest.raises(SystemOneConfigError):
        create_provider("typesafe", None)

    transient = ProviderTransportError("transient", latency_ms=3.5)
    assert transient.kind == "transient"
    assert transient.latency_ms == 3.5
    assert not isinstance(transient, SystemOneError)

    auth = ProviderTransportError("auth")
    assert auth.kind == "auth"

    with pytest.raises(ValueError):
        ProviderTransportError("catastrophic")


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        import json

        self._bytes = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._bytes

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _openrouter_config():
    from cure_system_one.config import resolve_system_one_config

    return resolve_system_one_config(
        {
            "enabled": True,
            "provider": "openrouter",
            "model": "M1",
            "api_key_env": "CURE_SYSTEM_ONE_TEST_KEY",  # pragma: allowlist secret
        }
    )


def test_openrouter_transport_status_taxonomy(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import urllib.error

    from cure_system_one.providers import ProviderTransportError
    from cure_system_one.providers import openrouter as openrouter_module
    from cure_system_one.providers.openrouter import OpenRouterAdapter

    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["headers"] = dict(request.headers)
        return _FakeResponse(
            {"choices": [{"message": {"content": json.dumps({"probability": 0.82})}}]}
        )

    monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", fake_urlopen)

    raw, latency = OpenRouterAdapter(_openrouter_config()).complete(
        "noul",
        {"messages": [{"role": "user", "content": "x"}], "schema": {"type": "object"}},
        deadline=None,
    )
    assert raw == {"probability": 0.82}
    assert isinstance(latency, float)
    assert latency >= 0.0

    body = json.loads(bytes(captured["data"]).decode("utf-8"))
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert body["model"] == "M1"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "noul"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert body["provider"] == {"require_parameters": True}

    def raising(code: int):
        def _urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, code, "err", None, None)

        return _urlopen

    for code in (429, 500, 503):
        monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", raising(code))
        with pytest.raises(ProviderTransportError) as transient:
            OpenRouterAdapter(_openrouter_config()).complete(
                "noul", {"messages": [], "schema": {}}, deadline=None
            )
        assert transient.value.kind == "transient"

    for code in (401, 402, 403):
        monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", raising(code))
        with pytest.raises(ProviderTransportError) as auth:
            OpenRouterAdapter(_openrouter_config()).complete(
                "noul", {"messages": [], "schema": {}}, deadline=None
            )
        assert auth.value.kind == "auth"


def _decisions_config(model: str = "typesafe/jev-1.13"):
    from cure_system_one.config import SystemOneConfig

    return SystemOneConfig(
        enabled=True,
        provider="openrouter-decisions",
        model=model,
        api_key_env="CURE_SYSTEM_ONE_TEST_KEY",  # pragma: allowlist secret
    )


def test_decisions_request_maps_primitives(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from cure_system_one.providers import decisions as decisions_module
    from cure_system_one.providers.decisions import DecisionsAdapter

    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["headers"] = dict(request.headers)
        return _FakeResponse(
            {
                "model": "typesafe/jev-1.13-20260917",
                "answers": {
                    "choice": {"type": "choice", "choice": "B", "confidence": 0.9}
                },
            }
        )

    monkeypatch.setattr(decisions_module.urllib.request, "urlopen", fake_urlopen)

    raw, latency = DecisionsAdapter(_decisions_config()).complete(
        "choice",
        {
            "input": {
                "subject": "Is it safe?",
                "options": ["A", "B"],
                "context": "ctx",
            }
        },
        deadline=None,
    )
    assert raw == {"selected": "B", "confidence": 0.9}
    assert isinstance(latency, float)

    body = json.loads(bytes(captured["data"]).decode("utf-8"))
    assert captured["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert body["model"] == "typesafe/jev-1.13"
    assert set(body["questions"]) == {"choice"}
    question = body["questions"]["choice"]
    assert question["type"] == "choice"
    assert question["instructions"]["subject"] == "Is it safe?"
    assert set(question["criteria"]) == {"A", "B"}


def _recorded_decisions(monkeypatch: pytest.MonkeyPatch, envelope: dict) -> None:
    from cure_system_one.providers import decisions as decisions_module

    monkeypatch.setattr(
        decisions_module.urllib.request,
        "urlopen",
        lambda request, timeout=None: _FakeResponse(envelope),
    )


def test_decisions_translates_typed_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cure_system_one.providers.decisions import DecisionsAdapter

    adapter = DecisionsAdapter(_decisions_config())

    _recorded_decisions(
        monkeypatch,
        {
            "answers": {
                "choice": {"type": "choice", "choice": "B", "confidence": 0.9}
            }
        },
    )
    raw, _ = adapter.complete(
        "choice", {"input": {"subject": "s", "options": ["A", "B"]}}, deadline=None
    )
    assert raw == {"selected": "B", "confidence": 0.9}

    _recorded_decisions(
        monkeypatch,
        {"answers": {"noul": {"type": "noul", "probability": 0.82, "confidence": 1}}},
    )
    raw, _ = adapter.complete(
        "noul", {"input": {"claim": "c", "evidence": "e"}}, deadline=None
    )
    assert raw == {"probability": 0.82}

    _recorded_decisions(
        monkeypatch,
        {"answers": {"score": {"type": "score", "score": 7, "confidence": 0.5}}},
    )
    raw, _ = adapter.complete(
        "score", {"input": {"subject": "s", "rubric": {"min": 0, "max": 10}}}, deadline=None
    )
    assert raw == {"score": 7, "confidence": 0.5}

    # A missing answer and a mismatched type both return the raw envelope.
    _recorded_decisions(monkeypatch, {"answers": {}})
    raw, _ = adapter.complete(
        "noul", {"input": {"claim": "c", "evidence": "e"}}, deadline=None
    )
    assert raw == {"answers": {}}

    _recorded_decisions(
        monkeypatch,
        {"answers": {"score": {"type": "choice", "choice": "A", "confidence": 1}}},
    )
    raw, _ = adapter.complete(
        "score", {"input": {"subject": "s", "rubric": {"min": 0, "max": 10}}}, deadline=None
    )
    assert raw["answers"]["score"]["type"] == "choice"


def _http_error_raiser(code: int, body: bytes = b"{}"):
    import io
    import urllib.error

    def _urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, code, "err", {}, io.BytesIO(body)
        )

    return _urlopen


def test_decisions_status_taxonomy(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import urllib.error

    from cure_system_one.providers import ProviderTransportError
    from cure_system_one.providers import decisions as decisions_module
    from cure_system_one.providers.decisions import DecisionsAdapter

    adapter = DecisionsAdapter(_decisions_config())

    for code in (401, 402, 403):
        monkeypatch.setattr(
            decisions_module.urllib.request, "urlopen", _http_error_raiser(code)
        )
        with pytest.raises(ProviderTransportError) as auth:
            adapter.complete(
                "noul", {"input": {"claim": "c", "evidence": "e"}}, deadline=None
            )
        assert auth.value.kind == "auth"

    for code in (429, 500, 503, 404, 422):
        monkeypatch.setattr(
            decisions_module.urllib.request, "urlopen", _http_error_raiser(code)
        )
        with pytest.raises(ProviderTransportError) as transient:
            adapter.complete(
                "noul", {"input": {"claim": "c", "evidence": "e"}}, deadline=None
            )
        assert transient.value.kind == "transient"

    def _network_error(request, timeout=None):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(
        decisions_module.urllib.request, "urlopen", _network_error
    )
    with pytest.raises(ProviderTransportError) as network:
        adapter.complete(
            "noul", {"input": {"claim": "c", "evidence": "e"}}, deadline=None
        )
    assert network.value.kind == "transient"

    # HTTP 400 is a deterministic rejection: the raw error body is returned.
    body = json.dumps({"error": "A Choice needs at least 2 options"}).encode("utf-8")
    monkeypatch.setattr(
        decisions_module.urllib.request, "urlopen", _http_error_raiser(400, body)
    )
    raw, latency = adapter.complete(
        "choice", {"input": {"subject": "s", "options": ["A"]}}, deadline=None
    )
    assert raw == {"error": "A Choice needs at least 2 options"}
    assert isinstance(latency, float)


def test_decisions_registered_with_defaults() -> None:
    from cure_system_one.config import (
        DEFAULT_MODEL,
        DEFAULT_PROVIDER,
        resolve_system_one_config,
    )
    from cure_system_one.providers import provider_names

    assert "openrouter-decisions" in provider_names()
    assert DEFAULT_PROVIDER == "openrouter-decisions"
    assert DEFAULT_MODEL == "typesafe/jev-1.13"

    resolved = resolve_system_one_config({"enabled": True})
    assert resolved.provider == "openrouter-decisions"
    assert resolved.model == "typesafe/jev-1.13"
