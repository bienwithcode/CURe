from __future__ import annotations

import pytest


def test_registry_and_transport_error() -> None:
    from cure_system_one.errors import SystemOneConfigError, SystemOneError
    from cure_system_one.providers import (
        ProviderTransportError,
        create_provider,
        provider_names,
    )

    assert set(provider_names()) == {"openrouter", "local"}

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
