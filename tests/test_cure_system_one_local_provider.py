"""Unit tests for the deterministic local System 1 provider."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest


def _local_config(**overrides: Any):
    from cure_system_one.config import resolve_system_one_config

    section = {"enabled": True, "provider": "local", "model": "M1"}
    section.update(overrides)
    return resolve_system_one_config(section)


def _provider(config: Any = None):
    from cure_system_one.providers.local import LocalProvider

    return LocalProvider(config or _local_config())


def _write_script(tmp_path: Path, mapping: dict[str, Any]) -> str:
    path = tmp_path / "script.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return str(path)


def _request(fields: dict[str, Any]) -> dict[str, Any]:
    return {"input": fields}


def test_canned_defaults_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_network(*args: Any, **kwargs: Any):  # pragma: no cover - guard
        raise AssertionError("local provider must not touch the network")

    monkeypatch.setattr("urllib.request.urlopen", _no_network)

    provider = _provider()
    assert provider.complete("noul", _request({}), deadline=None)[0] == {
        "probability": 0.5
    }
    assert provider.complete(
        "choice", _request({"options": ["A", "B"]}), deadline=None
    )[0] == {"selected": "A", "confidence": 0.5}
    assert provider.complete(
        "score", _request({"rubric": {"min": 0, "max": 10}}), deadline=None
    )[0] == {"score": 5.0, "confidence": 0.5}
    assert provider.complete(
        "batch_filter", _request({"items": ["a", "b"]}), deadline=None
    )[0] == {"filtered": ["a", "b"]}


def test_script_load_once_and_freeze(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cure_system_one.providers import ProviderTransportError

    path = _write_script(
        tmp_path,
        {"noul": [{"fault": "timeout"}, {"value": {"probability": 0.9}}]},
    )
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", path)

    provider = _provider()
    with pytest.raises(ProviderTransportError):
        provider.complete("noul", _request({}), deadline=None)
    assert provider.complete("noul", _request({}), deadline=None)[0] == {
        "probability": 0.9
    }

    # Rewrite the file: a frozen stage must not observe the change.
    Path(path).write_text(
        json.dumps({"noul": [{"value": {"probability": 0.1}}]}), encoding="utf-8"
    )
    assert provider.complete("noul", _request({}), deadline=None)[0] == {
        "probability": 0.5
    }

    # The next stage re-reads the script.
    next_provider = _provider()
    assert next_provider.complete("noul", _request({}), deadline=None)[0] == {
        "probability": 0.1
    }

    # Blank or unset means canned mode with no file access.
    def _fail_read(*args: Any, **kwargs: Any) -> str:  # pragma: no cover - guard
        raise AssertionError("canned mode must not read the script file")

    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", "")
    monkeypatch.setattr(Path, "read_text", _fail_read)
    assert _provider().complete("noul", _request({}), deadline=None)[0] == {
        "probability": 0.5
    }
    monkeypatch.delenv("CURE_SYSTEM_ONE_SCRIPT", raising=False)
    assert _provider().complete("noul", _request({}), deadline=None)[0] == {
        "probability": 0.5
    }


def test_fifo_consumption_and_exhaustion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cure_system_one.providers import ProviderTransportError

    path = _write_script(
        tmp_path,
        {
            "noul": [
                {"fault": "timeout"},
                {"fault": "http_429"},
                {"value": {"probability": 0.9}},
            ],
            "choice": [{"value": {"selected": "B", "confidence": 0.9}}],
        },
    )
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", path)
    provider = _provider()

    with pytest.raises(ProviderTransportError) as first:
        provider.complete("noul", _request({}), deadline=None)
    assert first.value.kind == "transient"
    with pytest.raises(ProviderTransportError) as second:
        provider.complete("noul", _request({}), deadline=None)
    assert second.value.kind == "transient"
    assert provider.complete("noul", _request({}), deadline=None)[0] == {
        "probability": 0.9
    }

    options = {"options": ["A", "B"]}
    assert provider.complete("choice", _request(options), deadline=None)[0] == {
        "selected": "B",
        "confidence": 0.9,
    }
    for _ in range(2):
        assert provider.complete("choice", _request(options), deadline=None)[0] == {
            "selected": "A",
            "confidence": 0.5,
        }


def test_whole_script_upfront_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cure_system_one.errors import SystemOneConfigError

    def _expect_config_error(script_path: str) -> None:
        monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", script_path)
        with pytest.raises(SystemOneConfigError):
            _provider().complete("noul", _request({}), deadline=None)

    # Missing file, unreadable file, and directory are all malformed.
    _expect_config_error(str(tmp_path / "missing.json"))
    # A root runner can read a 0o000 file, so the case only holds for non-root.
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        unreadable = tmp_path / "unreadable.json"
        unreadable.write_text("{}", encoding="utf-8")
        unreadable.chmod(0o000)
        _expect_config_error(str(unreadable))
        unreadable.chmod(0o644)
    directory = tmp_path / "directory"
    directory.mkdir()
    _expect_config_error(str(directory))

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    _expect_config_error(str(bad_json))

    unknown_fault = tmp_path / "unknown_fault.json"
    unknown_fault.write_text(json.dumps({"noul": [{"fault": "429x"}]}), encoding="utf-8")
    _expect_config_error(str(unknown_fault))

    wrong_shape = tmp_path / "wrong_shape.json"
    wrong_shape.write_text(json.dumps({"noul": [{"weird": 1}]}), encoding="utf-8")
    _expect_config_error(str(wrong_shape))

    unknown_operation = tmp_path / "unknown_operation.json"
    unknown_operation.write_text(
        json.dumps({"nope": [{"value": {}}]}), encoding="utf-8"
    )
    _expect_config_error(str(unknown_operation))

    # An empty script is degenerate-valid and yields canned results.
    empty_script = tmp_path / "empty.json"
    empty_script.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(empty_script))
    assert _provider().complete("noul", _request({}), deadline=None)[0] == {
        "probability": 0.5
    }


def test_concurrent_consumption_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    path = _write_script(
        tmp_path,
        {
            "score": [
                {"value": {"score": 1.0, "confidence": 0.5}},
                {"value": {"score": 2.0, "confidence": 0.5}},
            ]
        },
    )
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", path)
    provider = _provider()
    fields = {"rubric": {"min": 0, "max": 10}}

    def _call(_: int):
        return provider.complete("score", _request(fields), deadline=None)[0]["score"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_call, index) for index in range(2)]
        scores = [future.result(timeout=5) for future in futures]

    assert sorted(scores) == [1.0, 2.0]
    # Exhausted afterwards: the canned midpoint applies.
    assert provider.complete("score", _request(fields), deadline=None)[0] == {
        "score": 5.0,
        "confidence": 0.5,
    }


def test_fault_taxonomy_and_client_taxonomy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cure_system_one.client import SystemOneClient
    from cure_system_one.errors import (
        SystemOneProviderAuthError,
        SystemOneTransientError,
    )
    from cure_system_one.providers import ProviderTransportError

    transient_faults = ["timeout", "connection_reset", "http_429", "http_500"]
    auth_faults = ["http_401", "http_402", "http_403"]
    path = _write_script(
        tmp_path,
        {"noul": [{"fault": fault} for fault in transient_faults + auth_faults]},
    )
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", path)
    provider = _provider()
    for _ in transient_faults:
        with pytest.raises(ProviderTransportError) as transient:
            provider.complete("noul", _request({}), deadline=None)
        assert transient.value.kind == "transient"
    for _ in auth_faults:
        with pytest.raises(ProviderTransportError) as auth:
            provider.complete("noul", _request({}), deadline=None)
        assert auth.value.kind == "auth"

    # Through the client a transient fault is retried exactly once.
    retry_script = tmp_path / "retry.json"
    retry_script.write_text(
        json.dumps({"noul": [{"fault": "http_429"}, {"fault": "http_429"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(retry_script))
    config = _local_config()
    client = SystemOneClient(config, _provider(config))
    with pytest.raises(SystemOneTransientError) as transient_error:
        client.noul("a claim", "some evidence")
    assert transient_error.value.attempts == 2
    assert transient_error.value.provider == "local"
    assert transient_error.value.operation == "noul"

    # An auth fault never retries.
    auth_script = tmp_path / "auth.json"
    auth_script.write_text(
        json.dumps({"noul": [{"fault": "http_401"}]}), encoding="utf-8"
    )
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(auth_script))
    config = _local_config()
    client = SystemOneClient(config, _provider(config))
    with pytest.raises(SystemOneProviderAuthError) as auth_error:
        client.noul("a claim", "some evidence")
    assert auth_error.value.attempts == 1
    assert auth_error.value.provider == "local"
    assert auth_error.value.operation == "noul"


def test_instant_timeout_and_schema_violation_carrier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cure_system_one.client import SystemOneClient
    from cure_system_one.errors import SystemOneSchemaViolationError
    from cure_system_one.providers import ProviderTransportError

    # A scripted timeout is instantaneous even with a large timeout_ms.
    timeout_script = tmp_path / "timeout.json"
    timeout_script.write_text(
        json.dumps({"score": [{"fault": "timeout"}]}), encoding="utf-8"
    )
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(timeout_script))
    config = _local_config(timeout_ms=2500)
    with pytest.raises(ProviderTransportError) as transient:
        _provider(config).complete(
            "score", _request({"rubric": {"min": 0, "max": 10}}), deadline=None
        )
    assert transient.value.latency_ms < 100

    # A schema_violation outcome carries its value payload into the client.
    carrier_script = tmp_path / "carrier.json"
    carrier_script.write_text(
        json.dumps(
            {
                "choice": [
                    {
                        "fault": "schema_violation",
                        "value": {"selected": "OPTION-X", "confidence": 0.5},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(carrier_script))
    config = _local_config()
    client = SystemOneClient(config, _provider(config))
    with pytest.raises(SystemOneSchemaViolationError) as violation:
        client.choice("subject", ["A", "B"])
    assert violation.value.raw_payload == {"selected": "OPTION-X", "confidence": 0.5}
    assert violation.value.operation == "choice"
    assert violation.value.provider == "local"

    # A plain value outcome is returned exactly.
    value_script = tmp_path / "value.json"
    value_script.write_text(
        json.dumps({"noul": [{"value": {"probability": 0.42}}]}), encoding="utf-8"
    )
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(value_script))
    assert _provider().complete("noul", _request({}), deadline=None)[0] == {
        "probability": 0.42
    }

    # A schema_violation without a value falls back to a per-operation canned
    # invalid payload.
    fallback_script = tmp_path / "fallback.json"
    fallback_script.write_text(
        json.dumps(
            {
                "noul": [{"fault": "schema_violation"}],
                "choice": [{"fault": "schema_violation"}],
                "score": [{"fault": "schema_violation"}],
                "batch_filter": [{"fault": "schema_violation"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(fallback_script))
    provider = _provider()
    assert provider.complete("noul", _request({}), deadline=None)[0] == {
        "probability": 1.7
    }
    assert provider.complete(
        "choice", _request({"options": ["A", "B"]}), deadline=None
    )[0] == {"selected": "OPTION-X", "confidence": 0.5}
    score_payload = provider.complete(
        "score", _request({"rubric": {"min": 0, "max": 10}}), deadline=None
    )[0]
    assert score_payload["score"] > 10
    assert score_payload["confidence"] == 0.5
    assert provider.complete(
        "batch_filter", _request({"items": ["a", "b"]}), deadline=None
    )[0] == {"filtered": ["__foreign__"]}

    config = _local_config()
    client = SystemOneClient(config, _provider(config))
    with pytest.raises(SystemOneSchemaViolationError):
        client.noul("a claim", "some evidence")


def test_chunked_batch_one_outcome_per_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cure_system_one.client import SystemOneClient

    path = _write_script(
        tmp_path, {"batch_filter": [{"value": {"filtered": ["item-0"]}}]}
    )
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", path)
    config = _local_config(batch_chunk_size=100)
    client = SystemOneClient(config, _provider(config))

    items = [f"item-{index}" for index in range(250)]
    result = client.batch_filter(items)
    expected = ["item-0"] + items[100:200] + items[200:250]
    assert result["filtered"] == expected


def test_local_declares_all_supported_operations() -> None:
    provider = _provider()
    assert provider.supported_operations == frozenset(
        {"noul", "choice", "score", "batch_filter"}
    )
