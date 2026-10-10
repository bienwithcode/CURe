from __future__ import annotations

import threading
from typing import Any, Callable

import pytest


class _FakeAdapter:
    """Thread-safe provider double that delegates to a handler callable."""

    def __init__(self, handler: Callable[[str, dict], tuple[dict, float]]) -> None:
        self._handler = handler
        self._lock = threading.Lock()
        self.calls: list[tuple[str, dict]] = []

    def complete(
        self, operation: str, request_payload: dict, *, deadline: float | None = None
    ) -> tuple[dict, float]:
        with self._lock:
            self.calls.append((operation, request_payload))
        return self._handler(operation, request_payload)


def _outcomes(*outcomes: Any) -> Callable[[str, dict], tuple[dict, float]]:
    queue = list(outcomes)

    def handler(operation: str, request_payload: dict) -> tuple[dict, float]:
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, 1.0

    return handler


def _make_client(
    handler: Callable[[str, dict], tuple[dict, float]], **overrides: Any
) -> tuple[Any, _FakeAdapter, Any]:
    from cure_system_one.client import SystemOneClient
    from cure_system_one.config import resolve_system_one_config

    section = {"enabled": True, "provider": "local", "model": "M1"}
    section.update(overrides)
    config = resolve_system_one_config(section)
    adapter = _FakeAdapter(handler)
    return SystemOneClient(config, adapter), adapter, config


def test_schema_validation_all_primitives() -> None:
    from cure_system_one.errors import SystemOneSchemaViolationError

    client, _, _ = _make_client(_outcomes({"probability": 1.7}))
    with pytest.raises(SystemOneSchemaViolationError) as noul_error:
        client.noul("a claim", "some evidence")
    assert noul_error.value.operation == "noul"
    assert noul_error.value.raw_payload == {"probability": 1.7}

    client, _, _ = _make_client(_outcomes({"selected": "OPTION-X", "confidence": 0.9}))
    with pytest.raises(SystemOneSchemaViolationError):
        client.choice("subject", ["A", "B"])

    client, _, _ = _make_client(_outcomes({"score": 10, "confidence": 0.5}))
    scored = client.score("subject", {"min": 0, "max": 10})
    assert scored["score"] == 10

    client, _, _ = _make_client(_outcomes({"filtered": ["A", "Z"]}))
    with pytest.raises(SystemOneSchemaViolationError):
        client.batch_filter(["A", "B"])

    client, _, _ = _make_client(_outcomes({"filtered": ["A", "A"]}))
    with pytest.raises(SystemOneSchemaViolationError):
        client.batch_filter(["A", "B"])

    client, _, _ = _make_client(_outcomes({"selected": "A", "confidence": 0.4}))
    valid = client.choice("subject", ["A", "B"])
    assert valid["selected"] == "A"
    assert valid["truncated"] is False


def test_structural_input_contract_and_degenerate_batch() -> None:
    from cure_system_one.errors import SystemOneInputError

    client, adapter, _ = _make_client(
        _outcomes(
            {
                "probability": 0.5,
                "selected": "A",
                "confidence": 0.5,
                "score": 0,
                "filtered": [],
            }
        )
    )

    for claim, evidence in [
        (None, "e"),
        ("", "e"),
        (123, "e"),
        ("c", None),
        ("c", ""),
        ("c", 5),
    ]:
        with pytest.raises(SystemOneInputError):
            client.noul(claim, evidence)

    with pytest.raises(SystemOneInputError):
        client.choice("subject", [])
    with pytest.raises(SystemOneInputError):
        client.choice("subject", ["A", 1])
    with pytest.raises(SystemOneInputError):
        client.choice("subject", "not-a-list")
    with pytest.raises(SystemOneInputError):
        client.choice("", ["A"])
    with pytest.raises(SystemOneInputError):
        client.choice(None, ["A"])
    with pytest.raises(SystemOneInputError):
        client.batch_filter("not-a-list")
    with pytest.raises(SystemOneInputError):
        client.batch_filter([1, 2])
    with pytest.raises(SystemOneInputError):
        client.score("subject", {})
    with pytest.raises(SystemOneInputError):
        client.score("subject", "not-a-dict")
    with pytest.raises(SystemOneInputError):
        client.score("", {"min": 0, "max": 10})

    degenerate = client.batch_filter([])
    assert degenerate["filtered"] == []
    assert adapter.calls == []


def test_oversized_fields_trim_and_flag() -> None:
    client, adapter, config = _make_client(_outcomes({"probability": 0.9}))
    oversized = "x" * 200_000

    result = client.noul("a claim", oversized)

    assert result["probability"] == 0.9
    assert result["truncated"] is True
    sent_evidence = adapter.calls[0][1]["input"]["evidence"]
    assert len(sent_evidence) == config.max_evidence_chars


def test_retry_once_and_failure_taxonomy() -> None:
    from cure_system_one.errors import (
        SystemOneProviderAuthError,
        SystemOneSchemaViolationError,
        SystemOneTransientError,
    )
    from cure_system_one.providers import ProviderTransportError

    client, adapter, _ = _make_client(
        _outcomes(
            ProviderTransportError("transient", latency_ms=2.0),
            {"probability": 0.9},
        )
    )
    assert client.noul("c", "e")["probability"] == 0.9
    assert len(adapter.calls) == 2

    client, adapter, _ = _make_client(
        _outcomes(
            ProviderTransportError("transient", latency_ms=2.0),
            ProviderTransportError("transient", latency_ms=3.0),
        )
    )
    with pytest.raises(SystemOneTransientError) as transient:
        client.noul("c", "e")
    assert transient.value.attempts == 2
    assert len(adapter.calls) == 2

    client, adapter, _ = _make_client(
        _outcomes(ProviderTransportError("auth", latency_ms=1.0))
    )
    with pytest.raises(SystemOneProviderAuthError) as auth:
        client.noul("c", "e")
    assert auth.value.attempts == 1
    assert len(adapter.calls) == 1

    client, adapter, _ = _make_client(_outcomes({"probability": 5.0}))
    with pytest.raises(SystemOneSchemaViolationError):
        client.noul("c", "e")
    assert len(adapter.calls) == 1

    def handler(operation: str, request_payload: dict) -> tuple[dict, float]:
        items = request_payload["input"]["items"]
        if "i7" in items:
            raise ProviderTransportError("transient", latency_ms=1.0)
        return {"filtered": list(items)}, 1.0

    client, adapter, _ = _make_client(handler, batch_chunk_size=3)
    with pytest.raises(SystemOneTransientError):
        client.batch_filter([f"i{n}" for n in range(10)])
    assert len(adapter.calls) == 5


def test_frozen_snapshot_and_chunked_batch_order() -> None:
    from cure_system_one.client import SystemOneClient
    from cure_system_one.config import resolve_system_one_config

    section = {"enabled": True, "provider": "local", "model": "M1"}
    config = resolve_system_one_config(section)
    adapter = _FakeAdapter(
        _outcomes(
            {"probability": 0.9},
            {"probability": 0.9},
            {"probability": 0.9},
        )
    )
    client = SystemOneClient.from_snapshot(config, provider=adapter)

    section["provider"] = "openrouter"
    section["model"] = "M2"
    for _ in range(3):
        assert client.noul("c", "e")["probability"] == 0.9
    assert client.config.provider == "local"
    assert client.config.model == "M1"

    chunk_sizes: list[int] = []

    def handler(operation: str, request_payload: dict) -> tuple[dict, float]:
        items = request_payload["input"]["items"]
        chunk_sizes.append(len(items))
        return {"filtered": list(items)}, 1.0

    config = resolve_system_one_config(
        {"enabled": True, "provider": "local", "model": "M1", "batch_chunk_size": 100}
    )
    adapter = _FakeAdapter(handler)
    client = SystemOneClient.from_snapshot(config, provider=adapter)
    items = [f"i{index:03d}" for index in range(250)]
    result = client.batch_filter(items)
    assert result["filtered"] == items
    assert sorted(chunk_sizes) == [50, 100, 100]


def test_inclusive_threshold_and_domain_boundaries() -> None:
    from cure_system_one.client import (
        is_unit_interval,
        passes_finding_acceptance,
        passes_grounding_confidence,
    )

    client, _, _ = _make_client(
        _outcomes({"probability": 0.75}), finding_acceptance_probability=0.75
    )
    assert client.noul("c", "e")["passed"] is True

    assert passes_finding_acceptance(0.75, 0.75) is True
    assert passes_finding_acceptance(0.74, 0.75) is False
    assert passes_grounding_confidence(0.80, 0.80) is True
    assert passes_grounding_confidence(0.79, 0.80) is False
    assert is_unit_interval(0.0) is True
    assert is_unit_interval(1.0) is True
    assert is_unit_interval(-0.1) is False
    assert is_unit_interval(1.1) is False


def test_error_context_propagation() -> None:
    from cure_system_one.errors import (
        SystemOneSchemaViolationError,
        SystemOneTransientError,
    )
    from cure_system_one.providers import ProviderTransportError

    client, _, _ = _make_client(
        _outcomes(
            ProviderTransportError("transient", latency_ms=4.0),
            ProviderTransportError("transient", latency_ms=6.0),
        )
    )
    with pytest.raises(SystemOneTransientError) as transient:
        client.score("subject", {"min": 0, "max": 10})
    error = transient.value
    assert error.provider == "local"
    assert error.operation == "score"
    assert error.latency_ms > 0
    assert error.attempts == 2

    client, _, _ = _make_client(
        _outcomes({"selected": "OPTION-X", "confidence": 0.5})
    )
    with pytest.raises(SystemOneSchemaViolationError) as violation:
        client.choice("subject", ["A", "B"])
    assert violation.value.raw_payload == {"selected": "OPTION-X", "confidence": 0.5}
    assert violation.value.operation == "choice"


def test_client_core_integration() -> None:
    from cure_system_one import build_client
    from cure_system_one.config import resolve_system_one_config

    config = resolve_system_one_config(
        {"enabled": True, "provider": "local", "model": "M1", "batch_chunk_size": 2}
    )

    def handler(operation: str, request_payload: dict) -> tuple[dict, float]:
        if operation == "noul":
            return {"probability": 0.82}, 1.0
        if operation == "choice":
            return {"selected": "A", "confidence": 0.9}, 1.0
        if operation == "batch_filter":
            return {"filtered": list(request_payload["input"]["items"])}, 1.0
        raise AssertionError(operation)

    adapter = _FakeAdapter(handler)
    client = build_client(config, provider=adapter)
    assert client.config.provider == "local"
    assert client.config.model == "M1"
    assert client.noul("claim", "evidence")["probability"] == 0.82
    assert client.choice("subject", ["A", "B"])["selected"] == "A"
    result = client.batch_filter(["i0", "i1", "i2"])
    assert result["filtered"] == ["i0", "i1", "i2"]

    batch_chunks = [
        tuple(payload["input"]["items"])
        for operation, payload in adapter.calls
        if operation == "batch_filter"
    ]
    assert sorted(batch_chunks) == [("i0", "i1"), ("i2",)]


def test_batch_filter_thread_pool_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    import cure_system_one.client as client_module

    captured: list[int | None] = []

    class _FakeFuture:
        def __init__(self, fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
            self._fn = fn
            self._args = args

        def result(self) -> Any:
            return self._fn(*self._args)

    class _FakeExecutor:
        def __init__(self, max_workers: int | None = None) -> None:
            captured.append(max_workers)

        def __enter__(self) -> "_FakeExecutor":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def submit(self, fn: Callable[..., Any], *args: Any) -> _FakeFuture:
            return _FakeFuture(fn, args)

    monkeypatch.setattr(client_module, "ThreadPoolExecutor", _FakeExecutor)

    def handler(operation: str, request_payload: dict) -> tuple[dict, float]:
        return {"filtered": list(request_payload["input"]["items"])}, 1.0

    client, _, _ = _make_client(handler, batch_chunk_size=1)
    items = [f"i{index:02d}" for index in range(20)]
    result = client.batch_filter(items)

    assert result["filtered"] == items
    assert client_module.MAX_BATCH_FILTER_WORKERS == 8
    assert captured == [client_module.MAX_BATCH_FILTER_WORKERS]
