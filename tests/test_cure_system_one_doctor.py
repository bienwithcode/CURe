"""Feature tests for the System 1 doctor checks and the setup advisory."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import urllib.error
from unittest import mock

import pytest

import cure_commands
import cure_runtime


def _runtime(tmp_path: Path, system_one_toml: str | None) -> cure_runtime.ReviewflowRuntime:
    sandboxes = tmp_path / "sandboxes"
    cache = tmp_path / "cache"
    sandboxes.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "cure.toml"
    lines: list[str] = []
    if system_one_toml is not None:
        lines.append(system_one_toml)
        lines.append("")
    lines.extend(
        [
            "[paths]",
            f'sandbox_root = "{sandboxes}"',
            f'cache_root = "{cache}"',
        ]
    )
    cfg.write_text("\n".join(lines), encoding="utf-8")
    return cure_runtime.ReviewflowRuntime(
        config_path=cfg,
        config_source="cli",
        config_enabled=True,
        paths=cure_runtime.ReviewflowPaths(sandbox_root=sandboxes, cache_root=cache),
        sandbox_root_source="config",
        cache_root_source="config",
        codex_base_config_path=tmp_path / "codex.toml",
        codex_base_config_source="default",
    )


def _system_one_checks(runtime: cure_runtime.ReviewflowRuntime) -> dict[str, cure_runtime.DoctorCheck]:
    with mock.patch.object(cure_runtime.shutil, "which", return_value=None):
        checks = cure_runtime._doctor_runtime_checks(runtime)
    return {item.name: item for item in checks if item.name.startswith("system-one")}


def test_system_one_config_check_ok_and_fail(tmp_path: Path) -> None:
    valid = _system_one_checks(
        _runtime(
            tmp_path / "valid",
            '[system_one]\nenabled = true\nprovider = "openrouter-decisions"\nmodel = "M1"',
        )
    )["system-one-config"]
    assert valid.status == "ok"
    assert "openrouter-decisions" in valid.detail
    assert "M1" in valid.detail

    unknown = _system_one_checks(
        _runtime(
            tmp_path / "unknown",
            '[system_one]\nenabled = true\nprovider = "openrouter-decisions"\nfuture_key = 7',
        )
    )["system-one-config"]
    assert unknown.status == "ok"

    unsupported = _system_one_checks(
        _runtime(
            tmp_path / "unsupported",
            '[system_one]\nenabled = true\nprovider = "typesafe"',
        )
    )["system-one-config"]
    assert unsupported.status == "fail"
    assert "typesafe" in unsupported.detail
    assert "Traceback" not in unsupported.detail

    out_of_range = _system_one_checks(
        _runtime(
            tmp_path / "range",
            "[system_one]\nenabled = true\ngrounding_confidence_min = 1.5",
        )
    )["system-one-config"]
    assert out_of_range.status == "fail"
    assert "grounding_confidence_min" in out_of_range.detail

    non_table = _system_one_checks(
        _runtime(tmp_path / "non_table", 'system_one = "oops"')
    )["system-one-config"]
    assert non_table.status == "fail"
    assert "Traceback" not in non_table.detail


def test_system_one_checks_disabled_emits_none(tmp_path: Path) -> None:
    absent = _system_one_checks(_runtime(tmp_path / "absent", None))
    assert absent == {}

    omitted = _system_one_checks(
        _runtime(tmp_path / "omitted", '[system_one]\nprovider = "typesafe"')
    )
    assert omitted == {}

    disabled = _system_one_checks(
        _runtime(
            tmp_path / "disabled",
            '[system_one]\nenabled = false\nprovider = "typesafe"',
        )
    )
    assert disabled == {}

    non_table = _system_one_checks(
        _runtime(tmp_path / "non_table_disabled", 'system_one = "oops"')
    )
    assert non_table["system-one-config"].status == "fail"


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status
        self.headers = {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return b"{}"

    def getcode(self) -> int:
        return self.status


def _decisions_runtime(tmp_path: Path, **overrides: str) -> cure_runtime.ReviewflowRuntime:
    lines = ['[system_one]', 'enabled = true', 'provider = "openrouter-decisions"']
    for key, value in overrides.items():
        lines.append(f'{key} = "{value}"')
    return _runtime(tmp_path, "\n".join(lines))


def _install_urlopen(
    monkeypatch: pytest.MonkeyPatch, impl: object
) -> list[urllib.request.Request]:
    calls: list[urllib.request.Request] = []

    def _wrapped(request: urllib.request.Request, *args: object, **kwargs: object):
        calls.append(request)
        return impl(request, *args, **kwargs)  # type: ignore[operator]

    monkeypatch.setattr(cure_runtime.urllib.request, "urlopen", _wrapped)
    return calls


def test_system_one_api_key_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_probe(*args: object, **kwargs: object):
        raise AssertionError("this provider must not probe")

    monkeypatch.setattr(cure_runtime.urllib.request, "urlopen", _no_probe)
    local = _system_one_checks(
        _runtime(tmp_path / "local", '[system_one]\nenabled = true\nprovider = "local"')
    )["system-one-api-key"]
    assert local.status == "ok"
    assert "not required for local provider" in local.detail

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls = _install_urlopen(monkeypatch, _no_probe)
    unset = _system_one_checks(_decisions_runtime(tmp_path / "unset"))["system-one-api-key"]
    assert unset.status == "fail"
    assert "key unset" in unset.detail
    assert calls == []

    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    calls = _install_urlopen(monkeypatch, _no_probe)
    blank = _system_one_checks(_decisions_runtime(tmp_path / "blank"))["system-one-api-key"]
    assert blank.status == "fail"
    assert "key unset" in blank.detail
    assert calls == []

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    calls = _install_urlopen(monkeypatch, lambda request, *a, **k: _FakeResponse(200))
    ok = _system_one_checks(_decisions_runtime(tmp_path / "ok"))["system-one-api-key"]
    assert ok.status == "ok"
    assert calls and calls[0].full_url == "https://openrouter.ai/api/v1/key"
    assert calls[0].get_method() == "GET"

    def _http_error(code: int):
        def _raise(request: urllib.request.Request, *args: object, **kwargs: object):
            raise urllib.error.HTTPError(request.full_url, code, "err", {}, None)

        return _raise

    for code in (401, 402, 403):
        calls = _install_urlopen(monkeypatch, _http_error(code))
        rejected = _system_one_checks(_decisions_runtime(tmp_path / f"auth{code}"))[
            "system-one-api-key"
        ]
        assert rejected.status == "fail"
        assert str(code) in rejected.detail

    for code in (429, 500, 503, 418):
        calls = _install_urlopen(monkeypatch, _http_error(code))
        warned = _system_one_checks(_decisions_runtime(tmp_path / f"warn{code}"))[
            "system-one-api-key"
        ]
        assert warned.status == "warn"

    def _network_error(*args: object, **kwargs: object):
        raise urllib.error.URLError("boom Authorization: Bearer sk-test-key")

    calls = _install_urlopen(monkeypatch, _network_error)
    network = _system_one_checks(_decisions_runtime(tmp_path / "network"))["system-one-api-key"]
    assert network.status == "warn"
    assert "sk-test-key" not in network.detail

    def _timeout(*args: object, **kwargs: object):
        raise TimeoutError("timed out")

    calls = _install_urlopen(monkeypatch, _timeout)
    timeout = _system_one_checks(_decisions_runtime(tmp_path / "timeout"))["system-one-api-key"]
    assert timeout.status == "warn"
    assert "sk-test-key" not in timeout.detail


def test_system_one_selftest_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_network(*args: object, **kwargs: object):
        raise AssertionError("the self-test must not touch the network")

    monkeypatch.setattr(cure_runtime.urllib.request, "urlopen", _no_network)
    monkeypatch.delenv("CURE_SYSTEM_ONE_SCRIPT", raising=False)
    ok = _system_one_checks(
        _runtime(tmp_path / "local_ok", '[system_one]\nenabled = true\nprovider = "local"')
    )["system-one-selftest"]
    assert ok.status == "ok"

    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(tmp_path / "missing.json"))
    broken = _system_one_checks(
        _runtime(tmp_path / "broken_script", '[system_one]\nenabled = true\nprovider = "local"')
    )["system-one-selftest"]
    assert broken.status == "fail"
    assert "CURE_SYSTEM_ONE_SCRIPT" in broken.detail
    assert "Traceback" not in broken.detail

    monkeypatch.delenv("CURE_SYSTEM_ONE_SCRIPT", raising=False)
    failed_config = _system_one_checks(
        _runtime(
            tmp_path / "failed_config",
            '[system_one]\nenabled = true\nprovider = "typesafe"',
        )
    )
    assert "system-one-selftest" not in failed_config

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-super-secret")
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(tmp_path / "missing.json"))
    leaked = _system_one_checks(
        _runtime(tmp_path / "leak", '[system_one]\nenabled = true\nprovider = "local"')
    )["system-one-selftest"]
    assert leaked.status == "fail"
    assert "sk-super-secret" not in leaked.detail


def test_doctor_dependent_gating_summary_and_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CURE_SYSTEM_ONE_SCRIPT", raising=False)
    real_checks = cure_runtime._doctor_runtime_checks

    def _only_system_one(runtime: object, **kwargs: object) -> list[cure_runtime.DoctorCheck]:
        return [
            item
            for item in real_checks(runtime, **kwargs)  # type: ignore[arg-type]
            if item.name.startswith("system-one")
        ]

    def _run(runtime: cure_runtime.ReviewflowRuntime) -> tuple[int, dict[str, object]]:
        args = argparse.Namespace(json_output=True, pr_url=None, agent_runtime_profile=None)
        with mock.patch.object(cure_commands, "_doctor_runtime_checks", side_effect=_only_system_one), \
            mock.patch.object(cure_runtime.shutil, "which", return_value=None):
            rc = cure_commands.doctor_flow(args, runtime=runtime)
        return rc, json.loads(capsys.readouterr().out)

    healthy_rc, healthy = _run(
        _runtime(tmp_path / "healthy", '[system_one]\nenabled = true\nprovider = "local"')
    )
    names = [c["name"] for c in healthy["checks"]]  # type: ignore[index]
    assert names == ["system-one-config", "system-one-api-key", "system-one-selftest"]
    assert all(c["status"] == "ok" for c in healthy["checks"])  # type: ignore[index]
    assert healthy["summary"] == {"ok": 3, "warn": 0, "fail": 0}
    assert "system_one" not in healthy
    assert healthy_rc == 0

    failed_rc, failed = _run(
        _runtime(tmp_path / "failed", '[system_one]\nenabled = true\nprovider = "typesafe"')
    )
    assert [c["name"] for c in failed["checks"]] == ["system-one-config"]  # type: ignore[index]
    assert failed["summary"] == {"ok": 0, "warn": 0, "fail": 1}
    assert failed_rc == 1

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-warn-key")
    monkeypatch.setattr(
        cure_runtime.urllib.request,
        "urlopen",
        lambda request, *a, **k: (_ for _ in ()).throw(
            urllib.error.HTTPError(request.full_url, 429, "slow down", {}, None)
        ),
    )
    warn_rc, warned = _run(_decisions_runtime(tmp_path / "warn"))
    by_name = {c["name"]: c for c in warned["checks"]}  # type: ignore[index]
    assert by_name["system-one-api-key"]["status"] == "warn"
    assert warned["summary"]["fail"] == 0  # type: ignore[index]
    assert warned["summary"]["warn"] == 1  # type: ignore[index]
    assert warn_rc == 0


def test_setup_advisory_display_only(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "setup", None)
    stdout = io.StringIO()
    stderr = io.StringIO()
    stdin = io.StringIO()
    args = argparse.Namespace(
        cmd="setup",
        agent=None,
        force=False,
        skip_install=False,
        llm_preset=None,
    )
    with mock.patch.object(cure_commands, "_stream_is_tty", return_value=False), \
        mock.patch.object(
            cure_commands,
            "_resolve_bootstrap_agent_choice",
            return_value={"agent": None},
        ) as agent_choice, \
        mock.patch.object(
            cure_commands,
            "resolve_local_agent_selection",
            return_value={"blocking": False},
        ), \
        mock.patch.object(cure_commands.shutil, "which", return_value="/usr/bin/chunkhound"):
        rc = cure_commands.setup_flow(
            args, runtime=runtime, stdin=stdin, stdout=stdout, stderr=stderr
        )

    out = stdout.getvalue()
    assert rc == 0
    assert out.count("[system_one]") == 1
    assert "documentation" in out
    assert "cure doctor" in out
    assert "Select" not in out and "Select" not in stderr.getvalue()
    assert agent_choice.call_args.kwargs["interactive"] is False
    assert stdin.getvalue() == ""
    assert "system_one" not in runtime.config_path.read_text(encoding="utf-8")


def test_decisions_provider_doctor_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    calls = _install_urlopen(monkeypatch, lambda request, *a, **k: _FakeResponse(200))

    checks = _system_one_checks(
        _runtime(
            tmp_path / "decisions",
            '[system_one]\nenabled = true\nprovider = "openrouter-decisions"\n'
            'model = "typesafe/jev-1.13"',
        )
    )
    assert checks["system-one-config"].status == "ok"
    assert "openrouter-decisions" in checks["system-one-config"].detail
    assert checks["system-one-api-key"].status == "ok"
    assert "openrouter-decisions" in checks["system-one-api-key"].detail
    assert calls and calls[0].full_url == "https://openrouter.ai/api/v1/key"
    assert calls[0].get_method() == "GET"
