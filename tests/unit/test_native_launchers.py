"""Portable native check/launcher plumbing. Never launches Flutter or creates a model."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("returncode", [0, 1])
def test_disposable_native_check_preserves_interpreter_and_exit_status(
    tmp_path, monkeypatch, returncode
):
    module = load_script("check_native_shadow")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HANDSHAKE_MODEL_PACKAGE", "do-not-use-real-model")
    monkeypatch.setenv("HANDSHAKE_TEST_SOURCE", "0")
    monkeypatch.setenv("HANDSHAKE_SHADOW_SCREENSHOT", "do-not-write.png")
    monkeypatch.setenv("PYTHONPATH", "unrelated-host-injection")
    monkeypatch.setenv("PYTHONHOME", "unrelated-host-python")
    monkeypatch.setenv("FLUTTER_BIN", "portable-flutter")
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda v: "/sdk/flutter" if v == "portable-flutter" else None,
    )
    created = []

    def fixture(output, kind):
        assert kind == "tcn" and output.is_dir()
        created.append(output)
        return "a" * 64, "b" * 64

    def run(command, **kwargs):
        assert command == [
            "/sdk/flutter",
            "--no-version-check",
            "test",
            "integration_test/shadow_session_test.dart",
            "-d",
            "linux",
        ]
        assert kwargs["cwd"] == ROOT / "frontend/native"
        env = kwargs["env"]
        assert env["HANDSHAKE_ROOT"] == str(ROOT)
        assert env["HANDSHAKE_PYTHON"] == sys.executable
        assert env["HANDSHAKE_TEST_SOURCE"] == "1"
        assert env["HANDSHAKE_MODEL_PACKAGE"] == str(created[0] / "bundle")
        assert env["HANDSHAKE_MODEL_SHA256"] == "a" * 64
        assert env["HANDSHAKE_CALIBRATION_SHA256"] == "b" * 64
        assert (
            not {"PYTHONPATH", "PYTHONHOME", "HANDSHAKE_SHADOW_SCREENSHOT"} & env.keys()
        )
        assert kwargs["timeout"] == 45 and kwargs["check"] is False
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(module, "_make_fixture", fixture)
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main(["--model-type", "tcn", "--timeout", "45"]) == returncode
    assert not created[0].exists()


@pytest.mark.parametrize("args", [["--timeout", "0"], ["--flutter", "missing"]])
def test_native_check_preflight_does_not_allocate_assets(monkeypatch, args):
    module = load_script("check_native_shadow")
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        module, "_make_fixture", lambda *a: pytest.fail("no fixture before preflight")
    )
    with pytest.raises(SystemExit) as error:
        module.main(args)
    assert error.value.code == 2


def test_native_check_timeout_is_failure_and_cleans_up(monkeypatch):
    module = load_script("check_native_shadow")
    monkeypatch.setattr(module.shutil, "which", lambda _: "/sdk/flutter")
    created = []
    monkeypatch.setattr(
        module, "_make_fixture", lambda p, _: created.append(p) or ("a" * 64, "b" * 64)
    )

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(module.subprocess, "run", timeout)
    assert module.main(["--timeout", "1"]) == 124
    assert not created[0].exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--shadow-session"],
        ["--allow-synthetic-model"],
        ["--shadow-session", "--calibration", "missing"],
    ],
)
def test_shadow_launcher_rejects_partial_or_implicit_authorization(monkeypatch, args):
    module = load_script("run_native")
    monkeypatch.setattr(
        module.subprocess, "call", lambda *a, **k: pytest.fail("must not launch")
    )
    with pytest.raises(SystemExit) as error:
        module.main(args)
    assert error.value.code == 2


def test_shadow_launcher_only_forwards_explicit_assets(tmp_path, monkeypatch):
    module = load_script("run_native")
    executable = tmp_path / "native"
    executable.touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HANDSHAKE_ALLOW_SYNTHETIC_MODEL", "1")
    captured = {}
    monkeypatch.setattr(
        module.subprocess, "call", lambda cmd, **kw: captured.update(kw) or 0
    )
    assert (
        module.main(
            [
                "--bundle",
                str(executable),
                "--shadow-session",
                "--perception-model",
                "hand.task",
                "--model-package",
                "model",
                "--model-sha256",
                "a" * 64,
                "--calibration",
                "calibration.json",
                "--calibration-sha256",
                "b" * 64,
            ]
        )
        == 0
    )
    env = captured["env"]
    assert env["HANDSHAKE_SHADOW_SESSION"] == "1"
    assert env["HANDSHAKE_ALLOW_SYNTHETIC_MODEL"] == "0"
    assert env["HANDSHAKE_MODEL_PACKAGE"] == str(tmp_path / "model")
    assert env["HANDSHAKE_PYTHON"] == sys.executable
