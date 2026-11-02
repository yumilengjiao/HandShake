"""Workflow defaults, short CLI errors and dependency plumbing; never train."""

import sys

import pytest

from backend.workflows.cli import DESCRIPTIONS, main
from backend.workflows.workspace import (
    Settings,
    WorkflowError,
    Workspace,
    operation_lock,
    workspace,
    write_json,
)


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path, Settings())


def test_workspace_discovery_and_local_overrides_are_relative(tmp_path, monkeypatch):
    (tmp_path / "handshake.toml").write_text(
        '[train]\nbatch_size=32\n[app]\nflutter="custom-flutter"\n'
    )
    (tmp_path / "handshake.local.toml").write_text('[train]\ndevice="cuda:1"\n')
    child = tmp_path / "nested" / "deep"
    child.mkdir(parents=True)
    monkeypatch.chdir(child)
    value = workspace()
    assert value.root == tmp_path
    assert (
        value.settings.train.batch_size == 32
        and value.settings.train.device == "cuda:1"
    )
    assert value.path(value.settings.data.raw) == tmp_path / "data/raw/ipn-hands"
    assert value.settings.app.flutter == "custom-flutter"


def test_config_typos_fail_instead_of_ignoring_them(tmp_path, capsys):
    (tmp_path / "handshake.toml").write_text("[train]\nbatch_szie=32\n")
    assert main(["doctor", "--workspace", str(tmp_path)]) == 2
    assert "batch_szie" in capsys.readouterr().err


@pytest.mark.parametrize("command", DESCRIPTIONS)
def test_all_daily_commands_have_help_without_optional_dependencies(command):
    with pytest.raises(SystemExit) as result:
        main([command, "--help"])
    assert result.value.code == 0


@pytest.mark.parametrize(
    "command", ["train", "resume", "evaluate", "calibrate", "export", "prepare"]
)
def test_missing_data_has_a_short_next_step_without_traceback(ws, capsys, command):
    assert main([command, "--workspace", str(ws.root)]) == 2
    text = capsys.readouterr().err
    assert "建议" in text and "Traceback" not in text
    assert not ws.runs.exists()


@pytest.mark.parametrize("flag", ["--epochs", "--batch-size"])
def test_invalid_numeric_override_does_not_silently_use_default(ws, flag):
    with pytest.raises(SystemExit) as result:
        main(["train", flag, "0", "--workspace", str(ws.root)])
    assert result.value.code == 2


def test_operation_lock_is_exclusive_and_cleans_up(ws):
    with operation_lock(ws.cache, "prepare"):
        with pytest.raises(WorkflowError, match="已有"):
            with operation_lock(ws.cache, "prepare"):
                pytest.fail("must not get second lock")
    assert not (ws.cache / ".prepare.lock").exists()


def test_setup_gpu_is_explicit_and_preserves_local_config(ws, monkeypatch):
    from backend.workflows import runtime

    (ws.root / "pyproject.toml").write_text('[project]\nname="handshake-core"\n')
    local = ws.root / "handshake.local.toml"
    local.write_text("[train]\nbatch_size=16\n")
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/tools/uv")
    calls = []
    monkeypatch.setattr(
        runtime.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 0
    )
    assert runtime.setup(ws, gpu=True) == 0
    assert calls[0] == [
        "/tools/uv",
        "sync",
        "--locked",
        "--extra",
        "cuda",
        "--extra",
        "dev",
    ]
    configured = workspace(ws.root)
    assert configured.settings.train.batch_size == 16
    assert configured.settings.runtime.profile == "cuda"


def test_bootstrap_uses_locked_extra_and_avoids_recursive_install(ws, monkeypatch):
    from backend.workflows import runtime

    (ws.root / "pyproject.toml").write_text('[project]\nname="handshake-core"\n')
    monkeypatch.delenv("HANDSHAKE_WORKFLOW_EXTRA", raising=False)
    monkeypatch.setenv("PYTHONPATH", "host-injection")
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/tools/uv")
    calls = []
    monkeypatch.setattr(
        runtime.subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 7
    )
    assert runtime.ensure_training(ws, ["train"]) == 7
    assert calls[0][0][:5] == ["/tools/uv", "run", "--locked", "--extra", "training"]
    assert "PYTHONPATH" not in calls[0][1]["env"]
    assert calls[0][1]["env"]["HANDSHAKE_WORKFLOW_EXTRA"] == "training"
    monkeypatch.setenv("HANDSHAKE_WORKFLOW_EXTRA", "training")
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _: object())
    assert runtime.ensure_training(ws, ["train"]) is None
    assert len(calls) == 1


def test_nonlinux_app_fails_without_launching_or_opening_devices(ws, monkeypatch):
    from backend.workflows import system

    monkeypatch.setattr(system.platform, "system", lambda: "Windows")
    with pytest.raises(WorkflowError, match="Linux"):
        system.launch_app(ws)


def test_default_check_excludes_neural_training(ws, monkeypatch):
    from backend.workflows import system

    (ws.root / "tests").mkdir()
    monkeypatch.setattr(system.importlib.util, "find_spec", lambda _: object())
    calls = []
    monkeypatch.setattr(
        system.subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 0
    )
    assert system.check(ws) == 0
    assert "--ignore=tests/unit/test_training.py" in calls[0][0]
    assert calls[0][1]["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert calls[0][1]["env"]["HANDSHAKE_NO_TRAINING"] == "1"
    assert calls[0][1]["env"]["HANDSHAKE_HARDWARE_TRAINING"] == "0"
    assert "--ignore=tests/integration/test_accelerator_training.py" in calls[0][0]
    assert system.check(ws, training=True) == 0
    assert calls[1][0] == [sys.executable, "-m", "pytest"]
    assert calls[1][1]["env"]["HANDSHAKE_NO_TRAINING"] == "0"
    assert calls[1][1]["env"]["HANDSHAKE_HARDWARE_TRAINING"] == "1"


@pytest.mark.parametrize("stale", [False, True])
def test_app_model_uses_pinned_calibration_without_opening_camera(
    ws, monkeypatch, stale
):
    from backend.workflows import system, training
    import backend.native_launcher as launcher
    from algorithm.data.source_archive import file_sha256

    native = ws.root / "frontend/native"
    native.mkdir(parents=True)
    (native / "pubspec.yaml").touch()
    perception = ws.path(ws.settings.app.perception_model)
    perception.parent.mkdir(parents=True)
    perception.write_bytes(b"not a real detector; never loaded")
    root = ws.runs / "fixture-run"
    root.mkdir(parents=True)
    sidecar = root / "evaluations/fixture/calibration.json"
    write_json(sidecar, {"source": {"model_sha256": "b" * 64 if stale else "a" * 64}})
    write_json(
        root / "calibration-receipt.json",
        {
            "path": "evaluations/fixture/calibration.json",
            "sha256": file_sha256(sidecar),
        },
    )
    monkeypatch.setattr(system.platform, "system", lambda: "Linux")
    monkeypatch.setattr(system.shutil, "which", lambda _: "/tools/flutter")
    monkeypatch.setattr(system.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(training, "_bound_run", lambda _: (root, {}))
    monkeypatch.setattr(
        training,
        "export_run",
        lambda *a, **k: {"manifest_sha256": "a" * 64, "directory": "model"},
    )
    commands = []
    monkeypatch.setattr(launcher, "main", lambda args, **kw: commands.append(args) or 0)
    if stale:
        with pytest.raises(WorkflowError, match="旧模型"):
            system.launch_app(ws, model=True)
        assert not commands
    else:
        assert system.launch_app(ws, model=True) == 0
        args = commands[0]
        assert args[args.index("--calibration") + 1] == str(sidecar)
        assert "--shadow-session" not in args


def test_run_pointer_rejects_escape(ws):
    from backend.workflows.training import latest_run

    write_json(ws.latest, {"format_version": "workflow-run-v1", "run": "../../.."})
    with pytest.raises(WorkflowError, match="越出"):
        latest_run(ws)
