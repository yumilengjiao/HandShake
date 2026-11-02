"""External model handoff and desktop launch tests. Analytical forward only, never training."""

import subprocess
import sys

import pytest

from backend.workflows.cli import main
from backend.workflows.system import launch_app
from backend.workflows.workspace import Settings, WorkflowError, Workspace
from tests.integration.model_bridge_fixture import fixture_bundle


@pytest.fixture
def handed_off(tmp_path):
    assets = tmp_path / "handed off 模型"
    assets.mkdir()
    bundle, _, calibration_sha = fixture_bundle(assets, "mlp")
    ws = Workspace(tmp_path, Settings())
    return ws, assets, bundle.manifest_sha256, calibration_sha


@pytest.mark.parametrize("args", [
    ["--package", "model"], ["--sha256", "a" * 64], ["--shadow"],
    ["--verify-only"], ["--allow-synthetic-model"],
    ["--model", "--package", "model", "--sha256", "a" * 64],
    ["--calibration", "sidecar"],
])
def test_invalid_handoff_options_fail_before_bootstrap(args, monkeypatch):
    monkeypatch.setattr("backend.workflows.runtime.ensure_training",
                        lambda *a, **k: pytest.fail("invalid CLI must not install anything"))
    with pytest.raises(SystemExit) as error:
        main(["app", *args])
    assert error.value.code == 2


def test_handoff_preflight_needs_no_local_dataset_flutter_camera_or_training(handed_off, monkeypatch, capsys):
    ws, assets, sha, calibration_sha = handed_off
    monkeypatch.setattr("backend.workflows.runtime.ensure_training", lambda *a, **k: None)
    monkeypatch.setattr("backend.workflows.training._bound_run",
                        lambda *a: pytest.fail("external model must not read local training data"))
    monkeypatch.setattr("backend.native_launcher.main",
                        lambda *a, **k: pytest.fail("verification cannot open an app"))
    command = [
        "app", "--workspace", str(ws.root), "--package", str(assets / "bundle"),
        "--sha256", sha, "--calibration", str(assets / "calibration.json"),
        "--calibration-sha256", calibration_sha, "--shadow", "--allow-synthetic-model", "--verify-only",
    ]
    assert main(command) == 0
    assert "15类" in capsys.readouterr().out
    assert not ws.cache.exists() and not ws.runs.exists()
    # Other tests import trainer for guards. Check import isolation in a fresh process.
    result = subprocess.run([sys.executable, "-c",
        "import sys; from backend.workflows.cli import main; "
        "import backend.workflows.runtime as runtime; runtime.ensure_training=lambda *a, **k: None; "
        "assert main(sys.argv[1:]) == 0; "
        "assert not {'algorithm.training.trainer','mediapipe','backend.native_launcher'} & sys.modules.keys()",
        *command], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("bad", ["model_hash", "calibration_hash", "no_calibration", "no_opt_in"])
def test_external_handoff_fails_closed(handed_off, bad):
    ws, assets, sha, calibration_sha = handed_off
    with pytest.raises((WorkflowError, ValueError)):
        launch_app(ws, package=assets / "bundle", sha256="0" * 64 if bad == "model_hash" else sha,
                   calibration=None if bad == "no_calibration" else assets / "calibration.json",
                   calibration_sha256=None if bad == "no_calibration" else (
                       "0" * 64 if bad == "calibration_hash" else calibration_sha),
                   shadow=True, allow_synthetic=bad != "no_opt_in", verify_only=True)


def test_prebuilt_external_shadow_launch_preserves_pins_without_sdk(handed_off, monkeypatch):
    ws, assets, sha, calibration_sha = handed_off
    native = ws.root / "frontend/native"
    native.mkdir(parents=True)
    (native / "pubspec.yaml").touch()
    executable = ws.root / "release/native app"
    executable.parent.mkdir()
    executable.write_text("not executed")
    executable.chmod(0o700)
    perception = ws.path(ws.settings.app.perception_model)
    perception.parent.mkdir(parents=True)
    perception.touch()
    monkeypatch.setattr("backend.workflows.system.platform.system", lambda: "Linux")
    monkeypatch.setattr("backend.workflows.system.shutil.which", lambda _: None)
    monkeypatch.setattr("backend.workflows.system.importlib.util.find_spec", lambda _: object())
    calls = []
    monkeypatch.setattr("backend.native_launcher.main", lambda args, **kw: calls.append(args) or 0)
    assert launch_app(ws, package=assets / "bundle", sha256=sha,
                      calibration=assets / "calibration.json", calibration_sha256=calibration_sha,
                      shadow=True, allow_synthetic=True, bundle=executable) == 0
    assert calls[0][:2] == ["--bundle", str(executable)]
    assert "--flutter" not in calls[0]
    assert "--shadow-session" in calls[0]
    assert calls[0][calls[0].index("--model-sha256") + 1] == sha


def test_prebuilt_missing_binary_does_not_fall_back_to_sdk(tmp_path, monkeypatch):
    ws = Workspace(tmp_path, Settings())
    (tmp_path / "frontend/native").mkdir(parents=True)
    (tmp_path / "frontend/native/pubspec.yaml").touch()
    monkeypatch.setattr("backend.workflows.system.platform.system", lambda: "Linux")
    with pytest.raises(WorkflowError, match="不可执行"):
        launch_app(ws, bundle="missing")
