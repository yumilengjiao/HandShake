"""Native bridge tests: no camera, no training, no external input."""

import json
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from backend.desktop import decode_command
from backend.native_gesture import NativeGestureSession
from shared.native_gesture_protocol import NativeGestureCommand


class Clock:
    now = 1_000_000_000

    def __call__(self):
        return self.now


class Worker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.done, self.stop_requested = threading.Event(), threading.Event()
        self.thread = SimpleNamespace(is_alive=lambda: False, join=lambda timeout: None)
        self.adapter = self.error = self.latest = None
        self.starts = 0

    def start(self):
        self.starts += 1

    def take(self):
        value, self.latest = self.latest, None
        return value

    def request_stop(self):
        self.stop_requested.set()
        self.done.set()


ASSETS = {
    "package": "not-loaded",
    "sha256": "a" * 64,
    "calibration": "not-loaded",
    "calibration_sha256": "b" * 64,
}


def command(session, kind, snapshot=None):
    value = snapshot or session.snapshot()
    return NativeGestureCommand(
        version=1,
        session_id=value["session_id"],
        command_id=str(session.last_command_id + 1),
        camera_epoch=value["camera_epoch"],
        based_on_sequence=value["sequence"],
        kind=kind,
    )


def test_preview_survives_model_age_limit_but_stalled_camera_is_cleared():
    from backend.perception.preview import CameraFrame

    clock = Clock()
    session = NativeGestureSession(
        assets=ASSETS, perception_model="model", worker_factory=Worker, clock=clock
    )
    session.accept(command(session, "open_camera"))
    source_ns = clock.now
    session.worker.latest = CameraFrame(
        1, source_ns, 640, 480, b"display-only", (), "processed_hands", 30
    )
    for age in (0, 149_000_000, 151_000_000, 400_000_000, 750_000_000):
        clock.now = source_ns + age
        value = session.snapshot()
        assert value["camera_state"] == "running" and value["jpeg"]
        assert not value["eligible_owner"] and not value["actions_enabled"]
    clock.now += 1
    value = session.snapshot()
    assert value["reason"] == "input_stalled" and value["jpeg"] == ""
    assert not value["enabled"] and not value["actions_enabled"]
    session.close()


def test_future_preview_frame_is_not_displayed():
    from backend.perception.preview import CameraFrame

    clock = Clock()
    session = NativeGestureSession(
        assets=ASSETS, perception_model="model", worker_factory=Worker, clock=clock
    )
    session.accept(command(session, "open_camera"))
    session.worker.latest = CameraFrame(
        1, clock.now + 1, 640, 480, b"future", (), "processed_hands", 30
    )
    assert session.snapshot()["jpeg"] == ""
    session.close()


def test_initial_idle_and_no_model_has_no_camera_side_effects():
    session = NativeGestureSession(
        worker_factory=lambda **kw: pytest.fail("camera must not open")
    )
    value = session.snapshot()
    assert value["camera_state"] == "stopped" and value["camera_released"]
    assert not value["model_available"] and not value["actions_enabled"]
    for kind in [
        "heartbeat",
        "surface_ack",
        "enable",
        "choose_owner",
        "stop",
        "close_camera",
        "open_camera",
    ]:
        session.accept(command(session, kind))
    assert session.worker is None


def test_explicit_camera_open_and_single_worker_until_confirmed_release():
    session = NativeGestureSession(
        assets=ASSETS, perception_model="model", worker_factory=Worker
    )
    session.accept(command(session, "open_camera"))
    first = session.worker
    assert first.starts == 1 and first.kwargs["allow_application_local"] is True
    assert not session.snapshot()["enabled"]
    session.accept(command(session, "open_camera"))
    assert session.worker is first and first.starts == 1
    first.request_stop = lambda: first.stop_requested.set()
    session.accept(command(session, "close_camera"))
    assert session.snapshot()["camera_state"] == "stopping"
    session.accept(command(session, "open_camera"))
    assert session.worker is first
    first.done.set()
    assert session.snapshot()["camera_released"]
    session.accept(command(session, "open_camera"))
    assert session.worker is not first
    session.close()


def test_old_surface_and_old_generation_cannot_start_or_reenable():
    clock = Clock()
    session = NativeGestureSession(
        assets=ASSETS, perception_model="model", worker_factory=Worker, clock=clock
    )
    old = session.snapshot()
    clock.now += 500_000_000
    session.accept(command(session, "open_camera", old))
    assert session.worker is None
    current = session.snapshot()
    session.accept(command(session, "stop", current))
    session.accept(command(session, "open_camera", current))
    assert session.worker is None  # stop revoked the buffered surface token


def test_parent_timeout_and_stuck_release_never_claim_success():
    clock = Clock()
    session = NativeGestureSession(
        assets=ASSETS, perception_model="model", worker_factory=Worker, clock=clock
    )
    session.accept(command(session, "open_camera"))
    session.worker.request_stop = lambda: session.worker.stop_requested.set()
    clock.now += 1_500_000_000
    value = session.snapshot()
    assert value["camera_state"] == "stopping" and not value["actions_enabled"]
    clock.now += 2_000_000_000
    value = session.snapshot()
    assert session.closed and value["reason"] == "release_unconfirmed"
    assert not value["camera_released"]


@pytest.mark.parametrize(
    "extra",
    [
        {"kind": "next"},
        {"kind": "previous"},
        {"logits": [9]},
        {"class": "TRT"},
        {"version": True},
    ],
)
def test_wire_refuses_navigation_prediction_and_unknown_fields(extra):
    session = NativeGestureSession()
    data = command(session, "heartbeat").model_dump() | extra
    with pytest.raises(ValueError):
        decode_command(json.dumps(data).encode() + b"\n", NativeGestureCommand)


def test_real_child_connect_and_eof_stay_camera_free():
    result = subprocess.run(
        [sys.executable, "-m", "backend.native_gesture_cli"],
        check=False,
        input=b"",
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode()
    values = [json.loads(line) for line in result.stdout.splitlines()]
    assert values and all(
        v["camera_state"] == "stopped" and not v["enabled"] for v in values
    )
    assert b"camera_opening" not in result.stderr


def test_default_app_uses_existing_receipts_without_training_cache(
    tmp_path, monkeypatch
):
    from backend.workflows.system import launch_app
    from backend.workflows.workspace import Settings, Workspace, write_json

    ws = Workspace(tmp_path, Settings())
    (tmp_path / "frontend/native").mkdir(parents=True)
    (tmp_path / "frontend/native/pubspec.yaml").touch()
    perception = ws.path(ws.settings.app.perception_model)
    perception.parent.mkdir(parents=True)
    perception.touch()
    run = ws.runs / "known"
    write_json(ws.latest, {"format_version": "workflow-run-v1", "run": "known"})
    write_json(
        run / "export.json",
        {
            "format_version": "model-export-receipt-v1",
            "directory": "models/known",
            "manifest_sha256": "a" * 64,
        },
    )
    write_json(
        run / "calibration-receipt.json",
        {"path": "val/calibration.json", "sha256": "b" * 64},
    )
    before = ws.latest.read_bytes()
    monkeypatch.setattr("backend.workflows.system.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "backend.workflows.system.shutil.which", lambda _: "/tools/flutter"
    )
    monkeypatch.setattr(
        "backend.workflows.system.importlib.util.find_spec", lambda _: object()
    )
    monkeypatch.setattr(
        "backend.workflows.training._bound_run",
        lambda *a: pytest.fail("no cache access"),
    )
    monkeypatch.setattr(
        "backend.workflows.training.export_run",
        lambda *a, **kw: pytest.fail("no re-export"),
    )
    verified, launched = [], []
    monkeypatch.setattr(
        "backend.workflows.deployment.verify_assets",
        lambda *a, **kw: verified.append((a, kw)),
    )
    monkeypatch.setattr(
        "backend.native_launcher.main", lambda a, **kw: launched.append(a) or 0
    )
    assert launch_app(ws) == 0
    assert verified[0][1]["shadow"] is True  # checks real calibrated 15-class contract
    assert "--model-package" in launched[0] and "--calibration" in launched[0]
    assert "--shadow-session" not in launched[0]
    assert ws.latest.read_bytes() == before and not ws.cache.exists()
