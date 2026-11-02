"""HSP5 protocol and actual-network shadow adapter checks. All source/image inputs are procedural."""

import io
import json
import threading
from pathlib import Path

import pytest

from backend.controller import ApplicationController
from backend.interaction.public_replay import ReplayClock, synthetic_input
from backend.perception.preview import CameraFrame
from backend.perception.shadow import ShadowCameraWorker, ShadowClassifier
from backend.shadow_preview import NativeShadowSession
from shared.calibrated_preview_protocol import read_calibrated_packet
from shared.preview_protocol import PreviewHand, PreviewPoint
from shared.shadow_preview_protocol import (
    ShadowCommand,
    ShadowSnapshot,
    read_shadow_packet,
)
from tests.integration.model_bridge_fixture import fixture_bundle


def fixture():
    return Path("frontend/native/test/fixtures/shadow_preview_v5.bin").read_bytes()


class Rig:
    def __init__(self, tmp_path):
        self.bundle, self.calibration, self.sha = fixture_bundle(tmp_path, "mlp")
        self.clock = ReplayClock()
        self.stop = threading.Event()
        self.adapter = ShadowClassifier(
            self.bundle,
            self.calibration,
            self.sha,
            self.stop,
            allow_synthetic=True,
            clock=self.clock,
        )
        self.source = ApplicationController("native-source", timebase="monotonic")
        self.index = self.command = 0
        self.frame = self.notice = None

    def update(self, x=0.5, dt=30_000_000):
        self.clock.now += dt
        self.index += 1
        _, self.frame = synthetic_input(
            self.source.snapshot(),
            timestamp_ns=self.clock.now,
            frame_id=self.index,
            model_version="unused",
            x=x,
        )
        self.notice = self.adapter.update(self.frame)
        return self.notice

    def control(self, kind, **changes):
        self.command += 1
        raw = self.adapter.snapshot()
        values = dict(
            version=5,
            session_id="a" * 32,
            command_id=str(self.command),
            epoch="7",
            kind=kind,
            model_epoch=str(raw["epoch"]),
            model_sequence=str(raw["sequence"]),
            model_revision=str(raw["revision"]),
        )
        if kind == "request_owner":
            values["track_id"] = "operator"
        return self.adapter.control(ShadowCommand(**(values | changes)))

    def begin(self):
        assert self.update() is None
        assert self.control("surface_ack") and self.control("enable_shadow")
        for _ in range(32):
            self.update()
            self.control("surface_ack")

    def close(self):
        self.stop.set()
        assert self.adapter.close()


@pytest.fixture
def rig(tmp_path):
    value = Rig(tmp_path)
    yield value
    value.close()


def test_v5_roundtrip_separates_source_and_model_epochs_and_schema():
    packet = read_shadow_packet(io.BytesIO(fixture()))
    assert packet.encode() == fixture()
    assert packet.snapshot.epoch == "1" and packet.snapshot.shadow.model_epoch == "3"
    assert packet.snapshot.classification.context_epoch == "3"
    assert not packet.snapshot.shadow.actions_enabled
    with pytest.raises(ValueError):
        read_calibrated_packet(io.BytesIO(fixture()))
    schema = json.loads(Path("shared/shadow_preview_protocol.schema.json").read_text())
    assert schema == {
        "command": ShadowCommand.model_json_schema(),
        "snapshot": ShadowSnapshot.model_json_schema(),
    }


@pytest.mark.parametrize(
    "change",
    [
        {"actions_enabled": True},
        {"actions_enabled": 0},
        {"actual_page": 2},
        {"actual_page": 1.0},
        {"actual_page": True},
        {"model_revision": "1"},
        {"execution": "live"},
        {"model_epoch": "1"},
        {"model_version": "wrong"},
        {"eligible_owner": "unseen"},
        {"state": "DISABLED"},
        {"owner_track_id": "unexpected"},
    ],
)
def test_forged_or_inconsistent_shadow_state_rejected(change):
    data = read_shadow_packet(io.BytesIO(fixture())).snapshot.model_dump()
    data["shadow"].update(change)
    with pytest.raises(ValueError):
        ShadowSnapshot.model_validate(data)


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "request_owner"},
        {"kind": "enable_shadow"},
        {"kind": "surface_ack"},
        {"kind": "stop", "model_epoch": "1"},
        {"kind": "heartbeat", "model_sequence": "2"},
        {"kind": "start"},
        {"kind": "next_slide"},
        {"execution": "synthetic_application_local"},
    ],
)
def test_invalid_command_or_execution_request_rejected(change):
    with pytest.raises(ValueError):
        ShadowCommand(
            **(
                dict(
                    version=5,
                    session_id="a" * 32,
                    command_id="1",
                    epoch="0",
                    kind="heartbeat",
                )
                | change
            )
        )


def test_actual_network_requires_explicit_enable_and_owner_then_rewarms(rig):
    rig.begin()
    assert rig.adapter.session.controller.snapshot().owner_track_id is None
    assert rig.adapter.snapshot()["eligible_owner"] == "operator"
    assert rig.control("request_owner")
    assert rig.adapter.snapshot()["owner_request_pending"]
    assert (
        rig.update() is None
    )  # binding result retains OLD epoch; never attach to new frame epoch
    assert (
        rig.adapter.snapshot()["epoch"] == 2 and rig.adapter.snapshot()["model_warming"]
    )
    assert rig.control("surface_ack")
    for _ in range(32 + 15):
        rig.update()
        rig.control("surface_ack")
    for x in (0.56, 0.62, 0.68, 0.74):
        rig.update(x, 50_000_000)
        rig.control("surface_ack")
    snap = rig.adapter.snapshot()
    assert snap["candidate"].action == "next_slide" and snap["candidates_total"] == 1
    assert snap["page"] == 1 and snap["revision"] == 0 and snap["execution"] == "shadow"
    assert rig.adapter.session.host.backend.stop_requested.is_set()


def test_old_model_epoch_and_expired_surface_tokens_cannot_grant_owner(rig):
    rig.begin()
    assert not rig.control("request_owner", model_epoch="0")
    assert rig.adapter.session.controller.snapshot().owner_track_id is None
    old = rig.adapter.snapshot()
    rig.clock.now += 600_000_000
    rig.command += 1
    command = ShadowCommand(
        version=5,
        session_id="a" * 32,
        command_id=str(rig.command),
        epoch="7",
        kind="surface_ack",
        model_epoch=str(old["epoch"]),
        model_sequence=str(old["sequence"]),
        model_revision="0",
    )
    assert not rig.adapter.control(command)


def test_bad_or_unapproved_synthetic_model_fails_before_camera_access(tmp_path):
    bundle, _, sha = fixture_bundle(tmp_path, "mlp")

    def forbidden():
        pytest.fail("camera/detector must not be opened")

    worker = ShadowCameraWorker(
        model_package=tmp_path / "bundle",
        model_sha256=bundle.manifest_sha256,
        calibration_path=tmp_path / "calibration.json",
        calibration_sha256=sha,
        source_factory=forbidden,
        detector_factory=forbidden,
    )
    worker._run()
    assert (
        worker.model_failed and worker.done.is_set() and worker.error == "preview_error"
    )


class Worker:
    def __init__(self, rig):
        self.rig = rig
        self.done = threading.Event()
        self.error = None
        self.model_failed = False
        self.classification_max_age_ns = 150_000_000
        self.calibration = rig.calibration
        self.calibration_sha256 = rig.sha
        self.next = None

    def start(self):
        pass

    def take(self):
        result, self.next = self.next, None
        return result

    def request_stop(self):
        self.rig.stop.set()
        self.rig.adapter.stop()

    def shadow_snapshot(self):
        return self.rig.adapter.snapshot()

    def shadow_control(self, command):
        return self.rig.adapter.control(command)

    def publish(self):
        frame = self.rig.frame
        hands = tuple(
            PreviewHand(
                landmarks=tuple(
                    PreviewPoint(x=x, y=y, z=z) for x, y, z in o.landmarks_image
                ),
                validity=tuple(o.validity_mask),
                usable=o.observation_valid,
            )
            for o in frame.observations
        )
        jpg = read_shadow_packet(io.BytesIO(fixture())).jpeg
        self.next = CameraFrame(
            frame.info.frame_id,
            frame.info.timestamp_ns,
            160,
            120,
            jpg,
            hands,
            "processed_hands",
            30.0,
            self.rig.notice,
            self.rig.adapter.offer,
        )


def test_native_service_same_frame_offers_nested_epochs_and_stop_release(rig):
    worker = Worker(rig)
    session = NativeShadowSession(worker_factory=lambda: worker, clock=rig.clock)
    session.epoch = 6
    session.accept(
        ShadowCommand(
            version=5,
            session_id=session.session_id,
            command_id="1",
            epoch="6",
            kind="start_camera",
        )
    )
    rig.begin()
    worker.publish()
    packet = session.snapshot()
    assert packet.snapshot.epoch == "7" and packet.snapshot.shadow.model_epoch == "1"
    assert packet.snapshot.shadow.eligible_owner == "operator"
    assert packet.snapshot.probability is not None
    # Source generation guard must reject even a perfectly valid model token.
    snap = packet.snapshot.shadow
    session.accept(
        ShadowCommand(
            version=5,
            session_id=session.session_id,
            command_id="2",
            epoch="6",
            kind="request_owner",
            model_epoch=snap.model_epoch,
            model_sequence=snap.model_sequence,
            model_revision="0",
            track_id="operator",
        )
    )
    assert rig.adapter.session.controller.snapshot().owner_track_id is None
    assert session.snapshot().snapshot.shadow.control_status == "rejected"
    session.accept(
        ShadowCommand(
            version=5,
            session_id=session.session_id,
            command_id="3",
            epoch="0",
            kind="stop",
        )
    )
    stopped = session.snapshot().snapshot
    assert (
        stopped.state == "stopping"
        and stopped.classification is None
        and stopped.shadow is None
    )
    worker.done.set()
    assert session.snapshot().snapshot.state == "stopped"


def test_native_stop_returns_before_blocked_model_and_withholds_release_ack(rig):
    worker = Worker(rig)
    session = NativeShadowSession(worker_factory=lambda: worker, clock=rig.clock)
    session.accept(
        ShadowCommand(
            version=5,
            session_id=session.session_id,
            command_id="1",
            epoch="0",
            kind="start_camera",
        )
    )
    rig.begin()
    worker.publish()
    session.snapshot()
    entered, proceed = threading.Event(), threading.Event()
    original = rig.bundle.model.forward

    def blocked(value):
        entered.set()
        assert proceed.wait(2)
        return original(value)

    rig.bundle.model.forward = blocked
    model_thread = threading.Thread(target=rig.update)
    try:
        model_thread.start()
        assert entered.wait(1)
        stopper = threading.Thread(
            target=lambda: session.accept(
                ShadowCommand(
                    version=5,
                    session_id=session.session_id,
                    command_id="2",
                    epoch="0",
                    kind="stop",
                )
            )
        )
        stopper.start()
        stopper.join(0.5)
        assert not stopper.is_alive()
        packet = session.snapshot().snapshot
        assert (
            packet.state == "stopping"
            and packet.shadow is None
            and packet.classification is None
        )
        assert rig.adapter.session.snapshot()["model_busy"]
        proceed.set()
        model_thread.join(2)
        worker.done.set()
        assert session.snapshot().snapshot.state == "stopped"
    finally:
        proceed.set()
        model_thread.join(2)
        rig.bundle.model.forward = original


def test_new_model_epoch_or_expired_image_clears_frozen_classification_offer(rig):
    worker = Worker(rig)
    session = NativeShadowSession(worker_factory=lambda: worker, clock=rig.clock)
    session.accept(
        ShadowCommand(
            version=5,
            session_id=session.session_id,
            command_id="1",
            epoch="0",
            kind="start_camera",
        )
    )
    rig.begin()
    worker.publish()
    assert session.snapshot().snapshot.shadow.eligible_owner == "operator"
    assert rig.control("request_owner")
    rig.update()  # owner bound, but encoded image still belongs to the previous model epoch
    current = session.snapshot().snapshot
    assert current.shadow.model_epoch == "2" and current.shadow.model_warming
    assert (
        current.classification is None
        and current.probability is None
        and current.shadow.eligible_owner is None
    )
    rig.clock.now += 200_000_000
    assert session.snapshot().snapshot.classification is None
    session.fail("shutdown")
    worker.done.set()
