"""Actual camera-worker loop with fake capture/perception and a verified UNTRAINED network."""

import time

import numpy as np

from backend.controller import ApplicationController
from backend.interaction.public_replay import ReplayClock, synthetic_input
from backend.perception.shadow import ShadowCameraWorker
from backend.shadow_preview import NativeShadowSession
from shared.contracts import TransformMetadata
from shared.observations import CapturedFrame, FrameInfo
from shared.shadow_preview_protocol import ShadowCommand
from tests.integration.model_bridge_fixture import fixture_bundle


def test_worker_load_model_process_frames_and_release_without_real_devices(
    tmp_path, monkeypatch
):
    bundle, _, sha = fixture_bundle(tmp_path, "mlp")
    source_context = ApplicationController(
        "fake-native-camera", timebase="monotonic"
    ).snapshot()
    released = []
    # Test the worker/control integration, not scheduler performance. A virtual
    # source clock keeps CPU contention from impersonating a >100ms camera gap;
    # the outer wait still has a real wall-clock deadline.
    clock = ReplayClock()

    class Capture:
        def __init__(self, *args, **kwargs):
            self.done = False
            self.release_failed = False
            self.index = 0
            self.stats = self

        def start(self, timeout):
            pass

        def as_dict(self):
            return {"actual_capture_hz": 50.0}

        def take(self, timeout):
            time.sleep(0.02)
            self.index += 1
            clock.now += 20_000_000
            info = FrameInfo(
                source_context.stream_id,
                self.index,
                clock.now,
                640,
                480,
                TransformMetadata(),
                "monotonic",
            )
            return [], CapturedFrame(
                info, np.zeros((480, 640, 3), np.uint8), info.timestamp_ns
            )

        def stop(self, timeout):
            self.done = True
            released.append("capture")
            return True

    class Pipeline:
        def __init__(self, detector):
            pass

        def process(self, frame):
            return synthetic_input(
                source_context,
                timestamp_ns=frame.info.timestamp_ns,
                frame_id=frame.info.frame_id,
                model_version="unused",
            )[1]

        def close(self):
            released.append("pipeline")

    monkeypatch.setattr("backend.capture.worker.CaptureWorker", Capture)
    monkeypatch.setattr("backend.perception.pipeline.PerceptionPipeline", Pipeline)
    worker = ShadowCameraWorker(
        model_package=tmp_path / "bundle",
        model_sha256=bundle.manifest_sha256,
        calibration_path=tmp_path / "calibration.json",
        calibration_sha256=sha,
        source_factory=object,
        detector_factory=object,
        allow_synthetic_model=True,
        clock=clock,
    )
    session = NativeShadowSession(worker_factory=lambda: worker, clock=clock)
    command_id = 0

    def send(kind, snapshot=None):
        nonlocal command_id
        command_id += 1
        data = dict(
            version=5,
            session_id=session.session_id,
            command_id=str(command_id),
            epoch=str(session.epoch),
            kind=kind,
        )
        if kind not in {"start_camera", "stop"} and snapshot and snapshot.shadow:
            s = snapshot.shadow
            data.update(
                model_epoch=s.model_epoch,
                model_sequence=s.model_sequence,
                model_revision=s.model_revision,
            )
            if kind == "request_owner":
                data["track_id"] = s.eligible_owner
        session.accept(ShadowCommand(**data))

    def until(predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            snapshot = session.snapshot().snapshot
            assert snapshot.model_status != "error", worker.error
            send("heartbeat", snapshot)
            if snapshot.shadow:
                send(
                    "surface_ack", snapshot
                )  # simulated UI receipt, not a drawing claim
            if predicate(snapshot):
                return snapshot
            time.sleep(0.02)
        raise AssertionError(
            "fake camera/model session timed out: "
            f"error={worker.error}; state={snapshot.model_dump(mode='json')}"
        )

    try:
        send("start_camera")
        snapshot = until(lambda s: s.shadow is not None and s.shadow.surface_ready)
        assert snapshot.shadow.state == "DISABLED"
        send("enable_shadow", snapshot)
        snapshot = until(
            lambda s: s.shadow is not None and s.shadow.eligible_owner is not None
        )
        assert snapshot.classification.source_frame_id == snapshot.frame_id
        send("request_owner", snapshot)
        snapshot = until(
            lambda s: s.shadow is not None and s.shadow.owner_track_id is not None
        )
        assert (
            snapshot.shadow.model_epoch != snapshot.epoch
        )  # independent ownership generation
        until(
            lambda s: (
                s.shadow is not None
                and not s.shadow.model_warming
                and s.classification is not None
            )
        )
        assert worker.shadow_adapter.session.host.backend.page == 1
        assert not worker.shadow_adapter.session.host.commands
        send("stop")
        assert worker.done.wait(2)
        assert session.snapshot().snapshot.state == "stopped"
        assert released == ["capture", "pipeline"]
    finally:
        worker.request_stop()
        worker.done.wait(2)
