"""Virtual camera timing regressions; no physical camera or learned accuracy claims."""

import numpy as np

from algorithm.online import OnlineClassifier
from backend.controller import ApplicationController
from backend.interaction.public_replay import ReplayClock, synthetic_input
from backend.perception.preview import CameraFrame, CameraPreviewWorker
from shared.contracts import PerceptionStatus, TransformMetadata
from shared.observations import CapturedFrame, FrameInfo, ObservationFrame
from tests.integration.model_bridge_fixture import fixture_bundle


def test_old_drop_ledger_cannot_repeatedly_reset_fresh_camera_warmup(tmp_path, monkeypatch):
    bundle, _, _ = fixture_bundle(tmp_path, "mlp")
    clock = ReplayClock()
    context = ApplicationController("warmup-camera", timebase="monotonic").snapshot()
    notices = []
    updates = []

    class Capture:
        def __init__(self, *args, **kwargs):
            self.done = self.release_failed = False
            self.index = 0
            self.stats = self

        def start(self, timeout):
            pass

        def as_dict(self):
            return {"actual_capture_hz": 30.0}

        def take(self, timeout):
            self.index += 1
            if self.index > 40:
                self.done = True
                return (), None
            source_ns = self.index * 60_000_000
            # Retained evidence is 120ms old (<150ms), spaced by 60ms (<100ms).
            # An older replaced frame in the ledger is already 170ms old.
            clock.now = source_ns + 120_000_000
            info = FrameInfo(
                context.stream_id, self.index * 2, source_ns, 640, 480,
                TransformMetadata(), "monotonic",
            )
            old = FrameInfo(
                context.stream_id, self.index * 2 - 1, source_ns - 50_000_000,
                640, 480, TransformMetadata(), "monotonic",
            )
            dropped = ObservationFrame(
                old, old.status(PerceptionStatus.NOT_PROCESSED, reason="latest_frame_replaced")
            )
            return (dropped,), CapturedFrame(
                info, np.zeros((480, 640, 3), np.uint8), source_ns
            )

        def stop(self, timeout):
            self.done = True
            return True

    class Pipeline:
        def __init__(self, detector):
            pass

        def process(self, frame):
            return synthetic_input(
                context, timestamp_ns=frame.info.timestamp_ns,
                frame_id=frame.info.frame_id, model_version="unused",
            )[1]

        def close(self):
            pass

    class Classifier(OnlineClassifier):
        def update(self, frame):
            updates.append(frame.status.perception_status.value)
            result = super().update(frame)
            if frame.observations:
                notices.append(result)
            return result

    class Worker(CameraPreviewWorker):
        def _make_classifier(self, loaded):
            return Classifier(loaded, context_epoch=0, clock=clock)

    monkeypatch.setattr("algorithm.bundle.load_bundle", lambda *a, **kw: bundle)
    monkeypatch.setattr("backend.capture.worker.CaptureWorker", Capture)
    monkeypatch.setattr("backend.perception.pipeline.PerceptionPipeline", Pipeline)
    monkeypatch.setattr(
        "backend.perception.preview.encode_frame",
        lambda frame, observation, hz, **kw: CameraFrame(
            frame.info.frame_id, frame.info.timestamp_ns, 640, 480, b"", (),
            "processed_hands", hz, kw.get("classification"),
        ),
    )
    worker = Worker(
        source_factory=object, detector_factory=object,
        model_package="verified-in-memory-fixture", model_sha256=bundle.manifest_sha256,
    )
    worker.start()
    assert worker.done.wait(10), worker.error
    worker.thread.join(1)
    assert len(notices) == 40
    assert notices[31].state == "predicted"
    assert notices[31].window_length == 32
    assert "not_processed" not in updates


def test_demo_reports_actual_warmup_count_and_sampling_reset(tmp_path):
    import threading

    from backend.interaction.live_demo import LocalDemoSession
    from backend.interaction.public_replay import acknowledge
    from backend.perception.local_demo import DemoCameraWorker, DemoClassifier
    from tests.integration.test_local_gesture_demo import contract_inputs
    from tests.integration.test_model_session_bridge import Harness

    clock = ReplayClock()
    args = contract_inputs(tmp_path)
    adapter = DemoClassifier(
        args["bundle"], args["calibration"], args["calibration_sha256"],
        threading.Event(), clock=clock,
    )
    adapter.session = LocalDemoSession(
        allow_application_local=True, **args, clock=clock, watchdog=False,
    )
    h = Harness(adapter.session, clock)
    worker = DemoCameraWorker(
        allow_application_local=True,
        model_package="unused-fixture", model_sha256="a" * 64,
        calibration_path="unused-fixture", calibration_sha256="b" * 64,
    )
    try:
        h.enable()
        for i in range(6):
            notice = worker._update_classifier(
                adapter, h.frame(), capture_hz=30.0, perception_ms=18.5,
            )
            assert notice.state == "warming_up"
            diagnostics = adapter.snapshot()["diagnostics"]
            assert diagnostics["warmup_valid"] == i + 1
            assert diagnostics["capture_hz"] == 30.0
            assert diagnostics["perception_ms"] == 18.5
            assert diagnostics["model_ms"] == 0.0  # virtual clock, not real latency
            assert acknowledge(adapter.session.host)
        adapter.update(h.frame(dt=110_000_000))
        diagnostic = adapter.snapshot()["diagnostics"]
        assert diagnostic["warmup_valid"] == 1
        assert diagnostic["window_length"] == 7
        assert diagnostic["source_gap_ms"] == 110
        assert diagnostic["max_gap_ms"] == 100
        assert diagnostic["reset_reason"] == "sampling_gap"
        assert adapter.snapshot()["eligible_owner"] is None
        assert acknowledge(adapter.session.host)
        for _ in range(31):
            notice = adapter.update(h.frame())
            assert acknowledge(adapter.session.host)
        assert notice.state == "predicted"
        assert adapter.snapshot()["diagnostics"]["warmup_valid"] == 32
        # The last reset is history, not a claim that CURRENT timing is bad.
        assert adapter.snapshot()["diagnostics"]["source_gap_ms"] == 30
        assert adapter.snapshot()["diagnostics"]["reset_reason"] == "sampling_gap"
        adapter.stop()
        assert adapter.snapshot()["diagnostics"] is None
        assert not adapter.snapshot()["actions_enabled"]
    finally:
        adapter.close()
