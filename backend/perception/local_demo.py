"""Camera adapter for the separate, opt-in application-local demo (not shadow IPC)."""

import time

from backend.interaction.live_demo import LocalDemoSession
from backend.perception.preview import CameraPreviewWorker


class DemoClassifier:
    def __init__(
        self,
        bundle,
        calibration,
        calibration_sha256,
        stop_requested,
        *,
        deck=None,
        clock=time.monotonic_ns,
    ):
        if calibration.synthetic or calibration.rejection_status != "selected":
            raise ValueError("local demo needs feasible real-model calibration")
        self.bundle, self.calibration = bundle, calibration
        self.calibration_sha256 = calibration_sha256
        self.stop_requested, self.deck, self.clock = stop_requested, deck, clock
        self.session = None

    def update(self, frame):
        if self.stop_requested.is_set():
            return None
        if frame.info.timebase != "monotonic":
            raise ValueError("local demo requires monotonic camera observations")
        if self.session is None:
            self.session = LocalDemoSession(
                allow_application_local=True,
                deck=self.deck,
                bundle=self.bundle,
                calibration=self.calibration,
                calibration_sha256=self.calibration_sha256,
                clock=self.clock,
                stream_id=frame.info.stream_id,
            )
            if self.stop_requested.is_set():
                self.close()
                return None
        result = self.session.process(frame)
        notice = result.notice
        # Binding owner advances the model epoch. Never show the old result as current.
        if notice is not None and notice.context_epoch != str(
            self.session.controller.snapshot().context_epoch
        ):
            return None
        return notice

    def snapshot(self):
        return self.session.snapshot() if self.session is not None else None

    def acknowledge(self, snapshot):
        if self.session is None or self.stop_requested.is_set():
            return False
        return self.session.host.acknowledge_surface(
            sequence=snapshot["sequence"],
            epoch=snapshot["epoch"],
            revision=snapshot["revision"],
        )

    def enable(self, snapshot):
        if self.session is None or self.stop_requested.is_set():
            return False
        return self.session.enable(epoch=snapshot["epoch"])

    def request_owner(self, snapshot):
        if (
            self.session is None
            or self.stop_requested.is_set()
            or not snapshot["eligible_owner"]
        ):
            return False
        return self.session.request_owner(
            snapshot["eligible_owner"], epoch=snapshot["epoch"]
        )

    def stop(self):
        if self.session is not None:
            self.session.stop("local_demo_explicit_stop")

    def close(self):
        if self.session is None:
            return True
        self.session.close()
        return self.session.host.dispatcher.wait_released(timeout=0.3)


class DemoCameraWorker(CameraPreviewWorker):
    preview_interval_s = 0.03

    def __init__(self, *, allow_application_local=False, deck=None, **kwargs):
        if allow_application_local is not True:
            raise ValueError("local demo requires explicit opt-in")
        self.adapter = None
        self.deck = deck
        super().__init__(**kwargs)
        if not self.model_package or not self.calibration_path:
            raise ValueError("local demo requires model and calibration assets")

    def _make_classifier(self, bundle):
        self.adapter = DemoClassifier(
            bundle,
            self.calibration,
            self.calibration_sha256,
            self.stop_requested,
            deck=self.deck,
        )
        return self.adapter

    def request_stop(self):
        super().request_stop()
        if self.adapter is not None:
            self.adapter.stop()

    def _close_classifier(self, classifier):
        return classifier.close()
