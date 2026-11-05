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
        self.progress = None
        self.capture_hz = self.perception_ms = None
        self._last_source_ns = None
        self._last_identity = None
        self._last_reset = None

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
        model_started = self.clock()
        result = self.session.process(frame)
        model_ms = max(0, (self.clock() - model_started) / 1e6)
        notice = result.notice
        # Binding owner advances the model epoch. Never show the old result as current.
        if notice is not None and notice.context_epoch != str(
            self.session.controller.snapshot().context_epoch
        ):
            self.progress = None
            self._last_identity = None
            return None
        if notice is not None and frame.status.perception_status.value != "not_processed":
            self._record_progress(frame, notice, model_ms=model_ms)
        return notice

    def _record_progress(self, frame, notice, *, model_ms):
        # Diagnostic scalars only, never used to authorize or alter inference.
        classifier = self.session._classifier
        builder = classifier.builder if classifier is not None else None
        config = self.bundle.normalizer.config
        gap_ns = (
            None if self._last_source_ns is None
            else frame.info.timestamp_ns - self._last_source_ns
        )
        identity = (frame.info.stream_id, notice.context_epoch, notice.track_id)
        if notice.reason in {"stale", "out_of_order", "future_input", "perception_error"}:
            self._last_reset = notice.reason
        elif not frame.observations:
            self._last_reset = "no_hand"
        elif len(frame.observations) > 1:
            self._last_reset = "multiple_hands"
        elif not frame.observations[0].observation_valid:
            flags = frame.observations[0].quality_flags
            self._last_reset = "quality_invalid:" + (flags[0] if flags else "unknown")
        elif self._last_identity is not None and identity != self._last_identity:
            self._last_reset = "source_changed"
        elif gap_ns is not None and gap_ns > config.max_sampling_gap_ns:
            self._last_reset = "sampling_gap"
        self._last_source_ns = frame.info.timestamp_ns
        if notice.track_id is not None:
            self._last_identity = identity
        self.progress = {
            "window_length": len(builder.samples) if builder is not None else 0,
            "warmup_valid": min(32, builder._consecutive_valid) if builder is not None else 0,
            "source_gap_ms": None if gap_ns is None else min(60000, max(0, gap_ns / 1e6)),
            "frame_age_ms": min(60000, max(0, (self.clock() - frame.info.timestamp_ns) / 1e6)),
            "max_gap_ms": config.max_sampling_gap_ns / 1e6,
            "reset_reason": None if self._last_reset is None else self._last_reset[:128],
            "capture_hz": None if self.capture_hz is None else min(1000, max(0, self.capture_hz)),
            "perception_ms": None if self.perception_ms is None else min(60000, max(0, self.perception_ms)),
            "model_ms": min(60000, model_ms),
        }

    def snapshot(self):
        if self.session is None:
            return None
        return self.session.snapshot() | {"diagnostics": self.progress}

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
        self.progress = None
        self._last_source_ns = self._last_identity = self._last_reset = None
        if self.session is not None:
            self.session.stop("local_demo_explicit_stop")

    def close(self):
        if self.session is None:
            return True
        self.session.close()
        return self.session.host.dispatcher.wait_released(timeout=0.3)


class DemoCameraWorker(CameraPreviewWorker):
    # Cap only preview encoding, never processing cadence. Less JPEG/Flutter
    # churn leaves CPU headroom for the 100ms observation-gap contract.
    preview_interval_s = 0.1

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

    def _update_classifier(self, classifier, observation, *, capture_hz, perception_ms):
        # Timing is telemetry only. It never replaces source timestamps or
        # changes the model's sampling/readiness contract.
        classifier.capture_hz = capture_hz
        classifier.perception_ms = perception_ms
        return classifier.update(observation)

    def request_stop(self):
        super().request_stop()
        if self.adapter is not None:
            self.adapter.stop()

    def _close_classifier(self, classifier):
        return classifier.close()
