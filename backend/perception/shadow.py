"""Camera-worker adapter for the mandatory shadow model session. No action backend is exposed."""

import time
from dataclasses import replace

from backend.interaction.model_session import ModelPresentationSession
from backend.perception.preview import CameraPreviewWorker
from shared.classes import IPN_HANDS_VERSION
from shared.shadow_preview_protocol import ShadowCandidate


class ShadowClassifier:
    def __init__(
        self,
        bundle,
        calibration,
        calibration_sha256,
        stop_requested,
        *,
        allow_synthetic=False,
        clock=time.monotonic_ns,
    ):
        if (
            bundle.vocabulary.vocabulary_version != IPN_HANDS_VERSION
            or calibration.rejection_status != "selected"
        ):
            raise ValueError(
                "shadow owner session requires feasible IPN HandS calibration"
            )
        if calibration.synthetic and allow_synthetic is not True:
            raise ValueError("synthetic shadow model requires explicit opt-in")
        if (
            not 0
            < bundle.normalizer.config.max_window_span_ns
            + bundle.normalizer.config.max_age_ns
            <= 5_000_000_000
        ):
            raise ValueError("unreviewed model window timing")
        self.bundle, self.calibration, self.calibration_sha256 = (
            bundle,
            calibration,
            calibration_sha256,
        )
        self.stop_requested, self.allow_synthetic, self.clock = (
            stop_requested,
            allow_synthetic,
            clock,
        )
        self.session = None
        self.offer = None
        self.candidate = None
        self.candidates = 0
        self.control_status = "none"

    def update(self, frame):
        self.offer = None
        if self.stop_requested.is_set():
            return None
        if frame.info.timebase != "monotonic":
            raise ValueError("native source requires the monotonic clock domain")
        if self.session is None:
            self.session = ModelPresentationSession(
                bundle=self.bundle,
                calibration=self.calibration,
                calibration_sha256=self.calibration_sha256,
                clock=self.clock,
                timebase="monotonic",
                source_synthetic=False,
                execution="shadow",
                allow_synthetic=self.allow_synthetic,
                stream_id=frame.info.stream_id,
                watchdog=True,
            )
            if self.stop_requested.is_set():
                self.close()
                return None
        result = self.session.process(frame)
        if result.ticket is not None:
            self.session.stop("shadow_execution_invariant_failure")
            raise RuntimeError("shadow must never submit an application command")
        with self.session.controller.gate:
            current = self.session.snapshot()
            if result.event is not None:
                event = result.event
                self.candidates += 1
                self.candidate = ShadowCandidate(
                    event_id=event.event_id,
                    action=event.action,
                    source_frame_id=str(event.source_frame_id),
                    source_timestamp_ns=str(event.source_timestamp_ns),
                    model_epoch=str(event.context_epoch),
                )
            notice = result.notice
            if notice is None or notice.context_epoch != str(current["epoch"]):
                return None
            if (notice.source_frame_id, notice.source_timestamp_ns) != (
                str(frame.info.frame_id),
                str(frame.info.timestamp_ns),
            ):
                raise ValueError("shadow model result is not from this frame")
            self.offer = (str(current["epoch"]), current["eligible_owner"])
            return notice

    def snapshot(self):
        session = self.session
        if session is None:
            return None
        with session.controller.gate:
            raw = session.snapshot()
            context = session.controller.snapshot()
            if raw["execution"] != "shadow" or raw["page"] != 1 or raw["revision"] != 0:
                session.stop("shadow_execution_invariant_failure")
                raise RuntimeError("shadow state cannot change an application page")
            candidate = self.candidate
            if candidate is not None and (
                candidate.model_epoch != str(context.context_epoch)
                or not context.enabled
                or context.suspended
            ):
                candidate = None
            return raw | {
                "model_stream_id": context.stream_id,
                "owner_track_id": context.owner_track_id,
                "candidates_total": self.candidates,
                "candidate": candidate,
                "control_status": self.control_status,
            }

    def control(self, command):
        session = self.session
        if session is None or self.stop_requested.is_set():
            self.control_status = "rejected"
            return False
        with session.controller.gate:
            valid = session.host._receipt(
                int(command.model_sequence),
                int(command.model_epoch),
                int(command.model_revision),
            )
            accepted = False
            if valid:
                if command.kind == "surface_ack":
                    accepted = session.host.acknowledge_surface(
                        sequence=int(command.model_sequence),
                        epoch=int(command.model_epoch),
                        revision=int(command.model_revision),
                    )
                elif command.kind == "heartbeat":
                    accepted = session.host.heartbeat(
                        sequence=int(command.model_sequence),
                        epoch=int(command.model_epoch),
                        revision=int(command.model_revision),
                    )
                elif command.kind == "enable_shadow":
                    accepted = session.enable(epoch=int(command.model_epoch))
                elif command.kind == "request_owner":
                    accepted = session.request_owner(
                        command.track_id, epoch=int(command.model_epoch)
                    )
            if command.kind != "heartbeat":
                self.control_status = "accepted" if accepted else "rejected"
            return accepted

    def stop(self):
        session = self.session
        if session is not None:
            session.stop("preview_stop")

    def close(self):
        session = self.session
        if session is None:
            return True
        session.close()
        return session.host.dispatcher.wait_released(timeout=0.3)


class ShadowCameraWorker(CameraPreviewWorker):
    # The wire/UI still publishes at 10Hz. Keep its single encoded slot fresh
    # enough for the 150ms owner-evidence budget; never relax that safety budget.
    preview_interval_s = 0.03

    def __init__(
        self, *, allow_synthetic_model=False, clock=time.monotonic_ns, **kwargs
    ):
        self.shadow_adapter = None
        self.clock = clock
        self.allow_synthetic_model = allow_synthetic_model
        super().__init__(**kwargs)
        if not self.model_package or not self.calibration_path:
            raise ValueError("shadow camera requires model and calibration assets")

    def _make_classifier(self, bundle):
        self.shadow_adapter = ShadowClassifier(
            bundle,
            self.calibration,
            self.calibration_sha256,
            self.stop_requested,
            allow_synthetic=self.allow_synthetic_model,
            clock=self.clock,
        )
        return self.shadow_adapter

    def _decorate_frame(self, packet):
        return replace(packet, shadow_offer=self.shadow_adapter.offer)

    def _close_classifier(self, classifier):
        return classifier.close()

    def request_stop(self):
        super().request_stop()
        if self.shadow_adapter is not None:
            self.shadow_adapter.stop()

    def shadow_snapshot(self):
        return (
            self.shadow_adapter.snapshot() if self.shadow_adapter is not None else None
        )

    def shadow_control(self, command):
        return (
            self.shadow_adapter.control(command)
            if self.shadow_adapter is not None
            else False
        )
