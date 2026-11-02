"""Native HSP5 session: camera/model observations plus explicit SHADOW owner controls."""

from backend.desktop_preview import CameraPreviewSession
from shared.preview_protocol import PreviewCommand
from shared.shadow_preview_protocol import ShadowInfo, ShadowPacket, ShadowSnapshot


class NativeShadowSession(CameraPreviewSession):
    def __init__(self, *, worker_factory, **kwargs):
        super().__init__(worker_factory=worker_factory, **kwargs)
        self.control_status = "none"

    def accept(self, command):
        with self.lock:
            if self.closed:
                return
            if command.kind in {"surface_ack", "enable_shadow", "request_owner"}:
                if (
                    command.session_id != self.session_id
                    or int(command.command_id) <= self.last_command_id
                ):
                    raise ValueError("invalid shadow command identity/order")
                self.last_command_id = int(command.command_id)
                self._expire(self.clock())
                valid = (
                    not self.closed
                    and self.state == "preview"
                    and int(command.epoch) == self.epoch
                    and self.worker is not None
                )
                if valid:
                    self.last_heartbeat_ns = self.clock()
                accepted = valid and self.worker.shadow_control(command)
                if command.kind != "surface_ack":
                    self.control_status = "accepted" if accepted else "rejected"
                return
            super().accept(
                PreviewCommand(
                    version=2,
                    session_id=command.session_id,
                    command_id=command.command_id,
                    epoch=command.epoch,
                    kind=command.kind,
                )
            )
            if (
                command.kind == "heartbeat"
                and command.model_epoch is not None
                and int(command.epoch) == self.epoch
                and self.state == "preview"
                and self.worker is not None
            ):
                self.worker.shadow_control(command)
            if command.kind in {"stop", "shutdown", "start_camera"}:
                self.control_status = "none"

    def snapshot(self):
        import numpy as np

        from algorithm.evaluation.classification import log_probabilities
        from shared.calibrated_preview_protocol import ProbabilityNotice

        with self.lock:
            base = super().snapshot()
            shadow = classification = probability = None
            status = "unavailable"
            if self.worker is not None and getattr(self.worker, "model_failed", False):
                status = "error"
            elif base.snapshot.state == "preview" and self.camera_frame is not None:
                raw = self.worker.shadow_snapshot()
                if raw is not None:
                    status = "loaded_shadow"
                    current = self.camera_frame.classification
                    age = self.clock() - self.camera_frame.timestamp_ns
                    if (
                        current is not None
                        and current.context_epoch == str(raw["epoch"])
                        and current.source_frame_id == base.snapshot.frame_id
                        and current.source_timestamp_ns
                        == base.snapshot.frame_timestamp_ns
                        and 0 <= age <= self.worker.classification_max_age_ns
                        and raw["state"] not in {"DISABLED", "SUSPENDED"}
                    ):
                        classification = current
                    eligible = None
                    if (
                        classification is not None
                        and classification.state == "predicted"
                    ):
                        calibration = self.worker.calibration
                        probabilities = np.exp(
                            log_probabilities(
                                [classification.logits], calibration.temperature
                            )
                        )[0]
                        rejected = (
                            "unavailable"
                            if calibration.threshold is None
                            else "accepted"
                            if probabilities.max() >= calibration.threshold
                            else "rejected"
                        )
                        probability = ProbabilityNotice(
                            model_version=classification.model_version,
                            calibration_sha256=self.worker.calibration_sha256,
                            synthetic=calibration.synthetic,
                            temperature=calibration.temperature,
                            threshold=calibration.threshold,
                            probabilities=tuple(float(v) for v in probabilities),
                            rejection_status=rejected,
                        )
                        offer = self.camera_frame.shadow_offer
                        if (
                            offer is not None
                            and offer == (str(raw["epoch"]), raw["eligible_owner"])
                            and offer[1] == classification.track_id
                            and raw["state"] == "WAIT_OWNER"
                            and not raw["owner_request_pending"]
                            and probability.rejection_status == "accepted"
                        ):
                            eligible = offer[1]
                    shadow = ShadowInfo(
                        model_epoch=str(raw["epoch"]),
                        model_sequence=str(raw["sequence"]),
                        model_revision=str(raw["revision"]),
                        model_stream_id=raw["model_stream_id"],
                        model_version=raw["model_version"],
                        state=raw["state"],
                        reason=raw["bridge_reason"] + " / " + raw["reason"],
                        model_warming=raw["model_warming"],
                        model_busy=raw["model_busy"],
                        surface_ready=raw["rendered"],
                        eligible_owner=eligible,
                        owner_track_id=raw["owner_track_id"],
                        owner_request_pending=raw["owner_request_pending"],
                        candidates_total=str(raw["candidates_total"]),
                        candidate=raw["candidate"],
                        synthetic=raw["synthetic"],
                        control_status=self.control_status,
                    )
            values = base.snapshot.model_dump(exclude={"version", "model_status"})
            return ShadowPacket(
                ShadowSnapshot(
                    **values,
                    model_status=status,
                    classification=classification,
                    probability=probability,
                    shadow=shadow,
                ),
                base.jpeg,
            )
