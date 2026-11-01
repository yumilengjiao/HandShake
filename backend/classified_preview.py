"""Read-only model telemetry added to the camera endpoint, never to action dispatch."""

from backend.desktop_preview import CameraPreviewSession
from shared.inference_protocol import ClassificationNotice
from shared.model_preview_protocol import ModelPreviewPacket, ModelPreviewSnapshot


class ClassifiedCameraSession(CameraPreviewSession):
    def _make_worker(self):
        # Epoch is captured when work is created, not copied onto late completions.
        return self.worker_factory(context_epoch=self.epoch)

    def snapshot(self):
        with self.lock:
            base = super().snapshot()
            notice = (
                self.camera_frame.classification
                if self.camera_frame is not None and base.snapshot.state == "preview"
                else None
            )
            if (
                notice is not None
                and notice.state == "predicted"
                and self.clock() - int(notice.source_timestamp_ns)
                > getattr(self.worker, "classification_max_age_ns", 150_000_000)
            ):
                notice = ClassificationNotice.model_validate(
                    notice.model_dump()
                    | {
                        "state": "rejected",
                        "reason": "stale",
                        "logits": (),
                        "inference_ns": None,
                    }
                )
            status = "loaded_uncalibrated" if notice is not None else "unavailable"
            if self.worker is not None and getattr(self.worker, "model_failed", False):
                status = "error"
            values = base.snapshot.model_dump(exclude={"version", "model_status"})
            return ModelPreviewPacket(
                ModelPreviewSnapshot(
                    **values, model_status=status, classification=notice
                ),
                base.jpeg,
            )


class CalibratedCameraSession(ClassifiedCameraSession):
    """V4 conversion uses already-verified sidecar; old/stopped scores are never retained."""

    def snapshot(self):
        import numpy as np

        from algorithm.evaluation.classification import log_probabilities
        from shared.calibrated_preview_protocol import (
            CalibratedPreviewPacket,
            CalibratedPreviewSnapshot,
            ProbabilityNotice,
        )

        with self.lock:
            base = super().snapshot()
            probability = None
            notice = base.snapshot.classification
            if notice is not None and notice.state == "predicted":
                calibration = self.worker.calibration
                if (
                    calibration is None
                    or notice.model_version
                    != "sha256:" + calibration.source.model_sha256
                ):
                    raise ValueError("missing or mismatched verified calibration")
                values = np.exp(
                    log_probabilities([notice.logits], calibration.temperature)
                )[0]
                status = (
                    "unavailable"
                    if calibration.threshold is None
                    else "accepted"
                    if values.max() >= calibration.threshold
                    else "rejected"
                )
                probability = ProbabilityNotice(
                    calibration_sha256=self.worker.calibration_sha256,
                    model_version=notice.model_version,
                    synthetic=calibration.synthetic,
                    temperature=calibration.temperature,
                    threshold=calibration.threshold,
                    probabilities=tuple(float(v) for v in values),
                    rejection_status=status,
                )
            return CalibratedPreviewPacket(
                CalibratedPreviewSnapshot(
                    **base.snapshot.model_dump(exclude={"version"}),
                    probability=probability,
                ),
                base.jpeg,
            )
