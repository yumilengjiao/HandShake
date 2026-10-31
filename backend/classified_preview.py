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
