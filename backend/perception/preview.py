"""Camera/MediaPipe work off the protocol thread. Images stay in bounded RAM only."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from backend.perception.lifecycle import lifecycle
from shared.inference_protocol import ClassificationNotice
from shared.preview_protocol import MAX_JPEG_BYTES, PreviewHand, PreviewPoint


@dataclass(frozen=True)
class CameraFrame:
    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    jpeg: bytes
    hands: tuple[PreviewHand, ...]
    perception_status: str
    capture_hz: float | None
    classification: ClassificationNotice | None = None
    shadow_offer: tuple[str, str | None] | None = None


def encode_frame(frame, observation, capture_hz, *, cv=None, classification=None):
    if observation is not None and observation.info != frame.info:
        raise ValueError("image/observation frame identity mismatch")
    if classification is not None and (
        classification.stream_id != frame.info.stream_id
        or classification.source_frame_id != str(frame.info.frame_id)
        or classification.source_timestamp_ns != str(frame.info.timestamp_ns)
    ):
        raise ValueError("image/classification frame identity mismatch")
    # Lazy optional dependency: the synthetic UI works without OpenCV/MediaPipe.
    if cv is None:
        import cv2 as cv
    width, height = frame.info.width, frame.info.height
    ratio = min(1.0, 640 / width, 480 / height)
    width, height = max(1, round(width * ratio)), max(1, round(height * ratio))
    rgb = frame.rgb
    if (width, height) != (frame.info.width, frame.info.height):
        rgb = cv.resize(rgb, (width, height), interpolation=cv.INTER_AREA)
    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
    jpeg = b""
    for quality in (70, 50, 30):
        ok, encoded = cv.imencode(".jpg", bgr, [cv.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("JPEG encoding failed")
        jpeg = encoded.tobytes()
        if len(jpeg) <= MAX_JPEG_BYTES:
            break
    if len(jpeg) > MAX_JPEG_BYTES:
        raise ValueError("JPEG exceeds preview budget")
    hands = ()
    status = "unavailable"
    if observation is not None:
        status = observation.status.perception_status.value
        hands = tuple(
            PreviewHand(
                landmarks=tuple(
                    PreviewPoint(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                    for p in hand.landmarks_image
                ),
                validity=tuple(hand.validity_mask),
                usable=hand.observation_valid,
            )
            for hand in observation.observations
        )
    return CameraFrame(
        frame.info.frame_id,
        frame.info.timestamp_ns,
        width,
        height,
        jpeg,
        hands,
        status,
        capture_hz,
        classification,
    )


class CameraPreviewWorker:
    """One capture slot + one encoded slot. Stop confirmation requires thread exit.

    A blocked native driver/model is NOT called safely cancelled. The UI may kill
    its isolated core process after a stop-ACK deadline; no application actions exist.
    """

    preview_interval_s = 0.1

    def __init__(
        self,
        *,
        device=0,
        model_path=None,
        source_factory=None,
        detector_factory=None,
        model_package=None,
        model_sha256=None,
        context_epoch=0,
        calibration_path=None,
        calibration_sha256=None,
    ):
        self.device, self.model_path = device, model_path
        self.source_factory, self.detector_factory = source_factory, detector_factory
        self.model_package, self.model_sha256 = model_package, model_sha256
        if (
            bool(calibration_path) != bool(calibration_sha256)
            or calibration_path
            and not model_package
        ):
            raise ValueError("calibration requires a model package and its own SHA256")
        self.calibration_path, self.calibration_sha256 = (
            calibration_path,
            calibration_sha256,
        )
        self.calibration = None
        self.context_epoch = context_epoch
        self.model_failed = False
        self.classification_max_age_ns = 150_000_000
        self.stop_requested = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="preview-perception"
        )

    def start(self):
        self.thread.start()

    def request_stop(self):
        self.stop_requested.set()

    def take(self):
        with self.lock:
            result, self.latest = self.latest, None
            return result

    def _make_classifier(self, bundle):
        from algorithm.online import OnlineClassifier

        return OnlineClassifier(
            bundle,
            context_epoch=self.context_epoch,
            cancelled=self.stop_requested.is_set,
        )

    def _decorate_frame(self, packet):
        return packet

    def _close_classifier(self, classifier):
        return True

    def _run(self):
        capture = pipeline = classifier = None
        release_ok = True
        first_frame = True
        lifecycle("worker_started")
        try:
            from backend.capture.camera import OpenCVSource
            from backend.capture.worker import CaptureWorker
            from backend.perception.mediapipe_adapter import MediaPipeDetector
            from backend.perception.pipeline import PerceptionPipeline

            if self.model_package is not None:
                lifecycle("model_loading")
                try:
                    import torch

                    from algorithm.bundle import load_bundle

                    if self.model_path is None and self.detector_factory is None:
                        raise ValueError(
                            "a gesture classifier requires actual perception"
                        )
                    torch.set_num_threads(1)
                    bundle = load_bundle(
                        self.model_package, expected_manifest_sha256=self.model_sha256
                    )
                    if self.calibration_path is not None:
                        from algorithm.evaluation.calibration import load_for_bundle

                        self.calibration = load_for_bundle(
                            self.calibration_path,
                            expected_sha256=self.calibration_sha256,
                            bundle=bundle,
                        )
                    self.classification_max_age_ns = bundle.normalizer.config.max_age_ns
                    classifier = self._make_classifier(bundle)
                    lifecycle("model_ready")
                except Exception as error:  # optional model must fail before camera access
                    lifecycle("model_failed", error=error)
                    self.model_failed = True
                    self.error = "preview_error"
                    return

            if self.stop_requested.is_set():
                return
            # A supplied-but-broken asset is an error, never silent fallback.
            if self.model_path is not None or self.detector_factory is not None:
                lifecycle("perception_loading")
                try:
                    detector = (
                        self.detector_factory()
                        if self.detector_factory
                        else MediaPipeDetector(self.model_path)
                    )
                    pipeline = PerceptionPipeline(detector)
                    lifecycle("perception_ready")
                except Exception as error:  # SDK init faults never silently fall back
                    lifecycle("perception_failed", error=error)
                    self.error = "perception_unavailable"
                    return
            if self.stop_requested.is_set():
                return
            lifecycle("camera_opening")
            source = (
                self.source_factory()
                if self.source_factory
                else OpenCVSource(
                    self.device, width=640, height=480, preview_mirrored=False
                )
            )
            capture = CaptureWorker(source, status_capacity=32)
            try:
                capture.start(timeout=3)
                lifecycle("camera_ready")
            except (RuntimeError, TimeoutError) as error:
                lifecycle("camera_failed", error=error)
                self.error = "camera_unavailable"
                return
            next_frame_at = 0.0
            while not self.stop_requested.is_set():
                if classifier is None and self.stop_requested.wait(
                    max(0, next_frame_at - time.monotonic())
                ):
                    break
                skipped, frame = capture.take(timeout=0.1)
                if classifier is not None:
                    for dropped in skipped:
                        classifier.update(dropped)
                if frame is None:
                    if capture.done:
                        self.error = "camera_unavailable"
                        break
                    continue
                observation = pipeline.process(frame) if pipeline is not None else None
                notice = (
                    classifier.update(observation) if classifier is not None else None
                )
                # Process evidence at capture cadence; cap only JPEG/UI output. A 10Hz
                # sleep here would break the trained 100ms sampling-gap contract.
                if classifier is not None and time.monotonic() < next_frame_at:
                    continue
                packet = encode_frame(
                    frame,
                    observation,
                    capture.stats.as_dict()["actual_capture_hz"],
                    classification=notice,
                )
                packet = self._decorate_frame(packet)
                if self.stop_requested.is_set():
                    break
                with self.lock:
                    self.latest = packet
                if first_frame:
                    lifecycle("first_frame")
                    first_frame = False
                next_frame_at = time.monotonic() + self.preview_interval_s
                # Encoding cap only, not a camera FPS claim. Shadow uses a faster
                # latest slot so two independent 10Hz caps do not age owner offers.
        except Exception as error:  # worker boundary publishes failure and releases resources
            lifecycle("worker_failed", error=error)
            self.error = "preview_error"
        finally:
            lifecycle("release_started")
            if capture is not None:
                release_ok = capture.stop(timeout=0.3) and not capture.release_failed
            if pipeline is not None:
                try:
                    pipeline.close()
                except Exception:  # noqa: BLE001 - SDK close failure must withhold the stop ACK
                    release_ok = False
            if classifier is not None:
                try:
                    release_ok = self._close_classifier(classifier) and release_ok
                except Exception:
                    release_ok = False
            with self.lock:
                self.latest = None
            # Do NOT confirm stop if the capture thread still owns the camera.
            if release_ok:
                lifecycle("release_complete")
                self.done.set()
            else:
                lifecycle("release_failed")
                self.error = "preview_error"
