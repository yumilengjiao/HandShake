"""Camera/MediaPipe work off the protocol thread. Images stay in bounded RAM only."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

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
    ):
        self.device, self.model_path = device, model_path
        self.source_factory, self.detector_factory = source_factory, detector_factory
        self.model_package, self.model_sha256 = model_package, model_sha256
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

    def _run(self):
        capture = pipeline = classifier = None
        release_ok = True
        try:
            from backend.capture.camera import OpenCVSource
            from backend.capture.worker import CaptureWorker
            from backend.perception.mediapipe_adapter import MediaPipeDetector
            from backend.perception.pipeline import PerceptionPipeline

            if self.model_package is not None:
                try:
                    import torch

                    from algorithm.bundle import load_bundle
                    from algorithm.online import OnlineClassifier

                    if self.model_path is None and self.detector_factory is None:
                        raise ValueError(
                            "a gesture classifier requires actual perception"
                        )
                    torch.set_num_threads(1)
                    bundle = load_bundle(
                        self.model_package, expected_manifest_sha256=self.model_sha256
                    )
                    self.classification_max_age_ns = bundle.normalizer.config.max_age_ns
                    classifier = OnlineClassifier(
                        bundle,
                        context_epoch=self.context_epoch,
                        cancelled=self.stop_requested.is_set,
                    )
                except Exception:  # noqa: BLE001 - optional model must fail closed before camera access
                    self.model_failed = True
                    self.error = "preview_error"
                    return

            # A supplied-but-broken asset is an error, never silent fallback.
            if self.model_path is not None or self.detector_factory is not None:
                try:
                    detector = (
                        self.detector_factory()
                        if self.detector_factory
                        else MediaPipeDetector(self.model_path)
                    )
                    pipeline = PerceptionPipeline(detector)
                except Exception:  # noqa: BLE001 - translate arbitrary SDK init faults, never fall back
                    self.error = "perception_unavailable"
                    return
            if self.stop_requested.is_set():
                return
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
            except (RuntimeError, TimeoutError):
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
                if self.stop_requested.is_set():
                    break
                with self.lock:
                    self.latest = packet
                next_frame_at = (
                    time.monotonic() + 0.1
                )  # display cap, not a camera FPS claim
        except Exception:  # noqa: BLE001 - worker boundary must publish failure and release resources
            self.error = "preview_error"
        finally:
            if capture is not None:
                release_ok = capture.stop(timeout=0.3) and not capture.release_failed
            if pipeline is not None:
                try:
                    pipeline.close()
                except Exception:  # noqa: BLE001 - SDK close failure must withhold the stop ACK
                    release_ok = False
            with self.lock:
                self.latest = None
            # Do NOT confirm stop if the capture thread still owns the camera.
            if release_ok:
                self.done.set()
            else:
                self.error = "preview_error"
