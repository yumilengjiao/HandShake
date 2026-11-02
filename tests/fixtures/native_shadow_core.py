"""Opt-in native INTEGRATION TEST source. Never accesses a real camera or detector."""

import os
import time

import numpy as np

from backend.controller import ApplicationController
from backend.interaction.public_replay import synthetic_input
from shared.contracts import TransformMetadata
from shared.observations import CapturedFrame, FrameInfo


class Capture:
    def __init__(self, *args, **kwargs):
        self.done = self.release_failed = False
        self.index = 0
        self.first_ns = self.last_ns = None
        self.stats = self

    def start(self, timeout):
        pass

    def as_dict(self):
        hz = (
            (self.index - 1) * 1e9 / (self.last_ns - self.first_ns)
            if self.first_ns is not None and self.last_ns > self.first_ns
            else 0.0
        )
        return {"actual_capture_hz": hz}

    def take(self, timeout):
        time.sleep(0.02)
        self.index += 1
        self.last_ns = time.monotonic_ns()
        if self.first_ns is None:
            self.first_ns = self.last_ns
        info = FrameInfo(
            "procedural-native-test",
            self.index,
            self.last_ns,
            640,
            480,
            TransformMetadata(),
            "monotonic",
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, :] = [24, 49, 57]
        return [], CapturedFrame(info, image, info.timestamp_ns)

    def stop(self, timeout):
        self.done = True
        return True


class Pipeline:
    def __init__(self, detector):
        self.context = ApplicationController(
            "procedural-native-test", timebase="monotonic"
        ).snapshot()
        self.origin = None

    def process(self, frame):
        if self.origin is None:
            self.origin = frame.info.timestamp_ns
        elapsed = (frame.info.timestamp_ns - self.origin) / 1e9
        phase = (elapsed - 4) % 5
        # Constant physical test velocity, not index-based jumps whose velocity
        # depends on the machine's rendering/inference throughput.
        # Use the same well-separated 1.2 image-width/s diagnostic signal as the
        # bridge fixtures, staying inside the image throughout the motion.
        x = 0.35 + 1.2 * phase if elapsed >= 4 and phase < 0.4 else 0.35
        return synthetic_input(
            self.context,
            timestamp_ns=frame.info.timestamp_ns,
            frame_id=frame.info.frame_id,
            model_version="unused",
            x=x,
        )[1]

    def close(self):
        pass


def main():
    if os.environ.get("HANDSHAKE_TEST_SOURCE") != "1":
        raise SystemExit(
            "test source requires explicit HANDSHAKE_TEST_SOURCE=1; no devices opened"
        )
    import backend.capture.camera
    import backend.capture.worker
    import backend.perception.mediapipe_adapter
    import backend.perception.pipeline
    from backend.desktop_preview import main as serve

    backend.capture.camera.OpenCVSource = lambda *a, **k: object()
    backend.capture.worker.CaptureWorker = Capture
    backend.perception.mediapipe_adapter.MediaPipeDetector = lambda *a, **k: object()
    backend.perception.pipeline.PerceptionPipeline = Pipeline
    serve()


if __name__ == "__main__":
    main()
