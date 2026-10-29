"""T03 synchronous source adapters. A worker owns open/read/close on one thread."""
from pathlib import Path
from threading import Event
from time import monotonic_ns
from typing import Protocol

import numpy as np

from backend.capture.clock import StreamClock
from shared.contracts import TransformMetadata
from shared.observations import CapturedFrame, FrameInfo


class CaptureError(RuntimeError):
    pass


class CameraSource(Protocol):
    metadata: dict
    def open(self) -> None: ...
    def read(self) -> CapturedFrame: ...
    def close(self) -> None: ...


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise CaptureError("OpenCV unavailable; install the perception extra") from error
    return cv2


class OpenCVSource:
    """Camera read failures are faults; file EOF is StopIteration, not no-hand.

    File timing is explicitly either decoder POS_MSEC or index/FPS. There is no
    silent switch on bad timestamps. Driver buffer requests are reported, not guarantees.
    """
    def __init__(self, device=0, *, video=None, width=640, height=480, fps=30.,
                 rotation=0, preview_mirrored=True, backend=None, video_clock="fps",
                 now=monotonic_ns, cv=None):
        if min(width, height) <= 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("positive image size and FPS required")
        if rotation not in (0, 90, 180, 270) or video_clock not in ("fps", "decoder"):
            raise ValueError("invalid rotation/video clock")
        self.device, self.video = device, Path(video) if video is not None else None
        self.width, self.height, self.fps = width, height, fps
        self.transform = TransformMetadata(rotation_degrees=rotation, preview_mirrored=preview_mirrored)
        self.backend, self.video_clock, self.now = backend, video_clock, now
        self.cv, self.cap = cv, None
        self.clock = StreamClock(now)
        self.metadata = {}

    def open(self):
        if self.cap is not None:
            raise CaptureError("source already open")
        if self.video is not None and not self.video.is_file():
            raise CaptureError("video file does not exist")
        cv = self.cv or _cv2()
        self.cv = cv
        target = str(self.video) if self.video is not None else self.device
        cap = cv.VideoCapture(target) if self.backend is None else cv.VideoCapture(target, self.backend)
        self.cap = cap
        try:
            if not cap.isOpened():
                raise CaptureError("cannot open camera/video (device, permission or backend unavailable)")
            requested = {}
            if self.video is None:
                for key, prop, value in (("width", cv.CAP_PROP_FRAME_WIDTH, self.width),
                                         ("height", cv.CAP_PROP_FRAME_HEIGHT, self.height),
                                         ("fps", cv.CAP_PROP_FPS, self.fps),
                                         ("buffer_size", cv.CAP_PROP_BUFFERSIZE, 1)):
                    requested[key] = {"requested": value, "set_accepted": bool(cap.set(prop, value))}
            nominal_fps = float(cap.get(cv.CAP_PROP_FPS))
            if self.video is not None and self.video_clock == "fps" and (not np.isfinite(nominal_fps) or nominal_fps <= 0):
                raise CaptureError("file has no usable FPS; select decoder timestamps explicitly")
            self.media_fps = nominal_fps
            self.clock.reset()
            self.metadata = {
                "kind": "file" if self.video is not None else "camera", "opencv_version": cv.__version__,
                "backend": cap.getBackendName(), "requests": requested,
                "reported_size": [int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))],
                "reported_fps": nominal_fps if np.isfinite(nominal_fps) else None,
                "timestamp_kind": f"media_{self.video_clock}" if self.video else "software_read_complete_not_exposure",
                "source_format": "BGR8", "pixel_format": "RGB8", "crop": "none",
                "transform": self.transform.model_dump(mode="json"),
                "driver_drops": None, "exposure_age_known": False,
            }
        except BaseException:
            self.close()
            raise

    def read(self):
        if self.cap is None:
            raise CaptureError("source is closed")
        ok, bgr = self.cap.read()
        received = self.now()
        if not ok or bgr is None or not np.asarray(bgr).size:
            if self.video is not None:
                raise StopIteration
            raise CaptureError("camera read failed/empty frame; reconnect requires a new stream")
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise CaptureError("unsupported source pixel format; expected BGR8")
        timestamp = received
        if self.video is not None:
            if self.video_clock == "fps":
                timestamp = round((self.clock.frame_id + 1) * 1_000_000_000 / self.media_fps)
            else:
                milliseconds = self.cap.get(self.cv.CAP_PROP_POS_MSEC)
                if not np.isfinite(milliseconds) or milliseconds < 0:
                    raise CaptureError("invalid decoder timestamp")
                timestamp = round(milliseconds * 1_000_000)
        stream, index, timestamp = self.clock.stamp(timestamp)
        rgb = self.cv.cvtColor(bgr, self.cv.COLOR_BGR2RGB)
        if self.transform.rotation_degrees:
            rgb = np.rot90(rgb, -(self.transform.rotation_degrees // 90))
        info = FrameInfo(stream, index, timestamp, rgb.shape[1], rgb.shape[0], self.transform,
                         "recording_relative" if self.video else "monotonic")
        self.metadata["actual_size"] = [info.width, info.height]
        return CapturedFrame(info, rgb, received)

    def close(self):
        if self.cap is not None:
            cap, self.cap = self.cap, None
            cap.release()


class SyntheticSource:
    """Finite, visibly synthetic RGB source. No camera or user data is accessed."""
    def __init__(self, *, frames=90, fps=30., width=160, height=120, paced=False, now=monotonic_ns):
        if frames < 1 or fps <= 0 or not np.isfinite(fps) or min(width, height) <= 0:
            raise ValueError("invalid synthetic source settings")
        self.frames, self.fps, self.width, self.height = frames, fps, width, height
        self.paced, self.now = paced, now
        self.clock = StreamClock(now)
        self.opened = False
        self._stop = Event()
        self.metadata = {"kind": "synthetic", "synthetic": True, "timestamp_kind": "synthetic_index_fps", "pixel_format": "RGB8"}

    def open(self):
        if self.opened:
            raise CaptureError("source already open")
        self.clock.reset()
        self._stop.clear()
        self.opened = True
        self.started = self.now()

    def read(self):
        if not self.opened:
            raise CaptureError("source is closed")
        index = self.clock.frame_id + 1
        if index >= self.frames:
            raise StopIteration
        timestamp = round(index * 1_000_000_000 / self.fps)
        if self.paced and self._stop.wait(max(0, (self.started + timestamp - self.now()) / 1_000_000_000)):
            raise StopIteration
        stream, index, timestamp = self.clock.stamp(timestamp)
        image = np.zeros((self.height, self.width, 3), np.uint8)
        image[:, :, 0] = index % 256
        info = FrameInfo(stream, index, timestamp, self.width, self.height, TransformMetadata(), "synthetic")
        return CapturedFrame(info, image, self.now())

    def request_stop(self):
        self._stop.set()

    def close(self):
        self.opened = False
