"""Ordered VIDEO perception, quality and association; publishes an entire frame."""
from time import monotonic_ns

from backend.perception.assembler import ObservationAssembler
from backend.perception.quality import QualitySettings
from backend.tracking.association import TrackingSettings
from shared.contracts import PerceptionStatus
from shared.observations import ObservationFrame


class PerceptionPipeline:
    def __init__(self, detector, *, quality=QualitySettings(), tracking=TrackingSettings(), now=monotonic_ns):
        self.detector, self.quality, self.now = detector, quality, now
        self.assembler = ObservationAssembler(quality=quality, tracking=tracking)
        self.association = self.assembler.association
        self.stream = None
        self.last_ns = None
        self.last_ms = None
        self.origin_ns = None

    def process(self, frame):
        info = frame.info
        if getattr(self.detector, "metadata", {}).get("disabled"):
            return self.skipped(info, "model_not_loaded")
        if self.stream is None:
            self.stream, self.origin_ns = info.stream_id, info.timestamp_ns
        if info.stream_id != self.stream:
            raise ValueError("new stream requires a new pipeline/MediaPipe VIDEO instance")
        if self.last_ns is not None and info.timestamp_ns <= self.last_ns:
            return self.skipped(info, "duplicate_or_backwards_source_time")
        self.last_ns = info.timestamp_ns
        ms = (info.timestamp_ns - self.origin_ns) // 1_000_000
        if self.last_ms is not None and ms <= self.last_ms:
            return self.skipped(info, "same_mediapipe_millisecond")
        self.last_ms = ms
        try:
            raw = self.detector.detect(frame.rgb, ms)
            return self.assembler.assemble(info, raw, completion_clock=lambda: self._completed(frame),
                                           processing_time_origin="local_runtime")
        except Exception as error:
            return ObservationFrame(info, info.status(PerceptionStatus.ERROR, processed_at_ns=self._completed(frame),
                                                     reason=f"{type(error).__name__}: {error}"))

    def _completed(self, frame):
        # Local elapsed processing time expressed on the source axis. For media
        # this is a duration mapping, NOT a timestamp from another clock domain.
        return frame.info.timestamp_ns + max(0, self.now() - frame.received_at_ns)

    @staticmethod
    def skipped(info, reason):
        return ObservationFrame(info, info.status(PerceptionStatus.NOT_PROCESSED, reason=reason))

    def close(self):
        self.detector.close()
