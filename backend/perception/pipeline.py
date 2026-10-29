"""Ordered VIDEO perception, quality and association; publishes an entire frame."""
from time import monotonic_ns
import math

import numpy as np

from backend.perception.quality import QualitySettings, geometry
from backend.tracking.association import Association, TrackingSettings
from shared.contracts import HandObservation, PerceptionStatus
from shared.observations import ObservationFrame


class PerceptionPipeline:
    def __init__(self, detector, *, quality=QualitySettings(), tracking=TrackingSettings(), now=monotonic_ns):
        self.detector, self.quality, self.now = detector, quality, now
        self.association = Association(tracking, quality)
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
            if len(raw) > 2:
                raise ValueError("detector returned more than two hands")
            shapes = [geometry(h.landmarks, info.width, info.height, self.quality) for h in raw]
            handedness = [h.handedness if h.handedness in {"left", "right"} else "unknown" for h in raw]
            matches = self.association.associate(info, shapes, handedness)
            observations, worlds = [], []
            for h, g, hand, (track, flags) in zip(raw, shapes, handedness, matches):
                score = h.score
                if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                    score = None
                observations.append(HandObservation(
                    stream_id=info.stream_id, source_frame_id=info.frame_id, source_timestamp_ns=info.timestamp_ns,
                    timebase=info.timebase, track_id=track, landmarks_image=g.landmarks.tolist(),
                    validity_mask=g.mask.tolist(), observation_valid=not flags,
                    handedness=hand, handedness_score=score, quality=0. if flags else 1.,
                    quality_flags=flags, transform_metadata=info.transform))
                world = np.asarray(h.world) if h.world is not None else None
                worlds.append(tuple(tuple(float(v) for v in p) for p in world)
                              if world is not None and world.shape == (21,3) and np.isfinite(world).all() else None)
            status = info.status(PerceptionStatus.HANDS if raw else PerceptionStatus.NO_HAND,
                                 processed_at_ns=self._completed(frame))
            return ObservationFrame(info, status, tuple(observations), tuple(worlds))
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
