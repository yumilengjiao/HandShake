"""Causal, NumPy-only landmarks94-v1 extraction shared by replay and inference."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from shared import FEATURE_SCHEMA_VERSION
from shared.contracts import HandObservation
from shared.features.schema import (CX, CY, DS_DT, DT, MASK, MISSING, MOTION_VALID, QUALITY, SCALE, VX, VY, FeatureConfig)


def check_ns(value: int, name: str = "timestamp_ns") -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be integer nanoseconds in [0,2**63-1]")
    return value


def check_identity(stream_id: str, track_id: str, context_epoch: int) -> tuple[str, str, int]:
    if not isinstance(stream_id, str) or not stream_id or not isinstance(track_id, str) or not track_id:
        raise ValueError("stream_id and track_id must be nonempty strings")
    check_ns(context_epoch, "context_epoch")
    return stream_id, track_id, context_epoch


@dataclass(frozen=True)
class FeatureSample:
    values: np.ndarray
    timestamp_ns: int
    source_frame_id: int
    stream_id: str
    track_id: str
    context_epoch: int
    observation_valid: bool
    motion_valid: bool
    history_initialized: bool
    geometric_valid_mask: np.ndarray
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    timebase: str = "monotonic"

    def __post_init__(self):
        check_ns(self.timestamp_ns)
        check_ns(self.source_frame_id, "source_frame_id")
        check_identity(self.stream_id, self.track_id, self.context_epoch)
        if self.timebase not in {"monotonic", "recording_relative", "synthetic"}:
            raise ValueError("unsupported timebase")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported feature schema")
        for name in ("observation_valid", "motion_valid", "history_initialized"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        values = np.array(self.values, dtype=np.float64, copy=True)
        raw_mask = np.asarray(self.geometric_valid_mask)
        if values.shape != (94,) or not np.isfinite(values).all() or raw_mask.shape != (69,) or raw_mask.dtype != np.bool_:
            raise ValueError("FeatureSample requires finite values[94] and boolean geometric_valid_mask[69]")
        mask = raw_mask.copy()
        values.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "geometric_valid_mask", mask)

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.stream_id, self.track_id, self.context_epoch


class FeatureExtractor:
    """One stream/track/epoch at a time; instantiate independently per recording/track.

    An unprocessed source frame advances health only. A processed no-hand frame
    can append missing evidence only after an actual observation established a track.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.reset()

    def reset(self) -> None:
        self.identity = None
        self._timebase = "monotonic"
        self._previous_timestamp_ns = None
        self._health_timestamp_ns = None
        self._last_valid_timestamp_ns = None
        self._previous_geometry = None

    def _prepare(self, identity, timestamp_ns, source_frame_id, timebase):
        check_ns(timestamp_ns)
        check_ns(source_frame_id, "source_frame_id")
        check_identity(*identity)
        if identity != self.identity or timebase != self._timebase:
            self.reset()
            self.identity, self._timebase = identity, timebase
        if self._health_timestamp_ns is not None and timestamp_ns < self._health_timestamp_ns:
            raise ValueError("source timestamps must strictly increase and cannot precede the health clock")
        if self._previous_timestamp_ns is not None:
            if timestamp_ns <= self._previous_timestamp_ns:
                raise ValueError("processed source timestamps must strictly increase")
            if timestamp_ns - self._previous_timestamp_ns > self.config.reset_gap_ns:
                self._previous_timestamp_ns = None
                self._last_valid_timestamp_ns = None
                self._previous_geometry = None
        dt = 0.0 if self._previous_timestamp_ns is None else (timestamp_ns - self._previous_timestamp_ns) / 1e9
        self._health_timestamp_ns = timestamp_ns
        return dt

    def extract(self, observation: HandObservation, image_size: Sequence[int], *, context_epoch: int = 0) -> FeatureSample:
        if len(image_size) != 2 or any(type(x) is not int or x <= 0 for x in image_size):
            raise ValueError("image_size must be [positive integer width,height]")
        if observation.transform_metadata.model_mirrored:
            raise ValueError("features require unmirrored model coordinates")
        identity = check_identity(observation.stream_id, observation.track_id, context_epoch)
        dt = self._prepare(identity, observation.source_timestamp_ns, observation.source_frame_id, observation.timebase)
        points = np.asarray(observation.landmarks_image, dtype=np.float64).copy()
        points[:, 1] *= image_size[1] / image_size[0]
        point_mask = np.asarray(observation.validity_mask, dtype=bool) & np.isfinite(points).all(axis=1)
        key_valid = point_mask[[0, 5, 9, 13, 17]].all()
        scale = float(np.linalg.norm(points[[5, 9, 17], :2] - points[0, :2], axis=1).mean()) if key_valid else 0.0
        valid = bool(observation.observation_valid and key_valid and np.isfinite(scale)
                     and scale >= self.config.min_palm_scale and observation.quality >= self.config.min_geometry_quality)
        values = np.zeros(94, dtype=np.float64)
        geometry_mask = np.zeros(69, dtype=bool)
        quality = float(observation.quality) if key_valid and scale >= self.config.min_palm_scale else 0.0
        motion = False
        if valid:
            center = points[[0, 5, 9, 13, 17], :2].mean(axis=0)
            local = (points - points[0]) / max(scale, self.config.epsilon)
            local[~point_mask] = 0.0
            values[:63] = local.reshape(63)
            values[CX:CY + 1] = center
            values[SCALE] = scale
            geometry_mask[:63] = np.repeat(point_mask, 3)
            geometry_mask[[CX, CY, SCALE]] = True
            dt_ns = round(dt * 1e9)
            motion = bool(self._previous_geometry is not None and self.config.min_motion_dt_ns <= dt_ns <= self.config.max_motion_gap_ns)
            if motion:
                previous_center, previous_scale = self._previous_geometry
                values[VX:VY + 1] = (center - previous_center) / dt
                values[DS_DT] = (scale - previous_scale) / dt
                geometry_mask[[VX, VY, DS_DT]] = True
            values[MASK] = point_mask.astype(float)
            self._previous_geometry = center, scale
            self._last_valid_timestamp_ns = observation.source_timestamp_ns
        else:
            self._previous_geometry = None
        values[DT] = dt
        values[MISSING] = 0.0 if self._last_valid_timestamp_ns is None else (observation.source_timestamp_ns - self._last_valid_timestamp_ns) / 1e6
        values[QUALITY] = quality
        values[MOTION_VALID] = float(motion)
        self._previous_timestamp_ns = observation.source_timestamp_ns
        return FeatureSample(values, observation.source_timestamp_ns, observation.source_frame_id, *identity,
                             valid, motion, self._last_valid_timestamp_ns is not None, geometry_mask, timebase=observation.timebase)

    def missing(self, timestamp_ns: int, source_frame_id: int, stream_id: str, track_id: str, *, context_epoch: int = 0, quality: float = 0.0) -> FeatureSample | None:
        identity = check_identity(stream_id, track_id, context_epoch)
        check_ns(timestamp_ns)
        check_ns(source_frame_id, "source_frame_id")
        if type(quality) not in (float, int) or not np.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError("quality must be finite in [0,1]")
        if identity != self.identity:
            if self.identity is not None:
                self.reset()
            return None
        dt = self._prepare(identity, timestamp_ns, source_frame_id, self._timebase)
        values = np.zeros(94, dtype=np.float64)
        values[DT] = dt
        values[MISSING] = 0.0 if self._last_valid_timestamp_ns is None else (timestamp_ns - self._last_valid_timestamp_ns) / 1e6
        values[QUALITY] = quality
        self._previous_timestamp_ns = timestamp_ns
        self._previous_geometry = None
        return FeatureSample(values, timestamp_ns, source_frame_id, *identity, False, False,
                             self._last_valid_timestamp_ns is not None, np.zeros(69, dtype=bool), timebase=self._timebase)

    def unprocessed(self, timestamp_ns: int, *, stream_id: str | None = None, track_id: str | None = None, context_epoch: int | None = None) -> None:
        check_ns(timestamp_ns)
        if self.identity is not None and ((stream_id is not None and stream_id != self.identity[0])
                or (track_id is not None and track_id != self.identity[1])
                or (context_epoch is not None and context_epoch != self.identity[2])):
            self.reset()
        if self._health_timestamp_ns is not None and timestamp_ns < self._health_timestamp_ns:
            raise ValueError("health timestamps must not decrease")
        self._health_timestamp_ns = timestamp_ns
        if self._previous_timestamp_ns is not None and timestamp_ns - self._previous_timestamp_ns > self.config.reset_gap_ns:
            self._previous_timestamp_ns = None
            self._last_valid_timestamp_ns = None
            self._previous_geometry = None
