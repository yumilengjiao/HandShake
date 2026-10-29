"""Array-backed frame boundary for capture/perception; persisted observations stay portable.

FramePacket remains the v1 JSON wire format. Converting a live RGB image to nested
Python lists every frame is deliberately avoided; use to_packet only at that boundary.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from shared.contracts import FramePacket, FrameStatus, HandObservation, TransformMetadata


@dataclass(frozen=True)
class FrameInfo:
    stream_id: str
    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    transform: TransformMetadata
    timebase: Literal["monotonic", "recording_relative", "synthetic"] = "monotonic"

    def __post_init__(self):
        if not self.stream_id or any(type(v) is not int for v in (self.frame_id, self.timestamp_ns, self.width, self.height)):
            raise ValueError("frame identity, integer time and dimensions required")
        if self.frame_id < 0 or not 0 <= self.timestamp_ns < 2**63 or min(self.width, self.height) <= 0:
            raise ValueError("invalid frame identity/time/dimensions")
        if self.timebase not in {"monotonic", "recording_relative", "synthetic"}:
            raise ValueError("unknown frame timebase")

    def status(self, kind, *, processed_at_ns=None, reason=None):
        return FrameStatus(stream_id=self.stream_id, source_frame_id=self.frame_id,
                           source_timestamp_ns=self.timestamp_ns, timebase=self.timebase,
                           perception_status=kind, processed_at_ns=processed_at_ns,
                           drop_or_error_reason=reason)


@dataclass(frozen=True)
class CapturedFrame:
    info: FrameInfo
    rgb: np.ndarray
    received_at_ns: int  # local monotonic clock; NEVER substituted for media/source time

    def __post_init__(self):
        image = np.asarray(self.rgb)
        if image.dtype != np.uint8 or image.shape != (self.info.height, self.info.width, 3):
            raise ValueError("expected RGB uint8 [height,width,3]")
        image = np.array(image, copy=True, order="C")
        image.setflags(write=False)
        object.__setattr__(self, "rgb", image)

    def to_packet(self):
        return FramePacket(stream_id=self.info.stream_id, frame_id=self.info.frame_id,
                           capture_timestamp_ns=self.info.timestamp_ns, image=self.rgb.tolist(),
                           image_size=[self.info.width, self.info.height],
                           transform_metadata=self.info.transform, timebase=self.info.timebase)


@dataclass(frozen=True)
class ObservationFrame:
    info: FrameInfo
    status: FrameStatus
    observations: tuple[HandObservation, ...] = ()
    # world coordinates are optional SDK estimates, not measured camera-space positions
    world_landmarks: tuple[tuple[tuple[float, float, float], ...] | None, ...] = ()

    def __post_init__(self):
        if (self.status.stream_id, self.status.source_frame_id, self.status.source_timestamp_ns, self.status.timebase) != (
                self.info.stream_id, self.info.frame_id, self.info.timestamp_ns, self.info.timebase):
            raise ValueError("frame/status source mismatch")
        if (self.status.perception_status.value == "processed_hands") != bool(self.observations):
            raise ValueError("processed_hands must have observations; other statuses cannot")
        if self.world_landmarks and len(self.world_landmarks) != len(self.observations):
            raise ValueError("world/observation count mismatch")
        if len({o.track_id for o in self.observations}) != len(self.observations):
            raise ValueError("duplicate track in frame")
        for o in self.observations:
            if (o.stream_id, o.source_frame_id, o.source_timestamp_ns, o.timebase) != (
                    self.info.stream_id, self.info.frame_id, self.info.timestamp_ns, self.info.timebase):
                raise ValueError("observation source mismatch")
            if o.transform_metadata != self.info.transform:
                raise ValueError("observation transform mismatch")
        for world in self.world_landmarks:
            if world is not None and (np.asarray(world).shape != (21, 3) or not np.isfinite(world).all()):
                raise ValueError("invalid world landmarks")
