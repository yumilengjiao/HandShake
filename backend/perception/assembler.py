"""Shared whole-frame geometry/association for live SDK output and offline imports."""

import math

import numpy as np

from backend.perception.quality import QualitySettings, geometry
from backend.tracking.association import Association, TrackingSettings
from shared.contracts import FrameStatus, HandObservation, PerceptionStatus
from shared.observations import ObservationFrame


class ObservationAssembler:
    """No pixels, classifier, clock invention or device actions at this boundary."""

    def __init__(self, *, quality=QualitySettings(), tracking=TrackingSettings()):
        self.quality = quality
        self.association = Association(tracking, quality)
        self.stream = None
        self.last_ns = None

    def assemble(
        self,
        info,
        raw,
        *,
        processed_at_ns=None,
        processing_time_origin="unavailable_external",
        completion_clock=None,
    ):
        if self.stream is not None and self.stream != info.stream_id:
            raise ValueError("new stream requires a new observation assembler")
        if self.last_ns is not None and info.timestamp_ns <= self.last_ns:
            raise ValueError("raw observations must advance source time")
        if len(raw) > 2:
            raise ValueError("detector returned more than two hands")
        self.stream, self.last_ns = info.stream_id, info.timestamp_ns
        shapes = [
            geometry(hand.landmarks, info.width, info.height, self.quality)
            for hand in raw
        ]
        handedness = [
            hand.handedness if hand.handedness in {"left", "right"} else "unknown"
            for hand in raw
        ]
        matches = self.association.associate(info, shapes, handedness)
        observations, worlds = [], []
        for hand, shape, side, (track, flags) in zip(raw, shapes, handedness, matches):
            score = hand.score
            if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                score = None
            observations.append(
                HandObservation(
                    stream_id=info.stream_id,
                    source_frame_id=info.frame_id,
                    source_timestamp_ns=info.timestamp_ns,
                    timebase=info.timebase,
                    track_id=track,
                    landmarks_image=shape.landmarks.tolist(),
                    validity_mask=shape.mask.tolist(),
                    observation_valid=not flags,
                    handedness=side,
                    handedness_score=score,
                    quality=0.0 if flags else 1.0,
                    quality_flags=flags,
                    transform_metadata=info.transform,
                )
            )
            world = np.asarray(hand.world) if hand.world is not None else None
            worlds.append(
                tuple(tuple(float(v) for v in p) for p in world)
                if world is not None
                and world.shape == (21, 3)
                and np.isfinite(world).all()
                else None
            )
        status = FrameStatus(
            stream_id=info.stream_id,
            source_frame_id=info.frame_id,
            source_timestamp_ns=info.timestamp_ns,
            timebase=info.timebase,
            perception_status=PerceptionStatus.HANDS
            if raw
            else PerceptionStatus.NO_HAND,
            processed_at_ns=completion_clock() if completion_clock else processed_at_ns,
            processing_time_origin=processing_time_origin,
        )
        return ObservationFrame(info, status, tuple(observations), tuple(worlds))
