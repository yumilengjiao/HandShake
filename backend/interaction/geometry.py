"""Whole-frame geometry independent of neural class names or application intent."""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from shared.config import load_config


@dataclass(frozen=True)
class FrameGeometry:
    visible_tracks: tuple[str, ...]
    ambiguous: bool
    quality_valid: bool
    neutral: bool
    extended: int
    folded: int
    palm_xy: tuple[float, float] | None
    aspect_ratio: float


@lru_cache(maxsize=1)
def _neutral_thresholds():
    # Package defaults are fixed for this process. Do not reread and validate
    # seven YAML files on every geometry check (several checks per model frame).
    # Cache scalars/tuples only, not the config's mutable nested lists.
    protocol = load_config().gestures
    return (
        protocol.extended_pip_degrees_min,
        protocol.folded_pip_degrees_max,
        tuple(protocol.neutral_region_xy),
    )


def frame_geometry(frame, track_id):
    tracks = tuple(o.track_id for o in frame.observations)
    observation = next((o for o in frame.observations if o.track_id == track_id), None)
    ambiguous = any(
        "association" in flag or "ambiguous" in flag
        for o in frame.observations
        for flag in o.quality_flags
    )
    valid = (
        observation is not None
        and observation.observation_valid
        and observation.quality >= 0.5
        and all(observation.validity_mask)
    )
    neutral, center, extended, folded = False, None, 0, 0
    if valid:
        extended_min, folded_max, region = _neutral_thresholds()
        points = np.asarray(observation.landmarks_image, dtype=np.float64)
        center = tuple(float(v) for v in points[[0, 5, 9, 13, 17], :2].mean(0))
        corrected = points.copy()
        corrected[:, 1] *= frame.info.height / frame.info.width
        angles = []
        for mcp, pip, dip in ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)):
            a, b = corrected[mcp] - corrected[pip], corrected[dip] - corrected[pip]
            length = np.linalg.norm(a) * np.linalg.norm(b)
            angles.append(
                None
                if length <= 1e-12
                else math.degrees(
                    math.acos(float(np.clip(np.dot(a, b) / length, -1, 1)))
                )
            )
        extended = sum(
            a is not None and a >= extended_min for a in angles
        )
        folded = sum(
            a is not None and a <= folded_max for a in angles
        )
        xmin, xmax, ymin, ymax = region
        neutral = (
            1 <= extended <= 3
            and extended + folded == 4
            and xmin <= center[0] <= xmax
            and ymin <= center[1] <= ymax
        )
    return FrameGeometry(
        tracks,
        ambiguous,
        bool(valid),
        bool(neutral),
        extended,
        folded,
        center,
        frame.info.height / frame.info.width,
    )
