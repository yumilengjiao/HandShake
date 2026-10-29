"""Deterministic quality evidence; no invented per-landmark probabilities."""
from dataclasses import dataclass

import numpy as np

BONES = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
         (13,17),(0,17),(17,18),(18,19),(19,20))


@dataclass(frozen=True)
class QualitySettings:
    min_palm_scale: float = .025
    edge_margin: float = .05
    max_speed: float = 4.0  # image widths / second
    max_scale_ratio: float = 2.0
    max_shape_change: float = 1.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("quality thresholds must be positive finite draft values")


@dataclass
class Geometry:
    landmarks: np.ndarray
    mask: np.ndarray
    center: np.ndarray
    scale: float
    shape: np.ndarray
    bones: np.ndarray
    flags: list[str]


def geometry(points, width, height, settings=QualitySettings()):
    flags = []
    try:
        values = np.asarray(points, dtype=np.float64)
    except (ValueError, TypeError):
        values = np.empty((0, 3))
    if values.shape != (21, 3):
        values = np.zeros((21, 3), dtype=np.float64)
        mask = np.zeros(21, dtype=bool)
        flags.append("landmark_count_or_shape")
    else:
        mask = np.isfinite(values).all(axis=1)
        if not mask.all():
            flags.append("nonfinite_landmarks")
        values = np.where(mask[:, None], values, 0.)
    scaled = values.copy()
    scaled[:, 1] *= height / width
    center = scaled[[0, 5, 9, 13, 17], :2].mean(axis=0)
    scale = float(np.linalg.norm(scaled[[5, 9, 17], :2] - scaled[0, :2], axis=1).mean())
    if scale < settings.min_palm_scale:
        flags.append("palm_too_small")
    if ((values[:, :2] < -settings.edge_margin) | (values[:, :2] > 1 + settings.edge_margin)).any():
        flags.append("edge_out_of_bounds")
    normalized = (scaled - scaled[0]) / max(scale, 1e-6)
    bones = np.array([np.linalg.norm(normalized[a] - normalized[b]) for a, b in BONES])
    if (bones < 1e-6).any():
        flags.append("degenerate_bones")
    return Geometry(values, mask, center, scale, normalized, bones, flags)


def temporal_flags(current, previous, dt_ns, settings=QualitySettings()):
    if previous is None:
        return []
    if dt_ns <= 0:
        return ["invalid_motion_time"]
    flags = []
    if np.linalg.norm(current.center - previous.center) / (dt_ns / 1e9) > settings.max_speed:
        flags.append("abnormal_velocity")
    ratio = max(current.scale, previous.scale) / max(min(current.scale, previous.scale), 1e-6)
    if ratio > settings.max_scale_ratio:
        flags.append("abnormal_scale_change")
    if np.max(np.abs(current.bones - previous.bones)) > settings.max_shape_change:
        flags.append("abnormal_bone_change")
    return flags
