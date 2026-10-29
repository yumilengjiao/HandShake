"""B0 finite-history geometric baseline; its scores are not probabilities.

Thresholds are draft engineering settings to select on development/validation
data. They are not a trained classifier, an intent estimate, or event logic.
All distances use T11's image-width coordinate units (including corrected y).
"""

from dataclasses import asdict, dataclass
from math import acos, degrees, isfinite

import numpy as np

from shared.classes import load_classes
from shared.contracts import Gesture


@dataclass(frozen=True)
class RuleThresholds:
    """Published B0 settings, deliberately separate from annotation thresholds.

    The 150 ms history requirement applies to swipes, not to static event hold
    times. Static classes describe the current pose; the later shared event
    engine, not this classifier, will enforce wake/exit holds and releases.
    """

    history_size: int = 32
    extended_pip_degrees_min: float = 150.0
    folded_pip_degrees_max: float = 100.0
    extended_tip_to_pip_distance_ratio_min: float = 1.05
    folded_tip_to_mcp_distance_max: float = 1.5
    static_max_speed_widths_per_s: float = 0.05
    min_quality: float = 0.5
    max_sample_gap_ns: int = 100_000_000
    swipe_min_duration_ns: int = 150_000_000
    swipe_max_duration_ns: int = 1_500_000_000
    swipe_min_displacement_widths: float = 0.18
    swipe_max_vertical_span_widths: float = 0.15
    swipe_min_mean_speed_widths_per_s: float = 0.15
    swipe_min_current_speed_widths_per_s: float = 0.05
    swipe_min_direction_consistency: float = 0.8

    def __post_init__(self):
        if self.history_size != 32:
            raise ValueError("B0 uses the same 32-observation window as MLP/TCN")
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_sample_gap_ns", "swipe_min_duration_ns", "swipe_max_duration_ns"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be integer nanoseconds")
        if not 0 < self.folded_pip_degrees_max < self.extended_pip_degrees_min <= 180:
            raise ValueError("folded and extended angle ranges must be separated")
        if self.swipe_min_duration_ns >= self.swipe_max_duration_ns:
            raise ValueError("swipe duration range must increase")
        if self.min_quality > 1 or self.swipe_min_direction_consistency > 1:
            raise ValueError("quality and direction consistency must be in (0,1]")


def _pip_angle(points: np.ndarray, mcp: int, pip: int, dip: int) -> float | None:
    first, second = points[mcp] - points[pip], points[dip] - points[pip]
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None
    cosine = float(np.dot(first, second) / denominator)
    return degrees(acos(float(np.clip(cosine, -1.0, 1.0))))


class RuleBaseline:
    """Classify a bounded raw-94 feature history without accumulating state.

    ``classify`` always consumes a full 32-row window. Dynamic classes take
    priority over a matching hand pose. Inputs must be in geometric units;
    ``RulePredictor`` can invert T11 standardization before calling this class.
    ``logits`` returns one-hot decision scores solely to share the prediction
    contract. These are neither learned logits nor calibrated probabilities.
    """

    def __init__(self, thresholds: RuleThresholds | None = None, *, vocabulary=None):
        self.thresholds = thresholds or RuleThresholds()
        self.vocabulary = vocabulary or load_classes()
        self.class_tokens = tuple(self.vocabulary.tokens)

    def _valid(self, row: np.ndarray) -> bool:
        return bool(row[71] >= self.thresholds.min_quality and row[70] == 0.0 and np.all(row[73:] == 1.0))

    def classify(self, features, timestamps_ns) -> Gesture:
        values = np.asarray(features, dtype=np.float64)
        timestamps = np.asarray(timestamps_ns)
        if values.shape != (self.thresholds.history_size, 94) or not np.isfinite(values).all():
            raise ValueError("B0 requires finite [32,94] raw features")
        if timestamps.shape != (self.thresholds.history_size,) or not np.issubdtype(timestamps.dtype, np.integer):
            raise ValueError("B0 requires 32 integer nanosecond timestamps")
        if np.any(timestamps < 0) or np.any(timestamps > 2**63 - 1):
            raise ValueError("B0 timestamps must fit nonnegative int64 nanoseconds")
        timestamps = timestamps.astype(np.int64, copy=False)
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("B0 timestamps must be nonnegative and strictly increasing")
        current = values[-1]
        if not self._valid(current):
            return Gesture.NO_ACTION

        swipe = self._swipe(values, timestamps)
        if swipe is not None:
            return swipe
        if current[72] != 1.0 or float(np.linalg.norm(current[65:67])) > self.thresholds.static_max_speed_widths_per_s:
            return Gesture.NO_ACTION

        points = current[:63].reshape(21, 3)
        fingers = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
        angles = [_pip_angle(points, mcp, pip, dip) for mcp, pip, dip, _ in fingers]
        if any(angle is None for angle in angles):
            return Gesture.NO_ACTION
        extended = all(
            angle >= self.thresholds.extended_pip_degrees_min
            and np.linalg.norm(points[tip] - points[0]) >= self.thresholds.extended_tip_to_pip_distance_ratio_min * np.linalg.norm(points[pip] - points[0])
            for angle, (_, pip, _, tip) in zip(angles, fingers)
        )
        if extended:
            return Gesture.WAKE_OPEN
        folded = all(
            angle <= self.thresholds.folded_pip_degrees_max
            and np.linalg.norm(points[tip] - points[mcp]) <= self.thresholds.folded_tip_to_mcp_distance_max
            for angle, (mcp, _, _, tip) in zip(angles, fingers)
        )
        return Gesture.EXIT_POSE if folded else Gesture.NO_ACTION

    def _swipe(self, values: np.ndarray, timestamps: np.ndarray) -> Gesture | None:
        config = self.thresholds
        last = values[-1]
        if last[72] != 1.0 or abs(last[65]) < config.swipe_min_current_speed_widths_per_s:
            return None
        # Only a continuous valid suffix participates; a missing/poor observation
        # or invalid derivative blocks reaching into an earlier movement.
        for start in range(len(values) - 2, -1, -1):
            if not self._valid(values[start]) or values[start + 1, 72] != 1.0:
                break
            if timestamps[start + 1] - timestamps[start] > config.max_sample_gap_ns:
                break
            duration_ns = int(timestamps[-1]) - int(timestamps[start])
            if duration_ns > config.swipe_max_duration_ns:
                break
            if duration_ns < config.swipe_min_duration_ns:
                continue
            centers = values[start:, 63:65]
            dx = float(centers[-1, 0] - centers[0, 0])
            if abs(dx) < config.swipe_min_displacement_widths:
                continue
            if np.sign(dx) != np.sign(last[65]):
                continue
            if np.ptp(centers[:, 1]) > config.swipe_max_vertical_span_widths:
                continue
            travel = float(np.abs(np.diff(centers[:, 0])).sum())
            if travel <= 0 or abs(dx) / travel < config.swipe_min_direction_consistency:
                continue
            if abs(dx) / (duration_ns / 1e9) < config.swipe_min_mean_speed_widths_per_s:
                continue
            return Gesture.SWIPE_RIGHT if dx > 0 else Gesture.SWIPE_LEFT
        return None

    def logits(self, features, timestamps_ns) -> list[float]:
        scores = [0.0] * self.vocabulary.class_count
        scores[self.vocabulary.id_for(self.classify(features, timestamps_ns).value)] = 1.0
        return scores
