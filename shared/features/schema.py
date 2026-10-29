"""Draft, versioned feature layout and reproducible extraction configuration."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.resources import files
import json
from math import isfinite

from shared import FEATURE_SCHEMA_VERSION, PROTOCOL_VERSION
from shared.contracts import GESTURE_IDS

DIMENSION = 94
GEOMETRIC_DIMENSION = 69
LOCAL = slice(0, 63)
GLOBAL = slice(63, 73)
MASK = slice(73, 94)
CX, CY, VX, VY, SCALE, DS_DT, DT, MISSING, QUALITY, MOTION_VALID = range(63, 73)
FIELD_NAMES = tuple(f"local_{i}_{axis}" for i in range(21) for axis in "xyz") + (
    "cx", "cy", "vx", "vy", "s", "ds_dt", "dt_seconds", "missing_ms", "q_geom", "motion_valid",
) + tuple(f"mask_{i}" for i in range(21))


def canonical_hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FeatureConfig:
    """Implementation starting points; these thresholds have no real-data validation."""

    epsilon: float = 1e-6
    min_palm_scale: float = 0.005
    min_geometry_quality: float = 0.5
    min_motion_dt_ns: int = 1_000_000
    max_motion_gap_ns: int = 100_000_000
    max_sampling_gap_ns: int = 100_000_000
    reset_gap_ns: int = 500_000_000
    window_size: int = 32
    min_valid_ratio: float = 0.8
    max_window_span_ns: int = 3_200_000_000
    max_age_ns: int = 150_000_000
    dt_scale_seconds: float = 0.1
    dt_clip: float = 10.0
    missing_scale_ms: float = 1000.0
    missing_clip: float = 10.0
    std_epsilon: float = 1e-8

    def __post_init__(self):
        for key in ("min_motion_dt_ns", "max_motion_gap_ns", "max_sampling_gap_ns", "reset_gap_ns", "window_size", "max_window_span_ns", "max_age_ns"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("epsilon", "min_palm_scale", "dt_scale_seconds", "dt_clip", "missing_scale_ms", "missing_clip", "std_epsilon"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("min_geometry_quality", "min_valid_ratio"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{key} must be in (0,1]")
        if self.window_size != 32:
            raise ValueError("landmarks94-v1 uses exactly 32 observations per window")
        if not self.min_motion_dt_ns <= self.max_motion_gap_ns <= self.max_sampling_gap_ns < self.reset_gap_ns:
            raise ValueError("require min_motion_dt <= motion_gap <= sampling_gap < reset_gap")
        if self.min_palm_scale < self.epsilon:
            raise ValueError("min_palm_scale must not be less than epsilon")

    @property
    def config_hash(self) -> str:
        return canonical_hash(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)


def load_feature_schema() -> dict:
    schema = json.loads(files("shared.features").joinpath("feature_schema.json").read_text(encoding="utf-8"))
    expected_classes = {gesture.value: index for gesture, index in GESTURE_IDS.items()}
    if (schema.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or schema.get("dimension") != DIMENSION
            or schema.get("vocabulary_version") != PROTOCOL_VERSION or schema.get("class_ids") != expected_classes
            or schema.get("field_names") != list(FIELD_NAMES) or schema.get("default_config") != FeatureConfig().to_dict()):
        raise ValueError("unsupported or inconsistent feature schema/class table")
    return schema


def feature_schema_hash() -> str:
    return canonical_hash(load_feature_schema())
