"""Train-only field-aware statistics; missing times use fixed draft scales."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np

from shared import FEATURE_SCHEMA_VERSION, PROTOCOL_VERSION
from shared.classes import ClassVocabulary, vocabulary_for
from shared.features.geometry import FeatureSample
from shared.features.schema import (DT, FIELD_NAMES, MISSING, FeatureConfig, canonical_hash, feature_schema_hash)


class FeatureStatisticsWarning(UserWarning):
    """A training field has no usable samples or variance; fallback is explicit."""


@dataclass(frozen=True)
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray
    dataset_hash: str
    split_hash: str
    schema_hash: str
    config_hash: str
    fallback_fields: tuple[str, ...] = ()
    fitted_split: str = "train"

    def __post_init__(self):
        for name in ("dataset_hash", "split_hash", "schema_hash", "config_hash"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty provenance identifier")
        if self.fitted_split != "train":
            raise ValueError("statistics may only be fitted on train")
        for name in ("mean", "std", "count"):
            raw = np.asarray(getattr(self, name))
            if raw.shape != (69,):
                raise ValueError("statistics must have exactly 69 geometric fields")
            if name == "count" and (raw.dtype.kind not in "iu" or (raw < 0).any()):
                raise ValueError("field counts must be nonnegative integers")
            array = np.array(raw, dtype=np.int64 if name == "count" else np.float64, copy=True)
            if not np.isfinite(array).all() or (name == "std" and (array <= 0).any()):
                raise ValueError("statistics must be finite, with positive standard deviations")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if len(set(self.fallback_fields)) != len(self.fallback_fields) or not set(self.fallback_fields).issubset(FIELD_NAMES[:69]):
            raise ValueError("unknown or duplicate fallback field")

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(), "std": self.std.tolist(), "count": self.count.tolist(),
            "dataset_hash": self.dataset_hash, "split_hash": self.split_hash, "schema_hash": self.schema_hash,
            "config_hash": self.config_hash, "fallback_fields": list(self.fallback_fields), "fitted_split": self.fitted_split,
        }


class FeatureNormalizer:
    def __init__(self, stats: FeatureStats, config: FeatureConfig | None = None, *, vocabulary: ClassVocabulary | None = None):
        self.vocabulary = vocabulary or vocabulary_for()
        self.config = config or FeatureConfig()
        if stats.config_hash != self.config.config_hash or stats.schema_hash != feature_schema_hash():
            raise ValueError("statistics do not match the feature config/schema")
        self.stats = stats

    @classmethod
    def fit(cls, samples: Iterable[FeatureSample], *, dataset_hash: str, split_hash: str,
            split: str = "train", config: FeatureConfig | None = None,
            vocabulary: ClassVocabulary | None = None) -> "FeatureNormalizer":
        if split != "train":
            raise ValueError("statistics may only be fitted on train; validation/test fitting is forbidden")
        config = config or FeatureConfig()
        for name, value in (("dataset_hash", dataset_hash), ("split_hash", split_hash)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty provenance identifier")
        # Welford updates avoid cancellation for global centers with small motion.
        count = np.zeros(69, dtype=np.int64)
        mean = np.zeros(69, dtype=np.float64)
        m2 = np.zeros(69, dtype=np.float64)
        sample_count = 0
        for sample in samples:
            if not isinstance(sample, FeatureSample):
                raise TypeError("fit requires raw FeatureSample records with per-field validity")
            valid = sample.geometric_valid_mask
            count[valid] += 1
            delta = sample.values[:69][valid] - mean[valid]
            mean[valid] += delta / count[valid]
            m2[valid] += delta * (sample.values[:69][valid] - mean[valid])
            sample_count += 1
        if sample_count == 0:
            raise ValueError("cannot fit empty training data")
        variance = np.divide(m2, count, out=np.zeros(69), where=count > 0)
        std = np.sqrt(np.maximum(variance, 0.0))
        empty = count == 0
        degenerate = (~empty) & (std < config.std_epsilon)
        mean[empty] = 0.0
        std[empty | degenerate] = 1.0
        fallback = tuple(FIELD_NAMES[index] for index in np.flatnonzero(empty | degenerate))
        if fallback:
            warnings.warn("Training statistics fallback (empty mean=0/std=1; near-constant std=1): " + ", ".join(fallback), FeatureStatisticsWarning, stacklevel=2)
        stats = FeatureStats(mean, std, count, dataset_hash, split_hash, feature_schema_hash(), config.config_hash, fallback)
        return cls(stats, config, vocabulary=vocabulary)

    def transform(self, sample: FeatureSample | np.ndarray, geometric_valid_mask: np.ndarray | None = None) -> np.ndarray:
        if isinstance(sample, FeatureSample):
            if geometric_valid_mask is not None:
                raise ValueError("FeatureSample already supplies its geometric validity mask")
            values = sample.values
            mask = sample.geometric_valid_mask
        else:
            values = np.asarray(sample, dtype=np.float64)
            if geometric_valid_mask is None:
                raise ValueError("raw tensors require explicit per-field geometric_valid_mask")
            mask = np.asarray(geometric_valid_mask)
        if values.ndim < 1 or values.shape[-1] != 94 or not np.isfinite(values).all():
            raise ValueError("features must be finite [...,94]")
        if mask.shape != values.shape[:-1] + (69,) or mask.dtype != np.bool_:
            raise ValueError("geometric_valid_mask must be bool [...,69]")
        output = values.copy()
        output[..., :69] = np.where(mask, (values[..., :69] - self.stats.mean) / self.stats.std, 0.0)
        output[..., DT] = np.clip(values[..., DT] / self.config.dt_scale_seconds, 0.0, self.config.dt_clip)
        output[..., MISSING] = np.clip(values[..., MISSING] / self.config.missing_scale_ms, 0.0, self.config.missing_clip)
        if not np.isfinite(output).all():
            raise ValueError("normalization produced nonfinite values")
        return output

    def inverse_geometric(self, features: np.ndarray) -> np.ndarray:
        """Restore first 69 fields; masks/quality and fixed-scaled time fields stay as supplied.

        Use validity masks when interpreting restored zero-filled invalid geometry.
        """
        values = np.asarray(features, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 94 or not np.isfinite(values).all():
            raise ValueError("features must be finite [...,94]")
        output = values.copy()
        output[..., :69] = values[..., :69] * self.stats.std + self.stats.mean
        return output

    def to_dict(self) -> dict:
        body = {
            "format_version": 1 if self.vocabulary.vocabulary_version == PROTOCOL_VERSION else 2,
            "status": "draft", "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "vocabulary_version": self.vocabulary.vocabulary_version,
            "class_ids": {token: index for index, token in enumerate(self.vocabulary.tokens)},
            "config": self.config.to_dict(), "stats": self.stats.to_dict(),
        }
        return body | {"content_hash": canonical_hash(body)}

    @property
    def stats_hash(self) -> str:
        return self.to_dict()["content_hash"]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as output:
            output.write(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | Path, *, expected_dataset_hash: str | None = None,
             expected_split_hash: str | None = None, config: FeatureConfig | None = None) -> "FeatureNormalizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"format_version", "status", "feature_schema_version", "vocabulary_version", "class_ids", "config", "stats", "content_hash"}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("invalid normalizer document fields")
        body = {key: value for key, value in data.items() if key != "content_hash"}
        if data["content_hash"] != canonical_hash(body):
            raise ValueError("normalizer content hash mismatch")
        vocabulary = vocabulary_for(data["vocabulary_version"])
        expected_classes = {token: index for index, token in enumerate(vocabulary.tokens)}
        format_version = 1 if vocabulary.vocabulary_version == PROTOCOL_VERSION else 2
        if (type(data["format_version"]) is not int or data["format_version"] != format_version or data["status"] != "draft"
                or data["feature_schema_version"] != FEATURE_SCHEMA_VERSION
                or data["class_ids"] != expected_classes):
            raise ValueError("unsupported normalizer schema or class table")
        try:
            saved_config = FeatureConfig(**data["config"])
            stats_data = data["stats"]
            if set(stats_data) != {"mean", "std", "count", "dataset_hash", "split_hash", "schema_hash", "config_hash", "fallback_fields", "fitted_split"}:
                raise ValueError("invalid statistics fields")
            stats = FeatureStats(**(stats_data | {"fallback_fields": tuple(stats_data["fallback_fields"])}))
        except (TypeError, KeyError) as error:
            raise ValueError("invalid normalizer configuration/statistics") from error
        if config is not None and config.config_hash != saved_config.config_hash:
            raise ValueError("normalizer config does not match requested config")
        if expected_dataset_hash is not None and stats.dataset_hash != expected_dataset_hash:
            raise ValueError("normalizer dataset hash mismatch")
        if expected_split_hash is not None and stats.split_hash != expected_split_hash:
            raise ValueError("normalizer split hash mismatch")
        return cls(stats, saved_config, vocabulary=vocabulary)
