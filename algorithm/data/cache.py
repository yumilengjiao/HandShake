"""Rebuildable, hash-bound NPZ cache. Never pickle or overwrite a prior cache."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from algorithm.data.dataset import WindowDataset, canonical_digest
from shared import FEATURE_SCHEMA_VERSION, PROTOCOL_VERSION
from shared.classes import load_classes
from shared.features import feature_schema_hash


def write_cache(dataset: WindowDataset, directory: str | Path) -> dict:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=False)
    count = len(dataset)
    arrays = dict(
        x=np.stack([item["x"] for item in dataset.items]) if count else np.empty((0, 32, 94), dtype=np.float32),
        y=np.array([item["y"] for item in dataset.items], dtype=np.int64),
        class_valid=np.array([item["loss_masks"]["class_valid"] for item in dataset.items], dtype=bool),
        ready=np.array([item["ready"] for item in dataset.items], dtype=bool),
        valid_length=np.array([item["valid_length"] for item in dataset.items], dtype=np.int64),
    )
    npz = target / "features.npz"
    np.savez_compressed(npz, **arrays)
    dataset.normalizer.save(target / "stats.json")
    index = dict(
        format_version="t11-cache-v1", feature_schema_version=FEATURE_SCHEMA_VERSION,
        schema_sha256=feature_schema_hash(), vocabulary_version=PROTOCOL_VERSION,
        classes_sha256=load_classes().digest(), config_sha256=dataset.config.config_hash,
        dataset_sha256=dataset.manifest.dataset_sha256, split_sha256=dataset.manifest.digest(),
        stats_sha256=dataset.normalizer.stats_hash, split=dataset.split, stride=dataset.stride,
        synthetic=dataset.manifest.synthetic, npz_sha256=sha256(npz.read_bytes()).hexdigest(),
        source_ids=[item["source"] for item in dataset.items],
        endpoint_inventory=dataset.endpoint_inventory, coverage=dataset.coverage(),
    )
    index["index_sha256"] = canonical_digest(index)
    (target / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def load_cache(directory: str | Path, *, expected_dataset_sha256: str,
               expected_split_sha256: str, expected_stats_sha256: str,
               expected_config_sha256: str, expected_split: str) -> tuple[dict, dict]:
    root = Path(directory)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    claimed_hash = index.get("index_sha256")
    if claimed_hash != canonical_digest({key: value for key, value in index.items() if key != "index_sha256"}):
        raise ValueError("cache index hash mismatch")
    expected = dict(format_version="t11-cache-v1", feature_schema_version=FEATURE_SCHEMA_VERSION,
                    schema_sha256=feature_schema_hash(), vocabulary_version=PROTOCOL_VERSION,
                    classes_sha256=load_classes().digest(), dataset_sha256=expected_dataset_sha256,
                    split_sha256=expected_split_sha256, stats_sha256=expected_stats_sha256,
                    config_sha256=expected_config_sha256, split=expected_split)
    if any(index.get(key) != value for key, value in expected.items()):
        raise ValueError("cache schema/class/dataset/split/stats/config version mismatch")
    npz = root / "features.npz"
    if sha256(npz.read_bytes()).hexdigest() != index["npz_sha256"]:
        raise ValueError("cache NPZ content hash mismatch")
    from shared.features import FeatureNormalizer
    stats = FeatureNormalizer.load(root / "stats.json", expected_dataset_hash=expected_dataset_sha256,
                                   expected_split_hash=expected_split_sha256)
    if stats.stats_hash != expected_stats_sha256 or stats.config.config_hash != expected_config_sha256:
        raise ValueError("cache statistics provenance mismatch")
    with np.load(npz, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    if set(arrays) != {"x", "y", "class_valid", "ready", "valid_length"}:
        raise ValueError("invalid cache fields")
    count = len(index["source_ids"])
    if arrays["x"].shape != (count, 32, 94) or not np.isfinite(arrays["x"]).all():
        raise ValueError("cache features must be finite [N,32,94]")
    if any(arrays[name].shape != (count,) for name in arrays if name != "x"):
        raise ValueError("cache endpoint arrays disagree")
    if arrays["class_valid"].dtype != bool or arrays["ready"].dtype != bool:
        raise ValueError("cache masks must be boolean")
    if arrays["y"].dtype.kind not in "iu" or arrays["valid_length"].dtype.kind not in "iu":
        raise ValueError("cache labels/lengths must be integers")
    labels = arrays["y"]
    if np.any((arrays["class_valid"] & ((labels < 0) | (labels >= load_classes().class_count))) | (~arrays["class_valid"] & (labels != -1))):
        raise ValueError("cache labels do not match vocabulary/masks")
    if np.any((arrays["valid_length"] < 1) | (arrays["valid_length"] > 32)):
        raise ValueError("cache valid_length must be in [1,32]")
    return arrays, index
