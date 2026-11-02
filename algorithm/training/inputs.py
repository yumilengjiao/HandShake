"""Validate both T11 caches and provenance before constructing any optimizer."""
import json
from pathlib import Path

from algorithm.data.cache import load_cache
from algorithm.data.dataset import canonical_digest
from shared.classes import vocabulary_for


def load_training_inputs(train_path, val_path):
    roots = [Path(train_path), Path(val_path)]
    headers = [json.loads((root / "index.json").read_text(encoding="utf-8")) for root in roots]
    common = ("format_version", "dataset_sha256", "split_sha256", "stats_sha256", "config_sha256", "synthetic", "stride", "vocabulary_version", "classes_sha256")
    if any(headers[0].get(key) != headers[1].get(key) for key in common):
        raise ValueError("train/val cache provenance or stride mismatch")
    vocabulary = vocabulary_for(headers[0]["vocabulary_version"])
    arrays, indexes = [], []
    for root, split in zip(roots, ("train", "val")):
        header = headers[0]
        values, index = load_cache(root, expected_dataset_sha256=header["dataset_sha256"],
                                  expected_split_sha256=header["split_sha256"],
                                  expected_stats_sha256=header["stats_sha256"],
                                  expected_config_sha256=header["config_sha256"], expected_split=split, expected_vocabulary=vocabulary)
        if not len(values["y"]) or not values["class_valid"].any():
            raise ValueError(f"{split} requires nonempty supervised endpoints")
        arrays.append(values)
        indexes.append(index)
    sources = [index["groups"] if index["format_version"] == "t11-cache-v3" else index["source_ids"] for index in indexes]
    leak_keys = ("subject_id", "recording_id", "lineage_id", "original_sha256") if headers[0]["format_version"] == "t11-cache-v3" else ("subject_id", "recording_id")
    for key in leak_keys:
        groups = [{source[key] for source in values} for values in sources]
        if groups[0] & groups[1]:
            raise ValueError(f"train/val {key} leakage")
    provenance = {"train_index_sha256": indexes[0]["index_sha256"],
                  "val_index_sha256": indexes[1]["index_sha256"],
                  **{key: indexes[0][key] for key in common}}
    for split, values, index in zip(("train", "val"), arrays, indexes):
        provenance[split] = dict(endpoints=len(values["y"]), supervised=int(values["class_valid"].sum()),
                                class_counts=[int(((values["y"] == i) & values["class_valid"]).sum()) for i in range(vocabulary.class_count)],
                                coverage=index["coverage"], endpoint_sha256=(canonical_digest({
                                    "endpoints": index["files"]["endpoints.bin"]["sha256"],
                                    "groups": index["files"]["groups.json"]["sha256"]})
                                    if index["format_version"] == "t11-cache-v3" else canonical_digest(index["source_ids"])))
    return arrays[0], arrays[1], provenance
