"""Validate both T11 caches and provenance before constructing any optimizer."""
import json
from pathlib import Path

from algorithm.data.cache import load_cache
from algorithm.data.dataset import canonical_digest


def load_training_inputs(train_path, val_path):
    roots = [Path(train_path), Path(val_path)]
    headers = [json.loads((root / "index.json").read_text()) for root in roots]
    common = ("dataset_sha256", "split_sha256", "stats_sha256", "config_sha256", "synthetic", "stride")
    if any(headers[0].get(key) != headers[1].get(key) for key in common):
        raise ValueError("train/val cache provenance or stride mismatch")
    arrays, indexes = [], []
    for root, split in zip(roots, ("train", "val")):
        header = headers[0]
        values, index = load_cache(root, expected_dataset_sha256=header["dataset_sha256"],
                                  expected_split_sha256=header["split_sha256"],
                                  expected_stats_sha256=header["stats_sha256"],
                                  expected_config_sha256=header["config_sha256"], expected_split=split)
        if not len(values["y"]) or not values["class_valid"].any():
            raise ValueError(f"{split} requires nonempty supervised endpoints")
        arrays.append(values)
        indexes.append(index)
    for key in ("subject_id", "recording_id"):
        groups = [{source[key] for source in index["source_ids"]} for index in indexes]
        if groups[0] & groups[1]:
            raise ValueError(f"train/val {key} leakage")
    provenance = {"train_index_sha256": indexes[0]["index_sha256"],
                  "val_index_sha256": indexes[1]["index_sha256"],
                  **{key: indexes[0][key] for key in common}}
    for split, values, index in zip(("train", "val"), arrays, indexes):
        provenance[split] = dict(endpoints=len(values["y"]), supervised=int(values["class_valid"].sum()),
                                class_counts=[int(((values["y"] == i) & values["class_valid"]).sum()) for i in range(5)],
                                coverage=index["coverage"], endpoint_sha256=canonical_digest(index["source_ids"]))
    return arrays[0], arrays[1], provenance
