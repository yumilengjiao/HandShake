"""Bounded numeric score archives tied to a fixed model/cache endpoint inventory."""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from algorithm.data.dataset import canonical_digest
from algorithm.data.source_archive import file_sha256, strict_json
from shared.classes import vocabulary_for
from shared.desktop_protocol import WireModel

MAX_ROWS = 2_000_000
BATCH = 4096


class ScoreProvenance(WireModel):
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stats_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_version: str
    split: str
    data_synthetic: bool
    model_synthetic: bool
    scoring_policy: Literal["all_cache_endpoints"] = "all_cache_endpoints"

    @model_validator(mode="after")
    def validate_source(self):
        vocabulary_for(self.vocabulary_version)
        if self.split not in {"val", "test"}:
            raise ValueError("score archives are validation/test only")
        return self

    @property
    def synthetic(self):
        return self.data_synthetic or self.model_synthetic


def score_dtype(classes):
    return np.dtype(
        [
            ("label", "<i2"),
            ("class_valid", "u1"),
            ("ready", "u1"),
            ("available", "u1"),
            ("logits", "<f8", (classes,)),
        ]
    )


def validate_batch(rows, classes):
    if rows.dtype != score_dtype(classes) or rows.ndim != 1:
        raise ValueError("score layout mismatch")
    for field in ("class_valid", "ready", "available"):
        if np.any(rows[field] > 1):
            raise ValueError("score masks must be zero or one")
    labels, mask = rows["label"], rows["class_valid"].astype(bool)
    if np.any((mask & ((labels < 0) | (labels >= classes))) | (~mask & (labels != -1))):
        raise ValueError("score labels/masks mismatch")
    if not np.isfinite(rows["logits"]).all() or np.any(np.abs(rows["logits"]) > 1e6):
        raise ValueError("logits must be finite and within the numeric budget")
    if np.any(rows["logits"][rows["available"] == 0] != 0):
        raise ValueError("unavailable outputs cannot retain scores")


@dataclass
class Scores:
    rows: np.ndarray
    provenance: ScoreProvenance
    manifest_sha256: str | None = None
    guard: object = None

    def __post_init__(self):
        if (
            self.rows.dtype != score_dtype(self.classes)
            or self.rows.ndim != 1
            or len(self.rows) > MAX_ROWS
        ):
            raise ValueError("score layout/row budget exceeded")
        for batch in self.batches():
            validate_batch(batch, self.classes)

    @property
    def classes(self):
        return vocabulary_for(self.provenance.vocabulary_version).class_count

    def batches(self):
        if self.guard:
            self.guard()
        for start in range(0, len(self.rows), BATCH):
            if self.guard:
                self.guard()
            yield self.rows[start : start + BATCH]
        if self.guard:
            self.guard()


def write_scores(output, provenance, batches):
    from algorithm.bundle import digest, json_bytes

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    classes = vocabulary_for(provenance.vocabulary_version).class_count
    count = 0
    with (root / "scores.bin").open("xb") as stream:
        for batch in batches:
            validate_batch(batch, classes)
            count += len(batch)
            if len(batch) > BATCH or count > MAX_ROWS:
                raise ValueError("score batch/row budget exceeded")
            stream.write(batch.tobytes())
    manifest = {
        "format_version": "evaluation-scores-v1",
        "complete": True,
        "rows": count,
        "provenance": provenance.model_dump(mode="json"),
        "scope": "fixed_cache_represented_endpoints_not_entire_source_timeline",
        "score_sha256": file_sha256(root / "scores.bin"),
        "score_bytes": (root / "scores.bin").stat().st_size,
    }
    data = json_bytes(manifest)
    (root / "manifest.json").write_bytes(
        data
    )  # publish last; partial archives cannot be loaded
    return {
        "status": "scores_written",
        "manifest_sha256": digest(data),
        "rows": count,
        "synthetic": provenance.synthetic,
    }


def load_scores(directory, *, expected_manifest_sha256):
    from algorithm.bundle import digest, read_bounded

    root = Path(directory)
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_manifest_sha256
    ):
        raise ValueError("trusted score manifest SHA256 required")
    if root.is_symlink() or {p.name for p in root.iterdir()} != {
        "manifest.json",
        "scores.bin",
    }:
        raise ValueError("score archive file inventory mismatch")
    data = read_bounded(root / "manifest.json", 1_048_576)
    if digest(data) != expected_manifest_sha256:
        raise ValueError("score manifest hash mismatch")
    manifest = strict_json(data)
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "format_version",
            "complete",
            "rows",
            "provenance",
            "scope",
            "score_sha256",
            "score_bytes",
        }
        or manifest["format_version"] != "evaluation-scores-v1"
        or manifest["complete"] is not True
        or manifest["scope"]
        != "fixed_cache_represented_endpoints_not_entire_source_timeline"
        or type(manifest["rows"]) is not int
        or not 0 <= manifest["rows"] <= MAX_ROWS
        or type(manifest["score_bytes"]) is not int
        or not isinstance(manifest["score_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest["score_sha256"])
    ):
        raise ValueError("invalid/incomplete score manifest")
    provenance = ScoreProvenance.model_validate(manifest["provenance"])
    dtype = score_dtype(vocabulary_for(provenance.vocabulary_version).class_count)
    path = root / "scores.bin"
    info = path.stat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_size != manifest["rows"] * dtype.itemsize
        or info.st_size != manifest["score_bytes"]
    ):
        raise ValueError("score binary shape mismatch")
    stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if file_sha256(path) != manifest["score_sha256"]:
        raise ValueError("score binary hash mismatch")

    def guard():
        info = path.stat()
        if (
            path.is_symlink()
            or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != stamp
        ):
            raise ValueError("immutable scores changed during evaluation")

    guard()
    rows = (
        np.memmap(path, dtype=dtype, mode="r", shape=(manifest["rows"],))
        if manifest["rows"]
        else np.empty(0, dtype=dtype)
    )
    return Scores(rows, provenance, expected_manifest_sha256, guard)


def endpoint_digest(index):
    if index["format_version"] == "t11-cache-v3":
        return canonical_digest(
            {
                "endpoints": index["files"]["endpoints.bin"]["sha256"],
                "groups": index["files"]["groups.json"]["sha256"],
            }
        )
    return canonical_digest(index["source_ids"])
