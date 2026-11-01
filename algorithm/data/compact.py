"""T11 cache v3: one feature row per step; construct only requested batch windows.

Files are immutable local artifacts. Full hashes are verified once on open, and
size/inode/mtime changes are rejected before batch access. This is not a sandbox
against a malicious process that rewrites memory-mapped files concurrently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithm.data.dataset import canonical_digest
from algorithm.data.source_archive import file_sha256, strict_json
from shared import FEATURE_SCHEMA_VERSION
from shared.classes import load_classes, vocabulary_for
from shared.features import FeatureNormalizer, feature_schema_hash

FORMAT = "t11-cache-v3"
MAX_ROWS = 10_000_000
MAX_BATCH = 1024
STEP_DTYPE = np.dtype(
    [("frame_id", "<i8"), ("timestamp_ns", "<i8"), ("context_epoch", "<i8")]
)
ENDPOINT_DTYPE = np.dtype(
    {
        "names": [
            "row",
            "valid_length",
            "y",
            "class_valid",
            "ready",
            "frame_id",
            "timestamp_ns",
            "group",
            "context_epoch",
        ],
        "formats": ["<i8", "u1", "<i2", "?", "?", "<i8", "<i8", "<i4", "<i8"],
        "offsets": [0, 8, 10, 12, 13, 16, 24, 32, 40],
        "itemsize": 48,
    }
)
STORAGE = "little-endian float32[94] steps; fixed 48-byte endpoint-v1; no expanded window tensor"
MEMBERS = {
    "features.f32",
    "steps.bin",
    "endpoints.bin",
    "groups.json",
    "stats.json",
    "split_manifest.json",
    "classes.json",
    "inventory.jsonl.gz",
}


def _json(path):
    with path.open("rb") as stream:
        raw = stream.read(32 * 1024**2 + 1)
    if len(raw) > 32 * 1024**2:
        raise ValueError("cache metadata budget exceeded")
    return strict_json(raw)


def _stamp(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _mapped(path, dtype, shape):
    if not np.prod(shape):
        result = np.empty(shape, dtype=dtype)
        result.setflags(write=False)
        return result
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


class CompactWindows:
    dtype = np.dtype("float32")
    ndim = 3

    def __init__(self, features, endpoints, verify):
        self.features, self.endpoints, self.verify = features, endpoints, verify
        self.shape = (len(endpoints), 32, 94)

    def __len__(self):
        return len(self.endpoints)

    def __array__(self, *args, **kwargs):
        raise ValueError(
            "compact windows require bounded batch indexing, not full materialization"
        )

    def __getitem__(self, selected):
        self.verify()
        scalar = isinstance(selected, (int, np.integer)) and not isinstance(
            selected, bool
        )
        if scalar:
            indexes = np.array([selected], dtype=np.int64)
        elif isinstance(selected, slice):
            sequence = range(*selected.indices(len(self)))
            if len(sequence) > MAX_BATCH:
                raise ValueError("window batch budget exceeded")
            indexes = np.asarray(sequence, dtype=np.int64)
        else:
            if len(selected) > MAX_BATCH:
                raise ValueError("window batch budget exceeded")
            indexes = np.asarray(selected)
            if indexes.ndim != 1 or (indexes.size and indexes.dtype.kind not in "iu"):
                raise ValueError("window indexes must be an integer vector")
            indexes = indexes.astype(np.int64)
        if len(indexes) > MAX_BATCH:
            raise ValueError("window batch budget exceeded")
        indexes = np.where(indexes < 0, indexes + len(self), indexes)
        if np.any((indexes < 0) | (indexes >= len(self))):
            raise IndexError("window index out of range")
        if not len(indexes):
            return np.empty((0, 32, 94), dtype=np.float32)
        endpoints = self.endpoints[indexes]
        offsets = np.arange(32)[None, :]
        valid = offsets >= (32 - endpoints["valid_length"][:, None])
        rows = endpoints["row"][:, None] - 31 + offsets
        result = np.array(
            self.features[np.where(valid, rows, 0)], dtype=np.float32, copy=True
        )
        result[~valid] = 0
        return result[0] if scalar else result


def load_compact_cache(
    directory,
    *,
    expected_dataset_sha256,
    expected_split_sha256,
    expected_stats_sha256,
    expected_config_sha256,
    expected_split,
    expected_vocabulary=None,
):
    root = Path(directory)
    if root.is_symlink() or (root / "incomplete.json").exists():
        raise ValueError("incomplete or linked cache")
    index = _json(root / "index.json")
    if index.get("index_sha256") != canonical_digest(
        {k: v for k, v in index.items() if k != "index_sha256"}
    ):
        raise ValueError("cache index hash mismatch")
    vocabulary = expected_vocabulary or vocabulary_for(index["vocabulary_version"])
    expected = {
        "format_version": FORMAT,
        "complete": True,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_sha256": feature_schema_hash(),
        "vocabulary_version": vocabulary.vocabulary_version,
        "classes_sha256": vocabulary.digest(),
        "dataset_sha256": expected_dataset_sha256,
        "split_sha256": expected_split_sha256,
        "stats_sha256": expected_stats_sha256,
        "config_sha256": expected_config_sha256,
        "split": expected_split,
    }
    if any(index.get(k) != v for k, v in expected.items()):
        raise ValueError("compact cache provenance/version mismatch")
    for field in ("rows", "endpoints"):
        if type(index[field]) is not int or not 0 <= index[field] <= MAX_ROWS:
            raise ValueError("cache row budget exceeded")
    if index.get("complete") is not True or index.get("storage") != STORAGE:
        raise ValueError("unsupported or incomplete compact storage")
    if (
        type(index["synthetic"]) is not bool
        or type(index["stride"]) is not int
        or index["stride"] < 1
    ):
        raise ValueError("invalid cache policy")
    if set(index["files"]) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {
        "index.json"
    }:
        raise ValueError("cache file inventory mismatch")
    stamps = {}
    for name in sorted(MEMBERS | {"index.json"}):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("cache files must be regular, not links")
        before = _stamp(path)
        if name != "index.json":
            entry = index["files"][name]
            if before[2] != entry["bytes"] or file_sha256(path) != entry["sha256"]:
                raise ValueError("cache member hash/size mismatch: " + name)
        if _stamp(path) != before:
            raise ValueError("cache changed during verification")
        stamps[path] = before

    def verify():
        if any(_stamp(path) != stamp for path, stamp in stamps.items()):
            raise ValueError("immutable cache changed during use")

    if load_classes(root / "classes.json") != vocabulary:
        raise ValueError("cache vocabulary mismatch")
    stats = FeatureNormalizer.load(
        root / "stats.json",
        expected_dataset_hash=expected_dataset_sha256,
        expected_split_hash=expected_split_sha256,
    )
    if (
        stats.stats_hash != expected_stats_sha256
        or stats.config.config_hash != expected_config_sha256
        or stats.vocabulary != vocabulary
    ):
        raise ValueError("cache normalization mismatch")
    split = _json(root / "split_manifest.json")
    if (
        split.get("format_version") != "t11-stream-split-v1"
        or canonical_digest(split) != expected_split_sha256
        or split["dataset_sha256"] != expected_dataset_sha256
        or split["synthetic"] is not index["synthetic"]
        or split["vocabulary_version"] != vocabulary.vocabulary_version
        or split["status"] != "frozen"
        or not split["preregistration_reference"]
    ):
        raise ValueError("stream split manifest mismatch")
    if (
        split.get("digest_kind") != "stream-records-v1"
        or not isinstance(split["preregistration_reference"], str)
        or not split["preregistration_reference"].strip()
        or canonical_digest(
            {
                "format_version": "stream-records-v1",
                "import_manifest_sha256": split["import_manifest_sha256"],
                "record_stream_sha256": split["record_stream_sha256"],
            }
        )
        != expected_dataset_sha256
    ):
        raise ValueError("stream digest/review contract mismatch")
    expected_sizes = {
        "features.f32": index["rows"] * 94 * 4,
        "steps.bin": index["rows"] * STEP_DTYPE.itemsize,
        "endpoints.bin": index["endpoints"] * ENDPOINT_DTYPE.itemsize,
    }
    if any(stamps[root / name][2] != size for name, size in expected_sizes.items()):
        raise ValueError("binary cache shape/size mismatch")
    groups = _json(root / "groups.json")
    if not isinstance(groups, list) or len(groups) > 100_000:
        raise ValueError("group budget exceeded")
    features = _mapped(root / "features.f32", "<f4", (index["rows"], 94))
    steps = _mapped(root / "steps.bin", STEP_DTYPE, (index["rows"],))
    endpoints = _mapped(root / "endpoints.bin", ENDPOINT_DTYPE, (index["endpoints"],))
    row_cursor = endpoint_cursor = ready_count = 0
    identities = set()
    for number, group in enumerate(groups):
        identity = tuple(
            group[k]
            for k in (
                "recording_id",
                "subject_id",
                "session_id",
                "stream_id",
                "track_id",
            )
        )
        if (
            identity in identities
            or split["subject_splits"].get(group["subject_id"]) != expected_split
        ):
            raise ValueError("cache group identity/split mismatch")
        identities.add(identity)
        if (
            group["row_start"] != row_cursor
            or group["endpoint_start"] != endpoint_cursor
        ):
            raise ValueError("group ranges must be contiguous")
        row_stop, endpoint_stop = group["row_stop"], group["endpoint_stop"]
        if not row_cursor <= row_stop <= len(
            features
        ) or not endpoint_cursor <= endpoint_stop <= len(endpoints):
            raise ValueError("group range out of bounds")
        prior_time = prior_frame = prior_endpoint = -1
        for start in range(row_cursor, row_stop, 4096):
            stop = min(start + 4096, row_stop)
            block = steps[start:stop]
            if (
                not np.isfinite(features[start:stop]).all()
                or np.any(block["context_epoch"] < 0)
                or np.any(np.diff(block["timestamp_ns"]) <= 0)
                or np.any(np.diff(block["frame_id"]) <= 0)
                or block["timestamp_ns"][0] <= prior_time
                or block["frame_id"][0] <= prior_frame
            ):
                raise ValueError("invalid feature data or source ordering")
            prior_time, prior_frame = (
                int(block["timestamp_ns"][-1]),
                int(block["frame_id"][-1]),
            )
        for start in range(endpoint_cursor, endpoint_stop, MAX_BATCH):
            block = endpoints[start : min(start + MAX_BATCH, endpoint_stop)]
            rows, lengths = block["row"], block["valid_length"].astype(np.int64)
            if (
                np.any(block["group"] != number)
                or np.any((lengths < 1) | (lengths > 32))
                or np.any(
                    (rows < row_cursor)
                    | (rows >= row_stop)
                    | (rows - lengths + 1 < row_cursor)
                )
                or np.any(np.diff(rows) <= 0)
                or rows[0] <= prior_endpoint
            ):
                raise ValueError("endpoint windows cross groups or duplicate rows")
            source = steps[rows]
            if any(
                not np.array_equal(source[key], block[key])
                for key in ("frame_id", "timestamp_ns", "context_epoch")
            ):
                raise ValueError("endpoint/source identity mismatch")
            mask = block["class_valid"]
            if np.any(
                (mask & ((block["y"] < 0) | (block["y"] >= vocabulary.class_count)))
                | (~mask & (block["y"] != -1))
            ):
                raise ValueError("endpoint label/mask mismatch")
            if np.any(block["ready"] & (lengths != 32)):
                raise ValueError("ready endpoint must have full history")
            offsets = np.arange(32)[None, :]
            valid = offsets >= (32 - lengths[:, None])
            history = steps[
                np.where(valid, rows[:, None] - 31 + offsets, rows[:, None])
            ]
            if np.any(
                valid & (history["context_epoch"] != block["context_epoch"][:, None])
            ):
                raise ValueError("window crosses epoch")
            prior_endpoint = int(rows[-1])
            ready_count += int(block["ready"].sum())
        row_cursor, endpoint_cursor = row_stop, endpoint_stop
    if row_cursor != index["rows"] or endpoint_cursor != index["endpoints"]:
        raise ValueError("group inventory does not cover binary data")
    coverage = index["coverage"]
    for key in ("selected_endpoints", "represented", "ready", "unprocessed_frames"):
        if type(coverage[key]) is not int or not 0 <= coverage[key] <= MAX_ROWS:
            raise ValueError("invalid coverage counts")
    denominator = coverage["selected_endpoints"]
    if (
        coverage["represented"] != len(endpoints)
        or denominator < len(endpoints)
        or coverage["ready"] != ready_count
        or coverage["ready_coverage"]
        != (ready_count / denominator if denominator else 0.0)
    ):
        raise ValueError("cache coverage mismatch")
    verify()
    arrays = {
        name: endpoints[name] for name in ("y", "class_valid", "ready", "valid_length")
    }
    arrays["x"] = CompactWindows(features, endpoints, verify)
    arrays["_verify"] = verify
    index["groups"] = groups  # bounded group metadata, never a per-endpoint dict list
    return arrays, index


def inspect_compact_cache(directory, *, expected_index_sha256):
    """Read-only transfer preflight; the expected index hash comes from a trusted receipt."""
    root = Path(directory)
    header = _json(root / "index.json")
    if (
        header.get("format_version") != FORMAT
        or header.get("index_sha256") != expected_index_sha256
    ):
        raise ValueError("expected compact index version/hash mismatch")
    values, index = load_compact_cache(
        root,
        expected_dataset_sha256=header["dataset_sha256"],
        expected_split_sha256=header["split_sha256"],
        expected_stats_sha256=header["stats_sha256"],
        expected_config_sha256=header["config_sha256"],
        expected_split=header["split"],
    )
    return {
        "status": "cache_verified",
        "scope": "cache_integrity_only_not_source_or_review_approval",
        "format_version": FORMAT,
        "index_sha256": index["index_sha256"],
        "split": index["split"],
        "synthetic": index["synthetic"],
        "rows": index["rows"],
        "endpoints": index["endpoints"],
        "groups": len(index["groups"]),
        "has_supervision": bool(values["class_valid"].any()),
        "max_window_batch": MAX_BATCH,
        "feature_bytes": index["files"]["features.f32"]["bytes"],
        "logical_expanded_window_bytes": index["endpoints"] * 32 * 94 * 4,
    }
