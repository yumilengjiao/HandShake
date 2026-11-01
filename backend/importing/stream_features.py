"""Verified import -> disk-spooled audit -> train statistics -> compact cache v3."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from algorithm.data.compact import ENDPOINT_DTYPE, FORMAT, MEMBERS, STEP_DTYPE, STORAGE
from algorithm.data.dataset import _extract, canonical_digest
from algorithm.data.source_archive import file_sha256
from backend.importing.ipn_hands import _gzip_writer, _line
from backend.importing.spool import ImportSpool
from shared import FEATURE_SCHEMA_VERSION
from shared.classes import IPN_HANDS_VERSION, vocabulary_for
from shared.features import (
    FeatureConfig,
    FeatureExtractor,
    FeatureNormalizer,
    FeatureWindowBuilder,
    feature_schema_hash,
)


def _json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _training_samples(spool, config):
    for key, _ in spool.groups("train"):
        extractor = FeatureExtractor(config)
        for record in spool.group_records(key):
            sample = _extract(record, extractor)
            if sample is not None:
                yield sample


def build_stream_cache(
    directory,
    output,
    *,
    split,
    dataset_version,
    review_reference,
    stats=None,
    max_records=2_000_000,
    max_groups=50_000,
    stride=1,
):
    target = Path(output)
    if target.exists():
        raise FileExistsError("cache output already exists")
    if split not in {"train", "val", "test"} or type(stride) is not int or stride < 1:
        raise ValueError("invalid split/stride")
    if (
        not isinstance(dataset_version, str)
        or not dataset_version.strip()
        or not isinstance(review_reference, str)
        or not review_reference.strip()
    ):
        raise ValueError("dataset version and real review reference required")
    if split != "train" and stats is None:
        raise ValueError("validation/test requires --stats from train")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".handshake-spool-", dir=target.parent
    ) as temporary:
        spool = ImportSpool(
            directory,
            Path(temporary) / "audit.sqlite",
            max_records=max_records,
            max_groups=max_groups,
        )
        try:
            try:
                spool.audit()
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    "duplicate source frame or malformed import inventory"
                ) from error
            manifest = {
                "format_version": "t11-stream-split-v1",
                "status": "frozen",
                "dataset_version": dataset_version,
                "dataset_sha256": spool.dataset_hash,
                "synthetic": spool.manifest["synthetic"],
                "preregistration_reference": review_reference,
                "subject_splits": spool.manifest["subject_plan"]["subject_splits"],
                "vocabulary_version": IPN_HANDS_VERSION,
                "digest_kind": "stream-records-v1",
                "import_manifest_sha256": spool.manifest_hash,
                "record_stream_sha256": spool.payload_hash.hexdigest(),
            }
            split_hash = canonical_digest(manifest)
            vocabulary = vocabulary_for(IPN_HANDS_VERSION)
            if stats is None:
                config = FeatureConfig()
                normalizer = FeatureNormalizer.fit(
                    _training_samples(spool, config),
                    dataset_hash=spool.dataset_hash,
                    split_hash=split_hash,
                    config=config,
                    vocabulary=vocabulary,
                )
            else:
                normalizer = FeatureNormalizer.load(
                    stats,
                    expected_dataset_hash=spool.dataset_hash,
                    expected_split_hash=split_hash,
                )
                if normalizer.vocabulary != vocabulary:
                    raise ValueError("training normalizer vocabulary mismatch")
                config = normalizer.config
            target.mkdir(exist_ok=False)
            _json(
                target / "incomplete.json",
                {"format_version": FORMAT, "complete": False},
            )
            normalizer.save(target / "stats.json")
            _json(target / "split_manifest.json", manifest)
            _json(target / "classes.json", vocabulary.to_dict())
            rows = endpoints = selected_count = ready_count = unprocessed = 0
            groups = []
            with ExitStack() as stack:
                features_out = stack.enter_context((target / "features.f32").open("xb"))
                steps_out = stack.enter_context((target / "steps.bin").open("xb"))
                endpoints_out = stack.enter_context(
                    (target / "endpoints.bin").open("xb")
                )
                inventory = stack.enter_context(
                    _gzip_writer(target / "inventory.jsonl.gz")
                )
                for key, metadata in spool.groups(split):
                    group = metadata | {"row_start": rows, "endpoint_start": endpoints}
                    group_id = len(groups)
                    extractor, builder = (
                        FeatureExtractor(config),
                        FeatureWindowBuilder(config),
                    )
                    processed = 0
                    for record in spool.group_records(key):
                        status = record.frame_status
                        sample = _extract(record, extractor)
                        if status.perception_status.value == "not_processed":
                            unprocessed += 1
                            builder.unprocessed(
                                status.source_timestamp_ns,
                                stream_id=status.stream_id,
                                track_id=record.track_id,
                                context_epoch=record.context_epoch,
                            )
                            continue
                        selected = processed % stride == 0
                        processed += 1
                        window = builder.append(sample) if sample is not None else None
                        source = {
                            "recording_id": record.recording_id,
                            "subject_id": record.subject_id,
                            "session_id": record.session_id,
                            "stream_id": status.stream_id,
                            "track_id": record.track_id,
                            "context_epoch": record.context_epoch,
                            "source_frame_id": status.source_frame_id,
                            "timestamp_ns": status.source_timestamp_ns,
                            "timebase": record.timebase,
                        }
                        _line(
                            inventory,
                            source
                            | {
                                "selected": selected,
                                "represented": window is not None,
                                "ready": False if window is None else window.ready,
                                "class_id": record.class_id,
                                "class_valid": record.class_valid,
                            },
                        )
                        selected_count += selected
                        if window is None:
                            continue
                        values = np.asarray(normalizer.transform(sample), dtype="<f4")
                        if not np.isfinite(values).all():
                            raise ValueError("nonfinite float32 cached features")
                        features_out.write(values.tobytes())
                        steps_out.write(
                            np.array(
                                [
                                    (
                                        status.source_frame_id,
                                        status.source_timestamp_ns,
                                        record.context_epoch,
                                    )
                                ],
                                dtype=STEP_DTYPE,
                            ).tobytes()
                        )
                        if selected:
                            item = np.zeros(
                                1, dtype=ENDPOINT_DTYPE
                            )  # deterministic padding bytes
                            item[0] = (
                                rows,
                                len(window.features),
                                record.class_id,
                                record.class_valid,
                                window.ready,
                                status.source_frame_id,
                                status.source_timestamp_ns,
                                group_id,
                                record.context_epoch,
                            )
                            endpoints_out.write(item.tobytes())
                            endpoints += 1
                            ready_count += window.ready
                        rows += 1
                    groups.append(
                        group | {"row_stop": rows, "endpoint_stop": endpoints}
                    )
            _json(target / "groups.json", groups)
            index = {
                "format_version": FORMAT,
                "complete": True,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "schema_sha256": feature_schema_hash(),
                "vocabulary_version": IPN_HANDS_VERSION,
                "classes_sha256": vocabulary.digest(),
                "config_sha256": config.config_hash,
                "dataset_sha256": spool.dataset_hash,
                "split_sha256": split_hash,
                "stats_sha256": normalizer.stats_hash,
                "split": split,
                "stride": stride,
                "synthetic": spool.manifest["synthetic"],
                "rows": rows,
                "endpoints": endpoints,
                "coverage": {
                    "selected_endpoints": selected_count,
                    "represented": endpoints,
                    "ready": ready_count,
                    "ready_coverage": ready_count / selected_count
                    if selected_count
                    else 0.0,
                    "unprocessed_frames": unprocessed,
                },
                "files": {
                    name: {
                        "sha256": file_sha256(target / name),
                        "bytes": (target / name).stat().st_size,
                    }
                    for name in sorted(MEMBERS)
                },
                "storage": STORAGE,
            }
            index["index_sha256"] = canonical_digest(index)
            (target / "incomplete.json").unlink()
            _json(
                target / "index.json", index
            )  # publish last; a crash before this is not a cache
            return {
                "status": "cache_built",
                "scope": "compact_stream_cache_v3",
                "split": split,
                "records_audited": spool.records,
                "rows": rows,
                "endpoints": endpoints,
                "synthetic": index["synthetic"],
                "index_sha256": index["index_sha256"],
                "coverage": index["coverage"],
                "feature_bytes": index["files"]["features.f32"]["bytes"],
                "expanded_window_bytes_avoided": endpoints * 32 * 94 * 4,
            }
        except sqlite3.Error as error:
            raise ValueError("temporary cache spool failed") from error
        finally:
            spool.close()
