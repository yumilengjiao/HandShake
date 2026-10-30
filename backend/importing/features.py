"""Explicitly reviewed small-import bridge into the existing T11/T14 pipeline.

This adapter intentionally refuses a whole large corpus in memory. Sharded/
streaming full-corpus training is a later implementation, not an implicit OOM.
"""

from pathlib import Path

from algorithm.data.cache import write_cache
from algorithm.data.dataset import (
    SplitManifest,
    WindowDataset,
    fit_training_normalizer,
    records_digest,
)
from algorithm.data.source_archive import strict_json
from backend.importing.ipn_hands import iter_imported_records
from shared.classes import IPN_HANDS_VERSION
from shared.features import FeatureNormalizer


def build_import_cache(
    directory,
    output,
    *,
    split,
    dataset_version,
    review_reference,
    stats=None,
    max_records=10000,
):
    if type(max_records) is not int or max_records <= 0:
        raise ValueError("max_records must be a positive integer")
    if split not in {"train", "val", "test"}:
        raise ValueError("invalid split")
    if not isinstance(review_reference, str) or not review_reference.strip():
        raise ValueError(
            "a real quality/split review reference is required; no automatic freeze"
        )
    if Path(output).exists():
        raise FileExistsError("cache output already exists")
    if split != "train" and stats is None:
        raise ValueError("validation/test requires --stats from the training split")
    imported = strict_json(
        (Path(directory) / "manifest.json").read_text(encoding="utf-8")
    )
    records = []
    for record in iter_imported_records(directory):
        if len(records) >= max_records:
            raise ValueError(
                "small-import record budget exceeded; full-corpus streaming is not implemented"
            )
        records.append(record)
    manifest = SplitManifest(
        dataset_version=dataset_version,
        dataset_sha256=records_digest(records),
        status="frozen",
        synthetic=imported["synthetic"],
        preregistration_reference=review_reference,
        subject_splits=imported["subject_plan"]["subject_splits"],
        vocabulary_version=IPN_HANDS_VERSION,
    )
    manifest.validate_records(records)
    if stats is not None:
        normalizer = FeatureNormalizer.load(
            stats,
            expected_dataset_hash=manifest.dataset_sha256,
            expected_split_hash=manifest.digest(),
        )
    else:
        normalizer = fit_training_normalizer(records, manifest)
    dataset = WindowDataset(records, manifest, split=split, normalizer=normalizer)
    index = write_cache(dataset, output)
    (Path(output) / "split_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return {
        "status": "cache_built",
        "split": split,
        "records": len(records),
        "synthetic": manifest.synthetic,
        "vocabulary_version": IPN_HANDS_VERSION,
        "review_reference": review_reference,
        "index_sha256": index["index_sha256"],
        "coverage": index["coverage"],
        "scope": "bounded_small_import_not_full_corpus",
    }
