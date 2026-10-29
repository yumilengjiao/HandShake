"""Thin T11 draft JSON observation adapter → shared features → hash-bound NPZ."""

import argparse
import json
from pathlib import Path
import sys

from algorithm.data.cache import write_cache
from algorithm.data.dataset import ObservationRecord, SplitManifest, WindowDataset, fit_training_normalizer
from shared.features import FeatureConfig, FeatureNormalizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="JSON array of draft ObservationRecord objects")
    parser.add_argument("manifest", type=Path, help="preassigned SplitManifest, never randomizes participants")
    parser.add_argument("output", type=Path, help="new cache directory; never overwrites")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--stats", type=Path, help="train-only statistics; required for val/test")
    args = parser.parse_args(argv)
    try:
        records = [ObservationRecord.model_validate(row) for row in json.loads(args.records.read_text(encoding="utf-8"))]
        manifest = SplitManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
        config = FeatureConfig()
        if args.stats:
            normalizer = FeatureNormalizer.load(args.stats, expected_dataset_hash=manifest.dataset_sha256,
                                                 expected_split_hash=manifest.digest(), config=config)
        elif args.split == "train":
            normalizer = fit_training_normalizer(records, manifest, config)
        else:
            raise ValueError("validation/test requires --stats from the training split; fitting is forbidden")
        dataset = WindowDataset(records, manifest, split=args.split, normalizer=normalizer, config=config)
        index = write_cache(dataset, args.output)
        print(json.dumps(dict(output=str(args.output), synthetic=manifest.synthetic,
                              coverage=index["coverage"], index_sha256=index["index_sha256"])))
    except (OSError, ValueError, TypeError) as error:
        print(f"feature build failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
