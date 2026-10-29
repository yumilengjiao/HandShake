"""Actual trainer on class-coded synthetic geometry; no real gesture conclusions."""
from hashlib import sha256
import json
from pathlib import Path
import warnings

from algorithm.data.cache import write_cache
from algorithm.data.dataset import ObservationRecord, SplitManifest, WindowDataset, fit_training_normalizer, records_digest
from algorithm.training.config import TrainConfig
from algorithm.training.trainer import train
from shared.contracts import FrameStatus, HandObservation, TransformMetadata
from shared.features import FeatureStatisticsWarning


def _coded_points(cx):
    points = [(cx, .65, 0.)]
    for finger in range(5):
        for joint in range(4):
            points.append((cx + (finger - 2) * .035, .65 - (joint + 1) * .035 - .015 * finger, -.002 * joint))
    return points


def make_health_caches(root):
    records = []
    for subject, split in (("synthetic-A", "train"), ("synthetic-B", "val")):
        for label in range(5):
            recording = f"{subject}-class-{label}"
            for frame in range(8):
                t = frame * 33_333_333
                obs = HandObservation(stream_id=recording, track_id="hand", source_frame_id=frame,
                                      source_timestamp_ns=t, landmarks_image=_coded_points(.2 + label * .15),
                                      validity_mask=[True] * 21, observation_valid=True, handedness="unknown", quality=1.,
                                      transform_metadata=TransformMetadata(timebase="synthetic"), timebase="synthetic")
                status = FrameStatus(stream_id=recording, source_frame_id=frame, source_timestamp_ns=t,
                                     processed_at_ns=t, perception_status="processed_hands", timebase="synthetic")
                records.append(ObservationRecord(recording_id=recording, subject_id=subject, session_id="health",
                                                 lineage_id=recording, original_sha256=sha256(recording.encode()).hexdigest(),
                                                 visible_subjects=[subject], track_id="hand", context_epoch=0,
                                                 image_size=[640, 480], frame_status=status, observation=obs,
                                                 class_id=label, class_valid=True, timebase="synthetic"))
    manifest = SplitManifest(dataset_version="synthetic-class-code-health-v1", dataset_sha256=records_digest(records),
                             status="draft", synthetic=True, preregistration_reference="synthetic-train-val-only",
                             subject_splits={"synthetic-A": "train", "synthetic-B": "val"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FeatureStatisticsWarning)
        normalizer = fit_training_normalizer(records, manifest)
    for split in ("train", "val"):
        write_cache(WindowDataset(records, manifest, split=split, normalizer=normalizer), root / split)
    (root / "split.json").write_text(manifest.model_dump_json(indent=2) + "\n")


def run_health(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    make_health_caches(root)
    report = dict(scope="synthetic class-coded geometry, train/development only, no real gesture or generalization result",
                  synthetic=True, final_test_used=False, models={})
    for model in ("mlp", "tcn"):
        results = {}
        for shuffled in (False, True):
            config = TrainConfig(model_type=model, learning_rate=.01, batch_size=40, max_epochs=60,
                                 patience=60, min_delta=0., shuffle_labels_diagnostic=shuffled)
            run = train(root / "train", root / "val", root / f"{model}-{'shuffled' if shuffled else 'coded'}", config=config)
            results["shuffled_labels" if shuffled else "coded_labels"] = dict(val_macro_f1=run["best_val_macro_f1"],
                                                                                  best_epoch=run["best_epoch"], run_id=run["run_id"])
        report["models"][model] = results
    (root / "health.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new artifact directory")
    args = parser.parse_args(argv)
    print(json.dumps(run_health(args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
