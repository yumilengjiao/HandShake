from hashlib import sha256
import json
from pathlib import Path
import warnings
import subprocess
import sys

import numpy as np
import pytest

from algorithm.data.augmentation import augment_training_records
from algorithm.data.cache import load_cache, write_cache
from algorithm.data.dataset import ObservationRecord, SplitManifest, WindowDataset, fit_training_normalizer, records_digest
from shared.contracts import FrameStatus
from shared.features import FeatureStatisticsWarning
from shared.features.schema import VX
from tests.unit.test_features_geometry import observation


def example_records(subjects=("A", "B", "C"), count=40):
    records = []
    for number, subject in enumerate(subjects):
        for index in range(count):
            obs = observation(index * 30_000_000, index, offset=(number + index * 0.001, 0, 0))
            status = FrameStatus(stream_id=obs.stream_id, source_frame_id=index,
                                 source_timestamp_ns=obs.source_timestamp_ns, processed_at_ns=obs.source_timestamp_ns,
                                 perception_status="processed_hands", timebase="synthetic")
            records.append(ObservationRecord(recording_id="rec-" + subject, subject_id=subject,
                                             session_id="session-1", lineage_id="original-" + subject,
                                             original_sha256=sha256(subject.encode()).hexdigest(), visible_subjects=[subject],
                                             track_id="t", context_epoch=0, image_size=[640, 480], frame_status=status,
                                             observation=obs, class_id=2, class_valid=True, timebase="synthetic"))
    return records


def manifest_for(records, **changes):
    return SplitManifest(**(dict(dataset_version="synthetic-only-v1", dataset_sha256=records_digest(records),
                                status="draft", synthetic=True, preregistration_reference="synthetic-preassignment",
                                subject_splits={"A": "train", "B": "val", "C": "test"}) | changes))


def normalizer_for(records, manifest):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FeatureStatisticsWarning)
        return fit_training_normalizer(records, manifest)


def test_fixed_subject_split_precedes_stats_and_no_windows_cross_recordings():
    records = example_records()
    manifest = manifest_for(records)
    normalizer = normalizer_for(records, manifest)
    for split, subject in (("train", "A"), ("val", "B"), ("test", "C")):
        data = WindowDataset(list(reversed(records)), manifest, split=split, normalizer=normalizer)
        assert len(data) == 40
        assert all(item["source"]["subject_id"] == subject for item in data)
        assert data[0]["valid_length"] == 1 and not data[0]["ready"]
        assert data[31]["ready"] and data[31]["x"].shape == (32, 94)
        assert data[0]["loss_masks"]["class_valid"]  # readiness is not a truth mask
        assert data.coverage()["ready"] == 9
    train_cx = [r.observation.landmarks_image[0][0] for r in records if r.subject_id == "A"]
    assert normalizer.stats.mean[63] < 0.4
    assert normalizer.stats.dataset_hash == manifest.dataset_sha256


def test_unprocessed_does_not_forge_no_hand_and_endpoint_labels_remain_fixed():
    records = example_records(("A",))
    data = records[10].model_dump(mode="json")
    data["observation"] = None
    data["frame_status"].update(perception_status="not_processed", processed_at_ns=None, drop_or_error_reason="overload")
    records[10] = ObservationRecord(**data)
    manifest = manifest_for(records)
    dataset = WindowDataset(records, manifest, split="train", normalizer=normalizer_for(records, manifest))
    assert len(dataset) == 39 and dataset.coverage()["unprocessed_frames"] == 1
    assert len(dataset.records) == 40
    assert 10 not in [item["source"]["source_frame_id"] for item in dataset]
    assert dataset[10]["source"]["source_frame_id"] == 11


@pytest.mark.parametrize("case", ["lineage", "original", "visible", "unknown_subject"])
def test_cross_split_leakage_is_rejected_before_fitting(case):
    records = example_records(count=2)
    changed = records[2].model_dump(mode="json")
    if case == "lineage":
        changed["lineage_id"] = records[0].lineage_id
    elif case == "original":
        changed["original_sha256"] = records[0].original_sha256
    elif case == "visible":
        changed["visible_subjects"] = ["A", "B"]
    else:
        changed["visible_subjects"] = ["B", "unknown"]
    records[2] = ObservationRecord(**changed)
    with pytest.raises(ValueError):
        fit_training_normalizer(records, manifest_for(records))


def test_real_inputs_require_upstream_freeze_and_wrong_hash_is_rejected():
    records = example_records(count=2)
    with pytest.raises(ValueError, match="frozen"):
        manifest_for(records, synthetic=False)
    manifest = manifest_for(records, dataset_sha256="f" * 64)
    with pytest.raises(ValueError, match="hash"):
        fit_training_normalizer(records, manifest)


def test_cache_is_rebuildable_hash_bound_and_never_overwrites(tmp_path):
    records = example_records(("A",))
    manifest = manifest_for(records)
    normalizer = normalizer_for(records, manifest)
    dataset = WindowDataset(records, manifest, split="train", normalizer=normalizer)
    path = tmp_path / "cache"
    index = write_cache(dataset, path)
    kwargs = dict(expected_dataset_sha256=manifest.dataset_sha256, expected_split_sha256=manifest.digest(),
                  expected_stats_sha256=normalizer.stats_hash, expected_config_sha256=dataset.config.config_hash,
                  expected_split="train")
    arrays, reloaded = load_cache(path, **kwargs)
    assert arrays["x"].shape == (40, 32, 94) and reloaded == index
    with pytest.raises(FileExistsError):
        write_cache(dataset, path)
    with pytest.raises(ValueError, match="version"):
        load_cache(path, **(kwargs | {"expected_dataset_sha256": "0" * 64}))
    (path / "features.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content hash"):
        load_cache(path, **kwargs)


def test_raw_flip_changes_direction_then_shared_extractor_recomputes_velocity():
    records = example_records(("A",))
    flipped = augment_training_records(records, source_split="train", seed=42, horizontal_flip=True)
    assert all(row.class_id == 3 for row in flipped)
    assert all(row.source_split == "train" and row.parent_records_sha256 == records_digest(records) for row in flipped)
    assert records[0].class_id == 2
    first = records[0].observation.landmarks_image
    other = flipped[0].observation.landmarks_image
    np.testing.assert_allclose(np.array(other)[:, 0], 1 - np.array(first)[:, 0])
    np.testing.assert_array_equal(np.array(other)[:, 2], np.array(first)[:, 2])
    from shared.features import FeatureExtractor
    a, b = FeatureExtractor(), FeatureExtractor()
    for raw, flip in zip(records, flipped):
        left = a.extract(raw.observation, raw.image_size)
        right = b.extract(flip.observation, flip.image_size)
    assert left.values[VX] == pytest.approx(-right.values[VX])
    with pytest.raises(ValueError, match="training-only"):
        augment_training_records(records, source_split="test", seed=42)


def test_feature_builder_cli_and_train_stats_reuse(tmp_path):
    records = example_records(count=35)
    manifest = manifest_for(records)
    source, split = tmp_path / "observations.json", tmp_path / "split.json"
    source.write_text(json.dumps([record.model_dump(mode="json") for record in records]))
    split.write_text(manifest.model_dump_json())
    script = Path(__file__).resolve().parents[2] / "scripts/build_features.py"
    command = [sys.executable, str(script), str(source), str(split)]
    train = subprocess.run(command + [str(tmp_path / "train"), "--split", "train"], capture_output=True, text=True)
    assert train.returncode == 0, train.stderr
    invalid = subprocess.run(command + [str(tmp_path / "invalid"), "--split", "test"], capture_output=True, text=True)
    assert invalid.returncode == 2 and "--stats" in invalid.stderr
    valid = subprocess.run(command + [str(tmp_path / "val"), "--split", "val", "--stats", str(tmp_path / "train/stats.json")], capture_output=True, text=True)
    assert valid.returncode == 0, valid.stderr
    assert json.loads(valid.stdout)["coverage"]["selected_endpoints"] == 35


def test_dataset_and_online_use_identical_causal_features():
    from shared.features import FeatureExtractor, FeatureWindowBuilder
    records = example_records(("A",))
    manifest = manifest_for(records)
    normalizer = normalizer_for(records, manifest)
    dataset = WindowDataset(records, manifest, split="train", normalizer=normalizer)
    extractor, builder = FeatureExtractor(), FeatureWindowBuilder(normalizer=normalizer)
    for index, record in enumerate(records):
        window = builder.append(extractor.extract(record.observation, record.image_size))
        np.testing.assert_allclose(dataset[index]["x"][-len(window.features):], window.features, rtol=1e-6, atol=1e-7)
        assert dataset[index]["source"]["timestamp_ns"] == window.timestamps_ns[-1]
        assert dataset[index]["ready"] == window.ready


def test_test_geometry_cannot_change_training_statistics():
    records = example_records(count=4)
    baseline = normalizer_for(records, manifest_for(records))
    changed = []
    for record in records:
        data = record.model_dump(mode="json")
        if record.subject_id == "C":
            for point in data["observation"]["landmarks_image"]:
                point[0] += 100
        changed.append(ObservationRecord(**data))
    other = normalizer_for(changed, manifest_for(changed))
    np.testing.assert_array_equal(baseline.stats.mean, other.stats.mean)
    np.testing.assert_array_equal(baseline.stats.std, other.stats.std)
    np.testing.assert_array_equal(baseline.stats.count, other.stats.count)
    assert baseline.stats.dataset_hash != other.stats.dataset_hash


def test_initial_no_hand_never_invents_a_background_track():
    records = example_records(("A",))
    data = records[0].model_dump(mode="json")
    data["observation"] = None
    data["frame_status"]["perception_status"] = "processed_no_hand"
    records[0] = ObservationRecord(**data)
    manifest = manifest_for(records)
    dataset = WindowDataset(records, manifest, split="train", normalizer=normalizer_for(records, manifest))
    assert dataset.coverage()["selected_endpoints"] == 40
    assert len(dataset) == 39
    assert not dataset.endpoint_inventory[0]["represented"]
    assert dataset[0]["source"]["source_frame_id"] == 1
    assert len(dataset.records) == 40


def test_augmentation_cannot_move_an_assigned_test_subject_into_training():
    records = example_records(("C",), count=3)
    augmented = augment_training_records(records, source_split="train", seed=42, horizontal_flip=True)
    with pytest.raises(ValueError, match="source_split"):
        fit_training_normalizer(augmented, manifest_for(augmented))
