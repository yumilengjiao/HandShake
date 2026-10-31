"""Synthetic public-format fixtures only; no downloaded datasets or device access."""

import gzip
import json
import warnings
import zipfile

import pytest

from algorithm.data.cache import load_cache, write_cache
from algorithm.data.dataset import (
    SplitManifest,
    WindowDataset,
    fit_training_normalizer,
    records_digest,
)
from algorithm.data.ipn_hands import audit_annotations
from algorithm.data.source_archive import VerifiedArchive, file_sha256
from backend.importing.ipn_hands import import_ipn_hands, iter_imported_records
from backend.perception.synthetic import skeleton
from shared.classes import IPN_HANDS_VERSION, vocabulary_for
from shared.features import FeatureStatisticsWarning


def write_zip(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return file_sha256(path)


@pytest.fixture
def public_sources(tmp_path):
    vocabulary = vocabulary_for(IPN_HANDS_VERSION)
    videos = ["fixture_train", "fixture_val", "fixture_test"]
    annotations = {
        "classes.txt": "id,code,gesture\n"
        + "".join(
            f"{i},{token},synthetic description\n"
            for i, token in enumerate(vocabulary.tokens)
        ),
        "splits/train_list.txt": "\n".join(videos[:2]) + "\n",
        "splits/test_list.txt": videos[2] + "\n",
        "frame_counts.csv": "video,set,frames_ipnhands,frames_ipnhand_original\n"
        + "".join(
            f"{name},{'test' if i == 2 else 'train'},45,44\n"
            for i, name in enumerate(videos)
        ),
    }
    landmarks = {}
    for ordinal, name in enumerate(videos):
        annotations[f"temporal_annotations/{name}.txt"] = "".join(
            f"{i},{i * 3 + 1},{i * 3 + 3}\n" for i in range(15)
        )
        frames = {}
        for index in range(1, 46):
            points = skeleton(0.4 + ((index - 1) // 3) * 0.004 + ordinal * 0.0001)
            frames[f"{name}_{index:06d}.jpg"] = {
                "hand_landmarks": [
                    {
                        "handedness": "Right",
                        "landmarks": {str(i): list(p) for i, p in enumerate(points)},
                    }
                ]
            }
        landmarks[name + ".json"] = json.dumps(frames)
    plan = {
        "format_version": "subject-plan-v1",
        "status": "reviewed",
        "reference": "synthetic-only",
        "subject_splits": {
            f"subject-{role}": role for role in ("train", "val", "test")
        },
        "videos": {
            name: {"subject_id": f"subject-{role}", "session_id": name}
            for name, role in zip(videos, ("train", "val", "test"))
        },
    }
    ann, lm, subjects = (
        tmp_path / "annotations.zip",
        tmp_path / "landmarks.zip",
        tmp_path / "subjects.json",
    )
    subjects.write_text(json.dumps(plan))
    write_zip(ann, annotations)
    write_zip(lm, landmarks)
    return ann, lm, subjects, annotations, landmarks


def run_import(sources, output):
    ann, lm, subjects, *_ = sources
    return import_ipn_hands(
        ann,
        lm,
        subjects,
        output,
        annotations_sha256=file_sha256(ann),
        landmarks_sha256=file_sha256(lm),
        expected_videos=3,
        synthetic=True,
    )


def test_annotation_audit_exact_source_taxonomy_and_timeline(public_sources):
    ann = public_sources[0]
    report = audit_annotations(ann, expected_sha256=file_sha256(ann), expected_videos=3)
    assert report["frames"] == 135 and report["class_frames"] == [9] * 15
    assert report["source_split_videos"] == {"train": 2, "test": 1}
    assert not report["ready_for_training"] and not report["subjects_verified"]


def test_import_preserves_source_times_classes_unknown_sdk_time_and_no_actions(
    public_sources, tmp_path
):
    root = tmp_path / "import"
    manifest = run_import(public_sources, root)
    records = list(iter_imported_records(root))
    assert (
        manifest["complete"]
        and not manifest["ready_for_training"]
        and manifest["synthetic"]
    )
    assert manifest["action_backend"] == "none" and not manifest["context_available"]
    assert len(records) == 135 and set(r.class_id for r in records) == set(range(15))
    for r in records:
        assert (
            r.frame_status.source_timestamp_ns
            == (r.frame_status.source_frame_id - 1) * 1_000_000_000 // 30
        )
        assert (
            r.frame_status.processed_at_ns is None
            and r.frame_status.processing_time_origin == "unavailable_external"
        )
        assert (
            r.vocabulary_version == IPN_HANDS_VERSION
            and r.observation.handedness_score is None
        )
    assert len({r.original_sha256 for r in records}) == 3
    assert not list(root.rglob("*.jpg"))
    with pytest.raises(FileExistsError):
        run_import(public_sources, root)


@pytest.mark.parametrize(
    "problem",
    ["gap", "overlap", "wrong_class", "wrong_count", "split_overlap", "wrong_taxonomy"],
)
def test_bad_annotation_semantics_rejected(public_sources, problem):
    ann, _, _, entries, _ = public_sources
    if problem == "gap":
        entries["temporal_annotations/fixture_train.txt"] = "0,2,45\n"
    if problem == "overlap":
        entries["temporal_annotations/fixture_train.txt"] = "0,1,3\n1,3,45\n"
    if problem == "wrong_class":
        entries["temporal_annotations/fixture_train.txt"] = "15,1,45\n"
    if problem == "wrong_count":
        entries["frame_counts.csv"] = entries["frame_counts.csv"].replace(
            ",45,44", ",46,44"
        )
    if problem == "split_overlap":
        entries["splits/test_list.txt"] += "fixture_train\n"
    if problem == "wrong_taxonomy":
        entries["classes.txt"] = entries["classes.txt"].replace("7,TLT", "7,swipe_left")
    digest = write_zip(ann, entries)
    with pytest.raises(ValueError):
        audit_annotations(ann, expected_sha256=digest, expected_videos=3)


def test_checksum_failure_never_creates_output(public_sources, tmp_path):
    ann, lm, subjects, *_ = public_sources
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        import_ipn_hands(
            ann,
            lm,
            subjects,
            root,
            annotations_sha256="0" * 64,
            landmarks_sha256=file_sha256(lm),
            expected_videos=3,
        )
    assert not root.exists()


def test_subject_leakage_rejected_before_output(public_sources, tmp_path):
    subjects = public_sources[2]
    plan = json.loads(subjects.read_text())
    plan["videos"]["fixture_test"]["subject_id"] = "subject-train"
    subjects.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="isolation/subject overlap"):
        run_import(public_sources, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_missing_frame_retains_incomplete_import(public_sources, tmp_path):
    _, lm, _, _, entries = public_sources
    frames = json.loads(entries["fixture_val.json"])
    del frames["fixture_val_000020.jpg"]
    entries["fixture_val.json"] = json.dumps(frames)
    write_zip(lm, entries)
    root = tmp_path / "broken"
    with pytest.raises(ValueError, match="frame count"):
        run_import(public_sources, root)
    assert not json.loads((root / "manifest.json").read_text())["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        list(iter_imported_records(root))


def test_no_and_multiple_hands_keep_source_labels_without_guessing_target(
    public_sources, tmp_path
):
    _, lm, _, _, entries = public_sources
    frames = json.loads(entries["fixture_train.json"])
    frames["fixture_train_000002.jpg"]["hand_landmarks"] = []
    hands = frames["fixture_train_000004.jpg"]["hand_landmarks"]
    hands.append(hands[0].copy())
    entries["fixture_train.json"] = json.dumps(frames)
    write_zip(lm, entries)
    root = tmp_path / "import"
    manifest = run_import(public_sources, root)
    records = [
        r for r in iter_imported_records(root) if r.recording_id == "fixture_train"
    ]
    affected = [r for r in records if r.frame_status.source_frame_id in {2, 4}]
    assert affected and all(r.class_id == -1 and not r.class_valid for r in affected)
    entry = next(
        r for r in manifest["recordings"] if r["recording_id"] == "fixture_train"
    )
    with gzip.open(root / entry["frames_path"], "rt") as stream:
        inventory = [json.loads(line) for line in stream]
    assert len(inventory) == 45 and inventory[3]["source_class_id"] == 1
    assert entry["empty_prediction_frames"] == entry["multiple_prediction_frames"] == 1


@pytest.mark.parametrize(
    "member",
    ["../outside.txt", "/outside.txt", "nested\\outside.txt", "C:/outside.txt"],
)
def test_archive_path_traversal_rejected(tmp_path, member):
    path = tmp_path / "bad.zip"
    digest = write_zip(path, {member: "not executed"})
    with pytest.raises(ValueError, match="unsafe"):
        VerifiedArchive(path, expected_sha256=digest)
    assert not (tmp_path / "outside.txt").exists()


def test_archive_duplicate_and_size_limits(tmp_path):
    path = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("a.txt", "a")
            archive.writestr("a.txt", "b")
    with pytest.raises(ValueError, match="duplicate"):
        VerifiedArchive(path, expected_sha256=file_sha256(path))
    digest = write_zip(path, {"a.txt": "123456789"})
    with pytest.raises(ValueError, match="oversized"):
        VerifiedArchive(path, expected_sha256=digest, max_member_bytes=3)


def test_reviewed_import_cache_cli_and_budget(public_sources, tmp_path, capsys):
    from backend.importing.commands import main
    from backend.importing.features import build_import_cache

    root = tmp_path / "reviewed-import"
    run_import(public_sources, root)
    with pytest.raises(ValueError, match="budget exceeded"):
        build_import_cache(
            root,
            tmp_path / "too-big",
            split="train",
            dataset_version="fixture",
            review_reference="synthetic-only",
            max_records=10,
        )
    assert not (tmp_path / "too-big").exists()
    with pytest.raises(ValueError, match="requires --stats"):
        build_import_cache(
            root,
            tmp_path / "no-stats",
            split="val",
            dataset_version="fixture",
            review_reference="synthetic-only",
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FeatureStatisticsWarning)
        assert (
            main(
                [
                    "build-cache",
                    "--imported",
                    str(root),
                    "--output",
                    str(tmp_path / "cli-train"),
                    "--split",
                    "train",
                    "--dataset-version",
                    "fixture",
                    "--review-reference",
                    "synthetic-only",
                ]
            )
            == 0
        )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "cache_built" and result["synthetic"]
    assert (
        main(
            [
                "build-cache",
                "--imported",
                str(root),
                "--output",
                str(tmp_path / "cli-val"),
                "--split",
                "val",
                "--dataset-version",
                "fixture",
                "--review-reference",
                "synthetic-only",
                "--stats",
                str(tmp_path / "cli-train/stats.json"),
            ]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["vocabulary_version"] == IPN_HANDS_VERSION
    )


def test_import_reader_detects_shard_tampering(public_sources, tmp_path):
    root = tmp_path / "import"
    manifest = run_import(public_sources, root)
    path = root / manifest["recordings"][0]["records_path"]
    with path.open("ab") as output:
        output.write(b"tampered")
    with pytest.raises(ValueError, match="path/hash"):
        list(iter_imported_records(root))


@pytest.fixture
def public_caches(public_sources, tmp_path):
    root = tmp_path / "import"
    imported = run_import(public_sources, root)
    records = list(iter_imported_records(root))
    manifest = SplitManifest(
        dataset_version="synthetic-ipn-v1",
        dataset_sha256=records_digest(records),
        status="draft",
        synthetic=True,
        preregistration_reference="synthetic-only",
        subject_splits=imported["subject_plan"]["subject_splits"],
        vocabulary_version=IPN_HANDS_VERSION,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FeatureStatisticsWarning)
        normalizer = fit_training_normalizer(records, manifest)
    roots = []
    for split in ("train", "val", "test"):
        path = tmp_path / (split + "-cache")
        dataset = WindowDataset(records, manifest, split=split, normalizer=normalizer)
        write_cache(dataset, path)
        roots.append(path)
    return roots


def test_public_caches_bind_vocabulary_and_reject_cross_version(public_caches):
    root = public_caches[0]
    header = json.loads((root / "index.json").read_text())
    args = {
        "expected_" + k: header[k]
        for k in (
            "dataset_sha256",
            "split_sha256",
            "stats_sha256",
            "config_sha256",
            "split",
        )
    }
    values, index = load_cache(root, **args)
    assert index["format_version"] == "t11-cache-v2" and values["y"].max() == 14
    with pytest.raises(ValueError, match="version"):
        load_cache(root, **args, expected_vocabulary=vocabulary_for())
    classes = json.loads((root / "classes.json").read_text())
    classes["classes"][7]["token"] = "TRT"
    (root / "classes.json").write_text(json.dumps(classes))
    with pytest.raises(ValueError):
        load_cache(root, **args)


@pytest.mark.parametrize("kind", ["mlp", "tcn"])
def test_public_import_to_deep_training_and_resume(public_caches, tmp_path, kind):
    torch = pytest.importorskip("torch")
    from algorithm.training.config import TrainConfig
    from algorithm.training.trainer import train

    config = TrainConfig(model_type=kind, max_epochs=2, batch_size=16, seed=31)
    full = tmp_path / (kind + "-full")
    result = train(*public_caches[:2], full, config=config)
    assert (
        result["status"] == "completed"
        and result["synthetic"]
        and not result["final_test_used"]
    )
    assert result["identity"]["classes"]["class_count"] == 15
    checkpoint = torch.load(full / "last.pt", weights_only=True)
    assert len(checkpoint["history"][-1]["validation"]["confusion_matrix"]) == 15
    assert all(
        len(batch["class_counts"]) == 15
        for epoch in checkpoint["history"]
        for batch in epoch["batches"]
    )
    resumed = tmp_path / (kind + "-resumed")
    train(*public_caches[:2], resumed, config=config, stop_after_epoch=1)
    train(*public_caches[:2], resumed, config=config, resume=resumed / "last.pt")
    other = torch.load(resumed / "last.pt", weights_only=True)
    assert other["history"] == checkpoint["history"]
    for key in checkpoint["state_dict"]:
        assert torch.equal(checkpoint["state_dict"][key], other["state_dict"][key])
