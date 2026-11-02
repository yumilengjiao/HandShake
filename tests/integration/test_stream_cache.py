"""Compact-cache tests: procedural IPN-format fixtures, no network/devices."""

import gzip
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from algorithm.data.cache import load_cache
from algorithm.data.compact import ENDPOINT_DTYPE
from algorithm.data.dataset import (
    SplitManifest,
    WindowDataset,
    canonical_digest,
    records_digest,
)
from algorithm.data.source_archive import file_sha256
from algorithm.training.config import TrainConfig
from algorithm.training.trainer import train
from backend.importing.ipn_hands import iter_imported_records
from backend.importing.stream_features import build_stream_cache
from shared.classes import IPN_HANDS_VERSION
from shared.features import FeatureStatisticsWarning
from tests.integration.test_feature_dataset import normalizer_for
from tests.integration.test_public_data import public_sources, run_import

pytestmark = pytest.mark.filterwarnings(
    f"ignore::{FeatureStatisticsWarning.__module__}.{FeatureStatisticsWarning.__name__}"
)


@pytest.fixture
def imported(tmp_path):
    sources = public_sources.__wrapped__(tmp_path)
    root = tmp_path / "imported"
    run_import(sources, root)
    return root


def build(root, output, split="train", stats=None, **kwargs):
    return build_stream_cache(
        root,
        output,
        split=split,
        dataset_version="synthetic-scale-v1",
        review_reference="synthetic-only-review",
        stats=stats,
        **kwargs,
    )


def load(root):
    index = json.loads((root / "index.json").read_text())
    return load_cache(
        root,
        expected_dataset_sha256=index["dataset_sha256"],
        expected_split_sha256=index["split_sha256"],
        expected_stats_sha256=index["stats_sha256"],
        expected_config_sha256=index["config_sha256"],
        expected_split=index["split"],
    )


def repin(root):
    index = json.loads((root / "index.json").read_text())
    for name in index["files"]:
        index["files"][name] = {
            "sha256": file_sha256(root / name),
            "bytes": (root / name).stat().st_size,
        }
    index["index_sha256"] = canonical_digest(
        {k: v for k, v in index.items() if k != "index_sha256"}
    )
    (root / "index.json").write_text(json.dumps(index))


@pytest.mark.parametrize("stride", [1, 3])
def test_compact_windows_stats_and_coverage_match_legacy(imported, tmp_path, stride):
    result = build(imported, tmp_path / "compact", stride=stride)
    data, index = load(tmp_path / "compact")
    records = list(iter_imported_records(imported))  # tiny reference only
    manifest = SplitManifest(
        dataset_version="legacy-fixture",
        dataset_sha256=records_digest(records),
        synthetic=True,
        status="frozen",
        preregistration_reference="synthetic-only-review",
        subject_splits=json.loads((imported / "manifest.json").read_text())[
            "subject_plan"
        ]["subject_splits"],
        vocabulary_version=IPN_HANDS_VERSION,
    )
    normalizer = normalizer_for(records, manifest)
    dense = WindowDataset(
        records, manifest, split="train", normalizer=normalizer, stride=stride
    )
    saved = json.loads((tmp_path / "compact/stats.json").read_text())
    np.testing.assert_allclose(
        saved["stats"]["mean"], normalizer.stats.mean, atol=1e-12
    )
    np.testing.assert_allclose(saved["stats"]["std"], normalizer.stats.std, atol=1e-12)
    np.testing.assert_array_equal(saved["stats"]["count"], normalizer.stats.count)
    np.testing.assert_allclose(
        data["x"][:], np.stack([item["x"] for item in dense]), rtol=1e-5, atol=1e-6
    )
    for key in ("y", "ready", "valid_length"):
        np.testing.assert_array_equal(data[key], [item[key] for item in dense])
    assert index["coverage"] == dense.coverage()
    assert index["dataset_sha256"] != manifest.dataset_sha256
    assert result["feature_bytes"] == 45 * 94 * 4
    assert not (tmp_path / "compact/features.npz").exists()
    assert not list(tmp_path.glob(".handshake-spool-*"))


@pytest.mark.parametrize(
    "kwargs", [{"max_records": 100}, {"max_groups": 1}, {"max_records": 0}]
)
def test_limits_fail_without_publishing_cache(imported, tmp_path, kwargs):
    with pytest.raises(ValueError, match="budget"):
        build(imported, tmp_path / "rejected", **kwargs)
    assert not (tmp_path / "rejected").exists()
    assert not list(tmp_path.glob(".handshake-spool-*"))


def test_train_stats_required_and_cannot_cross_review(imported, tmp_path):
    with pytest.raises(ValueError, match="--stats"):
        build(imported, tmp_path / "val", split="val")
    build(imported, tmp_path / "train")
    with pytest.raises(ValueError, match="split hash"):
        build_stream_cache(
            imported,
            tmp_path / "val",
            split="val",
            dataset_version="different",
            review_reference="synthetic-only-review",
            stats=tmp_path / "train/stats.json",
        )
    with pytest.raises(FileExistsError):
        build(imported, tmp_path / "train")


@pytest.mark.parametrize(
    "problem", ["identity", "frame", "duplicate", "label", "visible", "original_leak"]
)
def test_rehashed_source_corruption_rejected(imported, tmp_path, problem):
    manifest = json.loads((imported / "manifest.json").read_text())
    entry = manifest["recordings"][0]
    path = imported / entry["records_path"]
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    if problem == "identity":
        rows[0]["lineage_id"] = "other-recording"
    elif problem == "frame":
        rows[0]["frame_status"]["source_frame_id"] += 1000
        rows[0]["observation"]["source_frame_id"] += 1000
    elif problem == "duplicate":
        rows.append(rows[0])
        entry["records"] += 1
        entry["supervised_records"] += 1
    elif problem == "label":
        rows[0]["class_id"] = (rows[0]["class_id"] + 1) % 15
    elif problem == "visible":
        rows[0]["visible_subjects"].append("subject-train")
    else:
        sha = manifest["recordings"][1]["original_landmark_member_sha256"]
        entry["original_landmark_member_sha256"] = sha
        for row in rows:
            row["original_sha256"] = sha
    with gzip.open(path, "wt") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
    entry["records_sha256"] = file_sha256(path)
    (imported / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        build(imported, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize(
    "problem", ["file_hash", "row", "length", "label", "epoch", "source", "finite"]
)
def test_rehashed_cache_corruption_rejected(imported, tmp_path, problem):
    cache = tmp_path / "cache"
    build(imported, cache)
    if problem == "file_hash":
        with (cache / "features.f32").open("ab") as stream:
            stream.write(b"extra")
    elif problem == "finite":
        values = np.memmap(cache / "features.f32", mode="r+", dtype="<f4")
        values[0] = np.nan
        values.flush()
        del values
        repin(cache)
    else:
        values = np.memmap(cache / "endpoints.bin", mode="r+", dtype=ENDPOINT_DTYPE)
        if problem == "row":
            values[0]["row"] = 100000
        if problem == "length":
            values[0]["valid_length"] = 32
        if problem == "label":
            values[0]["y"] = 15
        if problem == "epoch":
            values[0]["context_epoch"] = 1
        if problem == "source":
            values[0]["timestamp_ns"] = 1
        values.flush()
        del values
        repin(cache)
    with pytest.raises(ValueError):
        load(cache)


def test_indexing_budget_and_immutable_cache(imported, tmp_path):
    cache = tmp_path / "cache"
    build(imported, cache)
    data, _ = load(cache)
    with pytest.raises(ValueError, match="materialization"):
        np.asarray(data["x"])
    with pytest.raises(ValueError, match="budget"):
        data["x"][[0] * 1025]
    with pytest.raises(IndexError):
        data["x"][999]
    assert data["x"][[]].shape == (0, 32, 94)
    np.testing.assert_array_equal(data["x"][-1], data["x"][44])
    with (cache / "features.f32").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        data["x"][0]


@pytest.mark.parametrize("kind", ["mlp", "tcn"])
def test_compact_training_resume_and_model_export(imported, tmp_path, kind):
    from algorithm.bundle import export_bundle, load_bundle

    build(imported, tmp_path / "train")
    build(imported, tmp_path / "val", split="val", stats=tmp_path / "train/stats.json")
    config = TrainConfig(model_type=kind, max_epochs=3, batch_size=4, patience=10)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(tmp_path / "train", tmp_path / "val", full, config=config)
    train(
        tmp_path / "train", tmp_path / "val", resumed, config=config, stop_after_epoch=1
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())  # JSON is a YAML subset
    subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.cli",
            "train",
            "--train-cache",
            str(tmp_path / "train"),
            "--val-cache",
            str(tmp_path / "val"),
            "--output",
            str(resumed),
            "--resume",
            str(resumed / "last.pt"),
            "--training-config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    a, b = [torch.load(root / "last.pt", weights_only=True) for root in (full, resumed)]
    assert a["history"] == b["history"]
    for key in a["state_dict"]:
        torch.testing.assert_close(
            a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0
        )
    receipt = export_bundle(
        full, tmp_path / "train/stats.json", tmp_path / "bundle", allow_synthetic=True
    )
    assert load_bundle(
        tmp_path / "bundle", expected_manifest_sha256=receipt["manifest_sha256"]
    ).synthetic_training
    with pytest.raises(ValueError, match="1024"):
        train(
            tmp_path / "train",
            tmp_path / "val",
            tmp_path / "too-large",
            config=config.model_copy(update={"batch_size": 2048}),
        )
    assert not (tmp_path / "too-large").exists()


def test_truncated_gzip_and_frame_ledger_are_not_silently_accepted(imported, tmp_path):
    manifest = json.loads((imported / "manifest.json").read_text())
    path = imported / manifest["recordings"][0]["records_path"]
    with path.open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises((ValueError, OSError, EOFError)):
        build(imported, tmp_path / "no-cache")
    assert not (tmp_path / "no-cache").exists()


def test_cli_exposes_streaming_without_weakening_old_budget(imported, tmp_path, capsys):
    from backend.importing.commands import main

    common = [
        "--imported",
        str(imported),
        "--output",
        str(tmp_path / "cache"),
        "--split",
        "train",
        "--dataset-version",
        "fixture",
        "--review-reference",
        "synthetic-only-review",
    ]
    assert main(["build-cache", *common, "--max-records", "100"]) == 2
    assert not (tmp_path / "cache").exists()
    capsys.readouterr()
    assert main(["build-stream-cache", *common]) == 0
    assert json.loads(capsys.readouterr().out)["scope"] == "compact_stream_cache_v3"


def test_no_hand_and_multi_hand_preserve_partial_windows(tmp_path):
    from tests.integration.test_public_data import write_zip

    sources = public_sources.__wrapped__(tmp_path)
    entries = sources[4]
    frames = json.loads(entries["fixture_train.json"])
    frames["fixture_train_000001.jpg"]["hand_landmarks"] = []
    frames["fixture_train_000002.jpg"]["hand_landmarks"] = []
    frames["fixture_train_000009.jpg"]["hand_landmarks"] *= 2
    frames["fixture_train_000010.jpg"]["hand_landmarks"] = []
    entries["fixture_train.json"] = json.dumps(frames)
    write_zip(sources[1], entries)
    root = tmp_path / "multi-import"
    run_import(sources, root)
    build(root, tmp_path / "cache")
    data, index = load(tmp_path / "cache")
    assert index["coverage"]["selected_endpoints"] > len(data["y"])
    assert not data["class_valid"].all()
    assert (data["y"][~data["class_valid"]] == -1).all()
    endpoints = np.memmap(
        tmp_path / "cache/endpoints.bin", mode="r", dtype=ENDPOINT_DTYPE
    )
    for group_id in np.unique(endpoints["group"]):
        first = int(np.flatnonzero(endpoints["group"] == group_id)[0])
        assert data["valid_length"][first] == 1
        assert np.count_nonzero(data["x"][first][:-1]) == 0


@pytest.mark.parametrize("missing_resource", [False, True])
def test_probe_training_requires_explicit_opt_in(tmp_path, missing_resource):
    program = (
        [
            "-c",
            "import runpy, sys; sys.modules['resource'] = None; "
            "runpy.run_path('scripts/probe_stream_cache.py', run_name='__main__')",
        ]
        if missing_resource
        else ["scripts/probe_stream_cache.py"]
    )
    result = subprocess.run(
        [
            sys.executable,
            *program,
            "--output",
            str(tmp_path / "absent"),
            "--worker",
            "train",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2 and "--with-training" in result.stderr
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("system,expected", [("Linux", 7 * 1024), ("Darwin", 7)])
def test_probe_peak_rss_units(tmp_path, monkeypatch, system, expected):
    import platform
    import runpy
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=lambda _: SimpleNamespace(ru_maxrss=7),
        ),
    )
    monkeypatch.setattr(platform, "system", lambda: system)
    probe = runpy.run_path("scripts/probe_stream_cache.py")
    assert probe["_peak_bytes"]() == expected


def test_probe_cache_report_allows_unavailable_peak_rss(tmp_path, monkeypatch):
    import runpy
    from backend.importing import stream_features

    monkeypatch.setitem(sys.modules, "resource", None)
    probe = runpy.run_path("scripts/probe_stream_cache.py")
    assert probe["_peak_bytes"]() is None
    root = tmp_path / "probe"
    (root / "imported").mkdir(parents=True)
    (root / "imported/manifest.json").write_text(
        json.dumps({"synthetic": True}), encoding="utf-8"
    )
    calls = []

    def build_cache(*args, **kwargs):
        calls.append(kwargs["split"])
        return {"status": "synthetic-test-only"}

    monkeypatch.setattr(stream_features, "build_stream_cache", build_cache)
    probe["measure"](root, "cache")
    report = json.loads((root / "cache-report.json").read_text(encoding="utf-8"))
    assert report["peak_process_rss_bytes"] is None
    assert report["peak_process_rss_source"] == "unavailable"
    assert calls == ["train", "val"]
    assert not (root / "run").exists()


def test_validation_geometry_cannot_enter_training_statistics(imported, tmp_path):
    build(imported, tmp_path / "before")
    manifest = json.loads((imported / "manifest.json").read_text())
    entry = next(item for item in manifest["recordings"] if item["split"] == "val")
    path = imported / entry["records_path"]
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        for point in row["observation"]["landmarks_image"]:
            point[0] += 0.2
    with gzip.open(path, "wt") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
    entry["records_sha256"] = file_sha256(path)
    (imported / "manifest.json").write_text(json.dumps(manifest))
    build(imported, tmp_path / "after")
    before, after = [
        json.loads((tmp_path / part / "stats.json").read_text())
        for part in ("before", "after")
    ]
    for key in ("mean", "std", "count"):
        np.testing.assert_array_equal(before["stats"][key], after["stats"][key])
    assert before["stats"]["dataset_hash"] != after["stats"]["dataset_hash"]


def test_mixed_cache_versions_rejected_before_optimizer(imported, tmp_path):
    from algorithm.training.inputs import load_training_inputs
    from backend.importing.features import build_import_cache

    build(imported, tmp_path / "train")
    for split in ("train", "val"):
        build_import_cache(
            imported,
            tmp_path / ("old-" + split),
            split=split,
            dataset_version="fixture",
            review_reference="synthetic-only",
            stats=None if split == "train" else tmp_path / "old-train/stats.json",
        )
    with pytest.raises(ValueError, match="provenance"):
        load_training_inputs(tmp_path / "train", tmp_path / "old-val")


def test_metrics_merge_without_retaining_all_guesses():
    from algorithm.training.metrics import (
        classification_metrics,
        metrics_from_confusion,
    )

    a = classification_metrics([0, 1], [0, 0], 15)
    b = classification_metrics([1, 2], [1, 2], 15)
    merged = metrics_from_confusion(
        np.array(a["confusion_matrix"]) + np.array(b["confusion_matrix"])
    )
    assert merged == classification_metrics([0, 1, 1, 2], [0, 0, 1, 2], 15)


def test_transfer_preflight_requires_receipt_and_never_trains(
    imported, tmp_path, capsys
):
    from backend.importing.commands import main

    result = build(imported, tmp_path / "cache")
    args = ["inspect-cache", "--cache", str(tmp_path / "cache"), "--index-sha256"]
    assert main([*args, "0" * 64]) == 2
    capsys.readouterr()
    assert main([*args, result["index_sha256"]]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "cache_verified" and report["synthetic"]
    assert report["max_window_batch"] == 1024
    assert not (tmp_path / "run").exists()
