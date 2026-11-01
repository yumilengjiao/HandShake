"""Evaluation CLI integration with explicitly untrained zero-weight fixtures.

No trainer/checkpoint/optimizer is invoked. Procedural bundle metadata exists only
for exercising the loader protocol; its zero logits are not learned model results.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from algorithm.bundle import (
    CALIBRATION,
    EVENT_POLICY,
    _arrays_bytes,
    architecture_for,
    deployment_architecture,
    digest,
    json_bytes,
    runtime_contract,
)
from algorithm.data.cache import write_cache
from algorithm.data.dataset import WindowDataset, canonical_digest
from algorithm.evaluation.cli import main
from algorithm.evaluation.collect import collect_scores
from algorithm.evaluation.scores import load_scores
from algorithm.model_config import build_model
from shared.features.schema import feature_schema_hash, load_feature_schema
from tests.integration.test_feature_dataset import (
    example_records,
    manifest_for,
    normalizer_for,
)
from tests.unit.test_evaluation_events import inputs, prediction, target
from tests.unit.test_evaluation_scores import archive, rejection_scores


def zero_bundle(root, kind, normalizer, val_index):
    """Make a protocol fixture with zero parameters, not a fake training run."""
    vocabulary = normalizer.vocabulary
    model = build_model(kind, vocabulary=vocabulary)
    tensors = {
        name: np.zeros(tuple(t.shape), dtype=np.float32)
        for name, t in model.state_dict().items()
    }
    identity = {
        "config": {"shuffle_labels_diagnostic": False, "model_type": kind},
        "architecture": architecture_for(kind, vocabulary),
        "schema_sha256": feature_schema_hash(),
        "classes_sha256": vocabulary.digest(),
        "classes": vocabulary.to_dict(),
        "data": {
            key: val_index[key]
            for key in (
                "dataset_sha256",
                "split_sha256",
                "stats_sha256",
                "config_sha256",
                "synthetic",
                "vocabulary_version",
                "classes_sha256",
            )
        },
    }
    identity["data"]["val_index_sha256"] = val_index["index_sha256"]
    training = {
        "format_version": "training-run-v1",
        "status": "completed",
        "final_test_used": False,
        "synthetic": True,
        "identity": identity,
        "identity_sha256": canonical_digest(identity),
        "fixture_warning": "UNTRAINED zero-weight protocol fixture. No training evidence.",
    }
    members = {
        "weights.npz": _arrays_bytes(tensors),
        "architecture.json": json_bytes(deployment_architecture(kind, vocabulary)),
        "classes.json": json_bytes(vocabulary.to_dict()),
        "feature_schema.json": json_bytes(load_feature_schema()),
        "normalization.json": json_bytes(normalizer.to_dict()),
        "calibration.json": json_bytes(CALIBRATION),
        "event_thresholds.json": json_bytes(EVENT_POLICY),
        "training_manifest.json": json_bytes(training),
        "runtime_contract.json": json_bytes(runtime_contract()),
        "model_card.md": b"UNTRAINED TEST FIXTURE; NOT A TRAINING RESULT\n",
        "golden.npz": _arrays_bytes(
            {
                "x": np.zeros((2, 32, 94), dtype=np.float32),
                "logits": np.zeros((2, vocabulary.class_count), dtype=np.float32),
            }
        ),
    }
    root.mkdir()
    for name, raw in members.items():
        (root / name).write_bytes(raw)
    receipt = json_bytes(
        {
            "format_version": "inference-bundle-v1",
            "status": "uncalibrated_preview_only",
            "files": {
                name: {"sha256": digest(raw), "bytes": len(raw)}
                for name, raw in members.items()
            },
        }
    )
    (root / "checksums.json").write_bytes(receipt)
    return digest(receipt)


@pytest.fixture(params=["legacy", "compact"])
def scoring_fixture(request, tmp_path):
    if request.param == "legacy":
        records = example_records(count=35)
        manifest = manifest_for(records)
        normalizer = normalizer_for(records, manifest)
        indexes = {
            split: write_cache(
                WindowDataset(records, manifest, split=split, normalizer=normalizer),
                tmp_path / split,
            )
            for split in ("val", "test")
        }
    else:
        from shared.features import FeatureNormalizer
        from tests.integration.test_public_data import public_sources, run_import
        from tests.integration.test_stream_cache import build

        sources = public_sources.__wrapped__(tmp_path)
        imported = tmp_path / "imported"
        run_import(sources, imported)
        build(imported, tmp_path / "train")
        for split in ("val", "test"):
            build(
                imported,
                tmp_path / split,
                split=split,
                stats=tmp_path / "train/stats.json",
            )
        normalizer = FeatureNormalizer.load(tmp_path / "train/stats.json")
        indexes = {
            split: json.loads((tmp_path / split / "index.json").read_text())
            for split in ("val", "test")
        }
    return tmp_path, normalizer, indexes


@pytest.mark.parametrize("kind", ["mlp", "tcn"])
def test_real_bundle_loader_and_cache_scoring_no_training(
    scoring_fixture, kind, monkeypatch
):
    import torch

    root, normalizer, indexes = scoring_fixture
    sha = zero_bundle(root / "bundle", kind, normalizer, indexes["val"])

    def forbidden(*args, **kwargs):
        pytest.fail("evaluation must not load checkpoints or train")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    for split in ("val", "test"):
        result = collect_scores(
            root / "bundle",
            root / split,
            root / (split + "-scores"),
            model_sha256=sha,
            cache_index_sha256=indexes[split]["index_sha256"],
            split=split,
            allow_synthetic=True,
            batch_size=7,
        )
        scores = load_scores(
            root / (split + "-scores"),
            expected_manifest_sha256=result["manifest_sha256"],
        )
        assert np.all(scores.rows["logits"] == 0)
        assert np.all(scores.rows["available"] == 1)  # partial windows not removed
        assert scores.rows["ready"].sum() < len(scores.rows)
        assert (
            scores.provenance.split == split and scores.provenance.model_sha256 == sha
        )
        assert scores.provenance.synthetic
    assert not (root / "best.pt").exists()


def test_score_cli_rejects_wrong_pins_synthetic_and_split_without_output(
    scoring_fixture, capsys
):
    root, normalizer, indexes = scoring_fixture
    sha = zero_bundle(root / "bundle", "mlp", normalizer, indexes["val"])
    base = [
        "score",
        "--package",
        str(root / "bundle"),
        "--model-sha256",
        sha,
        "--cache",
        str(root / "val"),
        "--cache-index-sha256",
        indexes["val"]["index_sha256"],
        "--split",
        "val",
        "--output",
        str(root / "out"),
    ]
    with pytest.raises(SystemExit) as error:
        main(base)
    assert error.value.code == 2 and "synthetic" in capsys.readouterr().err
    wrong = base.copy()
    wrong[wrong.index("--cache-index-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit):
        main(wrong + ["--allow-synthetic"])
    assert not (root / "out").exists()
    wrong = base.copy()
    wrong[wrong.index("--split") + 1] = "test"
    with pytest.raises(SystemExit):
        main(wrong + ["--allow-synthetic"])
    assert not (root / "out").exists()
    assert main(base + ["--allow-synthetic"]) == 0
    capsys.readouterr()
    assert (root / "out/manifest.json").is_file()


def test_calibration_and_classification_cli_roundtrip(tmp_path, capsys):
    root, receipt, _ = archive(tmp_path, rejection_scores())
    selection = tmp_path / "selection"
    assert (
        main(
            [
                "calibrate",
                "--scores",
                str(root),
                "--sha256",
                receipt["manifest_sha256"],
                "--output",
                str(selection),
                "--allow-synthetic",
                "--temperatures",
                "1",
                "--thresholds",
                "0",
                ".6",
                "1",
                "--max-risk",
                "0",
                "--min-coverage",
                ".3",
                "--min-accepted",
                "1",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert (
        result["rejection_status"] == "selected" and result["actions_enabled"] is False
    )
    report = json.loads((selection / "report.json").read_text())
    assert report["selection"]["event_search_trials"] == 0
    assert report["selection"]["threshold_trials"] == 3
    assert report["selective"]["accepted_labeled"] == 1
    assert "actions_enabled: false" in (selection / "event_thresholds.yaml").read_text()
    out = tmp_path / "classification.json"
    assert (
        main(
            [
                "classify",
                "--scores",
                str(root),
                "--sha256",
                receipt["manifest_sha256"],
                "--output",
                str(out),
                "--calibration",
                str(selection / "calibration.json"),
                "--calibration-sha256",
                result["calibration_sha256"],
            ]
        )
        == 0
    )
    readback = json.loads(out.read_text())
    assert readback["selective"] == report["selective"]
    with pytest.raises(SystemExit):
        main(
            [
                "classify",
                "--scores",
                str(root),
                "--sha256",
                receipt["manifest_sha256"],
                "--output",
                str(out),
            ]
        )


def test_test_archive_cannot_fit_and_cli_cannot_tune_test_thresholds(tmp_path):
    scores = rejection_scores()
    scores.provenance = scores.provenance.model_copy(update={"split": "test"})
    root, receipt, _ = archive(tmp_path, scores)
    out = tmp_path / "not-created"
    with pytest.raises(SystemExit):
        main(
            [
                "calibrate",
                "--scores",
                str(root),
                "--sha256",
                receipt["manifest_sha256"],
                "--output",
                str(out),
                "--allow-synthetic",
            ]
        )
    assert not out.exists()
    with pytest.raises(SystemExit):
        main(
            [
                "classify",
                "--scores",
                str(root),
                "--sha256",
                receipt["manifest_sha256"],
                "--output",
                str(out),
                "--threshold",
                ".9",
            ]
        )
    assert not out.exists()


def test_events_cli_pinned_and_strict_json(tmp_path, capsys):
    path, out = tmp_path / "events.json", tmp_path / "report.json"
    raw = json_bytes(inputs([target()], [prediction()]).model_dump(mode="json"))
    path.write_bytes(raw)
    base = [
        "events",
        "--input",
        str(path),
        "--sha256",
        digest(raw),
        "--output",
        str(out),
    ]
    assert main(base) == 0
    report = json.loads(out.read_text())
    assert report["kinds"]["command"]["layers"]["submitted"]["tp"] == 1
    assert report["kinds"]["command"]["layers"]["external_injected"]["tp"] == 0
    assert json.loads(capsys.readouterr().out)["synthetic"] is True
    with pytest.raises(SystemExit):
        main(base[:-1] + [str(tmp_path / "bad")] + ["--sha256", "0" * 64])
    duplicate = raw.replace(
        b'"synthetic": true', b'"synthetic": false, "synthetic": true'
    )
    path.write_bytes(duplicate)
    with pytest.raises(SystemExit):
        main(
            [
                "events",
                "--input",
                str(path),
                "--sha256",
                digest(duplicate),
                "--output",
                str(tmp_path / "bad"),
            ]
        )


def test_documented_synthetic_event_fixture(tmp_path):
    from pathlib import Path

    path = Path(__file__).parents[1] / "fixtures/evaluation/events.synthetic.json"
    out = tmp_path / "report.json"
    assert (
        main(
            [
                "events",
                "--input",
                str(path),
                "--sha256",
                digest(path.read_bytes()),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads(out.read_text())
    layers = report["kinds"]["command"]["layers"]
    assert report["synthetic"] and layers["submitted"]["tp"] == 1
    assert layers["external_injected"]["tp"] == 0
    assert layers["submitted"]["zero_count_poisson_upper95_per_hour"] > 8.9
    assert report["availability"][0]["active_ns"] == 10_000_000_000


def test_help_and_score_metrics_do_not_import_torch_or_open_devices():
    code = "from algorithm.evaluation.cli import main; import sys; assert 'torch' not in sys.modules; main(['--help'])"
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0 and "calibrate" in result.stdout
    wrapper = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert wrapper.returncode == 0 and "events" in wrapper.stdout
