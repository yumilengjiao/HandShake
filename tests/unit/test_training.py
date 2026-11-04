import json
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from algorithm.data.cache import write_cache
from algorithm.data.dataset import ObservationRecord, WindowDataset
from algorithm.training.config import TrainConfig
from algorithm.training.inputs import load_training_inputs
from algorithm.training.metrics import classification_metrics
from algorithm.training.trainer import train
from tests.integration.test_feature_dataset import example_records, manifest_for, normalizer_for


@pytest.fixture
def caches(tmp_path):
    records = example_records(count=12)
    modified = []
    for i, row in enumerate(records):
        masked = i % 7 == 0
        modified.append(ObservationRecord.model_validate(row.model_dump() | dict(class_id=-1 if masked else i % 5, class_valid=not masked)))
    manifest = manifest_for(modified)
    normalizer = normalizer_for(modified, manifest)
    for split in ("train", "val"):
        write_cache(WindowDataset(modified, manifest, split=split, normalizer=normalizer), tmp_path / split)
    return tmp_path / "train", tmp_path / "val"


def test_metrics_fixed_classes_and_hand_computed_counts():
    metrics = classification_metrics([0, 1, 1, 2], [0, 1, 0, 2])
    assert metrics["confusion_matrix"][1] == [1, 1, 0, 0, 0]
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 5)
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["absent_classes"] == [3, 4]
    with pytest.raises(ValueError):
        classification_metrics([0], [5])


@pytest.mark.parametrize("model", ["mlp", "tcn"])
@pytest.mark.parametrize("balanced_plateau", [False, True])
@pytest.mark.parametrize("sampling", ["uniform", "supervised_uniform"])
def test_independent_process_epoch_resume_matches_uninterrupted_exactly(caches, tmp_path, model, balanced_plateau, sampling):
    cfg = tmp_path / "config.yaml"
    policy = dict(class_weighting="sqrt_inverse", scheduler="plateau", lr_patience=0, min_delta=1.0) if balanced_plateau else {}
    cfg.write_text(yaml.safe_dump(TrainConfig(model_type=model, max_epochs=3, batch_size=4, patience=10, sampling=sampling, **policy).model_dump()))
    common = [sys.executable, "-m", "backend.cli", "train", "--train-cache", str(caches[0]), "--val-cache", str(caches[1]), "--training-config", str(cfg)]
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    for command in (common + ["--output", str(full)],
                    common + ["--output", str(interrupted), "--stop-after-epoch", "1"],
                    common + ["--output", str(interrupted), "--resume", str(interrupted / "last.pt")]):
        run = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stderr
    a, b = [torch.load(root / "last.pt", weights_only=True) for root in (full, interrupted)]
    assert a["history"] == b["history"]
    assert a["global_step"] == b["global_step"]
    assert a["scheduler"] == b["scheduler"]
    assert a["identity"]["sampling"] == b["identity"]["sampling"]
    for row in a["history"]:
        assert row["sampling"] == sampling
        assert row["supervised"] == 10
        assert row["sampled_endpoints"] == (10 if sampling == "supervised_uniform" else 12)
        assert row["unsupervised_sampled"] == (0 if sampling == "supervised_uniform" else 2)
        assert row["optimizer_steps"] == 3
    if balanced_plateau:
        assert [row["learning_rate"] for row in a["history"]] == [.001, .001, .0005]
        assert a["history"][-1]["next_learning_rate"] == .00025
        for row in a["history"]:
            assert all(batch["learning_rate"] == row["learning_rate"] for batch in row["batches"])
        from algorithm.training.balancing import class_weights_from_counts
        identity = a["identity"]
        assert identity["objective"]["class_weights"] == class_weights_from_counts(identity["data"]["train"]["class_counts"], "sqrt_inverse")
        assert a["history"][0]["train_loss_kind"] == "sqrt_inverse_weighted_ce"
    for key in a["state_dict"]:
        assert torch.equal(a["state_dict"][key], b["state_dict"][key]), key
        assert torch.equal(a["best_state_dict"][key], b["best_state_dict"][key])
    assert torch.equal(a["rng"]["sampler"], b["rng"]["sampler"])
    manifest = json.loads((interrupted / "run_manifest.json").read_text())
    assert manifest["status"] == "completed" and not manifest["final_test_used"]
    assert manifest["synthetic"] and manifest["best_epoch"] <= 3


@pytest.mark.parametrize("variant", ["motion", "dropout35", "motion_dropout35"])
def test_experimental_variant_epoch_resume_is_exact(caches, tmp_path, variant):
    cfg = TrainConfig(model_variant=variant, max_epochs=3, batch_size=4, patience=10,
                      class_weighting="sqrt_inverse")
    full, paused = tmp_path / "full", tmp_path / "paused"
    train(*caches, full, config=cfg)
    train(*caches, paused, config=cfg, stop_after_epoch=1)
    with pytest.raises(ValueError, match="identity"):
        train(*caches, paused, config=cfg.model_copy(update={"model_variant": "baseline"}),
              resume=paused / "last.pt")
    train(*caches, paused, config=cfg, resume=paused / "last.pt")
    a, b = [torch.load(p / "last.pt", weights_only=True) for p in (full, paused)]
    assert a["history"] == b["history"]
    assert a["identity"]["architecture"]["model_variant"] == variant
    for name in a["state_dict"]:
        assert torch.equal(a["state_dict"][name], b["state_dict"][name]), name


def test_all_masked_batches_skip_updates_and_early_ties_keep_first(caches, tmp_path):
    cfg = TrainConfig(model_type="mlp", max_epochs=8, batch_size=1, patience=1, min_delta=1.0)
    result = train(*caches, tmp_path / "run", config=cfg)
    checkpoint = torch.load(tmp_path / "run/last.pt", weights_only=True)
    assert result["early_stopped"] and result["best_epoch"] == 1 and result["epochs"] == 2
    assert checkpoint["history"][0]["skipped_batches"] == 2
    assert checkpoint["history"][0]["sampled_endpoints"] == 12
    assert checkpoint["history"][0]["unsupervised_sampled"] == 2
    assert checkpoint["history"][0]["optimizer_steps"] == 10
    assert checkpoint["global_step"] == 20
    with pytest.raises(FileExistsError):
        train(*caches, tmp_path / "run", config=cfg)
    with pytest.raises(ValueError, match="identity"):
        train(*caches, tmp_path / "run", config=cfg.model_copy(update={"seed": 9}), resume=tmp_path / "run/last.pt")


def test_cache_tampering_rejected_before_creating_run(caches, tmp_path):
    (caches[1] / "features.npz").write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash"):
        train(*caches, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_test_cache_cannot_be_supplied_as_validation(caches):
    path = caches[1] / "index.json"
    data = json.loads(path.read_text())
    from algorithm.data.dataset import canonical_digest
    data["split"] = "test"
    data["index_sha256"] = canonical_digest({k: v for k, v in data.items() if k != "index_sha256"})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="version"):
        load_training_inputs(*caches)


def test_failed_gradient_run_keeps_failure_manifest(caches, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("nonfinite injected gradient")
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", broken)
    with pytest.raises(RuntimeError, match="nonfinite"):
        train(*caches, tmp_path / "failed", config=TrainConfig(max_epochs=1))
    manifest = json.loads((tmp_path / "failed/run_manifest.json").read_text())
    assert manifest["status"] == "failed" and "gradient" in manifest["failure"]


def test_resume_rejects_changed_checkpoint_hash(caches, tmp_path):
    root = tmp_path / "run"
    cfg = TrainConfig(model_type="mlp", max_epochs=2)
    train(*caches, root, config=cfg, stop_after_epoch=1)
    with (root / "last.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint hash"):
        train(*caches, root, config=cfg, resume=root / "last.pt")


def test_public_class_ids_are_not_silently_mapped_into_legacy_five_classes(caches, tmp_path):
    from hashlib import sha256
    from algorithm.data.dataset import canonical_digest
    path = caches[0]
    with np.load(path / "features.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["y"][arrays["class_valid"]] = 14  # Public-data-v1 is not migrated yet.
    np.savez_compressed(path / "features.npz", **arrays)
    index = json.loads((path / "index.json").read_text())
    index["npz_sha256"] = sha256((path / "features.npz").read_bytes()).hexdigest()
    index["index_sha256"] = canonical_digest({key: value for key, value in index.items() if key != "index_sha256"})
    (path / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="labels do not match vocabulary"):
        train(*caches, tmp_path / "must-not-train")
    assert not (tmp_path / "must-not-train").exists()


def test_weighted_training_keeps_validation_unweighted_and_all_labels(caches, tmp_path):
    from algorithm.training.trainer import _evaluate
    from algorithm.model_config import build_model
    from shared.classes import vocabulary_for
    cfg = TrainConfig(model_type="mlp", max_epochs=1, class_weighting="sqrt_inverse")
    result = train(*caches, tmp_path / "weighted", config=cfg)
    checkpoint = torch.load(tmp_path / "weighted/last.pt", weights_only=True)
    _, validation, provenance = load_training_inputs(*caches)
    model = build_model("mlp", vocabulary=vocabulary_for(provenance["vocabulary_version"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(validation["x"][:, -1]).float())
        mask = validation["class_valid"]
        expected = torch.nn.functional.cross_entropy(logits[mask], torch.from_numpy(validation["y"][mask]).long()).item()
    actual = checkpoint["history"][0]["validation"]
    assert actual["samples"] == int(mask.sum())
    assert actual["loss"] == pytest.approx(expected)
    assert _evaluate(model, validation, "mlp", 3)["loss"] == pytest.approx(expected)
    assert not result["final_test_used"]
    from algorithm.bundle import export_bundle, load_bundle
    receipt = export_bundle(tmp_path / "weighted", caches[0] / "stats.json", tmp_path / "bundle", allow_synthetic=True)
    bundle = load_bundle(tmp_path / "bundle", expected_manifest_sha256=receipt["manifest_sha256"])
    for name, tensor in checkpoint["state_dict"].items():
        torch.testing.assert_close(bundle.model.state_dict()[name], tensor, rtol=0, atol=0)


def test_plateau_restores_lr_after_reduction_and_respects_floor(caches, tmp_path):
    cfg = TrainConfig(model_type="mlp", max_epochs=5, class_weighting="sqrt_inverse",
                      scheduler="plateau", lr_patience=0, min_delta=1., min_learning_rate=.0003)
    root = tmp_path / "plateau"
    train(*caches, root, config=cfg, stop_after_epoch=3)
    train(*caches, root, config=cfg, resume=root / "last.pt")
    history = torch.load(root / "last.pt", weights_only=True)["history"]
    assert [row["learning_rate"] for row in history] == [.001, .001, .0005, .0003, .0003]
    assert history[-1]["next_learning_rate"] == .0003
    assert all(row["validation"]["samples"] == 10 for row in history)


@pytest.mark.parametrize("model", ["mlp", "tcn"])
def test_supervised_sampling_preserves_history_classes_and_validation(caches, tmp_path, monkeypatch, model):
    from hashlib import sha256
    from algorithm.training import trainer
    from algorithm.training.balancing import class_weights_from_counts
    training, validation, provenance = load_training_inputs(*caches)
    features = training["x"]
    original_labels = training["y"].copy()
    original_mask = training["class_valid"].copy()
    seen = []

    class ObservedWindows:
        def __getitem__(self, selected):
            seen.extend(selected.tolist())
            # The complete cached history, not a filtered timeline, is passed through.
            return features[selected]

    training["x"] = ObservedWindows()
    monkeypatch.setattr(trainer, "load_training_inputs", lambda *a: (training, validation, provenance))
    root = tmp_path / "supervised"
    cfg = TrainConfig(model_type=model, max_epochs=1, batch_size=4,
                      sampling="supervised_uniform", class_weighting="sqrt_inverse")
    result = train(*caches, root, config=cfg)
    checkpoint = torch.load(root / "last.pt", weights_only=True)
    row = checkpoint["history"][0]
    expected = np.flatnonzero(original_mask)
    assert sorted(seen) == expected.tolist()
    assert row["sampled_endpoints"] == row["supervised"] == len(expected)
    assert row["skipped_batches"] == row["unsupervised_sampled"] == 0
    assert row["optimizer_steps"] == 3
    assert [batch["supervised"] for batch in row["batches"]] == [4, 4, 2]
    np.testing.assert_array_equal(np.sum([batch["class_counts"] for batch in row["batches"]], axis=0), provenance["train"]["class_counts"])
    np.testing.assert_array_equal(training["y"], original_labels)
    np.testing.assert_array_equal(training["class_valid"], original_mask)
    assert row["validation"]["samples"] == provenance["val"]["supervised"]
    # This fixture includes supervised short/not-ready windows: none may be removed.
    assert (original_mask & ~training["ready"]).any()
    identity = result["identity"]
    assert identity["sampling"]["population_sha256"] == sha256(expected.astype("<i8").tobytes()).hexdigest()
    assert identity["sampling"]["excluded_unsupervised"] == 2
    assert identity["objective"]["class_weights"] == class_weights_from_counts(provenance["train"]["class_counts"], "sqrt_inverse")
    with pytest.raises(ValueError, match="identity"):
        train(*caches, root, config=cfg.model_copy(update={"sampling": "uniform"}), resume=root / "last.pt")
