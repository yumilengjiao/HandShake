import json
from pathlib import Path
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
def test_independent_process_epoch_resume_matches_uninterrupted_exactly(caches, tmp_path, model):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(TrainConfig(model_type=model, max_epochs=3, batch_size=4, patience=10).model_dump()))
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
    for key in a["state_dict"]:
        assert torch.equal(a["state_dict"][key], b["state_dict"][key]), key
        assert torch.equal(a["best_state_dict"][key], b["best_state_dict"][key])
    assert torch.equal(a["rng"]["sampler"], b["rng"]["sampler"])
    manifest = json.loads((interrupted / "run_manifest.json").read_text())
    assert manifest["status"] == "completed" and not manifest["final_test_used"]
    assert manifest["synthetic"] and manifest["best_epoch"] <= 3


def test_all_masked_batches_skip_updates_and_early_ties_keep_first(caches, tmp_path):
    cfg = TrainConfig(model_type="mlp", max_epochs=8, batch_size=1, patience=1, min_delta=1.0)
    result = train(*caches, tmp_path / "run", config=cfg)
    checkpoint = torch.load(tmp_path / "run/last.pt", weights_only=True)
    assert result["early_stopped"] and result["best_epoch"] == 1 and result["epochs"] == 2
    assert checkpoint["history"][0]["skipped_batches"] == 2
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
