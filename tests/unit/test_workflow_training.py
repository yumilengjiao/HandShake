"""Training command orchestration uses mocked trainers, never optimizes a network."""

from types import SimpleNamespace

import pytest

from algorithm.training.config import TrainConfig
from backend.workflows import training
from backend.workflows.workspace import Settings, Workspace, read_json, write_json


@pytest.fixture
def rig(tmp_path, monkeypatch):
    ws = Workspace(tmp_path, Settings.model_validate({"train": {"auto_export": False}}))
    receipt = {"receipt_sha256": "a" * 64}
    monkeypatch.setattr(
        training, "preflight", lambda *a: (TrainConfig(max_epochs=1), receipt)
    )
    monkeypatch.setattr(training, "dataset_receipt", lambda *a: receipt)
    monkeypatch.setattr(training, "ensure_training", lambda *a: None)
    return ws


def test_dry_run_never_imports_or_invokes_trainer(rig, monkeypatch):
    import algorithm.training.trainer as implementation

    monkeypatch.setattr(
        implementation, "train", lambda *a, **k: pytest.fail("must not train")
    )
    args = SimpleNamespace(command="train", dry_run=True)
    assert training.run_train(rig, args, ["train", "--dry-run"]) == 0
    assert not rig.runs.exists()


def test_new_runs_are_unique_and_latest_is_portable(rig, monkeypatch):
    import algorithm.training.trainer as implementation

    seen = []

    def fake_train(train, val, root, **kwargs):
        assert train == rig.cache / "train" and val == rig.cache / "val"
        root.mkdir()
        seen.append(root)
        write_json(
            root / "run_manifest.json",
            {
                "fixture": "mocked control flow, not training",
                "identity": {"config": kwargs["config"].model_dump()},
            },
        )
        kwargs["progress"](
            {"epoch": 1, "train_loss": 0.0, "validation": {"macro_f1": 0.0}}
        )
        return {"status": "completed", "best_epoch": 1, "best_val_macro_f1": 0.0}

    monkeypatch.setattr(implementation, "train", fake_train)
    for _ in range(2):
        assert (
            training.run_train(
                rig, SimpleNamespace(command="train", dry_run=False), ["train"]
            )
            == 0
        )
    assert seen[0] != seen[1]
    pointer = read_json(rig.latest)
    assert pointer["run"] == seen[1].name
    assert pointer["dataset_receipt_sha256"] == "a" * 64
    assert not (rig.runs / ".train.lock").exists()


def test_failed_training_preserves_pointer_for_resume(rig, monkeypatch):
    import algorithm.training.trainer as implementation

    def fake_train(train, val, root, **kwargs):
        root.mkdir()
        write_json(root / "run_manifest.json", {"fixture": True, "status": "failed"})
        raise RuntimeError("test-only simulated interruption")

    monkeypatch.setattr(implementation, "train", fake_train)
    with pytest.raises(RuntimeError, match="simulated"):
        training.run_train(
            rig, SimpleNamespace(command="train", dry_run=False), ["train"]
        )
    root, _ = training.latest_run(rig)
    assert read_json(root / "run_manifest.json")["status"] == "failed"
    assert not (rig.runs / ".train.lock").exists()


def test_export_after_resume_does_not_reuse_stale_best(rig, monkeypatch):
    import algorithm.bundle as bundle

    root = rig.runs / "fixture-run"
    root.mkdir(parents=True)
    monkeypatch.setattr(training, "_bound_run", lambda _: (root, {}))
    write_json(
        root / "run_manifest.json",
        {"status": "completed", "artifacts": {"best.pt": "b" * 64}},
    )
    old = root / "models" / "old"
    old.mkdir(parents=True)
    write_json(
        root / "export.json",
        {
            "checkpoint_sha256": "a" * 64,
            "directory": "models/old",
            "manifest_sha256": "c" * 64,
        },
    )
    outputs = []

    def fake_export(run, stats, output):
        output.mkdir()
        outputs.append(output)
        return {"manifest_sha256": "d" * 64}

    monkeypatch.setattr(bundle, "export_bundle", fake_export)
    monkeypatch.setattr(bundle, "load_bundle", lambda *a, **k: object())
    receipt = training.export_run(rig)
    assert receipt["checkpoint_sha256"] == "b" * 64
    assert old.is_dir() and training.model_directory(root, receipt) == outputs[0]
    assert training.export_run(rig) == receipt and len(outputs) == 1
