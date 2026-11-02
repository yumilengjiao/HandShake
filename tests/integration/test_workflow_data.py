"""Public-format fixtures exercise default directories and receipts; never train."""

import json
import shutil

import pytest

from algorithm.data.source_archive import file_sha256
from backend.workflows.data import dataset_receipt, prepare
from backend.workflows.download import ANNOTATIONS, LANDMARKS
from backend.workflows.workspace import Settings, WorkflowError, Workspace, write_json
from tests.integration.test_public_data import public_sources  # noqa: F401


@pytest.fixture
def workflow_source(public_sources, tmp_path):
    ann, lm, plan, *_ = public_sources
    root = tmp_path / "workspace"
    root.mkdir()
    settings = Settings.model_validate(
        {"data": {"review_reference": "synthetic-test-only"}}
    )
    ws = Workspace(root, settings)
    raw = ws.path(ws.settings.data.raw)
    raw.mkdir(parents=True)
    shutil.copyfile(ann, raw / ANNOTATIONS)
    shutil.copyfile(lm, raw / LANDMARKS)
    subjects = ws.path(ws.settings.data.subjects)
    subjects.parent.mkdir(parents=True)
    shutil.copyfile(plan, subjects)
    write_json(
        ws.path(ws.settings.data.catalog),
        {
            "schema_version": "public-dataset-sources-v1",
            "datasets": [
                {
                    "id": "ipn-hands",
                    "license": {"id": "CC-BY-4.0"},
                    "files": [
                        {"filename": ANNOTATIONS, "expected_sha256": file_sha256(ann)},
                        {"filename": LANDMARKS, "expected_sha256": file_sha256(lm)},
                    ],
                }
            ],
        },
    )
    return ws


def test_default_prepare_builds_three_splits_and_is_idempotent(workflow_source):
    ws = workflow_source
    first = prepare(ws, expected_videos=3, synthetic=True)
    assert set(first["splits"]) == {"train", "val", "test"}
    assert first["synthetic"] is True
    assert dataset_receipt(ws, allow_synthetic=True) == first
    assert prepare(ws, expected_videos=3, synthetic=True) == first
    with pytest.raises(WorkflowError, match="合成"):
        dataset_receipt(ws)
    imported = json.loads(
        (ws.path(ws.settings.data.imported) / "manifest.json").read_text()
    )
    assert all("\\" not in row["records_path"] for row in imported["recordings"])


def test_prepare_generates_draft_not_guessed_participants(workflow_source):
    ws = workflow_source
    path = ws.path(ws.settings.data.subjects)
    path.unlink()
    with pytest.raises(WorkflowError, match="模板"):
        prepare(ws, expected_videos=3, synthetic=True)
    plan = json.loads(path.read_text())
    assert plan["status"] == "draft" and plan["subject_splits"] == {}
    assert all(not item["subject_id"] for item in plan["videos"].values())
    assert not ws.path(ws.settings.data.imported).exists()
    with pytest.raises(WorkflowError, match="审核"):
        prepare(ws, expected_videos=3, synthetic=True)


def test_bad_source_cannot_produce_a_dataset_receipt(workflow_source):
    ws = workflow_source
    (ws.path(ws.settings.data.raw) / ANNOTATIONS).write_bytes(b"tampered")
    with pytest.raises(WorkflowError, match="校验"):
        prepare(ws, expected_videos=3, synthetic=True)
    assert not (ws.cache / "dataset.json").exists()


def test_prepare_rejects_a_new_configuration_without_overwriting(workflow_source):
    ws = workflow_source
    first = prepare(ws, expected_videos=3, synthetic=True)
    changed = Settings.model_validate(
        {"data": {"review_reference": "synthetic-test-only", "stride": 2}}
    )
    with pytest.raises(WorkflowError, match="不一致"):
        prepare(Workspace(ws.root, changed), expected_videos=3, synthetic=True)
    assert dataset_receipt(ws, allow_synthetic=True) == first


def test_tampered_receipt_does_not_disable_synthetic_protection(workflow_source):
    from algorithm.data.dataset import canonical_digest

    ws = workflow_source
    receipt = prepare(ws, expected_videos=3, synthetic=True)
    receipt["synthetic"] = False
    receipt["preparation"]["synthetic"] = False
    receipt["receipt_sha256"] = canonical_digest(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    )
    write_json(ws.cache / "dataset.json", receipt)
    with pytest.raises(WorkflowError, match="来源"):
        dataset_receipt(ws)


def test_train_receipt_validation_does_not_open_test_samples(workflow_source):
    ws = workflow_source
    prepare(ws, expected_videos=3, synthetic=True)
    (ws.cache / "test" / "features.f32").unlink()
    dataset_receipt(ws, allow_synthetic=True)
    with pytest.raises((ValueError, OSError)):
        dataset_receipt(
            ws, allow_synthetic=True, verify_splits=("train", "val", "test")
        )
