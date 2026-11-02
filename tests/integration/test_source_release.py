"""Project-reviewed frame release: no network, images, video or neural training."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import pytest

from algorithm.data.ipn_hands import VideoLabels
from algorithm.data.source_archive import file_sha256
from backend.importing.ipn_hands import validate_subject_plan
from backend.importing.source_release import (
    CAPTURE_GROUPING,
    capture_plan,
    load_release,
    resolve_source_entries,
    verify_release_annotations,
)
from backend.workflows.workspace import Settings, Workspace, WorkflowError
from tests.integration.test_public_data import public_sources  # noqa: F401
from tests.integration.test_workflow_data import workflow_source  # noqa: F401


def real_source():
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "data/sources.json").read_text(encoding="utf-8"))[
        "datasets"
    ][0]


def test_project_review_does_not_erase_or_weaken_publisher_pins():
    source = real_source()
    resolved = resolve_source_entries(source)
    original = source["files"][0]
    chosen = resolved[original["filename"]]
    assert original["expected_sha256"].startswith("4686")
    assert chosen["published_sha256"] == original["expected_sha256"]
    assert chosen["expected_sha256"].startswith("520d")
    source.pop("project_release")
    assert (
        resolve_source_entries(source)[original["filename"]]["expected_sha256"]
        == original["expected_sha256"]
    )


@pytest.mark.parametrize(
    "problem", ["review_hash", "review_id", "upstream", "rewritten_published_hash"]
)
def test_unreviewed_source_change_is_rejected(problem):
    source = real_source()
    if problem == "review_hash":
        source["project_release"]["sha256"] = "0" * 64
    if problem == "review_id":
        source["project_release"]["id"] = "unknown"
    if problem == "upstream":
        source["upstream_commit"] = "other"
    if problem == "rewritten_published_hash":
        source["files"][0]["expected_sha256"] = "520d" + "0" * 60
    with pytest.raises(ValueError):
        resolve_source_entries(source)


def test_fixed_release_plan_preserves_groups_test_and_is_order_independent():
    release = load_release(real_source()["project_release"])
    videos = {
        name: VideoLabels(name, item["source_split"], item["frames"], ())
        for name, item in release["sequences"].items()
    }
    plan = capture_plan(videos, release)
    assert validate_subject_plan(plan, videos) == plan
    assert capture_plan(dict(reversed(list(videos.items()))), release) == plan
    roles = {
        name: plan["subject_splits"][item["subject_id"]]
        for name, item in plan["videos"].items()
    }
    assert {
        role: list(roles.values()).count(role) for role in ("train", "val", "test")
    } == {"train": 116, "val": 32, "test": 52}
    assert len(plan["subject_splits"]) == 50
    assert {name for name, role in roles.items() if role == "test"} == {
        name for name, v in videos.items() if v.source_split == "test"
    }
    key = next(iter(videos))
    videos[key] = replace(videos[key], frame_count=1)
    with pytest.raises(ValueError, match="timing/split"):
        capture_plan(videos, release)


def test_member_review_cannot_be_replaced_by_a_matching_archive_hash(tmp_path):
    archive = tmp_path / "annotations.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("classes.txt", b"fixture")
    release = {
        "archives": {
            "IPN_HandS_annotations.zip": {
                "accepted_sha256": file_sha256(archive),
                "members": {
                    "classes.txt": {
                        "bytes": 7,
                        "sha256": sha256(b"fixture").hexdigest(),
                    }
                },
            }
        }
    }
    verify_release_annotations(archive, release)
    release["archives"]["IPN_HandS_annotations.zip"]["members"]["classes.txt"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="content changed"):
        verify_release_annotations(archive, release)


def fixture_release(ws):
    from backend.workflows.download import ANNOTATIONS
    from algorithm.data.ipn_hands import read_annotations
    from algorithm.data.source_archive import VerifiedArchive

    archive = ws.path(ws.settings.data.raw) / ANNOTATIONS
    with VerifiedArchive(archive, expected_sha256=file_sha256(archive)) as z:
        videos = read_annotations(z, expected_videos=3)
        members = {
            n: {
                "bytes": len(z.read_bytes(n)),
                "sha256": sha256(z.read_bytes(n)).hexdigest(),
            }
            for n in z.members
        }
    return {
        "release_id": "synthetic-review-only",
        "review_document": "synthetic-test-only",
        "grouping": {"validation_groups": 1},
        "archives": {
            ANNOTATIONS: {"accepted_sha256": file_sha256(archive), "members": members}
        },
        "sequences": {
            name: {
                "frames": v.frame_count,
                "source_split": v.source_split,
                "capture_group": "capture-group:" + name,
            }
            for name, v in videos.items()
        },
    }


def test_release_prepare_needs_no_handwritten_subjects_and_retains_scope(
    workflow_source, monkeypatch
):
    from backend.workflows import data

    ws = Workspace(workflow_source.root, Settings())
    ws.path(ws.settings.data.subjects).unlink()
    release = fixture_release(ws)
    entries = data.catalog(ws)
    entries["IPN_HandS_annotations.zip"]["project_release"] = {
        "id": "synthetic-review-only",
        "sha256": "a" * 64,
    }
    monkeypatch.setattr(data, "catalog", lambda _: entries)
    monkeypatch.setattr(data, "load_release", lambda _: release)
    receipt = data.prepare(ws, expected_videos=3, synthetic=True)
    assert receipt["grouping"] == CAPTURE_GROUPING
    assert not ws.path(ws.settings.data.subjects).exists()
    assert data.dataset_receipt(ws, allow_synthetic=True) == receipt
    assert data.prepare(ws, expected_videos=3, synthetic=True) == receipt
    from backend.importing.features import build_import_cache

    with pytest.raises(ValueError, match="v3"):
        build_import_cache(
            ws.path(ws.settings.data.imported),
            ws.root / "legacy-cache",
            split="train",
            dataset_version="fixture",
            review_reference="fixture-only",
        )
    from algorithm.training.inputs import load_training_inputs

    assert (
        load_training_inputs(ws.cache / "train", ws.cache / "val")[2]["grouping"]
        == CAPTURE_GROUPING
    )
    from backend.workflows.workspace import write_json
    from algorithm.data.dataset import canonical_digest

    receipt.pop("grouping")
    receipt["receipt_sha256"] = canonical_digest(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    )
    write_json(ws.cache / "dataset.json", receipt)
    with pytest.raises(WorkflowError, match="范围"):
        data.dataset_receipt(ws, allow_synthetic=True)
