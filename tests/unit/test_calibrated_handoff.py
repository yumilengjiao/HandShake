"""Calibrated handoff metadata/byte tests; model tensors are UNUSABLE fixtures."""

import json
import subprocess
import sys

import pytest

from algorithm.bundle import MEMBERS, digest, json_bytes
from algorithm.evaluation.calibration import fit_calibration, save_calibration
from backend.handoff import build_kit, inspect_kit
from tests.unit.test_evaluation_scores import rejection_scores
from tests.unit.test_handoff import inputs as inputs, repin


@pytest.fixture
def calibrated(tmp_path):
    scores = rejection_scores()
    p = scores.provenance
    data = {
        name: getattr(p, name)
        for name in (
            "dataset_sha256",
            "split_sha256",
            "stats_sha256",
            "config_sha256",
            "vocabulary_version",
        )
    }
    data.update(
        synthetic=True,
        val_index_sha256=p.cache_index_sha256,
        val={"endpoint_sha256": p.endpoint_sha256},
    )
    training = {"synthetic": True, "identity": {"data": data}}
    root = tmp_path / "model"
    root.mkdir()
    for name in MEMBERS:
        (root / name).write_bytes(
            json_bytes(training)
            if name == "training_manifest.json"
            else b"UNUSABLE TENSOR FIXTURE - NO REAL MODEL"
        )
    manifest = {
        "format_version": "inference-bundle-v1",
        "status": "uncalibrated_preview_only",
        "files": {
            name: {
                "bytes": (root / name).stat().st_size,
                "sha256": digest((root / name).read_bytes()),
            }
            for name in MEMBERS
        },
    }
    raw = json_bytes(manifest)
    (root / "checksums.json").write_bytes(raw)
    pin = digest(raw)
    scores.provenance = p.model_copy(update={"model_sha256": pin})
    value = fit_calibration(
        scores,
        allow_synthetic=True,
        temperatures=(1.0,),
        thresholds=(0.0, 0.6),
        max_risk=0.0,
        min_coverage=0.3,
        min_accepted=1,
    )
    sidecar = tmp_path / "validation.json"
    sha = save_calibration(value, sidecar)
    return dict(
        model=root,
        model_sha256=pin,
        model_license="synthetic-fixture-only",
        calibration=sidecar,
        calibration_sha256=sha,
    )


def test_v2_sidecar_roundtrip_without_model_execution(inputs, calibrated, tmp_path):
    root = tmp_path / "kit"
    receipt = build_kit(root, **inputs, **calibrated)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format_version"] == "offline-handoff-v2"
    assert manifest["calibration_sha256"] == calibrated["calibration_sha256"]
    assert (root / "calibration/validation.json").read_bytes() == calibrated[
        "calibration"
    ].read_bytes()
    assert (
        root / "model/calibration.json"
    ).read_bytes() == b"UNUSABLE TENSOR FIXTURE - NO REAL MODEL"
    report = inspect_kit(root, expected_sha256=receipt["manifest_sha256"])
    assert report["calibration_status"] == "selected"
    assert report["actions_enabled"] is False and report["launch_verified"] is False
    from backend.workflows.deployment import handoff_assets

    assets = handoff_assets(root)
    assert assets["package"] == root / "model"
    assert assets["calibration"] == root / "calibration/validation.json"
    assert assets["sha256"] == calibrated["model_sha256"]
    assert assets["calibration_sha256"] == calibrated["calibration_sha256"]
    code = (
        "import sys; from backend.handoff import inspect_kit; "
        "inspect_kit(sys.argv[1],expected_sha256=sys.argv[2]); "
        "assert not any(x in sys.modules for x in ('torch','cv2','mediapipe'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(root), receipt["manifest_sha256"]],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "change", ["without_model", "without_pin", "wrong_pin", "symlink", "oversize"]
)
def test_invalid_calibration_fails_before_staging(inputs, calibrated, tmp_path, change):
    if change == "without_model":
        calibrated.update(model=None, model_sha256=None)
    elif change == "without_pin":
        calibrated["calibration_sha256"] = None
    elif change == "wrong_pin":
        calibrated["calibration_sha256"] = "0" * 64
    elif change == "symlink":
        link = tmp_path / "linked.json"
        link.symlink_to(calibrated["calibration"])
        calibrated["calibration"] = link
    else:
        calibrated["calibration"].write_bytes(b" " * (1_048_576 + 1))
    with pytest.raises(ValueError):
        build_kit(tmp_path / "bad", **inputs, **calibrated)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "field", ["model_sha256", "cache_index_sha256", "endpoint_sha256", "dataset_sha256"]
)
def test_rehashed_sidecar_cannot_bind_another_model_or_validation_inventory(
    inputs, calibrated, tmp_path, field
):
    root = tmp_path / "kit"
    build_kit(root, **inputs, **calibrated)
    path = root / "calibration/validation.json"
    value = json.loads(path.read_text())
    value["source"][field] = "f" * 64
    path.write_bytes(json_bytes(value))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["calibration_sha256"] = digest(path.read_bytes())
    (root / "manifest.json").write_bytes(json_bytes(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        inspect_kit(root, expected_sha256=repin(root))


def test_legacy_v1_roundtrip_but_cannot_smuggle_unbound_sidecar(
    inputs, calibrated, tmp_path
):
    root = tmp_path / "kit"
    receipt = build_kit(root, **inputs)
    report = inspect_kit(root, expected_sha256=receipt["manifest_sha256"])
    assert report["calibration_status"] is None
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format_version"] == "offline-handoff-v1"
    path = root / "calibration/validation.json"
    path.parent.mkdir()
    path.write_bytes(calibrated["calibration"].read_bytes())
    manifest["files"]["calibration/validation.json"] = {
        "bytes": path.stat().st_size,
        "sha256": digest(path.read_bytes()),
        "executable": False,
    }
    (root / "manifest.json").write_bytes(json_bytes(manifest))
    with pytest.raises(ValueError, match="unbound calibration"):
        inspect_kit(root, expected_sha256=repin(root))


@pytest.mark.parametrize("name", ["export.json", "calibration-receipt.json"])
def test_rehashed_convenience_receipts_cannot_repoint_assets(
    inputs, calibrated, tmp_path, name
):
    root = tmp_path / "kit"
    build_kit(root, **inputs, **calibrated)
    path = root / name
    value = json.loads(path.read_text())
    value["directory" if name == "export.json" else "path"] = "../outside"
    path.write_bytes(json_bytes(value))
    with pytest.raises(ValueError, match="deployment receipt"):
        inspect_kit(root, expected_sha256=repin(root))


def test_rehashed_execution_authority_is_still_rejected(inputs, calibrated, tmp_path):
    root = tmp_path / "kit"
    build_kit(root, **inputs, **calibrated)
    path = root / "calibration/validation.json"
    value = json.loads(path.read_text())
    value["actions_enabled"] = True
    path.write_bytes(json_bytes(value))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["calibration_sha256"] = digest(path.read_bytes())
    (root / "manifest.json").write_bytes(json_bytes(manifest))
    with pytest.raises(ValueError):
        inspect_kit(root, expected_sha256=repin(root))
