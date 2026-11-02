"""V4 probability telemetry does not grant controls or train a network."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.classified_preview import CalibratedCameraSession
from backend.interaction.public_replay import synthetic_calibration
from backend.perception.preview import CameraFrame, CameraPreviewWorker
from shared.calibrated_preview_protocol import (
    CalibratedPreviewCommand,
    CalibratedPreviewSnapshot,
    ProbabilityNotice,
    read_calibrated_packet,
)
from shared.classes import vocabulary_for
from shared.model_preview_protocol import read_model_packet
from tests.unit.test_camera_preview import Clock, Worker
from tests.unit.test_model_preview import notice


def fixture():
    return Path("frontend/native/test/fixtures/calibrated_preview_v4.bin").read_bytes()


def test_roundtrip_schema_and_old_protocol_rejects_v4():
    packet = read_calibrated_packet(io.BytesIO(fixture()))
    assert packet.encode() == fixture()
    assert packet.snapshot.probability.rejection_status == "rejected"
    assert not packet.snapshot.probability.actions_enabled
    with pytest.raises(ValueError):
        read_model_packet(io.BytesIO(fixture()))
    schema = json.loads(
        Path("shared/calibrated_preview_protocol.schema.json").read_text()
    )
    assert schema == {
        "command": CalibratedPreviewCommand.model_json_schema(),
        "snapshot": CalibratedPreviewSnapshot.model_json_schema(),
    }


@pytest.mark.parametrize(
    "change",
    [
        {"actions_enabled": True},
        {"actions_enabled": 0},
        {"calibrated": 1},
        {"status": "frozen"},
        {"temperature": 0.0},
        {"temperature": float("nan")},
        {"probabilities": (0.2,) * 15},
        {"probabilities": (float("nan"),) * 15},
        {"rejection_status": "accepted"},
        {"model_version": "wrong"},
        {"synthetic": False},
        {"threshold": None},
    ],
)
def test_false_authority_bad_probability_and_source_mismatch_rejected(change):
    packet = read_calibrated_packet(io.BytesIO(fixture()))
    data = packet.snapshot.model_dump()
    data["probability"].update(change)
    with pytest.raises(ValueError):
        CalibratedPreviewSnapshot.model_validate(data)


def test_stale_and_stopped_frames_clear_probability(tmp_path):
    calibration, sha = synthetic_calibration(tmp_path)
    worker, clock = Worker(), Clock()
    worker.calibration, worker.calibration_sha256 = calibration, sha
    session = CalibratedCameraSession(
        worker_factory=lambda context_epoch: worker, clock=clock
    )
    session.accept(
        CalibratedPreviewCommand(
            version=4,
            session_id=session.session_id,
            command_id="1",
            epoch="0",
            kind="start_camera",
        )
    )
    base = read_calibrated_packet(io.BytesIO(fixture()))
    worker.frame = CameraFrame(
        17,
        clock.now,
        160,
        120,
        base.jpeg,
        base.snapshot.hands,
        "processed_hands",
        10.0,
        classification=notice(
            source_timestamp_ns=str(clock.now),
            model_version="sha256:" + calibration.source.model_sha256,
        ),
    )
    assert session.snapshot().snapshot.probability is not None
    clock.now += 200_000_000
    value = session.snapshot().snapshot
    assert value.classification.reason == "stale" and value.probability is None
    session.accept(
        CalibratedPreviewCommand(
            version=4,
            session_id=session.session_id,
            command_id="2",
            epoch="1",
            kind="stop",
        )
    )
    assert session.snapshot().snapshot.probability is None


def test_invalid_calibration_is_rejected_before_any_camera_or_detector_access(
    tmp_path, monkeypatch
):
    calibration, sha = synthetic_calibration(tmp_path)
    s = calibration.source
    bundle = SimpleNamespace(
        manifest_sha256=s.model_sha256,
        synthetic_training=True,
        vocabulary=vocabulary_for(s.vocabulary_version),
        normalizer=SimpleNamespace(
            stats=SimpleNamespace(
                dataset_hash=s.dataset_sha256, split_hash=s.split_sha256
            ),
            stats_hash=s.stats_sha256,
            config=SimpleNamespace(config_hash=s.config_sha256),
        ),
        training_identity={
            "data": {
                "synthetic": True,
                "val_index_sha256": s.cache_index_sha256,
                "val": {"endpoint_sha256": s.endpoint_sha256},
            }
        },
    )
    monkeypatch.setattr("algorithm.bundle.load_bundle", lambda *a, **k: bundle)

    def forbidden():
        pytest.fail("camera/detector must not be touched")

    worker = CameraPreviewWorker(
        model_package="unused",
        model_sha256=s.model_sha256,
        calibration_path=tmp_path / "calibration.json",
        calibration_sha256="0" * 64,
        detector_factory=forbidden,
        source_factory=forbidden,
    )
    worker._run()
    assert (
        worker.model_failed and worker.done.is_set() and worker.error == "preview_error"
    )


def test_infeasible_rejection_can_show_probabilities_but_never_accept():
    p = read_calibrated_packet(io.BytesIO(fixture())).snapshot.probability
    data = p.model_dump() | {"threshold": None, "rejection_status": "unavailable"}
    assert ProbabilityNotice.model_validate(data).rejection_status == "unavailable"


def test_launcher_clears_inherited_asset_selection(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "native_launcher", "scripts/run_native.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    executable = tmp_path / "native"
    executable.write_text("not executed")
    for key in (
        "HANDSHAKE_MODEL_PACKAGE",
        "HANDSHAKE_MODEL_SHA256",
        "HANDSHAKE_CALIBRATION",
        "HANDSHAKE_CALIBRATION_SHA256",
        "HANDSHAKE_PERCEPTION_MODEL",
        "HANDSHAKE_SHADOW_SESSION",
        "HANDSHAKE_ALLOW_SYNTHETIC_MODEL",
    ):
        monkeypatch.setenv(key, "must-not-inherit")
    captured = {}
    monkeypatch.setattr(
        module.subprocess, "call", lambda cmd, **kw: captured.update(kw) or 0
    )
    assert module.main(["--bundle", str(executable)]) == 0
    assert not any("must-not-inherit" == v for v in captured["env"].values())
    with pytest.raises(SystemExit):
        module.main(["--calibration", "missing"])
