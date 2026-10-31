"""Model preview wire/session tests; any camera payload uses procedural fixture pixels."""

import io
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.classified_preview import ClassifiedCameraSession
from backend.perception.preview import CameraFrame
from shared.inference_protocol import ClassificationNotice
from shared.model_preview_protocol import (
    ModelPreviewCommand,
    ModelPreviewSnapshot,
    read_model_packet,
)
from shared.preview_protocol import read_packet
from tests.unit.test_camera_preview import Clock, Worker


def notice(**changes):
    return ClassificationNotice(
        **(
            {
                "state": "predicted",
                "reason": "ready",
                "model_version": "sha256:" + "a" * 64,
                "model_type": "tcn",
                "vocabulary_version": "ipn-hands-v1",
                "synthetic_training": True,
                "stream_id": "synthetic-test-stream",
                "context_epoch": "1",
                "source_frame_id": "17",
                "source_timestamp_ns": "9007199254740999",
                "track_id": "test-hand",
                "window_length": 32,
                "inference_ns": "500000",
                "logits": tuple(float(i) / 10 for i in range(15)),
            }
            | changes
        )
    )


def test_v3_fixture_roundtrip_and_same_frame_binding():
    data = Path("frontend/native/test/fixtures/model_preview_v3.bin").read_bytes()
    packet = read_model_packet(io.BytesIO(data))
    assert packet.encode() == data
    assert packet.snapshot.classification == notice()
    with pytest.raises(ValueError, match="framing"):
        read_packet(io.BytesIO(data))
    with pytest.raises(ValidationError, match="match this frame"):
        ModelPreviewSnapshot(**(packet.snapshot.model_dump() | {"epoch": "2"}))


@pytest.mark.parametrize(
    "change",
    [
        {"calibrated": True},
        {"calibrated": 0},
        {"logits": (float("nan"),) * 15},
        {"logits": (0.0,) * 5},
        {"state": "warming_up"},
        {"window_length": 1},
        {"vocabulary_version": "unknown"},
        {"inference_ns": None},
    ],
)
def test_prediction_contract_rejects_invalid_scores_or_authority(change):
    with pytest.raises(ValidationError):
        notice(**change)


def test_stop_clears_v3_classification_without_restamping_epoch():
    worker, clock = Worker(), Clock()
    epochs = []

    def factory(context_epoch):
        epochs.append(context_epoch)
        return worker

    session = ClassifiedCameraSession(worker_factory=factory, clock=clock)
    session.accept(
        ModelPreviewCommand(
            version=3,
            session_id=session.session_id,
            command_id="1",
            epoch="0",
            kind="start_camera",
        )
    )
    fixture = read_packet(
        io.BytesIO(Path("frontend/native/test/fixtures/preview_v2.bin").read_bytes())
    )
    worker.frame = CameraFrame(
        17,
        clock.now,
        160,
        120,
        fixture.jpeg,
        fixture.snapshot.hands,
        "processed_hands",
        10.0,
        classification=notice(source_timestamp_ns=str(clock.now)),
    )
    packet = session.snapshot()
    assert epochs == [1] and packet.snapshot.classification.context_epoch == "1"
    clock.now += 200_000_000
    expired = session.snapshot().snapshot.classification
    assert expired.reason == "stale" and not expired.logits
    session.accept(
        ModelPreviewCommand(
            version=3,
            session_id=session.session_id,
            command_id="2",
            epoch="1",
            kind="stop",
        )
    )
    assert session.snapshot().snapshot.classification is None
    worker.done.set()
    stopped = session.snapshot().snapshot
    assert (
        stopped.state == "stopped"
        and stopped.epoch == "2"
        and stopped.model_status == "unavailable"
    )


def test_model_load_failure_is_explicit_no_prediction():
    worker = Worker()
    session = ClassifiedCameraSession(worker_factory=lambda context_epoch: worker)
    session.accept(
        ModelPreviewCommand(
            version=3,
            session_id=session.session_id,
            command_id="1",
            epoch="0",
            kind="start_camera",
        )
    )
    worker.model_failed = True
    worker.error = "preview_error"
    worker.done.set()
    packet = session.snapshot()
    assert packet.snapshot.model_status == "error"
    assert packet.snapshot.classification is None and not packet.jpeg


def test_v3_schema_is_generated_from_contract():
    schema = json.loads(Path("shared/model_preview_protocol.schema.json").read_text())
    assert schema == {
        "command": ModelPreviewCommand.model_json_schema(),
        "snapshot": ModelPreviewSnapshot.model_json_schema(),
    }
