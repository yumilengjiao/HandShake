import json
from pathlib import Path
import subprocess
import sys

import pytest

from backend.cli import main
from backend.controller import ApplicationController
from backend.mock import MockDispatcher, MockGestureEngine
from shared.contracts import ActionCommand, ModelPrediction
from shared.records import LogEnvelope, read_jsonl, write_jsonl


def context_for(prediction):
    controller = ApplicationController(prediction.stream_id, timebase=prediction.timebase)
    controller.enable(0)
    controller.request_mock_owner(prediction.track_id, 1)
    return controller


def test_synthetic_end_to_end_and_versioned_log(smoke_records, tmp_path):
    path = tmp_path / "smoke.jsonl"
    write_jsonl(path, smoke_records)
    assert read_jsonl(path) == smoke_records
    result = next(r.payload for r in smoke_records if r.record_type == "ActionResult")
    assert result["accepted"] and result["injection_status"] == "not_attempted"
    assert result["acknowledgement_level"] == "mock"
    assert all(r.synthetic and r.timebase == "synthetic" for r in smoke_records)
    assert smoke_records[-1].payload["reason"] == "stop"
    assert len({r.run_id for r in smoke_records}) == 1
    with pytest.raises(FileExistsError):
        write_jsonl(path, smoke_records)


@pytest.mark.parametrize("change", ["stop", "mode", "model", "owner", "suspend", "reconnect"])
def test_old_work_cannot_be_relabeled_after_context_change(smoke_records, config, change):
    prediction = ModelPrediction(**next(r.payload for r in smoke_records if r.record_type == "ModelPrediction"))
    controller = context_for(prediction)
    captured = controller.snapshot()
    assert captured.context_epoch == prediction.context_epoch
    now = prediction.inference_timestamp_ns
    if change == "stop":
        controller.stop(now)
    elif change == "mode":
        controller.change_mode("other", now)
    elif change == "model":
        controller.change_model(now)
    elif change == "owner":
        controller.request_mock_owner("other-track", now)
    elif change == "suspend":
        controller.suspend(now)
    else:
        controller.reconnect("new-stream", now)
    assert captured.context_epoch == prediction.context_epoch
    assert controller.snapshot().context_epoch > captured.context_epoch
    assert MockGestureEngine().update(prediction, controller.snapshot(), now_ns=now, max_age_ns=config.interaction.max_prediction_age_ns) == []


def test_stale_or_future_predictions_do_not_emit(smoke_records, config):
    prediction = ModelPrediction(**next(r.payload for r in smoke_records if r.record_type == "ModelPrediction"))
    context = context_for(prediction).snapshot()
    for now in (0, prediction.source_timestamp_ns + config.interaction.max_prediction_age_ns + 1):
        assert MockGestureEngine().update(prediction, context, now_ns=now, max_age_ns=config.interaction.max_prediction_age_ns) == []


def test_dispatch_rechecks_context_expiry_target_and_deduplicates(smoke_records, config):
    prediction = ModelPrediction(**next(r.payload for r in smoke_records if r.record_type == "ModelPrediction"))
    command = ActionCommand(**next(r.payload for r in smoke_records if r.record_type == "ActionCommand"))
    controller = context_for(prediction)
    dispatcher = MockDispatcher(config)
    assert dispatcher.dispatch(command, controller.snapshot(), command.expires_at_ns).error_code == "expired_or_future_command"
    assert dispatcher.dispatch(command, controller.snapshot(), command.issued_at_ns).accepted
    assert dispatcher.dispatch(command, controller.snapshot(), command.issued_at_ns).error_code == "duplicate_command"
    controller.change_model(command.issued_at_ns)
    assert not dispatcher.dispatch(command, controller.snapshot(), command.issued_at_ns).accepted


@pytest.mark.parametrize("mode", ["collect", "train", "evaluate", "replay", "live"])
def test_unimplemented_modes_fail_before_work(mode, capsys):
    assert main([mode]) == 2
    output = capsys.readouterr()
    assert not output.out and "unavailable" in output.err


def test_cli_works_without_camera_model_or_current_repo(tmp_path):
    result = subprocess.run([sys.executable, "-m", "backend.cli"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    records = [LogEnvelope.model_validate_json(line) for line in result.stdout.splitlines()]
    assert any(r.record_type == "ActionResult" for r in records)


def test_log_reader_rejects_tampering(smoke_records, tmp_path):
    bad = smoke_records[0].model_dump(mode="json")
    bad["payload"]["timebase"] = "monotonic"
    path = tmp_path / "tampered.jsonl"
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="timebase"):
        read_jsonl(path)
