import json

import pytest
from pydantic import ValidationError

from shared.contracts import ActionCommand, ActionResult, FeatureWindow, FramePacket, FrameStatus, HandObservation, ModelPrediction, TransformMetadata
from shared.records import LOG_TYPES


def test_all_logged_contracts_roundtrip(smoke_records):
    for envelope in smoke_records:
        cls = LOG_TYPES[envelope.record_type]
        record = cls.model_validate(envelope.payload)
        assert cls.model_validate_json(record.model_dump_json()) == record


def test_frame_shape_and_rgb_serialization():
    frame = FramePacket(stream_id="s", frame_id=0, capture_timestamp_ns=0, image=[[[0, 255, 0]]], image_size=[1, 1], transform_metadata=TransformMetadata())
    assert FramePacket.model_validate_json(frame.model_dump_json()) == frame
    for change in ({"image_size": [2, 1]}, {"image": [[[0, 0]]]}, {"image": [[[256, 0, 0]]]}, {"image": [[[True, 0, 0]]]}):
        with pytest.raises(ValidationError):
            FramePacket.model_validate(frame.model_dump() | change)


@pytest.mark.parametrize("bad", [-1, 1.5, "1000000", True])
def test_timestamps_are_strict_integer_ns(bad):
    with pytest.raises(ValidationError):
        FrameStatus(stream_id="s", source_frame_id=0, source_timestamp_ns=bad, perception_status="processed_no_hand", processed_at_ns=2_000_000)


def test_units_schema_and_temporal_order_are_checked():
    data = dict(stream_id="s", source_frame_id=0, source_timestamp_ns=100, perception_status="processed_no_hand", processed_at_ns=200)
    for change in ({"time_unit": "ms"}, {"schema_version": "99"}, {"processed_at_ns": 99}, {"extra": 1}):
        with pytest.raises(ValidationError):
            FrameStatus(**(data | change))


def test_processing_statuses_remain_distinct():
    base = dict(stream_id="s", source_frame_id=0, source_timestamp_ns=0)
    statuses = [
        FrameStatus(**base, perception_status="processed_no_hand", processed_at_ns=1),
        FrameStatus(**base, perception_status="not_processed", drop_or_error_reason="overload"),
        FrameStatus(**base, perception_status="error", processed_at_ns=1, drop_or_error_reason="perception_failed"),
    ]
    assert len({s.perception_status for s in statuses}) == 3
    with pytest.raises(ValidationError):
        FrameStatus(**base, perception_status="not_processed")


def test_landmarks_masks_and_missing_quality():
    data = dict(stream_id="s", source_frame_id=0, source_timestamp_ns=0, track_id="t", landmarks_image=[[0.0, 0.0, 0.0] for _ in range(21)], validity_mask=[True] * 21, observation_valid=True, handedness="unknown", quality=1.0, transform_metadata=TransformMetadata())
    assert HandObservation.model_validate_json(HandObservation(**data).model_dump_json()).handedness_score is None
    for change in ({"landmarks_image": [[0.0, 0.0]] * 21}, {"validity_mask": [True] * 20}, {"validity_mask": [True] * 20 + [False]}, {"quality": float("nan")}, {"observation_valid": False}):
        with pytest.raises(ValidationError):
            HandObservation(**(data | change))
    invalid = HandObservation(**(data | {"observation_valid": False, "validity_mask": [False] * 21, "quality_flags": ["occluded"]}))
    assert not invalid.observation_valid


@pytest.mark.parametrize("change", [
    {"features": [[0.0] * 93]}, {"features": [[0.0] * 94, [0.0] * 94]},
    {"timestamps_ns": [2, 1], "features": [[0.0] * 94] * 2},
    {"timestamps_ns": [1, 1], "features": [[0.0] * 94] * 2},
    {"feature_schema_version": "other"}, {"features": [[float("inf")] * 94]},
])
def test_feature_window_dimensions_and_order(change):
    with pytest.raises(ValidationError):
        FeatureWindow(**(dict(stream_id="s", track_id="t", source_frame_id=0, context_epoch=0, timestamps_ns=[1], features=[[0.0] * 94], ready=False, ready_reason="warming_up") | change))


def test_prediction_optional_heads_not_fabricated(smoke_records):
    data = next(r.payload for r in smoke_records if r.record_type == "ModelPrediction")
    pred = ModelPrediction(**data)
    assert pred.intent_score is None and pred.phase_scores is None and pred.calibrated_probabilities is None
    for change in ({"logits": [1.0] * 4}, {"calibrated_probabilities": [1.0] * 5}, {"inference_timestamp_ns": 0}, {"intent_score": 1.1}):
        with pytest.raises(ValidationError):
            ModelPrediction(**(data | change))


@pytest.mark.parametrize("change", [
    {"expires_at_ns": 0}, {"action": "shutdown"}, {"required_capabilities": []},
    {"required_capabilities": ["media"]}, {"control_stream_id": "c"},
    {"arguments": {"unexpected": 1}}, {"update_seq": True},
])
def test_commands_reject_invalid_or_unsafe_fields(smoke_records, change):
    data = next(r.payload for r in smoke_records if r.record_type == "ActionCommand")
    with pytest.raises(ValidationError):
        ActionCommand(**(data | change))


def test_uncertain_injection_is_preserved():
    data = dict(command_id="c", accepted=True, injection_status="uncertain", error_code="timeout_maybe_applied", completed_at_ns=1, acknowledgement_level="none")
    result = ActionResult(**data)
    assert json.loads(result.model_dump_json())["injection_status"] == "uncertain"
    for change in ({"acknowledgement_level": "application"}, {"accepted": False}, {"error_code": None}):
        with pytest.raises(ValidationError):
            ActionResult(**(data | change))


def test_injection_ack_is_not_application_ack():
    result = ActionResult(command_id="c", accepted=True, injection_status="confirmed", completed_at_ns=1, acknowledgement_level="injection")
    assert result.acknowledgement_level != "application"


@pytest.mark.parametrize("action,capability,arguments", [
    ("move_pointer", "pointer", {"x": 0.2, "y": 0.8}),
    ("set_volume", "volume", {"value": 0.5}),
    ("scroll", "scroll", {"delta": -1.0}),
])
def test_continuous_updates_require_identity_and_sequence(action, capability, arguments):
    data = dict(command_id="c", source_event_id="e", stream_id="s", context_epoch=0, action=action, arguments=arguments, issued_at_ns=1, expires_at_ns=2, required_capabilities=[capability], target_context="test-only")
    with pytest.raises(ValidationError, match="continuous"):
        ActionCommand(**data)
    command = ActionCommand(**data, control_stream_id="continuous-1", update_seq=0)
    assert ActionCommand.model_validate_json(command.model_dump_json()) == command
