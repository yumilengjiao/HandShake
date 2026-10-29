"""Finite, synthetic T02 smoke chain. No camera, network, model or OS injection.

The feature fixture and event engine below are test doubles, NOT T11 or T17.
"""

from uuid import uuid4

from algorithm.predictor import MockPredictor
from shared.config import ProjectConfig
from shared.contracts import ActionCommand, ActionResult, FeatureWindow, FramePacket, FrameStatus, GESTURE_IDS, Gesture, GestureEvent, ModelPrediction, RuntimeContext, TransformMetadata
from shared.records import LogEnvelope, make_log
from backend.controller import ApplicationController


def prediction_is_current(prediction: ModelPrediction, context: RuntimeContext, now_ns: int, max_age_ns: int) -> bool:
    return (
        prediction.stream_id == context.stream_id
        and prediction.context_epoch == context.context_epoch
        and prediction.timebase == context.timebase
        and context.enabled and not context.suspended
        and prediction.track_id == context.owner_track_id
        and 0 <= now_ns - prediction.source_timestamp_ns <= max_age_ns
        and prediction.inference_timestamp_ns <= now_ns
    )


class MockGestureEngine:
    """Only a scripted single-event adapter. It has no wake/hold/release logic."""

    def update(self, prediction: ModelPrediction, context: RuntimeContext, *, now_ns: int, max_age_ns: int) -> list[GestureEvent]:
        if not prediction_is_current(prediction, context, now_ns, max_age_ns):
            return []
        gesture = list(GESTURE_IDS)[max(range(len(prediction.logits)), key=prediction.logits.__getitem__)]
        if gesture not in {Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT}:
            return []
        return [GestureEvent(
            event_id=str(uuid4()), stream_id=prediction.stream_id, track_id=prediction.track_id,
            context_epoch=prediction.context_epoch, source_frame_id=prediction.source_frame_id,
            source_timestamp_ns=prediction.source_timestamp_ns, gesture_id=gesture,
            candidate_start_timestamp_ns=prediction.source_timestamp_ns,
            commit_timestamp_ns=now_ns, mode=context.mode,
            decision_reason="synthetic_script_only_not_interaction_state_machine", timebase=prediction.timebase,
        )]


class MockBackend:
    def execute(self, command: ActionCommand, now_ns: int) -> ActionResult:
        return ActionResult(
            command_id=command.command_id, accepted=True, injection_status="not_attempted",
            completed_at_ns=now_ns, acknowledgement_level="mock", timebase=command.timebase,
        )


class MockDispatcher:
    """Synchronous mock-only checks. Production queue/races/releases are T18/T24."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.backend = MockBackend()
        self._seen: set[str] = set()

    def dispatch(self, command: ActionCommand, context: RuntimeContext, now_ns: int) -> ActionResult:
        reason = None
        if not context.enabled or context.suspended or context.owner_track_id is None:
            reason = "inactive_context"
        elif (command.stream_id, command.context_epoch, command.timebase) != (context.stream_id, context.context_epoch, context.timebase):
            reason = "stale_context"
        elif not command.issued_at_ns <= now_ns < command.expires_at_ns:
            reason = "expired_or_future_command"
        elif command.target_context != context.target_context or command.target_context not in self.config.platform.allowed_targets:
            reason = "target_not_allowed"
        elif command.action not in self.config.actions.allowlist or not set(command.required_capabilities).issubset(self.config.platform.capabilities):
            reason = "capability_or_action_not_allowed"
        elif command.command_id in self._seen:
            reason = "duplicate_command"
        if reason:
            return ActionResult(command_id=command.command_id, accepted=False, injection_status="not_attempted", error_code=reason, completed_at_ns=now_ns, acknowledgement_level="none", timebase=command.timebase)
        self._seen.add(command.command_id)
        return self.backend.execute(command, now_ns)


def run_synthetic(config: ProjectConfig, *, code_version="unknown") -> list[LogEnvelope]:
    controller = ApplicationController("synthetic-stream", timebase="synthetic", target_context=config.runtime.target_context)
    controller.enable(0)
    controller.request_mock_owner("synthetic-track", 1)
    captured = controller.snapshot()
    frame = FramePacket(
        stream_id=captured.stream_id, frame_id=31, capture_timestamp_ns=1_033_333_323,
        image=[[[0, 0, 0]]], image_size=[1, 1],
        transform_metadata=TransformMetadata(timebase="synthetic"), timebase="synthetic",
    )
    status = FrameStatus(
        stream_id=frame.stream_id, source_frame_id=frame.frame_id,
        source_timestamp_ns=frame.capture_timestamp_ns, perception_status="processed_hands",
        processed_at_ns=frame.capture_timestamp_ns + 1_000, timebase=frame.timebase,
    )
    # Explicit fixture: no claim that pixels were perceived or that 94D features exist.
    window = FeatureWindow(
        stream_id=captured.stream_id, track_id=captured.owner_track_id,
        context_epoch=captured.context_epoch, source_frame_id=frame.frame_id,
        timestamps_ns=[frame.capture_timestamp_ns - (31 - i) * 33_333_333 for i in range(32)],
        features=[[0.0] * config.training.feature_dimension for _ in range(32)],
        ready=True, ready_reason="synthetic_fixture_only", timebase="synthetic",
    )
    prediction = MockPredictor().predict(window)
    now_ns = prediction.inference_timestamp_ns
    events = MockGestureEngine().update(prediction, captured, now_ns=now_ns, max_age_ns=config.interaction.max_prediction_age_ns)
    records = [*controller.changes, status, prediction]
    dispatcher = MockDispatcher(config)
    for event in events:
        action = config.actions.mappings[captured.mode][event.gesture_id]
        if action is None:
            continue
        from shared.contracts import ACTION_CAPABILITIES

        command = ActionCommand(
            command_id=str(uuid4()), source_event_id=event.event_id,
            stream_id=event.stream_id, context_epoch=event.context_epoch,
            action=action, arguments={}, issued_at_ns=now_ns,
            expires_at_ns=now_ns + config.runtime.command_ttl_ns,
            required_capabilities=[ACTION_CAPABILITIES[action]], target_context=captured.target_context,
            timebase=event.timebase,
        )
        # Re-read the controller at dispatch, never replace command's captured context.
        result = dispatcher.dispatch(command, controller.snapshot(), now_ns)
        records.extend([event, command, result])
    controller.stop(now_ns + 1)
    records.append(controller.changes[-1])
    run_id = str(uuid4())
    return [make_log(
        record, run_id=run_id, timestamp_ns=_record_time(record),
        config_sha256=config.digest(), code_version=code_version, synthetic=True,
    ) for record in records]


def _record_time(record) -> int:
    for field in ("timestamp_ns", "completed_at_ns", "issued_at_ns", "commit_timestamp_ns", "inference_timestamp_ns", "processed_at_ns"):
        value = getattr(record, field, None)
        if value is not None:
            return value
    raise ValueError("record has no occurrence timestamp")
